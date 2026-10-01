# ============================================================
# ingestion.py — Multimodal Video RAG Ingestion Engine
# Architecture: V = f(Visual, Audio, Temporal)
# ============================================================

import os
import cv2
import time
import whisper
import chromadb
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────
CHUNK_SIZE    = 5        # seconds per chunk
OVERLAP       = 1        # overlap between chunks (sliding window)
KEYFRAMES_DIR = "keyframes"
DATA_DIR      = "data"
DB_DIR        = "chroma_db"
COLLECTION    = "video_chunks_v2"   # NEW — multimodal collection
VISUAL_ALPHA  = 0.5                  # α: weight of visual vs audio in fusion
EMBED_DIM     = 512                  # CLIP output dimension


# ── Step 1: Temporal Chunking ─────────────────────────────────
def extract_chunks(video_path: str) -> list:
    """
    Sliding window temporal chunking.
    Extracts one keyframe per chunk (at midpoint).
    """
    print(f"\n[1] TEMPORAL CHUNKING — Reading: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    print(f"    FPS={fps:.0f} | Duration={duration:.1f}s | Frames={total_frames}")
    os.makedirs(KEYFRAMES_DIR, exist_ok=True)

    chunks = []
    start  = 0.0

    while start < duration - CHUNK_SIZE + OVERLAP:
        end = min(start + CHUNK_SIZE, duration)
        mid = (start + end) / 2.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(
                KEYFRAMES_DIR,
                f"frame_{start:.1f}_{end:.1f}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            chunks.append({
                "start":      round(start, 2),
                "end":        round(end,   2),
                "mid":        round(mid,   2),
                "frame_path": frame_path
            })

        start += (CHUNK_SIZE - OVERLAP)

    cap.release()
    print(f"    ✓ {len(chunks)} chunks created (chunk={CHUNK_SIZE}s, overlap={OVERLAP}s)")
    return chunks


# ── Step 2: Load All Models ───────────────────────────────────
def load_models():
    """
    Load three models:
      - Whisper: Speech-to-Text (Audio Track)
      - CLIP:    Image encoder — maps frames to 512-dim visual space
      - MiniLM:  Multilingual text encoder — maps transcripts to 384-dim
    """
    print("\n[2] LOADING MODELS...")
    
    whisper_model = whisper.load_model("base")
    print("    ✓ Whisper base — Audio Track (ASR)")
    
    clip_model = SentenceTransformer("clip-ViT-B-32")
    print("    ✓ CLIP ViT-B/32 — Visual Track (512-dim)")
    
    text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("    ✓ Multilingual MiniLM — Audio Text Track (384-dim)")
    
    return whisper_model, clip_model, text_model


# ── Step 3: Audio Track — Transcription ───────────────────────
def transcribe_chunk(whisper_model, video_path: str,
                     start: float, end: float, language: str = "ar") -> str:
    """Extract audio from chunk and transcribe with Whisper."""
    temp_audio = f"temp_{start:.1f}.wav"
    os.system(
        f'ffmpeg -y -ss {start} -to {end} -i "{video_path}" '
        f'-ac 1 -ar 16000 -vn "{temp_audio}" -loglevel quiet'
    )
    try:
        result = whisper_model.transcribe(temp_audio, fp16=False, language=language)
        transcript = result["text"].strip()
    except Exception:
        transcript = ""
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
    return transcript


# ── Step 4a: Visual Track — CLIP Image Encoding ───────────────
def encode_visual(clip_model, frame_path: str) -> np.ndarray:
    """
    Encode keyframe image → normalized 512-dim CLIP vector.
    CLIP maps images into a semantic visual space.
    """
    try:
        img = Image.open(frame_path).convert("RGB")
        vec = clip_model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)
    except Exception:
        return np.zeros(EMBED_DIM, dtype=np.float32)


# ── Step 4b: Audio Track — Multilingual Text Encoding ─────────
def encode_audio_text(text_model, transcript: str) -> np.ndarray:
    """
    Encode transcript → normalized 384-dim multilingual vector.
    Supports Arabic, English, and 50+ languages.
    """
    if not transcript or len(transcript.strip()) < 3:
        return np.zeros(384, dtype=np.float32)
    vec = text_model.encode(transcript, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


# ── Step 4c: Joint Fusion — V = f(Visual, Audio, Temporal) ────
def fuse_embeddings(visual_vec: np.ndarray,
                    audio_vec: np.ndarray,
                    alpha: float = VISUAL_ALPHA) -> list:
    """
    Joint Understanding Fusion:

      fused = α · visual_norm + (1-α) · audio_projected_norm

    - visual_vec:  512-dim CLIP image embedding
    - audio_vec:   384-dim multilingual text embedding
    - Projection:  zero-pad audio from 384 → 512 dims
    - Result:      unified 512-dim multimodal vector

    Mathematical formulation:
      V = f(Visual, Audio, Temporal)
      where Temporal is encoded as chunk metadata (start/end timestamps)
    """
    # Project audio embedding from 384 to 512 by zero-padding
    audio_projected = np.zeros(EMBED_DIM, dtype=np.float32)
    audio_projected[:len(audio_vec)] = audio_vec

    # Weighted fusion
    fused = alpha * visual_vec + (1.0 - alpha) * audio_projected

    # Re-normalize the fused multimodal vector
    norm = np.linalg.norm(fused)
    if norm > 1e-9:
        fused = fused / norm

    return fused.tolist()


# ── Step 5: Initialize Vector DB ──────────────────────────────
def init_db(collection_name: str = COLLECTION) -> chromadb.Collection:
    """Create a fresh ChromaDB collection with cosine similarity."""
    print(f"\n[3] INITIALIZING VECTOR DB (ChromaDB)...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Drop old collection if exists (clean re-index)
    try:
        client.delete_collection(collection_name)
        print(f"    Old collection '{collection_name}' dropped.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"    ✓ Collection '{collection_name}' created (cosine similarity, {EMBED_DIM}-dim)")
    return collection


# ── Main Ingestion Pipeline ────────────────────────────────────
def ingest_video(video_path: str, video_id: str = "video_01",
                 language: str = "ar") -> chromadb.Collection:
    """
    Full multimodal ingestion pipeline:
    Video → Chunks → [Visual Track + Audio Track] → Fusion → ChromaDB
    """
    t_total = time.time()

    # Phase 1: Temporal Chunking
    chunks = extract_chunks(video_path)

    # Phase 2: Load Models
    whisper_model, clip_model, text_model = load_models()

    # Phase 3: Initialize DB
    collection = init_db()

    # Phase 4: Process Each Chunk (Visual + Audio + Fusion)
    print(f"\n[4] MULTIMODAL PROCESSING — {len(chunks)} chunks...")
    print(f"    Visual α={VISUAL_ALPHA} | Audio α={1-VISUAL_ALPHA} | Dim={EMBED_DIM}")

    ids, embeddings, metadatas, documents = [], [], [], []

    t_ingest = time.time()
    for i, chunk in enumerate(chunks):
        print(
            f"    [{i+1:02d}/{len(chunks)}] Chunk {chunk['start']}s→{chunk['end']}s",
            end="\r"
        )

        # Audio Track: Whisper ASR
        transcript = transcribe_chunk(
            whisper_model, video_path,
            chunk["start"], chunk["end"], language
        )

        # Audio Text Embedding (multilingual 384-dim)
        audio_vec = encode_audio_text(text_model, transcript)

        # Visual Track: CLIP Frame Encoding (512-dim)
        visual_vec = encode_visual(clip_model, chunk["frame_path"])

        # Joint Fusion: V = f(Visual=512, Audio=384→512, Temporal=metadata)
        fused_vec = fuse_embeddings(visual_vec, audio_vec, alpha=VISUAL_ALPHA)

        # Collect
        ids.append(f"{video_id}_chunk_{i:04d}")
        embeddings.append(fused_vec)
        documents.append(transcript)
        metadatas.append({
            "video_id":        video_id,
            "video_path":      video_path,
            "start_timestamp": chunk["start"],
            "end_timestamp":   chunk["end"],
            "transcript":      transcript,
            "frame_path":      chunk["frame_path"],
            "chunk_index":     i,
            "has_visual":      str(True),
            "has_audio":       str(len(transcript.strip()) > 3),
        })

    # Batch insert into ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    ingest_time = time.time() - t_ingest
    total_time  = time.time() - t_total

    print(f"\n\n{'='*58}")
    print(f"  ✅  MULTIMODAL INGESTION COMPLETE")
    print(f"{'='*58}")
    print(f"  Video ID         : {video_id}")
    print(f"  Chunks indexed   : {len(chunks)}")
    print(f"  Embedding dim    : {EMBED_DIM} (fused multimodal)")
    print(f"  Fusion formula   : α·Visual + (1-α)·Audio_projected")
    print(f"  Fusion weights   : Visual={VISUAL_ALPHA} | Audio={1-VISUAL_ALPHA}")
    print(f"  Processing time  : {ingest_time:.1f}s ({ingest_time/len(chunks):.1f}s/chunk)")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  DB location      : ./{DB_DIR}/")
    print(f"{'='*58}")

    return collection


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH = os.path.join(DATA_DIR, "test_video.mp4")
    collection = ingest_video(
        video_path=VIDEO_PATH,
        video_id="test_video",
        language="ar"
    )
    print(f"\n[✓] Total items in DB: {collection.count()}")
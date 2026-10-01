# src/videorag/ingestion/pipeline.py — Ingestion Orchestrator (Optimized)
from __future__ import annotations
import time
from PIL import Image
from videorag.config import LANGUAGE
from videorag.logger import get_logger
from videorag.ingestion.chunker import extract_chunks
from videorag.ingestion.transcriber import Transcriber
from videorag.ingestion.encoder import MultimodalEncoder
from videorag.database.vector_store import VectorStore

log = get_logger(__name__)


def ingest_video(
    video_path: str,
    video_id:   str = "video_01",
    language:   str = LANGUAGE,
) -> VectorStore:
    """
    Full multimodal ingestion pipeline (optimized).

    Optimizations vs naive approach:
        - Whisper: 1 call for full video  (was: N calls, one per chunk)
        - CLIP:    1 batch call for all frames (was: N calls, one per frame)

    Steps:
        1. Temporal chunking   — sliding window (5s / 1s overlap)
        2. Audio transcription — ONE Whisper call on full audio
        3. Visual encoding     — CLIP batch encode all keyframes at once
        4. Joint fusion        — V = f(Visual, Audio, Temporal) per chunk
        5. Batch indexing      — ChromaDB cosine store

    Args:
        video_path: Path to the raw video file.
        video_id:   Unique string identifier for this video.
        language:   Whisper ASR language code.

    Returns:
        Populated VectorStore instance.
    """
    t_start = time.perf_counter()
    log.info(f"=== INGESTION START | video_id={video_id} ===")

    # ── Step 1: Temporal Chunking ─────────────────────────────────
    chunks = extract_chunks(video_path)
    if not chunks:
        raise RuntimeError(
            f"No chunks extracted from: {video_path}. "
            "Ensure the file is a valid video longer than 5 seconds."
        )
    log.info(f"Step 1 complete — {len(chunks)} chunks")

    # ── Step 2: Full-Video Whisper Transcription (1 call) ─────────
    t_asr = time.perf_counter()
    transcriber = Transcriber()
    segments    = transcriber.transcribe_full(video_path, language=language)
    log.info(f"Step 2 complete — ASR in {time.perf_counter()-t_asr:.1f}s")

    # ── Step 3: Batch CLIP Visual Encoding ────────────────────────
    t_clip = time.perf_counter()
    encoder = MultimodalEncoder()

    # Load all keyframe images at once
    images = []
    for chunk in chunks:
        try:
            images.append(Image.open(chunk.frame_path).convert("RGB"))
        except Exception:
            from PIL import Image as PILImage
            images.append(PILImage.new("RGB", (224, 224), color=(0, 0, 0)))

    # Batch encode — one GPU/CPU pass for ALL frames
    import numpy as np
    visual_vecs = encoder.clip_image.encode(
        images,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    ).astype(np.float32)

    log.info(f"Step 3 complete — CLIP batch in {time.perf_counter()-t_clip:.1f}s")

    # ── Step 4 & 5: Dual-Vector Indexing ──────────────────────────
    store = VectorStore()
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        transcript = transcriber.get_transcript_for_chunk(
            segments, chunk.start, chunk.end
        )

        base_meta = {
            "video_id":        video_id,
            "chunk_id":        f"{video_id}_chunk_{chunk.index:04d}",
            "video_path":      str(video_path),
            "start_timestamp": chunk.start,
            "end_timestamp":   chunk.end,
            "transcript":      transcript,
            "frame_path":      chunk.frame_path,
            "chunk_index":     chunk.index,
        }

        # 1. Store pure visual vector
        ids.append(f"{video_id}_chunk_{chunk.index:04d}_v")
        embeddings.append(visual_vecs[i].tolist())
        documents.append(transcript)
        
        v_meta = base_meta.copy()
        v_meta["modality"] = "visual"
        metadatas.append(v_meta)

        # 2. Store pure audio vector (if text exists)
        if len(transcript.strip()) > 3:
            # Must pad 384 to 512 because ChromaDB requires same dim for all items
            audio_raw = encoder.encode_audio(transcript)
            audio_vec = np.zeros(512, dtype=np.float32)
            audio_vec[: len(audio_raw)] = audio_raw
            
            ids.append(f"{video_id}_chunk_{chunk.index:04d}_a")
            embeddings.append(audio_vec.tolist())
            documents.append(transcript)
            
            a_meta = base_meta.copy()
            a_meta["modality"] = "audio"
            metadatas.append(a_meta)

    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    elapsed = time.perf_counter() - t_start
    log.info(
        f"=== INGESTION COMPLETE | chunks={len(chunks)} (dual-vectors) | "
        f"total={elapsed:.1f}s ({elapsed/len(chunks):.1f}s/chunk) ==="
    )
    return store

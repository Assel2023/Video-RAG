# src/videorag/ingestion/pipeline.py — Ingestion Orchestrator (Optimized)
from __future__ import annotations
import time
import numpy as np
from pathlib import Path
from PIL import Image
from videorag.config import COLLECTION_NAME, LANGUAGE
from videorag.logger import get_logger
from videorag.ingestion.chunker import extract_chunks
from videorag.ingestion.transcriber import Transcriber
from videorag.ingestion.encoder import MultimodalEncoder
from videorag.database.vector_store import VectorStore

log = get_logger(__name__)


def ingest_video(
    video_path: str,
    video_id:   str  = "video_01",
    language:   str | None = None,   # Bug 8 fix: None = auto-detect language
    encoder:    MultimodalEncoder | None = None,  # reuse server's encoder if provided
    transcriber: Transcriber | None = None,       # reuse server's transcriber if provided
    store: VectorStore | None = None,
    source_filename: str | None = None,
) -> tuple[VectorStore, int]:
    """
    Full multimodal ingestion pipeline (optimized).

    Optimizations:
        - Whisper: 1 call for full video (single-pass)
        - SigLIP 2: 1 batched call for all frames
        - Models:  reuse server-loaded encoder/transcriber (no reload per upload)

    Bug fixes:
        - Bug 1: frame filenames include video_id (no cross-video collision)
        - Bug 4: loop captures tail of video correctly
        - Bug 8: language=None lets Whisper auto-detect

    Args:
        video_path:  Path to the raw video file.
        video_id:    Unique string identifier for this video.
        language:    Whisper ASR language code, or None for auto-detect.
        encoder:     Optional pre-loaded MultimodalEncoder (avoids reload).
        transcriber: Optional pre-loaded Transcriber (avoids reload).

    Returns:
        Tuple of (VectorStore instance, number of video chunks indexed).
    """
    t_start = time.perf_counter()
    log.info(f"=== INGESTION START | video_id={video_id} ===")

    # ── Step 1: Temporal Chunking ─────────────────────────────────
    # Bug 1 fix: pass video_id so keyframe filenames are unique per video
    chunks = extract_chunks(video_path, video_id=video_id)
    if not chunks:
        raise RuntimeError(
            f"No chunks extracted from: {video_path}. "
            "Ensure the file is a valid video with at least 2 seconds of content."
        )
    log.info(f"Step 1 complete — {len(chunks)} chunks")

    # ── Step 2: Full-Video Whisper Transcription (1 call) ─────────
    t_asr = time.perf_counter()
    if transcriber is None:
        transcriber = Transcriber()
    # Bug 8 fix: language=None → Whisper auto-detects (Arabic, English, etc.)
    segments = transcriber.transcribe_full(video_path, language=language)
    log.info(f"Step 2 complete — ASR in {time.perf_counter()-t_asr:.1f}s | "
             f"language={'auto' if language is None else language} | {len(segments)} segments")

    # ── Step 3: Batch SigLIP 2 Visual Encoding ────────────────────
    t_visual = time.perf_counter()
    if encoder is None:
        encoder = MultimodalEncoder()

    images = []
    for chunk in chunks:
        try:
            images.append(Image.open(chunk.frame_path).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))

    visual_vecs = encoder.encode_visual_batch(images, batch_size=32)
    log.info(f"Step 3 complete — SigLIP 2 batch in {time.perf_counter()-t_visual:.1f}s")

    # ── Step 4: Delete old vectors for this video_id (Bug 2 fix) ──
    store = store or VectorStore(
        collection_name=COLLECTION_NAME,
        visual_dim=encoder.visual_dim,
    )
    store.delete_by_video_id(video_id)

    # ── Step 5: Dual-Vector Indexing ──────────────────────────────
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        transcript = transcriber.get_transcript_for_chunk(
            segments, chunk.start, chunk.end
        )

        chunk_id  = f"{video_id}_chunk_{chunk.index:04d}"
        base_meta = {
            "video_id":        video_id,
            "chunk_id":        chunk_id,
            "video_path":      str(video_path),
            "source_filename": source_filename or Path(video_path).name,
            "start_timestamp": chunk.start,
            "end_timestamp":   chunk.end,
            "transcript":      transcript,
            "frame_path":      chunk.frame_path,
            "chunk_index":     chunk.index,
            "has_audio":       str(len(transcript.strip()) > 3).lower(),
        }

        # One Qdrant point per chunk with independently sized named vectors.
        chunk_vectors = {"visual": visual_vecs[i].tolist()}
        documents.append(transcript)

        # Keep the audio vector at its native 384 dimensions.
        if len(transcript.strip()) > 3:
            audio_raw = encoder.encode_audio(transcript)
            chunk_vectors["audio"] = audio_raw.tolist()

        ids.append(chunk_id)
        embeddings.append(chunk_vectors)
        metadatas.append(base_meta)

    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    elapsed = time.perf_counter() - t_start
    log.info(
        f"=== INGESTION COMPLETE | video_chunks={len(chunks)} | "
        f"vectors_stored={len(ids)} | total={elapsed:.1f}s ==="
    )
    # Return store AND chunk count (Bug 3 fix: caller uses this for accurate status)
    return store, len(chunks)

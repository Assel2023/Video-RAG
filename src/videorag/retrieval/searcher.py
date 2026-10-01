# src/videorag/retrieval/searcher.py — Advanced Multimodal Search Orchestrator
from __future__ import annotations
import os
import math
import time
import numpy as np
from videorag.ingestion.encoder import MultimodalEncoder
from videorag.database.vector_store import VectorStore
from videorag.logger import get_logger

log = get_logger(__name__)


def calibrate_visual(raw_score: float) -> float:
    """
    CLIP cross-modal similarities are usually very low.
    Baseline ~ 0.15. Perfect match ~ 0.35.
    """
    score = (raw_score - 0.18) / (0.32 - 0.18)
    return float(np.clip(score, 0.0, 1.0))

def calibrate_audio(raw_score: float) -> float:
    """
    MiniLM text-text similarities are usually much higher.
    Baseline ~ 0.40. Perfect match ~ 0.85.
    """
    score = (raw_score - 0.45) / (0.85 - 0.45)
    return float(np.clip(score, 0.0, 1.0))

class Searcher:
    """
    Production-grade multimodal search orchestrator (Dual-Vector Retrieval).

    Searches both purely visual embeddings and purely textual embeddings
    in parallel, ensuring zero "dilution" between modalities.
    """

    def __init__(self):
        log.info("Initializing Searcher...")
        self.encoder = MultimodalEncoder()
        self.store   = VectorStore()
        log.info("Searcher ready")

    def search(
        self,
        query:    str,
        top_k:    int = 3,
        video_id: str | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        log.info(f"Query: '{query}'")

        # 1. Visual Search (CLIP Multilingual -> DB pure visual vectors)
        clip_vec = self.encoder.clip_text.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        where_visual = {"modality": "visual"}
        if video_id:
            where_visual = {"$and": [{"video_id": video_id}, {"modality": "visual"}]}
            
        hits_visual = self.store.search(clip_vec.tolist(), top_k=top_k * 2, where=where_visual)

        # 2. Audio Search (MiniLM Multilingual -> DB pure audio vectors padded to 512)
        text_raw = self.encoder.text_audio.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        text_vec = np.zeros(512, dtype=np.float32)
        text_vec[: len(text_raw)] = text_raw

        where_audio = {"modality": "audio"}
        if video_id:
            where_audio = {"$and": [{"video_id": video_id}, {"modality": "audio"}]}

        hits_audio = self.store.search(text_vec.tolist(), top_k=top_k * 2, where=where_audio)

        # 3. Merge hits by chunk_id
        combined_hits: dict[str, dict] = {}

        def process_hit(hit: dict, modality: str):
            meta     = hit["metadata"]
            chunk_id = meta["chunk_id"]
            raw_sim  = 1.0 - hit["distance"]
            
            # Calibrate differently based on modality
            calibrated = calibrate_visual(raw_sim) if modality == "visual" else calibrate_audio(raw_sim)

            if chunk_id not in combined_hits or calibrated > combined_hits[chunk_id]["best_sim"]:
                combined_hits[chunk_id] = {
                    "metadata": meta,
                    "best_sim": calibrated,
                    "raw_sim":  raw_sim,
                    "match_type": modality
                }

        for h in hits_visual:
            process_hit(h, "visual")
        for h in hits_audio:
            process_hit(h, "audio")

        # Filter out extremely weak hits (score 0.0)
        valid_chunks = [v for v in combined_hits.values() if v["best_sim"] > 0.05]

        # 4. Sort and Format
        sorted_chunks = sorted(
            valid_chunks,
            key=lambda x: x["best_sim"],
            reverse=True,
        )[:top_k]

        results = []
        for item in sorted_chunks:
            meta       = item["metadata"]
            frame_path = meta.get("frame_path", "")
            frame_url  = f"/frames/{os.path.basename(frame_path)}" if frame_path else ""
            
            # ensure precision formatting
            final_score = round(item["best_sim"], 4)

            results.append({
                "start_timestamp":    meta["start_timestamp"],
                "end_timestamp":      meta["end_timestamp"],
                "confidence_score":   final_score,
                "transcript_snippet": meta.get("transcript", "")[:200],
                "video_id":           meta.get("video_id", ""),
                "frame_url":          frame_url,
            })

        query_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info(f"Dual-vector search completed in {query_time_ms}ms")

        return {
            "results":               results,
            "query_time_ms":         query_time_ms,
            "total_chunks_searched": self.store.count(),
        }

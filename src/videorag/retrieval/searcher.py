# src/videorag/retrieval/searcher.py — Joint Multimodal Search Orchestrator
from __future__ import annotations
import hashlib
import os
import re
import time
from difflib import SequenceMatcher
import numpy as np
from videorag.ingestion.encoder import MultimodalEncoder
from videorag.database.vector_store import VectorStore
from videorag.config import COLLECTION_NAME
from videorag.logger import get_logger

log = get_logger(__name__)

# §2.1 & §2.2 weights for joint scoring
RRF_K = 60


def classify_query_intent(query: str) -> tuple[str, str]:
    """Infer whether a query targets frames, spoken content, or both.

    This intentionally uses transparent multilingual cues instead of a second
    generative model. Ambiguous queries fall back to hybrid retrieval.
    """
    terms = set(_normalize_transcript_text(query).split())
    visual_cues = {
        "صورة", "الصورة", "صوره", "الصوره", "صور", "الصور", "لقطة", "اللقطة",
        "لقطه", "اللقطه", "مشهد", "المشهد", "منظر", "المنظر", "شكل", "الشكل",
        "يظهر", "تظهر", "شاهد", "مرئي", "بصري", "خلفية", "الخلفية", "لون", "اللون", "الوان",
        "شاشة", "الشاشة", "مكتوب", "مكتوبة", "عنوان", "شعار", "وجه", "photo", "image",
        "picture", "frame", "scene", "visual", "shows", "showing", "appears",
        "background", "color", "logo", "screenshot",
    }
    audio_cues = {
        "قال", "قالت", "يقول", "تقول", "ذكر", "يذكر", "يتحدث", "تحدث",
        "كلام", "الكلام", "كلمة", "الكلمة", "كلمات", "العبارة", "عبارة", "جملة",
        "نص", "النص",
        "صوت", "الصوت", "سمع", "مسموع", "منطوق", "نطق", "التفريغ",
        "تفريغ", "transcript", "speech", "audio", "spoken", "said", "says",
        "voice", "quote", "quoted",
    }
    mixed_cues = {"مختلط", "مختلطة", "معا", "mixed", "multimodal"}
    wants_visual = bool(terms & visual_cues)
    wants_audio = bool(terms & audio_cues)

    if terms & mixed_cues or (wants_visual and wants_audio):
        return "combined", "الاستعلام يطلب الصورة والصوت معًا"
    if wants_visual:
        return "visual", "ظهرت كلمات تشير إلى صورة أو مشهد"
    if wants_audio:
        return "audio", "ظهرت كلمات تشير إلى كلام أو صوت منطوق"
    return "combined", "الاستعلام غير صريح؛ استُخدم البحث المختلط لتغطية الصورة والصوت"


def _normalize_transcript_text(text: str) -> str:
    """Normalize Arabic spelling and punctuation for literal transcript search."""
    text = text.lower().replace("ـ", "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text).replace("ى", "ي")
    text = text.replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _token_coverage(query: str, transcript: str) -> float:
    """Measure transcript coverage, tolerating common Arabic spelling variants."""
    query_tokens = set(_normalize_transcript_text(query).split())
    transcript_tokens = set(_normalize_transcript_text(transcript).split())
    if not query_tokens or not transcript_tokens:
        return 0.0

    matched = 0
    for token in query_tokens:
        if token in transcript_tokens:
            matched += 1
            continue
        if len(token) < 4:
            continue
        near_tokens = (
            candidate for candidate in transcript_tokens
            if candidate[0] == token[0] and abs(len(candidate) - len(token)) <= 1
        )
        if any(SequenceMatcher(None, token, candidate).ratio() >= 0.8 for candidate in near_tokens):
            matched += 1
    return matched / len(query_tokens)


class Searcher:
    """
    Video retrieval using visual, transcript, and reciprocal-rank fusion.
    """

    def __init__(self):
        log.info("Initializing Searcher...")
        self.encoder = MultimodalEncoder()
        self.store   = VectorStore(
            collection_name=COLLECTION_NAME,
            visual_dim=self.encoder.visual_dim,
        )
        log.info("Searcher ready")

    # ── Query Encoding ────────────────────────────────────────────
    def _encode_visual_query(self, query: str) -> list[float]:
        """Encode query as a SigLIP 2 visual search vector."""
        vec = self.encoder.encode_visual_text(query)
        return vec.reshape(-1).tolist()

    def _encode_audio_query(self, query: str) -> list[float]:
        """Encode query as a native 384-dimensional MiniLM vector."""
        raw = self.encoder.text_audio.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        return raw.tolist()

    def search_visual(
        self,
        query: str,
        top_k: int = 3,
        video_id: str | None = None,
    ) -> dict:
        """Rank chunks by this model's visual cosine similarity only."""
        t0 = time.perf_counter()
        query_vec = self._encode_visual_query(query)
        hits = self.store.search(
            query_vec, top_k=top_k, using="visual", video_id=video_id
        )
        results = []
        for hit in hits:
            meta = hit["metadata"]
            similarity = 1.0 - hit["distance"]
            results.append({
                "start_timestamp": meta["start_timestamp"],
                "end_timestamp": meta["end_timestamp"],
                # Required by the shared UI response schema; the UI displays
                # visual_similarity directly and does not call this confidence.
                "confidence_score": float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)),
                "visual_similarity": float(similarity),
                "transcript_snippet": meta.get("transcript", "")[:200],
                "video_id": meta.get("video_id", ""),
                "frame_url": (
                    f"/frames/{os.path.basename(meta['frame_path'])}"
                    if meta.get("frame_path") else ""
                ),
            })
        return {
            "results": results,
            "query_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "total_chunks_searched": self.store.count_chunks(),
        }

    def search_audio(
        self,
        query: str,
        top_k: int = 3,
        video_id: str | None = None,
    ) -> dict:
        """Rank chunks against transcript embeddings without visual fusion."""
        t0 = time.perf_counter()
        query_vec = self._encode_audio_query(query)
        hits = self.store.search(
            query_vec, top_k=max(top_k * 20, 50), using="audio", video_id=video_id
        )
        candidates: dict[str, dict] = {}
        for hit in hits:
            meta = hit["metadata"]
            chunk_id = meta.get("chunk_id", "")
            candidates[chunk_id] = {
                "metadata": meta,
                "similarity": 1.0 - hit["distance"],
            }

        normalized_query = _normalize_transcript_text(query)
        query_tokens = set(normalized_query.split())
        # Search the stored Whisper text too: embedding-only retrieval can miss
        # a verbatim spoken phrase even when that phrase exists in the transcript.
        for meta in self.store.list_transcripts(video_id=video_id):
            text = _normalize_transcript_text(meta.get("transcript", ""))
            if not text:
                continue
            chunk_id = meta.get("chunk_id", "")
            exact_match = bool(normalized_query and normalized_query in text)
            coverage = _token_coverage(query, text)
            if exact_match or coverage > 0:
                candidate = candidates.setdefault(
                    chunk_id, {"metadata": meta, "similarity": None}
                )
                candidate["text_match"] = exact_match
                candidate["coverage"] = coverage

        ranked = []
        for candidate in candidates.values():
            meta = candidate["metadata"]
            similarity = candidate["similarity"]
            similarity_for_rank = similarity if similarity is not None else 0.0
            exact_match = candidate.get("text_match", False)
            coverage = candidate.get("coverage", 0.0)
            fuzzy_phrase_match = len(query_tokens) >= 2 and coverage >= 0.75
            text_bonus = 2.0 if exact_match else 1.5 if fuzzy_phrase_match else 0.5 * coverage
            rank_score = similarity_for_rank + text_bonus
            ranked.append((rank_score, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)

        results = []
        for _, candidate in ranked[:top_k]:
            meta = candidate["metadata"]
            similarity = candidate["similarity"]
            score_value = similarity if similarity is not None else 0.0
            results.append({
                "start_timestamp": meta["start_timestamp"],
                "end_timestamp": meta["end_timestamp"],
                "confidence_score": float(np.clip((score_value + 1.0) / 2.0, 0.0, 1.0)),
                "audio_similarity": float(similarity) if similarity is not None else None,
                "text_match": bool(candidate.get("text_match", False)),
                "text_coverage": float(candidate.get("coverage", 0.0)),
                "transcript_snippet": meta.get("transcript", "")[:200],
                "video_id": meta.get("video_id", ""),
                "frame_url": (
                    f"/frames/{os.path.basename(meta['frame_path'])}"
                    if meta.get("frame_path") else ""
                ),
            })
        return {
            "results": results,
            "query_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "total_chunks_searched": self.store.count_chunks(),
        }

    def search_auto(
        self,
        query: str,
        top_k: int = 3,
        video_id: str | None = None,
    ) -> dict:
        """Infer query modality, then run its dedicated or hybrid retriever."""
        intent, reason = classify_query_intent(query)
        if intent == "combined" and reason.startswith("الاستعلام غير صريح"):
            best_transcript_coverage = max(
                (
                    _token_coverage(query, meta.get("transcript", ""))
                    for meta in self.store.list_transcripts(video_id=video_id)
                ),
                default=0.0,
            )
            query_token_count = len(_normalize_transcript_text(query).split())
            if best_transcript_coverage >= 0.75 or (
                query_token_count == 2 and best_transcript_coverage == 1.0
            ):
                intent = "audio"
                reason = "كلمات الاستعلام تطابق نصًا موجودًا في التفريغ الصوتي"
        if intent == "visual":
            result = self.search_visual(query, top_k=top_k, video_id=video_id)
            result["score_type"] = "visual_similarity"
        elif intent == "audio":
            result = self.search_audio(query, top_k=top_k, video_id=video_id)
            result["score_type"] = "audio_similarity"
        else:
            result = self.search(query, top_k=top_k, video_id=video_id)
            result["score_type"] = "joint"
        result["detected_intent"] = intent
        result["intent_reason"] = reason
        return result

    # ── Main Search ───────────────────────────────────────────────
    def search(
        self,
        query:    str,
        top_k:    int = 3,
        video_id: str | None = None,
    ) -> dict:
        """
        Execute a hybrid search using reciprocal-rank fusion (RRF).

        Visual and audio cosine similarities have different distributions, so
        adding calibrated raw scores caused one modality to swamp the other.
        RRF combines each modality's ranking. Chunks found by both get a boost,
        while a strong match in either modality can still rank.
        """
        t0 = time.perf_counter()
        log.info(f"Joint search | query='{query}' top_k={top_k} video_id={video_id}")

        # Retrieve broad shortlists before fusing the two rankings.
        candidate_limit = max(top_k * 20, 50)
        query_terms = set(_normalize_transcript_text(query).split())
        visual_cues = {
            "صورة", "صوره", "الصورة", "الصوره", "صور", "مشهد",
            "المشهد", "لقطة", "لقطه", "اللقطة", "اللقطه", "يظهر",
            "تظهر", "شكل", "رسم",
        }
        audio_cues = {
            "قال", "يقول", "تقول", "ذكر", "يذكر", "يتحدث", "تحدث",
            "كلام", "الكلام", "كلمة", "العبارة", "عبارة", "صوت",
            "الصوت", "سمع", "منطوق", "نطق",
        }
        visual_focused = bool(query_terms & visual_cues) and not bool(query_terms & audio_cues)
        audio_focused = bool(query_terms & audio_cues) and not bool(query_terms & visual_cues)
        if visual_focused:
            visual_weight, audio_weight = 0.6, 0.4
        elif audio_focused:
            visual_weight, audio_weight = 0.4, 0.6
        else:
            visual_weight, audio_weight = 0.5, 0.5

        # Remove modality instructions from the semantic content sent to both
        # encoders. Keep both modalities active: transcript evidence can
        # identify a person even when the user starts with "صورة".
        semantic_query = " ".join(
            term for term in query.split()
            if _normalize_transcript_text(term) not in visual_cues.union(audio_cues)
        ).strip() or query

        visual_vec = self._encode_visual_query(semantic_query)
        hits_vis = self.store.search(
            visual_vec, top_k=candidate_limit, using="visual", video_id=video_id
        )
        text_vec = self._encode_audio_query(semantic_query)
        hits_aud = self.store.search(
            text_vec, top_k=candidate_limit, using="audio", video_id=video_id
        )

        # ── Step 3: Build per-chunk score maps ────────────────────
        vis_ranks: dict[str, int] = {}
        aud_ranks: dict[str, int] = {}
        meta_lookup: dict[str, dict] = {}
        for rank, hit in enumerate(hits_vis, start=1):
            cid = hit["metadata"].get("chunk_id")
            if cid:
                vis_ranks[cid] = rank
                meta_lookup[cid] = hit["metadata"]
        for rank, hit in enumerate(hits_aud, start=1):
            cid = hit["metadata"].get("chunk_id")
            if cid:
                aud_ranks[cid] = rank
                meta_lookup[cid] = hit["metadata"]

        normalized_query = _normalize_transcript_text(semantic_query)
        exact_text: dict[str, dict] = {}
        if normalized_query:
            for meta in self.store.list_transcripts(video_id=video_id):
                transcript = _normalize_transcript_text(meta.get("transcript", ""))
                if normalized_query in transcript and meta.get("chunk_id"):
                    exact_text[meta["chunk_id"]] = meta
                    # Keep literal spoken matches even outside the semantic shortlist.
                    aud_ranks[meta["chunk_id"]] = 1
                    meta_lookup[meta["chunk_id"]] = meta

        # If the requested phrase is present verbatim in a transcript, return
        # those directly evidenced moments instead of padding the list with
        # merely related semantic candidates (the common false-positive case).
        all_chunk_ids = (
            set(exact_text)
            if exact_text
            else set(vis_ranks) | set(aud_ranks)
        )

        # ── Step 4: Calibrate and compute joint score ─────────────
        chunk_results: list[dict] = []
        for cid in all_chunk_ids:
            v_rank = vis_ranks.get(cid)
            a_rank = aud_ranks.get(cid)
            # Normalize each RRF contribution against a rank-1 result.
            v_norm = (RRF_K + 1) / (RRF_K + v_rank) if v_rank else 0.0
            a_norm = (RRF_K + 1) / (RRF_K + a_rank) if a_rank else 0.0
            joint = visual_weight * v_norm + audio_weight * a_norm
            if cid in exact_text:
                joint = min(1.0, joint + 0.1)

            chunk_results.append({
                "chunk_id":    cid,
                "joint_score": round(joint, 4),
                "vis_score":   round(v_norm, 4),
                "aud_score":   round(a_norm, 4),
                "text_match":  cid in exact_text,
                "metadata":    meta_lookup[cid],
            })

        # ── Step 5: Filter, sort, format ─────────────────────────
        # Literal transcript evidence is stronger than a semantic rank-only
        # candidate, even if the latter appears near the top of both lists.
        chunk_results.sort(
            key=lambda item: (item["text_match"], item["joint_score"]),
            reverse=True,
        )
        # Adjacent sliding windows overlap in time. Keep the strongest-ranked
        # hit for overlapping windows so one moment is not listed repeatedly.
        deduplicated: list[dict] = []
        frame_hashes: dict[str, str] = {}

        def frame_hash(meta: dict) -> str:
            path = meta.get("frame_path", "")
            if not path or not os.path.isfile(path):
                return ""
            if path not in frame_hashes:
                with open(path, "rb") as frame_file:
                    frame_hashes[path] = hashlib.file_digest(
                        frame_file, "sha256"
                    ).hexdigest()
            return frame_hashes[path]

        for item in chunk_results:
            meta = item["metadata"]
            current_frame_hash = frame_hash(meta)
            duplicate_overlap = any(
                (
                    kept["metadata"].get("video_id") == meta.get("video_id")
                    and max(
                        kept["metadata"]["start_timestamp"],
                        meta["start_timestamp"],
                    ) < min(
                        kept["metadata"]["end_timestamp"],
                        meta["end_timestamp"],
                    )
                )
                or (
                    # The same spoken phrase may be copied into overlapping
                    # windows, including duplicate uploads with different IDs
                    # and representative frames. Keep its strongest visual hit.
                    item["text_match"]
                    and kept["text_match"]
                    and max(
                        kept["metadata"]["start_timestamp"],
                        meta["start_timestamp"],
                    ) < min(
                        kept["metadata"]["end_timestamp"],
                        meta["end_timestamp"],
                    )
                )
                or (
                    current_frame_hash
                    and frame_hash(kept["metadata"]) == current_frame_hash
                    and (
                        visual_focused
                        or _normalize_transcript_text(
                            kept["metadata"].get("transcript", "")
                        ) == _normalize_transcript_text(meta.get("transcript", ""))
                    )
                    and max(
                        kept["metadata"]["start_timestamp"],
                        meta["start_timestamp"],
                    ) < min(
                        kept["metadata"]["end_timestamp"],
                        meta["end_timestamp"],
                    )
                )
                for kept in deduplicated
            )
            if not duplicate_overlap:
                deduplicated.append(item)

        top_results = deduplicated[:top_k]

        results = []
        for item in top_results:
            meta       = item["metadata"]
            frame_path = meta.get("frame_path", "")
            frame_url  = f"/frames/{os.path.basename(frame_path)}" if frame_path else ""

            results.append({
                "start_timestamp":    meta["start_timestamp"],
                "end_timestamp":      meta["end_timestamp"],
                "confidence_score":   item["joint_score"],
                "text_match":         item["text_match"],
                "transcript_snippet": meta.get("transcript", "")[:200],
                "video_id":           meta.get("video_id", ""),
                "frame_url":          frame_url,
            })

        query_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            f"Joint search done | {len(results)} results | {query_time_ms}ms | "
            f"candidates_evaluated={len(all_chunk_ids)}"
        )

        return {
            "results":               results,
            "query_time_ms":         query_time_ms,
            "total_chunks_searched": self.store.count_chunks(),
        }

# api/routes/search.py — Search Endpoints
import threading

from fastapi import APIRouter, HTTPException, Query
from api.models.schemas import (
    SearchRequest,
    GraphRAGRequest,
    SearchResponse,    ChunkResult,
    SearchResponseExtended, ChunkResultExtended,
)
from videorag.database.graph_store import create_graph_store
from videorag.retrieval.graph_rag import GraphRAG
from videorag.logger import get_logger

log    = get_logger(__name__)
router = APIRouter(tags=["Search"])

_searcher = None
_model_lock = threading.RLock()


def set_searcher(searcher) -> None:
    global _searcher
    with _model_lock:
        _searcher = searcher
        # Keep uploads and health endpoints attached to the active model/store.
        from api.routes import health as health_module
        from api.routes import upload as upload_module
        upload_module.set_searcher(searcher)
        health_module.set_store(searcher.store)


def get_searcher():
    with _model_lock:
        return _searcher


def _open_graph_store():
    try:
        return create_graph_store()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/graph/entities",
    summary="[Experiment] List entities extracted into the graph",
    include_in_schema=True,
)
def graph_entities(video_id: str | None = Query(None)) -> dict:
    graph = _open_graph_store()
    try:
        return {"entities": graph.list_fact_entities(video_id=video_id)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        graph.close()


@router.get(
    "/graph/communities",
    summary="[Experiment] List hierarchical GraphRAG communities",
    include_in_schema=True,
)
def graph_communities(level: int | None = Query(None, ge=0)) -> dict:
    graph = _open_graph_store()
    try:
        return {"communities": graph.community_reports(level=level)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        graph.close()


@router.post(
    "/search/graph-answer",
    summary="[Experiment] Evidence-grounded local/global GraphRAG answer",
    include_in_schema=True,
)
def graph_rag_answer(req: GraphRAGRequest) -> dict:
    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        searcher = _searcher
    graph = _open_graph_store()
    try:
        return GraphRAG(graph, searcher).answer(
            query=req.query,
            mode=req.mode,
            top_k=req.top_k,
            video_id=req.video_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        graph.close()


@router.post(
    "/search/graph-preview",
    summary="[Experiment] Evidence retrieval from the transcript graph",
    include_in_schema=True,
)
def search_graph_preview(req: SearchRequest) -> dict:
    """Search graph evidence separately; this does not affect normal retrieval."""
    graph = _open_graph_store()
    try:
        result = graph.search_facts(
            query=req.query, top_k=req.top_k, video_id=req.video_id
        )
        return {
            **result,
            "score_type": "graph_evidence_rank_not_confidence",
            "relation_note": "Experimental relations are kept only when a verbatim transcript quote contains the subject, predicate, and object in Arabic subject-verb-object or verb-subject-object order. A clear reversed subject-verb-object relation, including light Arabic spelling variants, is discarded; manual review is still needed.",
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        graph.close()


@router.get("/model", summary="Get the active visual model")
def active_model() -> dict:
    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        return {
            "visual_model": _searcher.encoder.visual_model,
            "chunks_indexed": _searcher.store.count_chunks(),
        }


# ── Graded Endpoint (Spec-Compliant) ─────────────────────────────────────────
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="[Graded] Multimodal Video Search",
    description=(
        "Search for specific moments in indexed videos using natural language.\n\n"
        "### Response\n"
        "Returns **exclusively**:\n"
        "- `start_timestamp` — segment start (seconds)\n"
        "- `end_timestamp` — segment end (seconds)\n"
        "- `confidence_score` — normalized ordering score, not a probability of correctness\n\n"
        "### Ranking\n"
        "```\n"
        "score = reciprocal-rank fusion of visual and audio results\n"
        "literal transcript matches receive a small boost\n"
        "```\n"
    ),
)
def search(req: SearchRequest) -> SearchResponse:
    """
    Spec-compliant endpoint.
    Returns ONLY: start_timestamp, end_timestamp, confidence_score.
    """
    log.info(f"[/search] query='{req.query}' top_k={req.top_k} video_id={req.video_id}")

    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        raw = _searcher.search(
            query=req.query,
            top_k=req.top_k,
            video_id=req.video_id,
        )

    results = [
        ChunkResult(
            start_timestamp  = hit["start_timestamp"],
            end_timestamp    = hit["end_timestamp"],
            confidence_score = hit["confidence_score"],
        )
        for hit in raw["results"]
    ]

    return SearchResponse(results=results)


# ── Extended Endpoint (Web UI only — not graded) ─────────────────────────────
@router.post(
    "/search/preview",
    response_model=SearchResponseExtended,
    summary="[UI] Extended search with keyframes and transcripts",
    include_in_schema=True,
)
def search_preview(req: SearchRequest) -> SearchResponseExtended:
    """
    Extended response for the Web UI.
    Includes frame_url, video_id, and transcript_snippet in addition to
    the three spec-required fields.
    """
    log.info(f"[/search/preview] query='{req.query}' top_k={req.top_k}")

    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        raw = _searcher.search(
            query=req.query,
            top_k=req.top_k,
            video_id=req.video_id,
        )

    results = [
        ChunkResultExtended(
            start_timestamp    = hit["start_timestamp"],
            end_timestamp      = hit["end_timestamp"],
            confidence_score   = hit["confidence_score"],
            video_id           = hit.get("video_id", ""),
            frame_url          = hit.get("frame_url", ""),
            transcript_snippet = hit.get("transcript_snippet", ""),
            text_match         = hit.get("text_match", False),
        )
        for hit in raw["results"]
    ]

    return SearchResponseExtended(
        results               = results,
        query_time_ms         = raw["query_time_ms"],
        total_chunks_searched = raw["total_chunks_searched"],
    )


@router.post(
    "/search/auto-preview",
    response_model=SearchResponseExtended,
    summary="[UI] Automatically infer search modality",
    include_in_schema=True,
)
def search_auto_preview(req: SearchRequest) -> SearchResponseExtended:
    """Infer image, spoken-text, or hybrid intent and search accordingly."""
    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        raw = _searcher.search_auto(
            query=req.query, top_k=req.top_k, video_id=req.video_id
        )

    results = [
        ChunkResultExtended(
            start_timestamp=hit["start_timestamp"],
            end_timestamp=hit["end_timestamp"],
            confidence_score=hit["confidence_score"],
            video_id=hit.get("video_id", ""),
            frame_url=hit.get("frame_url", ""),
            transcript_snippet=hit.get("transcript_snippet", ""),
            visual_similarity=hit.get("visual_similarity"),
            audio_similarity=hit.get("audio_similarity"),
            text_match=hit.get("text_match", False),
            text_coverage=hit.get("text_coverage", 0.0),
        )
        for hit in raw["results"]
    ]
    return SearchResponseExtended(
        results=results,
        query_time_ms=raw["query_time_ms"],
        total_chunks_searched=raw["total_chunks_searched"],
        score_type=raw["score_type"],
        detected_intent=raw["detected_intent"],
        intent_reason=raw["intent_reason"],
    )


@router.post(
    "/search/visual-preview",
    response_model=SearchResponseExtended,
    summary="[UI] Visual-only search with SigLIP 2",
    include_in_schema=True,
)
def search_visual_preview(req: SearchRequest) -> SearchResponseExtended:
    """Search with visual cosine only, excluding audio retrieval and fusion."""
    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        raw = _searcher.search_visual(
            query=req.query, top_k=req.top_k, video_id=req.video_id
        )
    results = [
        ChunkResultExtended(
            start_timestamp=hit["start_timestamp"],
            end_timestamp=hit["end_timestamp"],
            confidence_score=hit["confidence_score"],
            visual_similarity=hit["visual_similarity"],
            video_id=hit["video_id"],
            frame_url=hit["frame_url"],
            transcript_snippet=hit["transcript_snippet"],
        )
        for hit in raw["results"]
    ]
    return SearchResponseExtended(
        results=results,
        query_time_ms=raw["query_time_ms"],
        total_chunks_searched=raw["total_chunks_searched"],
        score_type="visual_similarity",
    )


@router.post(
    "/search/audio-preview",
    response_model=SearchResponseExtended,
    summary="[UI] Transcript-only search",
    include_in_schema=True,
)
def search_audio_preview(req: SearchRequest) -> SearchResponseExtended:
    """Search transcript embeddings without visual ranking or fusion."""
    with _model_lock:
        if _searcher is None:
            raise HTTPException(status_code=503, detail="Search model is not ready")
        raw = _searcher.search_audio(
            query=req.query, top_k=req.top_k, video_id=req.video_id
        )
    results = [
        ChunkResultExtended(
            start_timestamp=hit["start_timestamp"],
            end_timestamp=hit["end_timestamp"],
            confidence_score=hit["confidence_score"],
            audio_similarity=hit["audio_similarity"],
            text_match=hit["text_match"],
            text_coverage=hit["text_coverage"],
            video_id=hit["video_id"],
            frame_url=hit["frame_url"],
            transcript_snippet=hit["transcript_snippet"],
        )
        for hit in raw["results"]
    ]
    return SearchResponseExtended(
        results=results,
        query_time_ms=raw["query_time_ms"],
        total_chunks_searched=raw["total_chunks_searched"],
        score_type="audio_similarity",
    )

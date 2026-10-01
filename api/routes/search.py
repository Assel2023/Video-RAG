# api/routes/search.py — Search Endpoint
from fastapi import APIRouter
from api.models.schemas import SearchRequest, SearchResponse, ChunkResult
from videorag.logger import get_logger

log    = get_logger(__name__)
router = APIRouter(tags=["Search"])

_searcher = None


def set_searcher(searcher) -> None:
    global _searcher
    _searcher = searcher


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Multimodal video search",
    description=(
        "Search for specific moments in indexed videos using natural language.\n\n"
        "The query is encoded using both **CLIP** (visual) and **MiniLM** (audio) "
        "encoders, fused into a unified 512-dim embedding, and matched against "
        "indexed chunks using cosine similarity."
    ),
)
def search(req: SearchRequest) -> SearchResponse:
    """
    Execute a multimodal spatio-temporal search.

    Returns the top-K video chunks with precise timestamps,
    mathematical confidence scores, keyframe images, and transcript snippets.
    """
    log.info(f"Search request — query='{req.query}', top_k={req.top_k}")

    raw = _searcher.search(
        query=req.query,
        top_k=req.top_k,
        video_id=req.video_id,
    )

    return SearchResponse(
        results               = [ChunkResult(**hit) for hit in raw["results"]],
        query_time_ms         = raw["query_time_ms"],
        total_chunks_searched = raw["total_chunks_searched"],
    )

# api/routes/health.py — Health & Status Endpoints
from fastapi import APIRouter
from api.models.schemas import HealthResponse
from videorag.config import VISUAL_ALPHA, VISUAL_MODEL

router = APIRouter(tags=["Health"])

_store = None


def set_store(store) -> None:
    global _store
    _store = store


@router.get("/", response_model=HealthResponse, summary="System health check")
def health() -> HealthResponse:
    """Returns system status, architecture details, and indexed chunk count."""
    return HealthResponse(
        status         = "online",
        version        = "2.0.0",
        architecture   = "V = f(Visual, Audio, Temporal)",
        chunks_indexed = _store.count_chunks() if _store else 0,
        embedding_dim  = _store.visual_dim if _store else 0,
        modalities     = [
            f"visual — {VISUAL_MODEL} ({_store.visual_dim if _store else 0}-dim)",
            "audio   — Whisper ASR + MiniLM (384-dim)",
            "temporal — Sliding window timestamps",
        ],
        fusion_weights = {
            "visual": VISUAL_ALPHA,
            "audio":  round(1 - VISUAL_ALPHA, 2),
        },
    )


@router.get("/videos", summary="List indexed video IDs")
def list_videos() -> dict:
    """Returns indexed videos ordered by their display filename."""
    if not _store:
        return {"videos": []}
    return {"videos": _store.list_videos()}

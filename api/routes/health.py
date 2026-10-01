# api/routes/health.py — Health & Status Endpoints
from fastapi import APIRouter
from api.models.schemas import HealthResponse
from videorag.config import EMBED_DIM, VISUAL_ALPHA

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
        chunks_indexed = _store.count() if _store else 0,
        embedding_dim  = EMBED_DIM,
        modalities     = [
            f"visual  — CLIP ViT-B/32 ({EMBED_DIM}-dim)",
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
    """Returns a list of all distinct video IDs currently indexed."""
    if not _store:
        return {"videos": []}
    try:
        results = _store._col.get(include=["metadatas"])
        ids = sorted({m["video_id"] for m in results["metadatas"] if "video_id" in m})
        return {"videos": ids}
    except Exception:
        return {"videos": []}

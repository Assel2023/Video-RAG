# api/routes/upload.py — Video Upload & Ingestion Endpoint
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from videorag.config import DATA_DIR
from videorag.logger import get_logger

log    = get_logger(__name__)
router = APIRouter(tags=["Upload"])

_searcher = None   # injected from main.py


def set_searcher(searcher) -> None:
    global _searcher
    _searcher = searcher


# Track ingestion status per video
_ingestion_status: dict[str, dict] = {}


class UploadResponse(BaseModel):
    video_id:  str
    filename:  str
    size_mb:   float
    message:   str
    status:    str   # "queued" | "processing" | "done" | "error"


class StatusResponse(BaseModel):
    video_id: str
    status:   str
    message:  str
    chunks:   int


def _run_ingestion(video_path: str, video_id: str, language: str) -> None:
    """Background task: run ingestion pipeline and update status."""
    from videorag.ingestion.pipeline import ingest_video
    try:
        _ingestion_status[video_id]["status"]  = "processing"
        _ingestion_status[video_id]["message"] = "Extracting chunks and encoding..."

        store = ingest_video(
            video_path=video_path,
            video_id=video_id,
            language=language,
        )

        # Refresh the searcher's store reference
        if _searcher:
            _searcher.store._col = store._col

        _ingestion_status[video_id]["status"]  = "done"
        _ingestion_status[video_id]["message"] = f"Indexed {store.count()} chunks successfully"
        _ingestion_status[video_id]["chunks"]  = store.count()
        log.info(f"Ingestion complete for video_id={video_id}")

    except Exception as exc:
        log.error(f"Ingestion failed for video_id={video_id}: {exc}")
        _ingestion_status[video_id]["status"]  = "error"
        _ingestion_status[video_id]["message"] = str(exc)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a video for indexing",
    description=(
        "Upload a raw video file (.mp4, .avi, .mov). "
        "The system will automatically run the full multimodal "
        "ingestion pipeline in the background."
    ),
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file:     UploadFile = File(..., description="Video file to index"),
    language: str        = "ar",
) -> UploadResponse:
    """
    Accept a video file upload and trigger background ingestion.

    The ingestion pipeline runs asynchronously. Poll /status/{video_id}
    to track progress.
    """
    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Save uploaded file to data/
    video_id   = Path(file.filename).stem.replace(" ", "_")
    save_path  = DATA_DIR / file.filename

    log.info(f"Receiving upload: {file.filename} → {save_path}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    log.info(f"Saved {file.filename} ({size_mb:.1f} MB)")

    # Register status
    _ingestion_status[video_id] = {
        "status":  "queued",
        "message": "Queued for ingestion",
        "chunks":  0,
    }

    # Kick off background ingestion
    background_tasks.add_task(
        _run_ingestion, str(save_path), video_id, language
    )

    return UploadResponse(
        video_id = video_id,
        filename = file.filename,
        size_mb  = round(size_mb, 2),
        message  = "Video uploaded. Ingestion started in background.",
        status   = "queued",
    )


@router.get(
    "/status/{video_id}",
    response_model=StatusResponse,
    summary="Check ingestion status",
)
def ingestion_status(video_id: str) -> StatusResponse:
    """Poll the ingestion status of an uploaded video."""
    if video_id not in _ingestion_status:
        raise HTTPException(
            status_code=404,
            detail=f"No ingestion job found for video_id='{video_id}'"
        )
    info = _ingestion_status[video_id]
    return StatusResponse(
        video_id = video_id,
        status   = info["status"],
        message  = info["message"],
        chunks   = info.get("chunks", 0),
    )

# api/routes/upload.py — Video Upload & Ingestion Endpoint
import os
import hashlib
import uuid
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
    visual_model: str


class StatusResponse(BaseModel):
    video_id: str
    status:   str
    message:  str
    chunks:   int


def _run_ingestion(video_path: str, video_id: str, language: str | None, searcher) -> None:
    """Background task: run ingestion pipeline and update status."""
    from videorag.ingestion.pipeline import ingest_video
    try:
        _ingestion_status[video_id]["status"]  = "processing"
        _ingestion_status[video_id]["message"] = "Extracting chunks and encoding..."

        # Pin the selected model/store for this job even if global state changes.
        enc = searcher.encoder if searcher else None
        trn = getattr(searcher, "transcriber", None) if searcher else None

        store, chunk_count = ingest_video(
            video_path  = video_path,
            video_id    = video_id,
            language    = language,   # None = auto-detect (Bug 8 fix)
            encoder     = enc,
            transcriber = trn,
            store       = searcher.store if searcher else None,
            source_filename=_ingestion_status[video_id].get("filename"),
        )

        # Bug 3 fix: report actual chunk count, not total vector count
        _ingestion_status[video_id]["status"]  = "done"
        _ingestion_status[video_id]["message"] = (
            f"Ingestion complete! Indexed {chunk_count} chunks "
            f"({store.count()} vectors total)."
        )
        _ingestion_status[video_id]["chunks"]  = chunk_count
        log.info(f"Ingestion complete for video_id={video_id} | chunks={chunk_count}")

    except Exception as exc:
        log.error(f"Ingestion failed for video_id={video_id}: {exc}")
        _ingestion_status[video_id]["status"]  = "error"
        _ingestion_status[video_id]["message"] = f"خطأ: {exc}"


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
def upload_video(
    background_tasks: BackgroundTasks,
    file:     UploadFile = File(..., description="Video file to index"),
    language: str        = "",   # Bug 8 fix: empty = auto-detect
) -> UploadResponse:
    """
    Accept a video file upload and trigger background ingestion.
    Poll /status/{video_id} to track progress.
    """
    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    # Bug 6 fix: use only the basename, never allow path traversal
    safe_filename = Path(file.filename).name
    suffix        = Path(safe_filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Hash content so repeated uploads of the same source video keep one ID.
    temp_path = DATA_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    digest = hashlib.sha256()
    log.info(f"Receiving upload: {safe_filename}")
    with open(temp_path, "wb") as f:
        while block := file.file.read(1024 * 1024):
            digest.update(block)
            f.write(block)

    video_id = f"video_{digest.hexdigest()[:16]}"
    save_path = DATA_DIR / f"{video_id}{suffix}"
    size_mb = os.path.getsize(temp_path) / (1024 ** 2)

    # Bug 8 fix: treat empty string as None (auto-detect)
    lang = language.strip() or None

    # Pin the single active SigLIP 2 searcher before queuing ingestion.
    from api.routes import search as search_module
    with search_module._model_lock:
        pending = _ingestion_status.get(video_id, {}).get("status")
        if pending in {"queued", "processing"}:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail="This exact video is already being indexed; wait for it to finish.",
            )
        if _searcher is None:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail="Search model is not ready")
        pinned_searcher = search_module.get_searcher()
        selected_model = pinned_searcher.encoder.visual_model
        os.replace(temp_path, save_path)

        _ingestion_status[video_id] = {
            "status":  "queued",
            "message": f"Queued for {selected_model} ingestion",
            "chunks":  0,
            "visual_model": selected_model,
            "filename": safe_filename,
        }
        background_tasks.add_task(
            _run_ingestion, str(save_path), video_id, lang, pinned_searcher
        )

    return UploadResponse(
        video_id = video_id,
        filename = safe_filename,
        size_mb  = round(size_mb, 2),
        message  = "Video uploaded. Ingestion started in background.",
        status   = "queued",
        visual_model = selected_model,
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

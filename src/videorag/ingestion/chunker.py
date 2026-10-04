# src/videorag/ingestion/chunker.py — Temporal Sliding-Window Chunking
from __future__ import annotations
import shutil
import tempfile
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from videorag.config import CHUNK_SIZE, OVERLAP, KEYFRAMES_DIR
from videorag.logger import get_logger

log = get_logger(__name__)


@dataclass
class VideoChunk:
    index:      int
    start:      float
    end:        float
    mid:        float
    frame_path: str


def _safe_open_video(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file safely on Windows, even if the path contains
    Arabic or Unicode characters (which OpenCV cannot handle directly).
    Copies the file to a safe ASCII temp path first.
    """
    p = Path(video_path)
    try:
        str(p).encode("ascii")
        # Pure ASCII path — open directly
        return cv2.VideoCapture(video_path)
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Non-ASCII path: copy to a temp file with a safe name
    suffix = p.suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    shutil.copy2(video_path, tmp.name)
    log.info(f"Copied video to safe temp path for OpenCV: {tmp.name}")
    return cv2.VideoCapture(tmp.name)


def extract_chunks(video_path: str, video_id: str = "video") -> list[VideoChunk]:
    """
    Sliding-window temporal chunking.

    Splits the video into overlapping chunks of CHUNK_SIZE seconds,
    extracting one representative keyframe from the midpoint of each chunk.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of VideoChunk objects with saved keyframe images.
    """
    log.info(f"Opening video: {video_path}")
    cap = _safe_open_video(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS={fps} for video: {video_path}. "
            "The file may be corrupted or unsupported."
        )

    duration = total_frames / fps

    log.info(
        f"Video metadata — FPS={fps:.1f}, "
        f"Duration={duration:.1f}s, Frames={total_frames}"
    )

    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[VideoChunk] = []
    start = 0.0
    index = 0

    # Bug 4 fix: use > 0 so we capture the tail of the video too
    while start < duration:
        end = min(start + CHUNK_SIZE, duration)
        mid = (start + end) / 2.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
        ret, frame = cap.read()

        if ret:
            # Bug 1 fix: include video_id so frames from different videos never collide
            frame_path = str(
                KEYFRAMES_DIR / f"{video_id}_frame_{start:.1f}_{end:.1f}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            chunks.append(
                VideoChunk(
                    index=index,
                    start=round(start, 2),
                    end=round(end, 2),
                    mid=round(mid, 2),
                    frame_path=frame_path,
                )
            )
            index += 1

        start += CHUNK_SIZE - OVERLAP

    cap.release()
    log.info(
        f"Chunking complete — {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}s, overlap={OVERLAP}s)"
    )
    return chunks

#!/usr/bin/env python
# scripts/ingest.py — CLI Ingestion Script
"""
Run the full multimodal ingestion pipeline from the command line.

Usage:
    python scripts/ingest.py --video data/test_video.mp4 --id my_video --lang ar
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from videorag.ingestion.pipeline import ingest_video
from videorag.logger import get_logger

log = get_logger("scripts.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VideoRAG — Multimodal Video Ingestion"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--id",    default="video_01", help="Video identifier")
    parser.add_argument("--lang",  default="ar",       help="ASR language code")
    args = parser.parse_args()

    log.info(f"Starting ingestion: {args.video}")
    store, chunks = ingest_video(
        video_path=args.video,
        video_id=args.id,
        language=args.lang,
    )
    log.info(f"Done. Total indexed: {chunks} chunks ({store.count()} vectors).")


if __name__ == "__main__":
    main()

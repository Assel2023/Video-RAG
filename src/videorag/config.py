# src/videorag/config.py — Centralized Configuration
from pathlib import Path
import os

# ── Project Root ──────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
STORAGE_DIR   = ROOT_DIR / "storage"
KEYFRAMES_DIR = STORAGE_DIR / "keyframes"
QDRANT_PATH   = STORAGE_DIR / "qdrant"
QDRANT_URL    = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DATA_DIR      = ROOT_DIR / "data"
LOGS_DIR      = ROOT_DIR / "logs"

# ── Ingestion Settings ────────────────────────────────────────
CHUNK_SIZE = 5      # seconds per chunk
OVERLAP    = 1      # overlap between chunks
LANGUAGE   = "ar"   # default ASR language

# ── Model Settings ────────────────────────────────────────────
WHISPER_MODEL     = "small"
TEXT_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME   = "video_chunks_v5_siglip2_v2"
EMBED_DIM         = 768
VISUAL_ALPHA      = 0.5
VISUAL_MODEL      = "siglip2"
SIGLIP2_MODEL     = "google/siglip2-base-patch16-224"

# ── API Settings ──────────────────────────────────────────────
API_HOST      = "0.0.0.0"
API_PORT      = 8000
TOP_K_DEFAULT = 3

# ── Ensure directories exist ──────────────────────────────────
for _dir in [STORAGE_DIR, KEYFRAMES_DIR, QDRANT_PATH, DATA_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

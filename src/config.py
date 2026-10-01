# src/config.py — Central Configuration
from pathlib import Path

# ── المسارات ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
KEYFRAMES_DIR = BASE_DIR / "keyframes"
DB_DIR        = BASE_DIR / "chroma_db"
LOG_DIR       = BASE_DIR / "logs"

# ── إعدادات التقطيع ──────────────────────────────────────────
CHUNK_SIZE    = 5      # ثواني
OVERLAP       = 1      # ثواني تداخل

# ── إعدادات النماذج ──────────────────────────────────────────
WHISPER_MODEL   = "base"
EMBEDDER_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "video_chunks"

# ── إعدادات الـ API ───────────────────────────────────────────
API_HOST      = "0.0.0.0"
API_PORT      = 8000
TOP_K_DEFAULT = 3
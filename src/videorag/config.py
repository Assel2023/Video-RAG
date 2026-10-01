# src/videorag/config.py — Centralized Configuration
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
STORAGE_DIR   = ROOT_DIR / "storage"
KEYFRAMES_DIR = STORAGE_DIR / "keyframes"
DB_DIR        = STORAGE_DIR / "chroma_db"
DATA_DIR      = ROOT_DIR / "data"
LOGS_DIR      = ROOT_DIR / "logs"

# ── Ingestion Settings ────────────────────────────────────────
CHUNK_SIZE = 5      # seconds per chunk
OVERLAP    = 1      # overlap between chunks
LANGUAGE   = "ar"   # default ASR language

# ── Model Settings ────────────────────────────────────────────
WHISPER_MODEL     = "small"
CLIP_IMAGE_MODEL  = "clip-ViT-B-32"                                    # encodes images (English CLIP)
CLIP_TEXT_MODEL   = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # encodes text (Arabic + multilingual)
TEXT_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME   = "video_chunks_v4"
EMBED_DIM         = 512
VISUAL_ALPHA      = 0.5

# ── API Settings ──────────────────────────────────────────────
API_HOST      = "0.0.0.0"
API_PORT      = 8000
TOP_K_DEFAULT = 3

# ── Ensure directories exist ──────────────────────────────────
for _dir in [STORAGE_DIR, KEYFRAMES_DIR, DB_DIR, DATA_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

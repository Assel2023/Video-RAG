import os
import sys
from pathlib import Path

# Enable offline mode for cached HuggingFace models (instant startup)
# os.environ["HF_HUB_OFFLINE"] = "1"

# Make src/ importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from videorag.config import KEYFRAMES_DIR, VISUAL_MODEL
from videorag.retrieval.searcher import Searcher
from videorag.logger import get_logger
from api.routes import health as health_module
from api.routes import search as search_module
from api.routes import upload as upload_module

log = get_logger("api.main")

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, release resources at shutdown."""
    log.info("Starting up — loading models and connecting to vector store...")
    searcher = Searcher()
    search_module.set_searcher(searcher)
    upload_module.set_searcher(searcher)
    health_module.set_store(searcher.store)
    log.info(f"Startup complete — {searcher.store.count()} chunks ready")
    yield
    log.info("Shutting down...")
    active_searcher = search_module.get_searcher()
    if active_searcher is not None:
        active_searcher.store.close()


app = FastAPI(
    title       = "Spatio-Temporal Video RAG",
    description = (
        "## Multimodal Video Retrieval System\n\n"
        "**Architecture:** `V = f(Visual, Audio, Temporal)`\n\n"
        "| Component | Model | Dimensions |\n"
        "|-----------|-------|------------|\n"
        f"| Visual Track | {VISUAL_MODEL} | model-dependent |\n"
        "| Audio Track | Whisper Small + MiniLM | 384-dim |\n"
        "| Fusion | α·Visual + (1-α)·Audio | 512-dim |\n"
        "| Vector DB | Qdrant (named cosine vectors) | — |\n"
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
)

# Static files — serve keyframe images at /frames/<filename>
app.mount(
    "/frames",
    StaticFiles(directory=str(KEYFRAMES_DIR)),
    name="frames",
)

# Web UI
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/ui", include_in_schema=False)
def ui(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)

# Register routers
app.include_router(health_module.router)
app.include_router(search_module.router)
app.include_router(upload_module.router)

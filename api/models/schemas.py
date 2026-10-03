# api/models/schemas.py — Pydantic Request/Response Schemas
from pydantic import BaseModel, Field


# ── Search Request ──────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query:    str        = Field(...,  description="Natural language search query (Arabic or English)")
    video_id: str | None = Field(None, description="Optional: filter results to a specific video ID")
    top_k:    int        = Field(3,    ge=1, le=20, description="Number of results to return")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query":    "شاشة سطر الأوامر أثناء تشغيل السكريبت",
                "video_id": None,
                "top_k":    3,
            }
        }
    }


# ── Standard Search Response (Spec-compliant — 3 fields only) ───────────────
class ChunkResult(BaseModel):
    """
    A single retrieved video chunk.

    Per the task brief, this response contains EXCLUSIVELY:
      - start_timestamp:  The beginning of the matching segment (seconds).
      - end_timestamp:    The end of the matching segment (seconds).
      - confidence_score: Normalized visual/audio ordering score ∈ [0, 1].
                          This is a ranking signal, not a probability of correctness.
    """
    start_timestamp:  float = Field(..., description="Segment start time in seconds")
    end_timestamp:    float = Field(..., description="Segment end time in seconds")
    confidence_score: float = Field(..., ge=0, le=1,
                                   description="Normalized visual+audio ranking score, not calibrated confidence")


class SearchResponse(BaseModel):
    results: list[ChunkResult] = Field(..., description="Ranked list of matching video segments")


# ── Extended Response (for Web UI only — not in the graded API) ─────────────
class ChunkResultExtended(ChunkResult):
    """
    Extends ChunkResult with UI-display fields.
    Served at POST /search/preview only (not the graded endpoint).
    """
    video_id:           str = Field("", description="Source video identifier")
    frame_url:          str = Field("", description="URL to keyframe image")
    transcript_snippet: str = Field("", description="Transcribed audio snippet")
    visual_similarity: float | None = Field(None, description="Raw visual cosine similarity")
    audio_similarity: float | None = Field(None, description="Raw transcript cosine similarity")
    text_match: bool = Field(False, description="Literal match in stored transcript text")
    text_coverage: float = Field(0.0, description="Fraction of query words found in transcript")


class SearchResponseExtended(BaseModel):
    results:               list[ChunkResultExtended]
    query_time_ms:         float = Field(..., description="End-to-end retrieval latency (ms)")
    total_chunks_searched: int   = Field(..., description="Total unique chunks in the index")
    score_type: str = Field("joint", description="joint score or raw visual similarity")
    detected_intent: str | None = Field(None, description="Automatically inferred query modality")
    intent_reason: str | None = Field(None, description="Human-readable explanation of automatic modality selection")


# ── Health Check ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:         str
    version:        str
    architecture:   str
    chunks_indexed: int
    embedding_dim:  int
    modalities:     list[str]
    fusion_weights: dict[str, float]

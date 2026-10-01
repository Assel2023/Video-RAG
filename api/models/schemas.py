# api/models/schemas.py — Pydantic Request/Response Schemas
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query:    str        = Field(...,  description="Natural language search query")
    video_id: str | None = Field(None, description="Optional: filter by video ID")
    top_k:    int        = Field(3,    ge=1, le=20, description="Number of results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query":    "terminal screen running python script",
                "video_id": None,
                "top_k":    3,
            }
        }
    }


class ChunkResult(BaseModel):
    start_timestamp:    float = Field(..., description="Chunk start time in seconds")
    end_timestamp:      float = Field(..., description="Chunk end time in seconds")
    confidence_score:   float = Field(..., ge=0, le=1, description="Similarity ∈ [0,1]")
    transcript_snippet: str   = Field(..., description="Transcribed audio snippet")
    video_id:           str   = Field(..., description="Source video identifier")
    frame_url:          str   = Field("",  description="URL to keyframe image")


class SearchResponse(BaseModel):
    results:               list[ChunkResult]
    query_time_ms:         float = Field(..., description="Retrieval latency in ms")
    total_chunks_searched: int   = Field(..., description="Total indexed chunks")


class HealthResponse(BaseModel):
    status:         str
    version:        str
    architecture:   str
    chunks_indexed: int
    embedding_dim:  int
    modalities:     list[str]
    fusion_weights: dict[str, float]

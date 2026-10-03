# RFC: Spatio-Temporal Video RAG Architecture

## 1. Overview
The Spatio-Temporal Video RAG is a production-grade system designed to search within videos using natural language. It supports multimodal querying (visual and audio) across English and Arabic, outputting exact timestamps where the requested action or dialogue occurs.

## 2. System Architecture

### 2.1 Ingestion Pipeline
1. **Chunking**: Videos are sliced into 5-second overlapping chunks (stride 4s) using OpenCV. A keyframe is extracted from the midpoint.
2. **Audio Transcription**: `ffmpeg` extracts the audio track, which is transcribed in a single pass using OpenAI's **Whisper (small)**. This avoids the overhead of instantiating inference per chunk.
3. **Visual Encoding**: Keyframes are batched and encoded into 512-dim vectors using `clip-ViT-B-32`.
4. **Text Encoding**: Chunk transcripts are encoded into 384-dim vectors using `paraphrase-multilingual-MiniLM-L12-v2` (padded to 512-dim).
5. **Storage**: Vectors are stored in **Qdrant** with cosine similarity. Each video chunk is one point with named `visual` (512-dim) and `audio` (384-dim) vectors, plus shared metadata payload. Audio-less chunks omit the audio vector.

The default development mode uses Qdrant's embedded local storage at `storage/qdrant`. Set `QDRANT_URL` (and optionally `QDRANT_API_KEY`) to use a separately managed Qdrant server. The collection is `video_chunks_v5`; existing ChromaDB data is not migrated automatically and must be re-indexed from source videos.

### 2.2 Retrieval Pipeline (Joint Scoring)
1. **Query Encoding**:
   - The query is encoded via `clip-ViT-B-32-multilingual-v1` into a 512-dim visual query vector.
   - The query is encoded via `paraphrase-multilingual-MiniLM-L12-v2` into a 384-dim audio query vector (padded to 512).
2. **Parallel Search**: Both named vectors are queried independently in Qdrant and filtered by `video_id` when requested.
3. **Independent Calibration**:
   - CLIP similarities naturally peak around 0.35. They are linearly normalized using `(raw_sim - 0.10) / (0.35 - 0.10)`.
   - MiniLM similarities peak much higher. They are linearly normalized using `(raw_sim - 0.05) / (0.75 - 0.05)`.
4. **Joint Scoring**: For every candidate chunk, the final score is computed as:
   `joint_score = 0.6 * calibrated_visual + 0.4 * calibrated_audio`
   This heavily rewards chunks where the event is *both* seen and spoken about simultaneously, precisely satisfying the spatio-temporal requirement.

## 3. Justification of Choices

### Models
- **CLIP (ViT-B/32)**: Standard robust baseline for image-text similarity.
- **Multilingual CLIP Text Encoder**: A student model aligned to English CLIP, allowing native Arabic visual searches without latency-inducing translation APIs.
- **Whisper Small**: Excellent trade-off between ASR transcription speed and accuracy. Supports auto language detection seamlessly.
- **MiniLM (L12-v2)**: State-of-the-art multilingual dense text retriever, highly optimized for semantic search.

### Multi-Vector Indexing vs. Early Fusion
An early version of this project fused visual and audio embeddings via weighted addition before database insertion. This resulted in "dilution"—purely visual queries suffered a 50% penalty against chunks with audio. Switching to **Dual-Vector Indexing** ensures visual matches are scored against pure visual embeddings, preserving mathematical integrity and resulting in much higher practical accuracy.

## 4. API Design
The Search API returns `results` containing timestamps and confidence scores (as requested). Extra fields (`frame_url`, `transcript_snippet`, `video_id`, `query_time_ms`) are included to hydrate the Web UI and provide crucial debugging context without requiring secondary database round-trips.

## 5. Limitations & Future Work
- **Temporal Resolution**: Fixed 5-second windows may clip dialogue boundaries. Moving to Whisper word-level timestamps (`word_timestamps=True`) could yield precise sub-second bounds.
- **Visual Context**: Using mean-pooling over 3 frames per chunk (start, mid, end) instead of 1 keyframe would reduce the chance of missing fast visual actions.

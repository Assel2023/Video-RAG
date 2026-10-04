# Review Notes — Spatio-Temporal Video RAG

Reviewed on 2026-10-01 against the task brief. The project was run locally (Python 3.12, CPU only), three test videos were indexed, and the API was exercised end to end.

---

## 1. Verdict against the deliverables

| Deliverable | Status | Notes |
|---|---|---|
| **RFC / architecture design document** | ❌ Missing | Only a short README exists. No data-flow diagram, no vector DB schema, no justification of model / chunking choices. |
| **Ingestion engine** | ✅ Works, with bugs | Upload → chunk → Whisper → CLIP → ChromaDB works. See bugs in §3. |
| **Search API (JSON)** | ⚠️ Works, not to spec | Returns timestamps + confidence, but the brief says the response must contain those fields **exclusively**. Extra fields are returned (`transcript_snippet`, `video_id`, `frame_url`, `query_time_ms`, `total_chunks_searched`). |
| **Performance report** | ❌ Missing | A benchmark script exists, but there is no written report, and memory **during indexing** is never measured. |
| **Goal: joint visual + audio understanding** | ❌ Not achieved | Visual and audio are searched separately and the best single modality wins. See §2.1. This is the most important issue. |

---

## 2. Critical issues (these affect the grade the most)

### 2.1 There is no real "joint understanding" — the system does OR, not AND
The task's example query is: *"the moment where the command-line screen appears **while** the exit function is being explained"*. That needs a chunk where the **visual** matches (terminal) **and** the **audio** matches (explaining exit) at the same time.

What the code actually does (`src/videorag/retrieval/searcher.py`):
1. Search visual vectors → top hits.
2. Search audio vectors → top hits.
3. For each chunk, keep **the max** of the two scores.

So a chunk that matches only the picture can beat a chunk that matches both picture and speech. The system can never reward "both at the same moment".

Also:
- `MultimodalEncoder.fuse()` and `encode_query()` (the α·Visual + (1-α)·Audio formula) are **dead code**. Nothing calls them except a unit test.
- The API description in `api/main.py` and the pipeline docstring ("Step 4: Joint fusion") claim fusion that does not happen. A reviewer will notice this mismatch.

**Suggested fix:**
- Retrieve candidates from both modalities, then **re-score every candidate chunk in both modalities**, fetching the missing vector by `chunk_id`.
- Combine with a weighted sum, e.g. `score = w_v·visual + w_a·audio`, or use Reciprocal Rank Fusion.
- Bonus: split the query into a *visual part* and a *spoken part* (rules or a small LLM), e.g. "terminal screen running a script" + "explaining the exit function". Encode each part with the right encoder. This directly matches the task's example.

### 2.2 The confidence score is miscalibrated, so speech search almost never wins
`calibrate_audio()` maps raw similarity 0.45 → 0 and 0.85 → 1, and says "baseline ~0.40". Measured on a real video:

| Query vs. text | Raw MiniLM similarity |
|---|---|
| "pushing code to the server" vs. lyric *"One commit, one push, the server's taking the load now"* (correct match) | **0.48** |
| "cooking pasta recipe" vs. the same lyric (unrelated) | **0.06** |

So the real baseline is ~0.05, not 0.40. A correct speech match gets calibrated to **0.08**. Meanwhile CLIP is calibrated generously (0.18 → 0, 0.32 → 1), so an unrelated video's title card "THE CODE REPORT" scored **0.68** for the same query and won.

**Suggested fix:**
- Re-fit both calibration ranges on a small labelled set, or replace the hand-picked numbers with something principled (min-max over the candidates, a softmax over the top-K, or rank fusion).
- Document the formula in the RFC. The brief asks for a *mathematical* confidence score, so the reviewer will want to see how it is derived.

### 2.3 Missing RFC
This is a whole deliverable. It should cover at least:
- System architecture diagram (ingestion path + query path).
- Chunking strategy and why (5 s window / 1 s overlap; alternatives considered).
- Model choices and why (CLIP ViT-B/32 multilingual, Whisper small, MiniLM).
- **Vector DB schema**: collections, vector dims, metadata fields, IDs, distance metric.
- Scoring / fusion formula and the confidence-score math.
- Limitations and future work.

### 2.4 Missing performance report
- `scripts/benchmark.py` measures query latency and RAM **after** querying only. The brief asks for memory **during indexing and querying**.
- Its test queries ("شرح الشبكات العصبية"…) don't relate to any indexed video, so the scores mean nothing.
- The root `benchmark.py` points at the old `./chroma_db` collection `video_chunks`, which the current code no longer uses.

Numbers I measured (CPU only, no GPU), as a starting point:

| Metric | Value |
|---|---|
| Ingestion time, 2 min 25 s video | ~42 s (~0.3× real time) |
| Ingestion time, 10 s video | ~17 s (mostly model loading) |
| Query latency | 25–60 ms |
| Server peak RAM (after ingesting 3 videos) | **~7.4 GB** |
| Server RAM after ingestion finished | ~7.0 GB (not released) |

The report should include peak RAM for ingestion and for querying separately, latency p50/p95, and **accuracy metrics** (see §5).

---

## 3. Bugs found during testing

| # | Bug | Where | Effect |
|---|---|---|---|
| 1 | **Keyframes from different videos overwrite each other.** Filenames are `frame_{start}_{end}.jpg` with no video id. | `ingestion/chunker.py:96` | Search results show the wrong thumbnail, e.g. the `fir` video's 0–5 s result shows a football fan. There are 36 image files for 56 chunks. Fix: `f"{video_id}_frame_{start}_{end}.jpg"`. |
| 2 | **Re-uploading a video is silently ignored.** `collection.add()` skips IDs that already exist. | `database/vector_store.py` | If a video is re-uploaded with new content, the old vectors stay. Fix: delete by `video_id` first, then `upsert`. |
| 3 | **Status reports the wrong chunk count.** It returns `store.count()` for the whole DB, and counts vectors (visual + audio), not chunks. | `api/routes/upload.py` | A 10 s video (2 chunks) reports "Indexed 92 chunks". |
| 4 | **End of video is dropped.** The loop condition `start < duration - CHUNK_SIZE + OVERLAP` skips the tail, and videos shorter than ~4 s produce zero chunks. | `ingestion/chunker.py` | Up to ~4 s at the end is never searchable. The error message says "longer than 5 seconds", which isn't accurate either. |
| 5 | **ffmpeg failures are silent.** It runs with `check=False` and errors are swallowed. | `ingestion/transcriber.py` | If ffmpeg is missing or the video has no audio, transcription silently returns nothing and the video is indexed visual-only with no clear message. The status should say "no audio track" explicitly. |
| 6 | **The uploaded filename is used unsanitised as the save path.** | `api/routes/upload.py` | A filename like `../../x.mp4` writes outside `data/` (path traversal). Use `Path(file.filename).name` or generate a UUID. |
| 7 | **Video id comes from the filename.** | `api/routes/upload.py` | Two different videos with the same name overwrite each other. |
| 8 | **Language is forced to Arabic by default.** | `upload.py`, `config.py` | An English video uploaded with defaults is transcribed as Arabic and comes out as garbage. Whisper can auto-detect the language (`language=None`). Make that the default. |
| 9 | **Nonsense queries still return results.** The weak-match cutoff is 0.05. | `searcher.py` | "a cat playing with a ball" returns 3 results. Better to return an empty list, or flag low confidence. |

---

## 4. Architecture and code quality

- **One collection with zero-padded vectors.** Audio vectors (384-dim) are padded with zeros to 512 so they fit in the same Chroma collection as CLIP vectors, then filtered with `where={"modality": ...}`. Two collections (`visual` 512-dim, `audio` 384-dim) linked by `chunk_id` is cleaner, faster (no filter during the HNSW search), and easier to explain in the RFC.
- **Models are reloaded on every upload.** `ingest_video()` creates a new `MultimodalEncoder` and `Transcriber` each time, even though the server already holds the encoders in memory. That's duplicate RAM (part of the 7.4 GB peak) and ~30 s of extra loading per upload. Reuse the server's encoder, load Whisper once, and run ingestion jobs **one at a time through a queue**. Two simultaneous uploads currently load everything twice in parallel.
- **One keyframe per 5 s chunk.** Short visual events are easily missed. Options:
  - 2–3 frames per chunk, with mean or max pooling.
  - Scene-change detection (e.g. PySceneDetect) for keyframes.
- **Timestamp precision.** The brief asks for "exact" moments. Whisper segment or word timestamps (`word_timestamps=True`) could tighten `start/end` to where the speech actually happens, instead of fixed 5 s windows.
- **Dead or legacy code to remove:**
  - `src/videorag/retrieval/translator.py` (187 lines, never imported).
  - `encoder.fuse()` / `encode_query()` (unless used for the §2.1 fix).
  - Root `ingestion.py`, root `benchmark.py`, `src/config.py`, `src/logger.py`: an older version of the project.
  - `chroma_db/` and `keyframes/` at the repo root are committed but **never used**. The code reads `storage/` instead, which `.gitignore` excludes. They make the repo look like it ships a pre-built index when it doesn't.
- **Upload hack:** `_searcher.store._col = store._col` reaches into private attributes. Not needed if ingestion reuses the same store.
- **README claims without evidence:** "100% accuracy" and "80% faster indexing". Either back them up with numbers in the performance report or remove them. A reviewer will challenge them.
- **README setup is incomplete:**
  - No Python version.
  - No mention that **ffmpeg must be installed**. Without it, audio search silently stops working.
  - No note about CPU vs. GPU PyTorch.
  - No pinned dependency versions.

---

## 5. Testing and evaluation

- `tests/test_retrieval.py` only checks imports and the unused `fuse()` function. Both pass, but they don't test retrieval at all.
- There is **no accuracy evaluation**, yet the task explicitly grades "accuracy of outputs". Suggested:
  - A small evaluation set: 2–3 videos with ~20 hand-labelled queries, each with a ground-truth time range. Include visual-only, speech-only, and **combined** queries.
  - Metrics: **Recall@1 / Recall@5**, **MRR**, and **temporal IoU** (overlap between the returned and correct time ranges).
  - Show results before and after the fusion and calibration fixes. That makes a strong section in the performance report.
- Record one test video that matches the brief's example exactly: a screen recording of running a script in a terminal while explaining what `exit()` does. Use it as the demo query.

---

## 6. Prioritised to-do list

1. Implement real joint scoring (§2.1) and fix calibration (§2.2).
2. Write the RFC (§2.3).
3. Build the eval set and write the performance report with latency, indexing and query memory, and accuracy metrics (§2.4, §5).
4. Fix bugs 1, 2, 6 and 8 (thumbnails, re-upload, path traversal, language auto-detect).
5. Make the search response match the spec (timestamps + confidence only), or justify the extra fields in the RFC.
6. Remove dead and legacy code and the unused committed `chroma_db/` and `keyframes/`.
7. Reuse models during ingestion and run uploads through a queue.
8. Update the README: setup requirements (ffmpeg, Python version), and remove the unsupported claims.

---

## 7. What is good

- Clean project layout (`api/`, `src/videorag/ingestion|retrieval|database`), with good separation of concerns.
- Sensible model choices for Arabic + English: multilingual CLIP text encoder and multilingual MiniLM.
- Single-pass Whisper and batch CLIP encoding are real, sensible optimisations.
- Keeping visual and audio vectors separate (instead of averaging them into one vector) is the right idea. It just needs a proper way to combine the scores at query time.
- The web UI and the upload → background ingestion → status-polling flow work well for a demo.
- Image search works in both English and Arabic. "football fans in the stadium" and "مشجعين كرة قدم في الملعب" both found the right clip at the top.

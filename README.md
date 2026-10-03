# 🎬 Spatio-Temporal Video RAG

نظام استرجاع لقطات الفيديو بالبحث الدلالي متعدد الوسائط (Multimodal Semantic Search).

## المعمارية

```
query ──► [SigLIP 2 Text Encoder] ──► visual_vec ──┐
      └─► [MiniLM Text Encoder] ──► audio_vec  ─┤
                     Qdrant point (named vectors: visual + audio)
                                                                      ▼
                                          Reciprocal Rank Fusion (RRF)
                                                                      ▼
                                                           top-K chunks
```

| المكوّن | النموذج | الأبعاد |
|---|---|---|
| Visual and visual-query encoder | `google/siglip2-base-patch16-224` | 768-dim |
| Audio Encoder (النصوص الصوتية) | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim |
| ASR | Whisper `small` (single-pass على الفيديو كامل) | — |
| قاعدة البيانات المتجهة | Qdrant (named cosine vectors) | — |
| الخادم | FastAPI + Uvicorn | — |

## المتطلبات

- **Python 3.10+**
- **ffmpeg** مثبّت ومضاف لـ PATH (مطلوب لاستخراج الصوت)
  - Windows: `winget install ffmpeg` أو من [ffmpeg.org](https://ffmpeg.org/download.html)
  - Mac: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
- بطاقة رسومية (GPU) موصى بها لتسريع Whisper وSigLIP 2، لكن النظام يعمل على CPU أيضاً.

## تشغيل المشروع

```bash
# 1. إنشاء بيئة افتراضية وتثبيت المتطلبات
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. تشغيل الخادم
python -m uvicorn api.main:app --reload --port 8000

# 3. افتح الواجهة
# http://127.0.0.1:8000/ui
```

Qdrant runs embedded and stores its data under `storage/qdrant` by default; no separate server is required for local development. To connect to a Qdrant server instead, set `QDRANT_URL` and optionally `QDRANT_API_KEY` before starting the app. The application uses the fixed `video_chunks_v5_siglip2_v2` collection, with one point per video chunk and named `visual` (768 dimensions) and `audio` (384 dimensions) vectors. Existing collections and videos are not deleted. Videos indexed with another visual encoder must be re-indexed into the SigLIP 2 collection before they can be searched by this application.

ترمّز الواجهة كل فيديو جديد باستخدام SigLIP 2 تلقائيًا. يتطلب أول استخدام تنزيل أوزان النموذج؛ وبعد اكتمال الفهرسة سيظهر الفيديو في قائمة البحث.

### فهرسة فيديو من سطر الأوامر
```bash
python scripts/ingest.py --video "data/video.mp4" --id "my_video"
# Language auto-detected by default. Override with: --lang ar
```

## الأداء (CPU only — مقاس على جهاز فعلي)

| المقياس | القيمة |
|---|---|
| سرعة الفهرسة (فيديو 2.5 دقيقة) | ~42 ثانية |
| زمن الاستعلام (Latency) | 25 – 60 ms |
| ذاكرة RAM بعد تحميل النماذج | ~7 GB |

## هيكل المشروع

```
video_rag_project/
├── src/videorag/
│   ├── ingestion/
│   │   ├── chunker.py       # Sliding-window temporal chunking (5s/1s)
│   │   ├── transcriber.py   # Whisper single-pass transcription
│   │   ├── encoder.py       # SigLIP 2 visual + MiniLM transcript encoders
│   │   └── pipeline.py      # Ingestion orchestrator
│   ├── retrieval/
│   │   └── searcher.py      # Joint visual+audio search & scoring
│   └── database/
│       └── vector_store.py  # Qdrant abstraction
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── routes/
│   │   ├── upload.py        # Upload & background ingestion
│   │   ├── search.py        # Search endpoint
│   │   └── health.py        # Health check + video list
│   └── templates/index.html # Web UI
├── scripts/
│   └── ingest.py            # CLI ingestion tool
├── requirements.txt
└── README.md
```

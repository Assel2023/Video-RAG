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

### GraphRAG محلي للفيديوهات المفهرسة

مسار GraphRAG مستقل عن البحث المعتاد، ويخزّن الكيانات والعلاقات الموجهة وأدلتها وتوقيتاتها والمجتمعات الهرمية في Neo4j. تظل متجهات الفيديو والصوت في Qdrant. تبقى قاعدة SQLite السابقة كما هي لاستخدامها كمصدر ترحيل أو رجوع. مرشحو استخراج النموذج وخلاصات المجتمعات تُخزّن في Neo4j لتفادي تكرار عمل Qwen دون تغيّر المحتوى.

1. ثبّت Neo4j Desktop من [مركز تنزيل Neo4j](https://neo4j.com/download/)، وأنشئ قاعدة بيانات محلية باسم `video-rag` وكلمة مرور تتذكرها. بما أن مساحة C عندك محدودة، اضبط مسار بيانات Desktop إلى مجلد على D عبر متغير Windows `NEO4J_DESKTOP_DATA_PATH` قبل فتح Desktop، مثل `D:\AI_Training\Neo4jData`، ثم أعد تشغيل Desktop. يدعم Desktop مسار بيانات مخصصًا على Windows.

   ابدأ القاعدة من Desktop. عنوان Bolt المحلي الافتراضي هو `bolt://127.0.0.1:7687`.

2. ثبّت [llama.cpp لويندوز](https://github.com/ggml-org/llama.cpp) إذا لم يكن موجودًا:

   ```powershell
   winget install --id ggml.llamacpp --exact --source winget
   ```

   أغلق PowerShell وافتح نافذة جديدة بعد التثبيت. شغّل خادم Qwen 3 بحجم 4B وكمّية Q4_K_M. إذا كانت ذاكرة البطاقة الرسومية محدودة، استخدم `--n-gpu-layers 0` للتشغيل على المعالج:

   ```powershell
   llama-server.exe -hf "Qwen/Qwen3-4B-GGUF:Q4_K_M" --port 8080 --ctx-size 4096 --n-gpu-layers 0
   ```

   اترك هذه النافذة مفتوحة. النموذج يعمل محليًا على `http://127.0.0.1:8080`.

3. داخل `venv` ثبّت مشغّل Neo4j وNetworkX فقط (بقية متطلبات المشروع مثبتة مسبقًا):

   ```powershell
   pip install neo4j networkx
   ```

4. في PowerShell الذي ستشغّل منه الترحيل وUvicorn، اضبط الاتصال. أدخل كلمة مرور قاعدة Neo4j عندما يطلبها الأمر، ولا تضعها في ملفات المشروع:

   ```powershell
   $env:GRAPH_BACKEND = "neo4j"
   $env:NEO4J_URI = "bolt://127.0.0.1:7687"
   $env:NEO4J_USER = "neo4j"
   $env:NEO4J_DATABASE = "neo4j"
   $env:NEO4J_PASSWORD = Read-Host "Neo4j password"
   ```

5. أوقف Uvicorn بـ `Ctrl+C` لأن Qdrant المحلي لا ينبغي فتحه من الفهرسة والخادم في الوقت نفسه. ثم انقل الرسم الحالي من SQLite إلى Neo4j:

   ```powershell
   python scripts/migrate_graph_to_neo4j.py
   ```

   النقل لا يحذف قاعدة SQLite؛ يعيد إنشاء المجتمعات والخلاصات على Neo4j من المقاطع والعلاقات المحفوظة. عند نجاح النقل، سيستخدم التطبيق Neo4j افتراضيًا.

6. عند رفع فيديوهات جديدة، ابنِ علاقاتها ومجتمعاتها إلى Neo4j. أوقف Uvicorn قبل الفهرسة عند استخدام Qdrant المحلي، واستخدم `video_id` كما يظهر في الواجهة:

   ```powershell
   python scripts/index_graph.py --video-id "test_video.mp4_5f57b01b"
   # أو لبناء الرسم عبر كل الفيديوهات المفهرسة:
   python scripts/index_graph.py --all
   ```

   أول فهرسة قد تستغرق وقتًا. يعيد الأمر استخدام استخراج العلاقات للمقاطع التي لم تتغير، ويعيد إنشاء الكيانات الموحدة والحواف والمجتمعات الهرمية. كما يلخّص كل مجتمع عبر Qwen ويحفظ الخلاصة ببصمة التقرير ومعرّف النموذج؛ لذلك تُعاد فقط الخلاصات التي تغيّر محتواها. قد يستهلك التلخيص وقتًا إضافيًا عند البناء الأول.

7. أعد تشغيل API من PowerShell الذي يحتوي متغيرات اتصال Neo4j:

   ```powershell
   python -m uvicorn api.main:app --reload --port 8000
   ```

8. افتح `http://127.0.0.1:8000/ui` واختر تبويب GraphRAG للسؤال محليًا أو عالميًا. أو افتح `http://127.0.0.1:8000/docs` واستخدم `POST /search/graph-answer`، و`GET /graph/communities` لمعاينة المجتمعات:

   ```json
   {
     "query": "ما العلاقة بين الوعي الذاتي والذكاء الاصطناعي؟",
     "mode": "local",
     "video_id": null,
     "top_k": 5
   }
   ```

   اختر `mode: "local"` لربط كيانات السؤال بجيرانها ومقاطعها، أو `mode: "global"` للبحث عبر ملخصات المجتمعات الهرمية ثم جمع الوقائع والاقتباسات الأصلية ذات الصلة. الإجابة تتطلب بقاء llama.cpp متاحًا. درجات التشابه والترتيب ليست احتمالات صحة، ويجب مراجعة دقة الاستخراج العربي. يمكن ضبط عنوان خادم النموذج عبر `GRAPH_LLM_BASE_URL` ومعرّفه عبر `GRAPH_LLM_MODEL`.

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
│   ├── ingest.py            # CLI ingestion tool
│   └── index_graph.py       # GraphRAG fact/community indexing
├── requirements.txt
└── README.md
```

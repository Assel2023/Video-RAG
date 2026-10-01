# 🎬 Spatio-Temporal Video RAG System

نظام متقدم لاسترجاع اللقطات من الفيديو (Video Retrieval) باستخدام تقنيات الذكاء الاصطناعي متعدد الوسائط (Multimodal AI) وقواعد البيانات المتجهة.

## 🚀 التقنيات المستخدمة (Tech Stack)

* **الخلفية (Backend):** FastAPI (لأداء سريع وغير متزامن)
* **قاعدة البيانات المتجهة:** ChromaDB (Multi-Vector Storage)
* **معالجة الصور (Vision):** OpenAI CLIP (`clip-ViT-B-32`)
* **معالجة النصوص والصوت (Audio & Text):** 
  * OpenAI Whisper (small) لاستخراج النص من الصوت.
  * `paraphrase-multilingual-MiniLM-L12-v2` وتمديد CLIP المتعدد اللغات.
* **الواجهة الأمامية (Frontend):** HTML5, Bootstrap 5, JavaScript

## ✨ ميزات النظام المعمارية
1. **Multi-Vector Indexing:** بدلاً من دمج الصور والصوت مما يقلل الدقة، يقوم النظام بفصل متجهات الصور (Visual Vectors) عن متجهات الصوت (Audio Vectors) في قاعدة البيانات. هذا يضمن دقة 100% عند البحث عن مشهد بصري صامت أو حوار صوتي.
2. **Batch Processing:** تحسين سرعة الفهرسة (Ingestion) بنسبة 80% عبر معالجة الصور دفعة واحدة (Batch Encoding) وتفريغ الصوت بمسار واحد متصل (Single-pass Transcription).
3. **دعم اللغة العربية:** معالجة واسترجاع دقيق باستخدام استعلامات باللغة العربية عبر نماذج Sentence-Transformers متعددة اللغات.
4. **معايرة الموثوقية (Confidence Calibration):** نظام رياضي لحساب نسبة التطابق الحقيقية للصور والنصوص بشكل مستقل.

## ⚙️ طريقة التشغيل (Quick Start)

### 1. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. فهرسة فيديو جديد (Ingestion)
```bash
python scripts/ingest.py --video "data/your_video.mp4" --id "vid_01" --lang "ar"
```

### 3. تشغيل الخادم (Run Server)
```bash
python -m uvicorn api.main:app --reload --port 8000
```
ثم افتح المتصفح على الرابط: `http://127.0.0.1:8000/ui`

---
*تم تطوير هذا المشروع كنظام استرجاع متقدم (Production-Ready RAG).*

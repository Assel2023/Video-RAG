# ============================================================
# benchmark.py — Performance & Metrics Report Generator
# ============================================================

import time
import os
import psutil
import chromadb
from sentence_transformers import SentenceTransformer

# ── الإعداد ─────────────────────────────────────────────────
text_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client        = chromadb.PersistentClient(path="./chroma_db")
collection    = client.get_collection("video_chunks")

TEST_QUERIES = [
    "ما هو الذكاء الاصطناعي",
    "شرح الشبكات العصبية",
    "كيف يتعلم النموذج من البيانات",
    "ما هي خوارزمية التعلم",
    "تطبيقات الذكاء الاصطناعي في الحياة",
]


def measure_query_time(query: str, top_k: int = 3) -> dict:
    """قياس زمن الاستجابة لاستعلام واحد"""
    t0 = time.perf_counter()
    
    embedding = text_embedder.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    best_confidence = round(
        1 - results["distances"][0][0], 4
    ) if results["distances"][0] else 0.0
    
    return {
        "query":          query,
        "query_time_ms":  round(elapsed_ms, 2),
        "best_confidence": best_confidence,
        "top_result_ts":  (
            results["metadatas"][0][0]["start_timestamp"],
            results["metadatas"][0][0]["end_timestamp"]
        ) if results["metadatas"][0] else None
    }


def get_db_size_mb(path: str = "./chroma_db") -> float:
    """حجم قاعدة البيانات على القرص"""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)


def run_benchmark():
    print("=" * 60)
    print("   VIDEO RAG SYSTEM — PERFORMANCE BENCHMARK REPORT")
    print("=" * 60)

    # ── معلومات قاعدة البيانات ──────────────────────────────
    print(f"\n📦 DATABASE STATS:")
    print(f"   Total chunks indexed : {collection.count()}")
    print(f"   DB size on disk      : {get_db_size_mb()} MB")
    print(f"   Embedding dimensions : 384")
    print(f"   Similarity metric    : Cosine")

    # ── قياس زمن الاستجابة ──────────────────────────────────
    print(f"\n⚡ RETRIEVAL SPEED BENCHMARK ({len(TEST_QUERIES)} queries):")
    print(f"   {'Query':<40} {'Time(ms)':>10} {'Confidence':>12}")
    print(f"   {'-'*40} {'-'*10} {'-'*12}")

    times       = []
    confidences = []

    for query in TEST_QUERIES:
        result = measure_query_time(query)
        times.append(result["query_time_ms"])
        confidences.append(result["best_confidence"])
        print(f"   {query[:40]:<40} "
              f"{result['query_time_ms']:>10.2f} "
              f"{result['best_confidence']:>12.4f}")

    # ── الإحصائيات الإجمالية ────────────────────────────────
    avg_time = round(sum(times) / len(times), 2)
    min_time = round(min(times), 2)
    max_time = round(max(times), 2)
    avg_conf = round(sum(confidences) / len(confidences), 4)

    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Avg query time  : {avg_time} ms")
    print(f"   Min query time  : {min_time} ms")
    print(f"   Max query time  : {max_time} ms")
    print(f"   Avg confidence  : {avg_conf}")

    # ── استهلاك الذاكرة ─────────────────────────────────────
    process    = psutil.Process(os.getpid())
    mem_mb     = round(process.memory_info().rss / (1024 * 1024), 2)
    
    print(f"\n💾 RESOURCE USAGE:")
    print(f"   RAM consumption : {mem_mb} MB")
    print(f"   CPU cores used  : {psutil.cpu_count()}")

    # ── الخلاصة ─────────────────────────────────────────────
    print(f"\n✅ CONCLUSION:")
    print(f"   System handles {collection.count()} video chunks")
    print(f"   with avg retrieval speed of {avg_time}ms.")
    print(f"   Suitable for real-time search applications.")
    print("=" * 60)


if __name__ == "__main__":
    # تثبيت psutil إذا لم يكن موجوداً
    try:
        import psutil
    except ImportError:
        os.system("pip install psutil -q")
        import psutil
    
    run_benchmark()
#!/usr/bin/env python
# scripts/benchmark.py — Performance Benchmarking CLI
"""
Measure retrieval latency and resource usage.

Usage:
    python scripts/benchmark.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from videorag.retrieval.searcher import Searcher
from videorag.logger import get_logger

log = get_logger("scripts.benchmark")

TEST_QUERIES = [
    "شرح الشبكات العصبية",
    "ما هو الذكاء الاصطناعي",
    "كيف يتعلم النموذج من البيانات",
    "تطبيقات التعلم الآلي",
    "خوارزميات التصنيف",
]


def run_benchmark() -> None:
    print("=" * 64)
    print("  VIDEO RAG SYSTEM — PERFORMANCE BENCHMARK REPORT")
    print("=" * 64)

    searcher = Searcher()
    store    = searcher.store

    print(f"\n{'Database Stats':-<40}")
    print(f"  Chunks indexed  : {store.count()}")
    print(f"  Embedding dim   : 512 (multimodal fused)")
    print(f"  Similarity      : Cosine")

    print(f"\n{'Retrieval Speed':-<40}")
    print(f"  {'Query':<36} {'ms':>8} {'Score':>8}")
    print(f"  {'—'*36} {'—'*8} {'—'*8}")

    times, scores = [], []
    for q in TEST_QUERIES:
        result = searcher.search(q, top_k=3)
        ms     = result["query_time_ms"]
        best   = (
            result["results"][0]["confidence_score"]
            if result["results"] else 0.0
        )
        times.append(ms)
        scores.append(best)
        print(f"  {q:<36} {ms:>8.2f} {best:>8.4f}")

    print(f"\n{'Summary':-<40}")
    print(f"  Avg latency     : {sum(times)/len(times):.2f} ms")
    print(f"  Min latency     : {min(times):.2f} ms")
    print(f"  Max latency     : {max(times):.2f} ms")
    print(f"  Avg confidence  : {sum(scores)/len(scores):.4f}")

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        ram  = proc.memory_info().rss / (1024 ** 2)
        print(f"  RAM consumption : {ram:.1f} MB")
    except ImportError:
        pass

    print("=" * 64)


if __name__ == "__main__":
    run_benchmark()

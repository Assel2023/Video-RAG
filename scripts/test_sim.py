import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from videorag.retrieval.searcher import Searcher
import argparse

def main():
    searcher = Searcher()
    queries = [
        "شجرة",
        "شخص",
        "طبيعة",
        "tree",
        "person",
        "nature"
    ]
    
    print(f"Chunks in DB: {searcher.store.count()}")
    for q in queries:
        print(f"\n--- Query: {q} ---")
        
        # Pure visual query
        visual_vec = searcher.encoder.encode_visual_text(q).reshape(-1)
        
        hits = searcher.store.search(
            visual_vec.tolist(), top_k=3, using="visual"
        )
        for h in hits:
            meta = h["metadata"]
            raw_sim = 1.0 - h["distance"]
            print(f"[{meta['video_id']}] {meta['start_timestamp']}s - {meta['end_timestamp']}s | Sim: {raw_sim:.4f} | HasAudio: {meta.get('has_audio')} | {meta.get('transcript', '')[:30]}")

if __name__ == "__main__":
    main()

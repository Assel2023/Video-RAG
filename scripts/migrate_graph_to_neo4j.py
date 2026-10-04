"""Copy the existing SQLite graph into Neo4j without deleting the source DB."""
from __future__ import annotations

import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from videorag.config import GRAPH_PATH
from videorag.database.neo4j_graph_store import Neo4jGraphStore
from index_graph import _get_model_id, _summarize_communities


def main() -> None:
    if not GRAPH_PATH.exists():
        raise SystemExit(f"ملف SQLite غير موجود: {GRAPH_PATH}")
    source = sqlite3.connect(GRAPH_PATH)
    source.row_factory = sqlite3.Row
    neo4j = Neo4jGraphStore()
    try:
        rows = source.execute(
            """SELECT c.chunk_id, c.video_id, c.start_timestamp, c.end_timestamp,
                      c.transcript, f.subject, f.predicate, f.object, f.evidence
               FROM graph_chunks c LEFT JOIN graph_facts f ON f.chunk_id = c.chunk_id
               ORDER BY c.video_id, c.start_timestamp"""
        ).fetchall()
        chunks_by_video: dict[str, OrderedDict[str, dict]] = {}
        for row in rows:
            video_id = str(row["video_id"])
            chunks = chunks_by_video.setdefault(video_id, OrderedDict())
            chunk = chunks.setdefault(str(row["chunk_id"]), {
                "chunk_id": str(row["chunk_id"]),
                "start_timestamp": float(row["start_timestamp"]),
                "end_timestamp": float(row["end_timestamp"]),
                "transcript": str(row["transcript"] or ""),
                "facts": [],
            })
            if row["subject"]:
                chunk["facts"].append({
                    "subject": str(row["subject"]),
                    "predicate": str(row["predicate"]),
                    "object": str(row["object"]),
                    "evidence": str(row["evidence"]),
                })
        if not chunks_by_video:
            raise SystemExit("قاعدة SQLite لا تحتوي مقاطع Graph؛ لم يتم تعديل Neo4j.")

        for video_id, chunks in chunks_by_video.items():
            neo4j.replace_video_facts(video_id, list(chunks.values()))
            print(f"Migrated {video_id}: {len(chunks)} chunks")

        print(f"Graph: {neo4j.rebuild_knowledge_graph()}")
        print(f"Communities: {neo4j.rebuild_communities()}")
        try:
            model_id = _get_model_id()
        except Exception as exc:
            raise RuntimeError(
                "نُقلت المقاطع والعلاقات إلى Neo4j، لكن تلخيص المجتمعات يحتاج "
                "تشغيل llama.cpp على المنفذ 8080. أعد تشغيل هذا الأمر بعد تشغيله."
            ) from exc
        generated, reused = _summarize_communities(neo4j, model_id)
        print(f"Community summaries: generated {generated}, reused {reused}")
        print(f"Neo4j migration complete; SQLite source preserved at {GRAPH_PATH}")
    finally:
        source.close()
        neo4j.close()


if __name__ == "__main__":
    main()

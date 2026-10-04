"""Evidence-first SQLite graph sidecar for transcript concepts and relations."""
from __future__ import annotations

import re
import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

from videorag.config import GRAPH_BACKEND, GRAPH_PATH


def _normalize_entity(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(value.split())


class GraphStore:
    """Store transcript chunks and evidence-linked graph facts."""

    def __init__(self, path: str | Path = GRAPH_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_chunks (
                chunk_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL NOT NULL,
                transcript TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_graph_chunks_video
                ON graph_chunks(video_id);
            CREATE TABLE IF NOT EXISTS graph_entities (
                entity_id INTEGER PRIMARY KEY,
                normalized_name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                UNIQUE(normalized_name, entity_type)
            );
            CREATE TABLE IF NOT EXISTS graph_mentions (
                chunk_id TEXT NOT NULL REFERENCES graph_chunks(chunk_id) ON DELETE CASCADE,
                entity_id INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
                mention_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY(chunk_id, entity_id, mention_text)
            );
            CREATE TABLE IF NOT EXISTS graph_relations (
                chunk_id TEXT NOT NULL REFERENCES graph_chunks(chunk_id) ON DELETE CASCADE,
                source_entity_id INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
                target_entity_id INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL CHECK(relation_type = 'co_mentioned'),
                evidence TEXT NOT NULL,
                PRIMARY KEY(chunk_id, source_entity_id, target_entity_id),
                CHECK(source_entity_id < target_entity_id)
            );
            CREATE TABLE IF NOT EXISTS graph_facts (
                fact_id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL REFERENCES graph_chunks(chunk_id) ON DELETE CASCADE,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                evidence TEXT NOT NULL,
                UNIQUE(chunk_id, subject, predicate, object, evidence)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_facts_chunk ON graph_facts(chunk_id);
            CREATE TABLE IF NOT EXISTS graph_extraction_cache (
                video_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                transcript_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(video_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS kg_entities (
                entity_id INTEGER PRIMARY KEY,
                entity_key TEXT NOT NULL UNIQUE,
                canonical_name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kg_entity_aliases (
                alias_key TEXT PRIMARY KEY,
                alias_text TEXT NOT NULL,
                entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS kg_fact_edges (
                edge_id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL REFERENCES graph_chunks(chunk_id)
                    ON DELETE CASCADE,
                subject_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
                predicate TEXT NOT NULL,
                predicate_key TEXT NOT NULL,
                object_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
                evidence TEXT NOT NULL,
                UNIQUE(chunk_id, subject_id, predicate_key, object_id, evidence)
            );
            CREATE INDEX IF NOT EXISTS idx_kg_edges_subject ON kg_fact_edges(subject_id);
            CREATE INDEX IF NOT EXISTS idx_kg_edges_object ON kg_fact_edges(object_id);
            CREATE INDEX IF NOT EXISTS idx_kg_edges_predicate ON kg_fact_edges(predicate_key);
            CREATE TABLE IF NOT EXISTS kg_communities (
                community_id TEXT PRIMARY KEY,
                level INTEGER NOT NULL,
                parent_id TEXT REFERENCES kg_communities(community_id)
                    ON DELETE CASCADE,
                member_count INTEGER NOT NULL,
                report TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kg_community_members (
                community_id TEXT NOT NULL REFERENCES kg_communities(community_id)
                    ON DELETE CASCADE,
                entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id)
                    ON DELETE CASCADE,
                PRIMARY KEY(community_id, entity_id)
            );
            CREATE TABLE IF NOT EXISTS kg_community_summaries (
                community_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._connection.commit()

    def get_cached_candidates(
        self,
        video_id: str,
        chunk_id: str,
        transcript_hash: str,
        model_id: str,
        prompt_hash: str,
    ) -> list[dict] | None:
        """Return the original model candidates for an unchanged chunk, if cached."""
        row = self._connection.execute(
            """SELECT candidates_json FROM graph_extraction_cache
               WHERE video_id = ? AND chunk_id = ? AND transcript_hash = ?
                 AND model_id = ? AND prompt_hash = ?""",
            (video_id, chunk_id, transcript_hash, model_id, prompt_hash),
        ).fetchone()
        if row is None:
            return None
        try:
            candidates = json.loads(row["candidates_json"])
        except (json.JSONDecodeError, TypeError):
            return None
        return candidates if isinstance(candidates, list) else None

    def existing_facts_as_candidates(
        self, video_id: str, chunk_id: str, transcript: str
    ) -> list[dict] | None:
        """Reuse already-indexed facts when the source transcript is unchanged."""
        row = self._connection.execute(
            "SELECT transcript FROM graph_chunks WHERE video_id = ? AND chunk_id = ?",
            (video_id, chunk_id),
        ).fetchone()
        if row is None or str(row["transcript"]) != transcript:
            return None
        facts = self._connection.execute(
            """SELECT subject, predicate, object, evidence FROM graph_facts
               WHERE chunk_id = ?""",
            (chunk_id,),
        ).fetchall()
        return [
            {
                "chunk_id": chunk_id,
                "subject": str(fact["subject"]),
                "predicate": str(fact["predicate"]),
                "object": str(fact["object"]),
                "evidence": str(fact["evidence"]),
            }
            for fact in facts
        ]

    def cache_candidates(
        self,
        video_id: str,
        chunk_id: str,
        transcript_hash: str,
        model_id: str,
        prompt_hash: str,
        candidates: list[dict],
    ) -> None:
        """Persist raw extraction candidates so validation changes need no LLM rerun."""
        with self._connection:
            self._connection.execute(
                """INSERT INTO graph_extraction_cache
                   (video_id, chunk_id, transcript_hash, model_id, prompt_hash, candidates_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(video_id, chunk_id) DO UPDATE SET
                     transcript_hash = excluded.transcript_hash,
                     model_id = excluded.model_id,
                     prompt_hash = excluded.prompt_hash,
                     candidates_json = excluded.candidates_json,
                     updated_at = CURRENT_TIMESTAMP""",
                (video_id, chunk_id, transcript_hash, model_id, prompt_hash,
                 json.dumps(candidates, ensure_ascii=False)),
            )

    def replace_video(self, video_id: str, chunks: list[dict]) -> None:
        """Atomically replace one video's graph evidence; other videos are untouched."""
        db = self._connection
        with db:
            db.execute("DELETE FROM graph_chunks WHERE video_id = ?", (video_id,))
            for chunk in chunks:
                chunk_id = str(chunk["chunk_id"])
                transcript = str(chunk.get("transcript") or "").strip()
                db.execute(
                    "INSERT INTO graph_chunks VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, video_id, float(chunk["start_timestamp"]),
                     float(chunk["end_timestamp"]), transcript),
                )
                mentions = chunk.get("entities", [])
                entity_ids: set[int] = set()
                for mention in mentions:
                    name = str(mention.get("text") or "").strip()
                    normalized = _normalize_entity(name)
                    entity_type = str(mention.get("label") or "MISC")
                    if not normalized:
                        continue
                    db.execute(
                        """INSERT INTO graph_entities(normalized_name, display_name, entity_type)
                           VALUES (?, ?, ?)
                           ON CONFLICT(normalized_name, entity_type) DO UPDATE SET
                             display_name = CASE
                               WHEN length(excluded.display_name) > length(display_name)
                               THEN excluded.display_name ELSE display_name END""",
                        (normalized, name, entity_type),
                    )
                    row = db.execute(
                        "SELECT entity_id FROM graph_entities WHERE normalized_name = ? AND entity_type = ?",
                        (normalized, entity_type),
                    ).fetchone()
                    entity_id = int(row["entity_id"])
                    entity_ids.add(entity_id)
                    db.execute(
                        "INSERT OR IGNORE INTO graph_mentions VALUES (?, ?, ?, ?)",
                        (chunk_id, entity_id, name, float(mention.get("score", 0.0))),
                    )

                ordered_ids = sorted(entity_ids)
                for index, source_id in enumerate(ordered_ids):
                    for target_id in ordered_ids[index + 1:]:
                        db.execute(
                            "INSERT OR IGNORE INTO graph_relations VALUES (?, ?, ?, 'co_mentioned', ?)",
                            (chunk_id, source_id, target_id, transcript),
                        )

    def replace_video_facts(self, video_id: str, chunks: list[dict]) -> None:
        """Replace one video's graph with quote-grounded subject/relation/object facts."""
        db = self._connection
        with db:
            db.execute("DELETE FROM graph_chunks WHERE video_id = ?", (video_id,))
            db.execute(
                "DELETE FROM graph_entities WHERE NOT EXISTS "
                "(SELECT 1 FROM graph_mentions gm WHERE gm.entity_id = graph_entities.entity_id)"
            )
            for chunk in chunks:
                chunk_id = str(chunk["chunk_id"])
                transcript = str(chunk.get("transcript") or "").strip()
                db.execute(
                    "INSERT INTO graph_chunks VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, video_id, float(chunk["start_timestamp"]),
                     float(chunk["end_timestamp"]), transcript),
                )
                for fact in chunk.get("facts", []):
                    subject = str(fact.get("subject") or "").strip()
                    predicate = str(fact.get("predicate") or "").strip()
                    object_ = str(fact.get("object") or "").strip()
                    evidence = str(fact.get("evidence") or "").strip()
                    if not all((subject, predicate, object_, evidence)):
                        continue
                    db.execute(
                        "INSERT OR IGNORE INTO graph_facts "
                        "(chunk_id, subject, predicate, object, evidence) VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, subject, predicate, object_, evidence),
                    )

    def remove_videos_except(self, video_ids: set[str]) -> list[str]:
        """Remove graph evidence for videos no longer present in Qdrant."""
        current = {
            str(row[0]) for row in self._connection.execute(
                "SELECT DISTINCT video_id FROM graph_chunks"
            ).fetchall()
        }
        stale = sorted(current - video_ids)
        if stale:
            with self._connection:
                self._connection.executemany(
                    "DELETE FROM graph_chunks WHERE video_id = ?",
                    [(video_id,) for video_id in stale],
                )
                self._connection.execute(
                    "DELETE FROM graph_entities WHERE NOT EXISTS "
                    "(SELECT 1 FROM graph_mentions gm WHERE gm.entity_id = graph_entities.entity_id)"
                )
        return stale

    def rebuild_knowledge_graph(self) -> dict[str, int]:
        """Build a normalized, corpus-wide entity graph from evidence-backed facts."""
        db = self._connection
        with db:
            db.execute("DELETE FROM kg_communities")
            db.execute("DELETE FROM kg_fact_edges")
            db.execute("DELETE FROM kg_entity_aliases")
            db.execute("DELETE FROM kg_entities")

            facts = db.execute(
                """SELECT gf.chunk_id, gf.subject, gf.predicate, gf.object, gf.evidence
                   FROM graph_facts gf JOIN graph_chunks gc ON gc.chunk_id = gf.chunk_id
                   ORDER BY gc.video_id, gc.start_timestamp, gf.fact_id"""
            ).fetchall()
            entity_ids: dict[str, int] = {}

            def entity_id_for(name: str) -> int:
                key = _normalize_entity(name)
                entity_id = entity_ids.get(key)
                if entity_id is not None:
                    return entity_id
                db.execute(
                    "INSERT INTO kg_entities(entity_key, canonical_name) VALUES (?, ?)",
                    (key, name),
                )
                entity_id = int(db.execute(
                    "SELECT entity_id FROM kg_entities WHERE entity_key = ?", (key,)
                ).fetchone()[0])
                entity_ids[key] = entity_id
                db.execute(
                    "INSERT OR IGNORE INTO kg_entity_aliases(alias_key, alias_text, entity_id) "
                    "VALUES (?, ?, ?)", (key, name, entity_id)
                )
                return entity_id

            for fact in facts:
                subject = str(fact["subject"]).strip()
                predicate = str(fact["predicate"]).strip()
                object_ = str(fact["object"]).strip()
                subject_id = entity_id_for(subject)
                object_id = entity_id_for(object_)
                db.execute(
                    """INSERT OR IGNORE INTO kg_fact_edges
                       (chunk_id, subject_id, predicate, predicate_key, object_id, evidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (fact["chunk_id"], subject_id, predicate,
                     _normalize_entity(predicate), object_id, fact["evidence"]),
                )
            edge_count = int(db.execute("SELECT count(*) FROM kg_fact_edges").fetchone()[0])
            return {"entities": len(entity_ids), "facts": edge_count}

    def rebuild_communities(self, seed: int = 42) -> dict[str, int]:
        """Detect hierarchical graph communities and store evidence-grounded reports."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "Community detection requires NetworkX; install project requirements."
            ) from exc

        rows = self._connection.execute(
            """SELECT e.subject_id, e.object_id, count(*) AS weight
               FROM kg_fact_edges e WHERE e.subject_id != e.object_id
               GROUP BY e.subject_id, e.object_id"""
        ).fetchall()
        graph = nx.Graph()
        names = {
            int(row["entity_id"]): str(row["canonical_name"])
            for row in self._connection.execute(
                "SELECT entity_id, canonical_name FROM kg_entities"
            ).fetchall()
        }
        graph.add_nodes_from(names)
        for row in rows:
            graph.add_edge(
                int(row["subject_id"]), int(row["object_id"]),
                weight=int(row["weight"]),
            )

        if graph.number_of_nodes() <= 1 or graph.number_of_edges() == 0:
            partitions = [{node} for node in graph.nodes]
        else:
            partitions = list(nx.community.louvain_partitions(
                graph, weight="weight", seed=seed
            ))
        if not partitions and graph.number_of_nodes():
            partitions = [{node} for node in graph.nodes]

        # Louvain yields increasingly coarse partitions; connect each fine
        # community to the smallest containing community at the next level.
        level_rows: list[list[tuple[str, set[int], str, str | None]]] = []
        for level, partition in enumerate(partitions):
            current = []
            for members in partition:
                member_ids = {int(member) for member in members}
                ordered_names = sorted(names[member] for member in member_ids)
                digest = hashlib.sha1("\0".join(ordered_names).encode("utf-8")).hexdigest()[:12]
                community_id = f"L{level}-{digest}"
                current.append((community_id, member_ids, "، ".join(ordered_names[:6]), None))
            level_rows.append(current)

        for level, communities in enumerate(level_rows[:-1]):
            next_level = level_rows[level + 1]
            for index, (community_id, members, title, _) in enumerate(communities):
                parent_id = next(
                    (candidate[0] for candidate in next_level if members.issubset(candidate[1])),
                    None,
                )
                communities[index] = (community_id, members, title, parent_id)

        with self._connection:
            self._connection.execute("DELETE FROM kg_communities")
            for level in range(len(level_rows) - 1, -1, -1):
                communities = level_rows[level]
                for community_id, member_ids, title, parent_id in communities:
                    placeholders = ",".join("?" for _ in member_ids) or "NULL"
                    facts = self._connection.execute(
                        f"""SELECT DISTINCT s.canonical_name AS subject, e.predicate,
                                   o.canonical_name AS object, e.evidence, e.chunk_id,
                                   c.video_id, c.start_timestamp, c.end_timestamp
                            FROM kg_fact_edges e
                            JOIN kg_entities s ON s.entity_id = e.subject_id
                            JOIN kg_entities o ON o.entity_id = e.object_id
                            JOIN graph_chunks c ON c.chunk_id = e.chunk_id
                            WHERE e.subject_id IN ({placeholders})
                              AND e.object_id IN ({placeholders})
                            ORDER BY c.video_id, c.start_timestamp LIMIT 80""",
                        (*member_ids, *member_ids),
                    ).fetchall() if member_ids else []
                    report_lines = [f"مجتمع مفاهيم: {title}"]
                    for fact in facts:
                        report_lines.append(
                            f"- {fact['subject']} — {fact['predicate']} — {fact['object']} "
                            f"[video={fact['video_id']} {fact['start_timestamp']:.1f}-"
                            f"{fact['end_timestamp']:.1f}s] الدليل: {fact['evidence']}"
                        )
                    report = "\n".join(report_lines)
                    content_hash = hashlib.sha256(report.encode("utf-8")).hexdigest()
                    self._connection.execute(
                        """INSERT INTO kg_communities
                           (community_id, level, parent_id, member_count, report, content_hash)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (community_id, level, parent_id, len(member_ids), report, content_hash),
                    )
                    self._connection.executemany(
                        "INSERT INTO kg_community_members(community_id, entity_id) VALUES (?, ?)",
                        [(community_id, member_id) for member_id in member_ids],
                    )
        return {
            "levels": len(level_rows),
            "communities": sum(len(level) for level in level_rows),
            "entities": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }

    def entity_catalog(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT entity_id, canonical_name FROM kg_entities ORDER BY canonical_name COLLATE NOCASE"
        ).fetchall()
        return [dict(row) for row in rows]

    def community_reports(
        self, level: int | None = None, video_id: str | None = None
    ) -> list[dict]:
        filters = []
        params: list[object] = []
        if level is not None:
            filters.append("c.level = ?")
            params.append(level)
        if video_id:
            filters.append(
                "EXISTS (SELECT 1 FROM kg_community_members m "
                "JOIN kg_fact_edges e ON e.subject_id = m.entity_id "
                "OR e.object_id = m.entity_id "
                "JOIN graph_chunks gc ON gc.chunk_id = e.chunk_id "
                "WHERE m.community_id = c.community_id AND gc.video_id = ?)"
            )
            params.append(video_id)
        clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self._connection.execute(
            f"""SELECT c.community_id, c.level, c.parent_id, c.member_count,
                       c.report, c.content_hash, s.summary, s.model_id AS summary_model_id
                FROM kg_communities c
                LEFT JOIN kg_community_summaries s
                  ON s.community_id = c.community_id AND s.content_hash = c.content_hash
                {clause}
                ORDER BY c.level DESC, c.member_count DESC""",
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_community_summary(
        self, community_id: str, content_hash: str, model_id: str, summary: str
    ) -> None:
        """Cache an LLM summary only for the exact report content and model."""
        with self._connection:
            self._connection.execute(
                """INSERT INTO kg_community_summaries
                   (community_id, content_hash, model_id, summary)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(community_id) DO UPDATE SET
                     content_hash = excluded.content_hash,
                     model_id = excluded.model_id,
                     summary = excluded.summary,
                     updated_at = CURRENT_TIMESTAMP""",
                (community_id, content_hash, model_id, summary),
            )

    def cached_community_summary(
        self, community_id: str, content_hash: str, model_id: str
    ) -> str | None:
        row = self._connection.execute(
            """SELECT summary FROM kg_community_summaries
               WHERE community_id = ? AND content_hash = ? AND model_id = ?""",
            (community_id, content_hash, model_id),
        ).fetchone()
        return str(row["summary"]) if row else None

    def fact_edges_for_entities(
        self,
        entity_ids: list[int],
        top_k: int = 30,
        video_id: str | None = None,
    ) -> list[dict]:
        """Fetch one-hop fact evidence around semantically matched graph entities."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        video_clause = "AND c.video_id = ?" if video_id else ""
        params = (*entity_ids, *entity_ids, *((video_id,) if video_id else ()), top_k)
        rows = self._connection.execute(
            f"""SELECT e.edge_id, e.chunk_id, c.video_id, c.start_timestamp,
                       c.end_timestamp, s.canonical_name AS subject, e.predicate,
                       o.canonical_name AS object, e.evidence
                FROM kg_fact_edges e
                JOIN kg_entities s ON s.entity_id = e.subject_id
                JOIN kg_entities o ON o.entity_id = e.object_id
                JOIN graph_chunks c ON c.chunk_id = e.chunk_id
                WHERE (e.subject_id IN ({placeholders}) OR e.object_id IN ({placeholders}))
                  {video_clause}
                ORDER BY c.start_timestamp LIMIT ?""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def facts_for_community(
        self, community_id: str, top_k: int = 60, video_id: str | None = None
    ) -> list[dict]:
        members = self._connection.execute(
            "SELECT entity_id FROM kg_community_members WHERE community_id = ?",
            (community_id,),
        ).fetchall()
        entity_ids = [int(row["entity_id"]) for row in members]
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        video_clause = "AND c.video_id = ?" if video_id else ""
        params = (*entity_ids, *entity_ids, *((video_id,) if video_id else ()), top_k)
        rows = self._connection.execute(
            f"""SELECT e.edge_id, e.chunk_id, c.video_id, c.start_timestamp,
                       c.end_timestamp, s.canonical_name AS subject, e.predicate,
                       o.canonical_name AS object, e.evidence
                FROM kg_fact_edges e
                JOIN kg_entities s ON s.entity_id = e.subject_id
                JOIN kg_entities o ON o.entity_id = e.object_id
                JOIN graph_chunks c ON c.chunk_id = e.chunk_id
                WHERE e.subject_id IN ({placeholders})
                  AND e.object_id IN ({placeholders}) {video_clause}
                ORDER BY c.video_id, c.start_timestamp LIMIT ?""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _query_terms(value: str) -> set[str]:
        tokens = re.findall(r"[\w]+", _normalize_entity(value), flags=re.UNICODE)
        terms = set()
        for token in tokens:
            terms.add(token)
            if token.startswith("ال") and len(token) > 4:
                terms.add(token[2:])
        return terms

    def search_facts(self, query: str, top_k: int = 5, video_id: str | None = None) -> dict:
        """Retrieve facts sharing a meaningful subject/object/relation term with the query."""
        query_terms = self._query_terms(query)
        if not query_terms:
            return {"query_terms": [], "matched_terms": [], "results": []}
        where = "WHERE gc.video_id = ?" if video_id else ""
        params = (video_id,) if video_id else ()
        rows = self._connection.execute(
            f"""SELECT gc.chunk_id, gc.video_id, gc.start_timestamp,
                       gc.end_timestamp, gc.transcript, gf.subject,
                       gf.predicate, gf.object, gf.evidence
                FROM graph_facts gf
                JOIN graph_chunks gc ON gc.chunk_id = gf.chunk_id
                {where}
                ORDER BY gc.start_timestamp""",
            params,
        ).fetchall()
        grouped: dict[str, dict] = {}
        matched_terms: set[str] = set()
        for row in rows:
            fact_terms = self._query_terms(
                f"{row['subject']} {row['predicate']} {row['object']}"
            )
            matches = query_terms & fact_terms
            if not matches:
                continue
            matched_terms.update(matches)
            item = grouped.setdefault(row["chunk_id"], {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "evidence": row["transcript"],
                "facts": [],
                "_matched_terms": set(),
            })
            item["facts"].append({
                "subject": row["subject"], "predicate": row["predicate"],
                "object": row["object"], "evidence": row["evidence"],
            })
            item["_matched_terms"].update(matches)

        results = list(grouped.values())
        for item in results:
            item["graph_score"] = len(item.pop("_matched_terms"))
        results.sort(key=lambda item: item["graph_score"], reverse=True)
        return {
            "query_terms": sorted(query_terms),
            "matched_terms": sorted(matched_terms),
            "results": results[:top_k],
        }

    def list_fact_entities(self, video_id: str | None = None) -> list[dict]:
        """List subject/object concepts with the number of evidence chunks."""
        where = "WHERE gc.video_id = ?" if video_id else ""
        params = (video_id,) if video_id else ()
        rows = self._connection.execute(
            f"""SELECT entity_name AS name, count(DISTINCT chunk_id) AS chunk_count
                FROM (
                    SELECT gf.subject AS entity_name, gf.chunk_id, gc.video_id
                    FROM graph_facts gf JOIN graph_chunks gc ON gc.chunk_id = gf.chunk_id
                    UNION ALL
                    SELECT gf.object AS entity_name, gf.chunk_id, gc.video_id
                    FROM graph_facts gf JOIN graph_chunks gc ON gc.chunk_id = gf.chunk_id
                ) gc
                {where}
                GROUP BY entity_name
                ORDER BY chunk_count DESC, name COLLATE NOCASE""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def fact_summary(self, video_id: str) -> dict[str, int]:
        row = self._connection.execute(
            """SELECT count(DISTINCT gc.chunk_id) AS chunks,
                      (SELECT count(DISTINCT concept) FROM (
                          SELECT gf2.subject AS concept
                          FROM graph_facts gf2 JOIN graph_chunks gc2
                            ON gc2.chunk_id = gf2.chunk_id
                          WHERE gc2.video_id = ?
                          UNION
                          SELECT gf3.object AS concept
                          FROM graph_facts gf3 JOIN graph_chunks gc3
                            ON gc3.chunk_id = gf3.chunk_id
                          WHERE gc3.video_id = ?
                      )) AS entities,
                      count(gf.fact_id) AS relations
               FROM graph_chunks gc LEFT JOIN graph_facts gf ON gf.chunk_id = gc.chunk_id
               WHERE gc.video_id = ?""",
            (video_id, video_id, video_id),
        ).fetchone()
        return {key: int(row[key]) for key in ("chunks", "entities", "relations")}

    def summary(self, video_id: str) -> dict[str, int]:
        row = self._connection.execute(
            """SELECT
                 (SELECT count(*) FROM graph_chunks WHERE video_id = ?) AS chunks,
                 (SELECT count(DISTINCT gm.entity_id) FROM graph_mentions gm
                    JOIN graph_chunks gc ON gc.chunk_id = gm.chunk_id WHERE gc.video_id = ?) AS entities,
                 (SELECT count(*) FROM graph_relations gr
                    JOIN graph_chunks gc ON gc.chunk_id = gr.chunk_id WHERE gc.video_id = ?) AS relations""",
            (video_id, video_id, video_id),
        ).fetchone()
        return {key: int(row[key]) for key in ("chunks", "entities", "relations")}

    def search(self, query: str, top_k: int = 5, video_id: str | None = None) -> dict:
        """Retrieve transcript evidence seeded by entity names stated in the query."""
        query_tokens = set(re.findall(r"[\w]+", _normalize_entity(query), flags=re.UNICODE))
        if not query_tokens:
            return {"query_entities": [], "results": []}

        entities = self._connection.execute(
            "SELECT entity_id, normalized_name, display_name FROM graph_entities"
        ).fetchall()
        matched = {
            int(entity["entity_id"]): entity["display_name"]
            for entity in entities
            if len(entity["normalized_name"]) >= 3
            and set(entity["normalized_name"].split()).issubset(query_tokens)
        }
        if not matched:
            return {"query_entities": [], "results": []}

        ids = tuple(matched)
        placeholders = ",".join("?" for _ in ids)
        video_clause = " AND gc2.video_id = ?" if video_id else ""
        params = (*ids, *((video_id,) if video_id else ()))
        rows = self._connection.execute(
            f"""SELECT gc.chunk_id, gc.video_id, gc.start_timestamp,
                       gc.end_timestamp, gc.transcript,
                       ge.entity_id, ge.display_name, ge.entity_type
                FROM graph_mentions gm
                JOIN graph_chunks gc ON gc.chunk_id = gm.chunk_id
                JOIN graph_entities ge ON ge.entity_id = gm.entity_id
                WHERE gm.chunk_id IN (
                    SELECT gm2.chunk_id FROM graph_mentions gm2
                    JOIN graph_chunks gc2 ON gc2.chunk_id = gm2.chunk_id
                    WHERE gm2.entity_id IN ({placeholders}){video_clause}
                )
                ORDER BY gc.start_timestamp""",
            params,
        ).fetchall()

        grouped: dict[str, dict] = {}
        for row in rows:
            item = grouped.setdefault(row["chunk_id"], {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "evidence": row["transcript"],
                "entities": [],
                "matched_entities": [],
            })
            entity = {"name": row["display_name"], "type": row["entity_type"]}
            item["entities"].append(entity)
            if int(row["entity_id"]) in matched:
                item["matched_entities"].append(row["display_name"])

        results = list(grouped.values())
        for item in results:
            item["matched_entities"] = sorted(set(item["matched_entities"]))
            item["entities"] = list({
                (entity["name"], entity["type"]): entity
                for entity in item["entities"]
            }.values())
            # This is an evidence ranking count, not a calibrated probability.
            item["graph_score"] = (
                2 * len(item["matched_entities"])
                + max(0, len({entity["name"] for entity in item["entities"]})
                      - len(item["matched_entities"]))
            )
        results.sort(key=lambda item: item["graph_score"], reverse=True)
        return {"query_entities": sorted(set(matched.values())), "results": results[:top_k]}

    def list_entities(self, video_id: str | None = None) -> list[dict]:
        """List entities present in one video, ordered by evidence count."""
        where = "WHERE gc.video_id = ?" if video_id else ""
        params = (video_id,) if video_id else ()
        rows = self._connection.execute(
            f"""SELECT ge.display_name AS name, ge.entity_type AS type,
                       count(DISTINCT gm.chunk_id) AS chunk_count
                FROM graph_mentions gm
                JOIN graph_chunks gc ON gc.chunk_id = gm.chunk_id
                JOIN graph_entities ge ON ge.entity_id = gm.entity_id
                {where}
                GROUP BY ge.entity_id
                ORDER BY chunk_count DESC, ge.display_name COLLATE NOCASE""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._connection.close()


# Keep SQLite as an explicit migration source and rollback backend. New app and
# indexing code should obtain its configured implementation through this factory.
SQLiteGraphStore = GraphStore


def create_graph_store():
    if GRAPH_BACKEND == "neo4j":
        from videorag.database.neo4j_graph_store import Neo4jGraphStore

        return Neo4jGraphStore()
    if GRAPH_BACKEND == "sqlite":
        return SQLiteGraphStore()
    raise RuntimeError(
        f"GRAPH_BACKEND غير معروف: {GRAPH_BACKEND}. اختر neo4j أو sqlite."
    )

"""Neo4j-backed transcript knowledge graph and GraphRAG sidecar."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from videorag.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(value.split())


class Neo4jGraphStore:
    """GraphStore-compatible adapter using native Neo4j nodes and relationships."""

    def __init__(self):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Neo4j Python Driver غير مثبت. نفّذ: pip install neo4j"
            ) from exc
        if not NEO4J_PASSWORD:
            raise RuntimeError(
                "عرّف NEO4J_PASSWORD في PowerShell قبل تشغيل التطبيق أو الفهرسة."
            )
        self.database = NEO4J_DATABASE
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        try:
            self.driver.verify_connectivity()
            self._run(
                "CREATE CONSTRAINT video_chunk_id IF NOT EXISTS "
                "FOR (n:VideoChunk) REQUIRE n.chunk_id IS UNIQUE"
            )
            self._run(
                "CREATE CONSTRAINT graph_entity_key IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.entity_key IS UNIQUE"
            )
            self._run(
                "CREATE CONSTRAINT graph_fact_key IF NOT EXISTS "
                "FOR (n:Fact) REQUIRE n.fact_key IS UNIQUE"
            )
            self._run(
                "CREATE CONSTRAINT graph_community_id IF NOT EXISTS "
                "FOR (n:Community) REQUIRE n.community_id IS UNIQUE"
            )
            self._run(
                "CREATE CONSTRAINT graph_cache_key IF NOT EXISTS "
                "FOR (n:ExtractionCache) REQUIRE n.cache_key IS UNIQUE"
            )
            self._run("CREATE INDEX graph_fact_video IF NOT EXISTS FOR (n:Fact) ON (n.video_id)")
            self._run("CREATE INDEX graph_chunk_video IF NOT EXISTS FOR (n:VideoChunk) ON (n.video_id)")
        except Exception as exc:
            self.driver.close()
            raise RuntimeError(
                f"تعذر الاتصال بـ Neo4j عند {NEO4J_URI}. تحقق من أن قاعدة البيانات "
                "تعمل، وأن NEO4J_USER وNEO4J_PASSWORD صحيحان."
            ) from exc

    def _run(self, query: str, **parameters: Any) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            query, parameters_=parameters, database_=self.database
        )
        return [record.data() for record in records]

    @staticmethod
    def _cache_key(video_id: str, chunk_id: str) -> str:
        return hashlib.sha256(f"{video_id}\0{chunk_id}".encode()).hexdigest()

    def get_cached_candidates(
        self, video_id: str, chunk_id: str, transcript_hash: str,
        model_id: str, prompt_hash: str,
    ) -> list[dict] | None:
        rows = self._run(
            "MATCH (c:ExtractionCache {cache_key: $key}) "
            "WHERE c.transcript_hash = $transcript_hash AND c.model_id = $model_id "
            "AND c.prompt_hash = $prompt_hash RETURN c.candidates_json AS payload",
            key=self._cache_key(video_id, chunk_id), transcript_hash=transcript_hash,
            model_id=model_id, prompt_hash=prompt_hash,
        )
        if not rows:
            return None
        try:
            value = json.loads(rows[0]["payload"])
            return value if isinstance(value, list) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def cache_candidates(
        self, video_id: str, chunk_id: str, transcript_hash: str,
        model_id: str, prompt_hash: str, candidates: list[dict],
    ) -> None:
        self._run(
            "MERGE (c:ExtractionCache {cache_key: $key}) SET c.video_id = $video_id, "
            "c.chunk_id = $chunk_id, c.transcript_hash = $transcript_hash, "
            "c.model_id = $model_id, c.prompt_hash = $prompt_hash, "
            "c.candidates_json = $candidates_json, c.updated_at = datetime()",
            key=self._cache_key(video_id, chunk_id), video_id=video_id,
            chunk_id=chunk_id, transcript_hash=transcript_hash, model_id=model_id,
            prompt_hash=prompt_hash,
            candidates_json=json.dumps(candidates, ensure_ascii=False),
        )

    def existing_facts_as_candidates(
        self, video_id: str, chunk_id: str, transcript: str
    ) -> list[dict] | None:
        rows = self._run(
            "MATCH (c:VideoChunk {chunk_id: $chunk_id, video_id: $video_id}) "
            "WHERE c.transcript = $transcript "
            "OPTIONAL MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact {chunk_id: $chunk_id}) "
            "-[:OBJECT_OF]->(o:Entity) "
            "RETURN c.chunk_id AS found, collect(CASE WHEN f IS NULL THEN NULL ELSE "
            "{chunk_id: f.chunk_id, subject: s.name, predicate: f.predicate, "
            "object: o.name, evidence: f.evidence} END) AS facts",
            video_id=video_id, chunk_id=chunk_id, transcript=transcript,
        )
        if not rows or rows[0]["found"] is None:
            return None
        return [fact for fact in rows[0]["facts"] if fact]

    def replace_video_facts(self, video_id: str, chunks: list[dict]) -> None:
        """Replace one video's transcript units and fact graph transactionally."""
        records = []
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            facts = []
            for fact in chunk.get("facts", []):
                subject = str(fact.get("subject") or "").strip()
                predicate = str(fact.get("predicate") or "").strip()
                object_ = str(fact.get("object") or "").strip()
                evidence = str(fact.get("evidence") or "").strip()
                if not all((subject, predicate, object_, evidence)):
                    continue
                fact_key = hashlib.sha256(
                    json.dumps(
                        [chunk_id, _normalize(subject), _normalize(predicate),
                         _normalize(object_), evidence], ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                facts.append({
                    "fact_key": fact_key, "subject": subject,
                    "subject_key": _normalize(subject), "predicate": predicate,
                    "predicate_key": _normalize(predicate),
                    "object": object_, "object_key": _normalize(object_),
                    "evidence": evidence,
                })
            records.append({
                "chunk_id": chunk_id,
                "start_timestamp": float(chunk["start_timestamp"]),
                "end_timestamp": float(chunk["end_timestamp"]),
                "transcript": str(chunk.get("transcript") or "").strip(),
                "facts": facts,
            })

        def write(tx):
            tx.run(
                "MATCH (f:Fact {video_id: $video_id}) DETACH DELETE f",
                video_id=video_id,
            )
            tx.run(
                "MATCH (c:VideoChunk {video_id: $video_id}) DETACH DELETE c",
                video_id=video_id,
            )
            tx.run(
                "UNWIND $chunks AS row "
                "CREATE (c:VideoChunk {chunk_id: row.chunk_id, video_id: $video_id, "
                "start_timestamp: row.start_timestamp, end_timestamp: row.end_timestamp, "
                "transcript: row.transcript}) "
                "WITH c, row UNWIND row.facts AS fact "
                "MERGE (s:Entity {entity_key: fact.subject_key}) "
                "ON CREATE SET s.name = fact.subject, s.aliases = [fact.subject] "
                "ON MATCH SET s.aliases = CASE WHEN fact.subject IN coalesce(s.aliases, []) "
                "THEN s.aliases ELSE coalesce(s.aliases, []) + [fact.subject] END "
                "WITH c, fact, s "
                "MERGE (o:Entity {entity_key: fact.object_key}) "
                "ON CREATE SET o.name = fact.object, o.aliases = [fact.object] "
                "ON MATCH SET o.aliases = CASE WHEN fact.object IN coalesce(o.aliases, []) "
                "THEN o.aliases ELSE coalesce(o.aliases, []) + [fact.object] END "
                "WITH c, fact, s, o "
                "MERGE (f:Fact {fact_key: fact.fact_key}) "
                "SET f.chunk_id = c.chunk_id, f.video_id = $video_id, "
                "f.start_timestamp = c.start_timestamp, f.end_timestamp = c.end_timestamp, "
                "f.predicate = fact.predicate, f.predicate_key = fact.predicate_key, "
                "f.evidence = fact.evidence "
                "MERGE (s)-[:SUBJECT_OF]->(f) "
                "MERGE (f)-[:OBJECT_OF]->(o) "
                "MERGE (f)-[:SUPPORTED_BY]->(c)",
                video_id=video_id, chunks=records,
            )

        with self.driver.session(database=self.database) as session:
            session.execute_write(write)

    def remove_videos_except(self, video_ids: set[str]) -> list[str]:
        rows = self._run("MATCH (c:VideoChunk) RETURN DISTINCT c.video_id AS video_id")
        stale = sorted({str(row["video_id"]) for row in rows} - video_ids)
        if stale:
            self._run(
                "MATCH (f:Fact) WHERE f.video_id IN $videos DETACH DELETE f",
                videos=stale,
            )
            self._run(
                "MATCH (c:VideoChunk) WHERE c.video_id IN $videos DETACH DELETE c",
                videos=stale,
            )
        return stale

    def rebuild_knowledge_graph(self) -> dict[str, int]:
        self._run("MATCH (c:Community) DETACH DELETE c")
        self._run("MATCH (e:Entity) WHERE NOT (e)<-[:SUBJECT_OF]-(:Fact) "
                  "AND NOT (e)<-[:OBJECT_OF]-(:Fact) DETACH DELETE e")
        rows = self._run(
            "MATCH (e:Entity) WITH count(e) AS entities "
            "MATCH (f:Fact) RETURN entities, count(f) AS facts"
        )
        chunks = self._run("MATCH (c:VideoChunk) RETURN count(c) AS chunks")
        return {
            "entities": int(rows[0]["entities"]) if rows else 0,
            "facts": int(rows[0]["facts"]) if rows else 0,
            "chunks": int(chunks[0]["chunks"]) if chunks else 0,
        }

    def rebuild_communities(self, seed: int = 42) -> dict[str, int]:
        import networkx as nx

        rows = self._run(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)-[:OBJECT_OF]->(o:Entity) "
            "RETURN s.entity_key AS subject_id, o.entity_key AS object_id, count(f) AS weight"
        )
        entities = self._run("MATCH (e:Entity) RETURN e.entity_key AS id, e.name AS name")
        names = {str(row["id"]): str(row["name"]) for row in entities}
        graph = nx.Graph()
        graph.add_nodes_from(names)
        for row in rows:
            if row["subject_id"] != row["object_id"]:
                graph.add_edge(str(row["subject_id"]), str(row["object_id"]), weight=int(row["weight"]))
        if graph.number_of_nodes() <= 1 or graph.number_of_edges() == 0:
            partitions = [{node} for node in graph.nodes]
        else:
            partitions = list(nx.community.louvain_partitions(graph, weight="weight", seed=seed))
        if not partitions and graph.number_of_nodes():
            partitions = [{node} for node in graph.nodes]

        level_rows = []
        for level, partition in enumerate(partitions):
            current = []
            for members in partition:
                member_ids = {str(member) for member in members}
                ordered_keys = sorted(member_ids)
                community_id = f"L{level}-" + hashlib.sha1(
                    "\0".join(ordered_keys).encode("utf-8")
                ).hexdigest()[:12]
                current.append((community_id, member_ids, None))
            level_rows.append(current)
        for level, communities in enumerate(level_rows[:-1]):
            for index, (community_id, members, _) in enumerate(communities):
                parent = next(
                    (row[0] for row in level_rows[level + 1] if members.issubset(row[1])),
                    None,
                )
                communities[index] = (community_id, members, parent)

        fact_rows = self._run(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)-[:OBJECT_OF]->(o:Entity) "
            "MATCH (f)-[:SUPPORTED_BY]->(c:VideoChunk) "
            "RETURN s.entity_key AS subject_id, o.entity_key AS object_id, s.name AS subject, "
            "f.predicate AS predicate, o.name AS object, f.evidence AS evidence, "
            "c.video_id AS video_id, c.start_timestamp AS start_timestamp, "
            "c.end_timestamp AS end_timestamp ORDER BY c.video_id, c.start_timestamp"
        )
        self._run("MATCH (c:Community) DETACH DELETE c")
        total = 0
        for level in range(len(level_rows) - 1, -1, -1):
            for community_id, members, parent_id in level_rows[level]:
                member_names = sorted(names[member] for member in members)
                title = "، ".join(member_names[:6])
                relevant = [
                    fact for fact in fact_rows
                    if str(fact["subject_id"]) in members
                    and str(fact["object_id"]) in members
                ][:80]
                report = "\n".join(
                    [f"مجتمع مفاهيم: {title}"] + [
                        f"- {row['subject']} — {row['predicate']} — {row['object']} "
                        f"[video={row['video_id']} {float(row['start_timestamp']):.1f}-"
                        f"{float(row['end_timestamp']):.1f}s] الدليل: {row['evidence']}"
                        for row in relevant
                    ]
                )
                content_hash = hashlib.sha256(report.encode("utf-8")).hexdigest()
                video_ids = sorted({str(row["video_id"]) for row in relevant})
                self._run(
                    "CREATE (c:Community {community_id: $community_id, level: $level, "
                    "parent_id: $parent_id, member_count: $member_count, title: $title, "
                    "report: $report, content_hash: $content_hash, video_ids: $video_ids}) "
                    "WITH c UNWIND $members AS entity_key "
                    "MATCH (e:Entity {entity_key: entity_key}) "
                    "MERGE (e)-[:MEMBER_OF]->(c)",
                    community_id=community_id, level=level, parent_id=parent_id,
                    member_count=len(members), title=title, report=report,
                    content_hash=content_hash, video_ids=video_ids,
                    members=sorted(members),
                )
                if parent_id:
                    self._run(
                        "MATCH (child:Community {community_id: $child}), "
                        "(parent:Community {community_id: $parent}) "
                        "MERGE (child)-[:CHILD_OF]->(parent)",
                        child=community_id, parent=parent_id,
                    )
                total += 1
        return {
            "levels": len(level_rows), "communities": total,
            "entities": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        }

    def entity_catalog(self) -> list[dict]:
        return self._run("MATCH (e:Entity) RETURN e.entity_key AS entity_id, e.name AS canonical_name ORDER BY canonical_name")

    def community_reports(
        self, level: int | None = None, video_id: str | None = None
    ) -> list[dict]:
        return self._run(
            "MATCH (c:Community) WHERE ($level IS NULL OR c.level = $level) "
            "AND ($video_id IS NULL OR $video_id IN coalesce(c.video_ids, [])) "
            "RETURN c.community_id AS community_id, c.level AS level, "
            "c.parent_id AS parent_id, c.member_count AS member_count, "
            "c.report AS report, c.content_hash AS content_hash, "
            "c.summary AS summary, c.summary_model_id AS summary_model_id "
            "ORDER BY level DESC, member_count DESC",
            level=level, video_id=video_id,
        )

    def save_community_summary(
        self, community_id: str, content_hash: str, model_id: str, summary: str
    ) -> None:
        self._run(
            "MATCH (c:Community {community_id: $community_id, content_hash: $content_hash}) "
            "SET c.summary = $summary, c.summary_model_id = $model_id, "
            "c.summary_updated_at = datetime()",
            community_id=community_id, content_hash=content_hash,
            model_id=model_id, summary=summary,
        )

    def cached_community_summary(
        self, community_id: str, content_hash: str, model_id: str
    ) -> str | None:
        rows = self._run(
            "MATCH (c:Community {community_id: $community_id, content_hash: $content_hash, "
            "summary_model_id: $model_id}) RETURN c.summary AS summary",
            community_id=community_id, content_hash=content_hash, model_id=model_id,
        )
        return str(rows[0]["summary"]) if rows and rows[0]["summary"] else None

    def fact_edges_for_entities(
        self, entity_ids: list[str], top_k: int = 30, video_id: str | None = None
    ) -> list[dict]:
        return self._run(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)-[:OBJECT_OF]->(o:Entity) "
            "MATCH (f)-[:SUPPORTED_BY]->(c:VideoChunk) "
            "WHERE (s.entity_key IN $ids OR o.entity_key IN $ids) "
            "AND ($video_id IS NULL OR c.video_id = $video_id) "
            "RETURN id(f) AS edge_id, f.chunk_id AS chunk_id, c.video_id AS video_id, "
            "c.start_timestamp AS start_timestamp, c.end_timestamp AS end_timestamp, "
            "s.name AS subject, f.predicate AS predicate, o.name AS object, "
            "f.evidence AS evidence ORDER BY c.start_timestamp LIMIT $limit",
            ids=entity_ids, video_id=video_id, limit=top_k,
        )

    def facts_for_community(
        self, community_id: str, top_k: int = 60, video_id: str | None = None
    ) -> list[dict]:
        return self._run(
            "MATCH (cmt:Community {community_id: $community_id}) "
            "MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)-[:OBJECT_OF]->(o:Entity) "
            "MATCH (f)-[:SUPPORTED_BY]->(c:VideoChunk) "
            "WHERE EXISTS { MATCH (s)-[:MEMBER_OF]->(cmt) } "
            "AND EXISTS { MATCH (o)-[:MEMBER_OF]->(cmt) } "
            "AND ($video_id IS NULL OR c.video_id = $video_id) "
            "RETURN id(f) AS edge_id, f.chunk_id AS chunk_id, c.video_id AS video_id, "
            "c.start_timestamp AS start_timestamp, c.end_timestamp AS end_timestamp, "
            "s.name AS subject, f.predicate AS predicate, o.name AS object, "
            "f.evidence AS evidence ORDER BY c.video_id, c.start_timestamp LIMIT $limit",
            community_id=community_id, video_id=video_id, limit=top_k,
        )

    def list_fact_entities(self, video_id: str | None = None) -> list[dict]:
        rows = self._run(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)-[:OBJECT_OF]->(o:Entity) "
            "MATCH (f)-[:SUPPORTED_BY]->(c:VideoChunk) "
            "WHERE ($video_id IS NULL OR c.video_id = $video_id) "
            "RETURN s.name AS subject, o.name AS object, f.chunk_id AS chunk_id",
            video_id=video_id,
        )
        counts: dict[str, set[str]] = {}
        for row in rows:
            counts.setdefault(str(row["subject"]), set()).add(str(row["chunk_id"]))
            counts.setdefault(str(row["object"]), set()).add(str(row["chunk_id"]))
        return [
            {"name": name, "chunk_count": len(chunk_ids)}
            for name, chunk_ids in sorted(counts.items(), key=lambda pair: (-len(pair[1]), pair[0].casefold()))
        ]

    def search_facts(self, query: str, top_k: int = 5, video_id: str | None = None) -> dict:
        query_terms = self._query_terms(query)
        rows = self._run(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)-[:OBJECT_OF]->(o:Entity) "
            "MATCH (f)-[:SUPPORTED_BY]->(c:VideoChunk) "
            "WHERE ($video_id IS NULL OR c.video_id = $video_id) "
            "RETURN f.chunk_id AS chunk_id, c.video_id AS video_id, "
            "c.start_timestamp AS start_timestamp, c.end_timestamp AS end_timestamp, "
            "c.transcript AS transcript, s.name AS subject, f.predicate AS predicate, "
            "o.name AS object, f.evidence AS fact_evidence",
            video_id=video_id,
        )
        groups: dict[str, dict] = {}
        matched_terms: set[str] = set()
        for row in rows:
            matches = query_terms & self._query_terms(
                f"{row['subject']} {row['predicate']} {row['object']}"
            )
            if not matches:
                continue
            matched_terms.update(matches)
            item = groups.setdefault(row["chunk_id"], {
                "chunk_id": row["chunk_id"], "video_id": row["video_id"],
                "start_timestamp": row["start_timestamp"], "end_timestamp": row["end_timestamp"],
                "evidence": row["transcript"], "facts": [], "_matched_terms": set(),
            })
            item["facts"].append({
                "subject": row["subject"], "predicate": row["predicate"],
                "object": row["object"], "evidence": row["fact_evidence"],
            })
            item["_matched_terms"].update(matches)
        results = list(groups.values())
        for item in results:
            item["graph_score"] = len(item.pop("_matched_terms"))
        results.sort(key=lambda row: row["graph_score"], reverse=True)
        return {
            "query_terms": sorted(query_terms), "matched_terms": sorted(matched_terms),
            "results": results[:top_k],
        }

    @staticmethod
    def _query_terms(value: str) -> set[str]:
        tokens = re.findall(r"[\w]+", _normalize(value), flags=re.UNICODE)
        terms = set(tokens)
        terms.update(token[2:] for token in tokens if token.startswith("ال") and len(token) > 4)
        return terms

    def fact_summary(self, video_id: str) -> dict[str, int]:
        rows = self._run(
            "MATCH (f:Fact {video_id: $video_id})-[:SUPPORTED_BY]->(c:VideoChunk) "
            "RETURN count(DISTINCT c.chunk_id) AS chunks, count(f) AS relations",
            video_id=video_id,
        )
        entities = self.list_fact_entities(video_id)
        return {
            "chunks": int(rows[0]["chunks"]) if rows else 0,
            "entities": len(entities),
            "relations": int(rows[0]["relations"]) if rows else 0,
        }

    def close(self) -> None:
        self.driver.close()

from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from videorag.config import (
    COLLECTION_NAME,
    EMBED_DIM,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
)
from videorag.logger import get_logger

log = get_logger(__name__)


class VectorStore:
    """Qdrant store with one point per chunk and separately sized named vectors."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        reset: bool = False,
        visual_dim: int = EMBED_DIM,
    ):
        if QDRANT_URL:
            self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            log.info(f"Connecting to Qdrant at {QDRANT_URL}")
        else:
            self._client = QdrantClient(path=str(QDRANT_PATH))
            log.info(f"Connecting to local Qdrant at {QDRANT_PATH}")

        if reset and self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "visual": models.VectorParams(
                        size=visual_dim, distance=models.Distance.COSINE
                    ),
                    "audio": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    ),
                },
            )

        self._collection = collection_name
        self.visual_dim = visual_dim
        self._client.create_payload_index(
            collection_name=collection_name,
            field_name="video_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        log.info(f"Qdrant ready — {self.count()} chunks indexed")

    def add(
        self,
        ids: list[str],
        embeddings: list[dict[str, list[float]]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert chunks as Qdrant points with named visual/audio vectors."""
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, point_id)),
                vector=vectors,
                payload={**metadata, "document": document},
            )
            for point_id, vectors, document, metadata
            in zip(ids, embeddings, documents, metadatas)
        ]
        if points:
            self._client.upsert(
                collection_name=self._collection, points=points, wait=True
            )
        log.info(f"Upserted {len(points)} chunks — total: {self.count()}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        using: str = "visual",
        video_id: str | None = None,
    ) -> list[dict]:
        query_filter = None
        if video_id:
            query_filter = models.Filter(must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ])
        points = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            using=using,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points
        hits = []
        for point in points:
            score = float(point.score)
            hits.append({
                "metadata": dict(point.payload or {}),
                "distance": 1.0 - score,
                "confidence_score": round(max(0.0, min(1.0, score)), 4),
            })
        return hits

    def count(self) -> int:
        return self._client.count(
            collection_name=self._collection, exact=True
        ).count

    def count_chunks(self) -> int:
        return self.count()

    def count_by_video_id(self, video_id: str) -> int:
        return self._client.count(
            collection_name=self._collection,
            count_filter=models.Filter(must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ]),
            exact=True,
        ).count

    def delete_by_video_id(self, video_id: str) -> int:
        deleted = self.count_by_video_id(video_id)
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ])),
            wait=True,
        )
        log.info(f"Deleted {deleted} old chunks for video_id={video_id}")
        return deleted

    def list_video_ids(self) -> list[str]:
        video_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                offset=offset,
                limit=256,
                with_payload=["video_id"],
                with_vectors=False,
            )
            for point in points:
                if point.payload and "video_id" in point.payload:
                    video_ids.add(point.payload["video_id"])
            if offset is None:
                return sorted(video_ids)

    def list_videos(self) -> list[dict[str, str]]:
        """Return indexed IDs with user-facing source filenames."""
        videos: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                offset=offset,
                limit=256,
                with_payload=["video_id", "source_filename", "video_path"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                video_id = payload.get("video_id")
                if video_id:
                    filename = payload.get("source_filename") or Path(
                        payload.get("video_path", "")
                    ).name or video_id
                    videos[str(video_id)] = str(filename)
            if offset is None:
                result = [
                    {"video_id": video_id, "filename": videos[video_id]}
                    for video_id in sorted(videos)
                ]
                result.sort(key=lambda item: (item["filename"].casefold(), item["video_id"]))
                return result

    def list_transcripts(self, video_id: str | None = None) -> list[dict]:
        """Scroll transcript payloads for exact/keyword matching."""
        query_filter = None
        if video_id:
            query_filter = models.Filter(must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ])
        transcripts = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=query_filter,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            transcripts.extend(dict(point.payload or {}) for point in points)
            if offset is None:
                return transcripts

    def close(self) -> None:
        self._client.close()

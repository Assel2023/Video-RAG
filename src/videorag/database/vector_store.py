# src/videorag/database/vector_store.py — ChromaDB Abstraction Layer
from __future__ import annotations
import chromadb
from videorag.config import DB_DIR, COLLECTION_NAME
from videorag.logger import get_logger

log = get_logger(__name__)


class VectorStore:
    """
    Clean abstraction over ChromaDB for multimodal vector storage.

    Provides batch indexing and cosine-similarity nearest-neighbor search
    over 512-dim fused embeddings.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        reset: bool = False,
    ):
        log.info(f"Connecting to ChromaDB at: {DB_DIR}")
        self._client = chromadb.PersistentClient(path=str(DB_DIR))

        if reset:
            try:
                self._client.delete_collection(collection_name)
                log.info(f"Dropped collection: {collection_name}")
            except Exception:
                pass
            self._col = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"Created collection: {collection_name}")
        else:
            try:
                self._col = self._client.get_collection(collection_name)
            except Exception:
                self._col = self._client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

        log.info(f"VectorStore ready — {self._col.count()} items indexed")

    # ── Write ─────────────────────────────────────────────────────
    def add(
        self,
        ids:        list[str],
        embeddings: list[list[float]],
        documents:  list[str],
        metadatas:  list[dict],
    ) -> None:
        """Batch-insert embeddings into the collection."""
        self._col.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        log.info(f"Indexed {len(ids)} chunks — total: {self._col.count()}")

    # ── Read ──────────────────────────────────────────────────────
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Cosine nearest-neighbor search.

        Returns:
            List of dicts: {metadata, distance, confidence_score}
        """
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )
        hits = []
        for meta, dist in zip(
            results["metadatas"][0], results["distances"][0]
        ):
            hits.append(
                {
                    "metadata":         meta,
                    "distance":         dist,
                    "confidence_score": round(
                        max(0.0, min(1.0, 1.0 - dist)), 4
                    ),
                }
            )
        return hits

    def count(self) -> int:
        return self._col.count()

"""
ChromaDB Vector Store Wrapper.

Manages all interactions with ChromaDB — the "Feature Store" pattern from M3.
Pre-computed embeddings are stored here for fast retrieval at query time,
analogous to how a feature store caches ML features.

Key operations:
    - add_documents: Batch insert chunks with embeddings and metadata
    - search: Cosine similarity search for query embeddings
    - delete_document: Remove all chunks for a re-ingested document
    - get_stats: Collection statistics for health checks

Continuation Note:
    This module is complete. The ChromaDB client uses persistent storage
    at the path defined in config.CHROMA_DB_PATH. Collection name is
    config.CHROMA_COLLECTION_NAME.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb

from app.config import get_settings
from app.monitoring.logging_config import get_logger

logger = get_logger("vector_store")


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float  # Lower = more similar (cosine distance)
    similarity_score: float  # Converted to 0-1 similarity

    @property
    def document_name(self) -> str:
        return self.metadata.get("document_name", "unknown")

    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", -1)

    @property
    def section(self) -> str:
        return self.metadata.get("section", "unknown")


class VectorStore:
    """
    ChromaDB-backed vector store for AUTOSAR document embeddings.

    Implements the Feature Store pattern: pre-computed embedding vectors
    are stored and indexed for fast similarity-based retrieval at query time.
    """

    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=str(settings.chroma_db_absolute_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        logger.info(
            "vector_store_initialized",
            path=str(settings.chroma_db_absolute_path),
            collection=settings.CHROMA_COLLECTION_NAME,
            count=self._collection.count(),
        )

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Batch upsert chunks with their embeddings and metadata.

        Args:
            ids: Unique IDs for each chunk (format: "{doc_name}_chunk_{i}")
            texts: Raw text of each chunk
            embeddings: Dense vector embeddings from the embedding model
            metadatas: Metadata dicts (document_name, page_number, section, etc.)

        Returns:
            Number of documents added/updated
        """
        if not ids:
            return 0

        # ChromaDB has a batch limit, process in chunks of 500
        batch_size = 500
        total_added = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            self._collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            total_added += len(batch_ids)

        logger.info("documents_added", count=total_added)
        return total_added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform cosine similarity search.

        Args:
            query_embedding: Dense vector of the user's query
            top_k: Number of nearest neighbors to retrieve
            where_filter: Optional metadata filter (e.g., {"document_name": "SWS_COM"})

        Returns:
            List of SearchResult ordered by similarity (highest first)
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": (
                min(top_k, self._collection.count()) if self._collection.count() > 0 else top_k
            ),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self._collection.query(**query_params)

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert cosine distance to similarity score (0-1)
                similarity = 1.0 - distance

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        distance=distance,
                        similarity_score=round(similarity, 4),
                    )
                )

        logger.info(
            "search_completed",
            top_k=top_k,
            results_found=len(search_results),
            top_score=search_results[0].similarity_score if search_results else 0,
        )
        return search_results

    def delete_document(self, document_name: str) -> int:
        """
        Delete all chunks belonging to a specific document.

        Used for re-ingestion: delete old chunks before adding new ones.
        """
        # Get all IDs matching this document
        results = self._collection.get(
            where={"document_name": document_name},
            include=[],
        )
        if results and results["ids"]:
            self._collection.delete(ids=results["ids"])
            count = len(results["ids"])
            logger.info("document_deleted", document=document_name, chunks_removed=count)
            return count
        return 0

    def get_stats(self) -> dict:
        """Get collection statistics for health checks."""
        count = self._collection.count()
        return {
            "collection_name": self._collection.name,
            "total_chunks": count,
            "metadata": self._collection.metadata,
        }

    def get_all_document_names(self) -> List[str]:
        """Get list of all unique document names in the store."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        if results and results["metadatas"]:
            names = set()
            for meta in results["metadatas"]:
                if "document_name" in meta:
                    names.add(meta["document_name"])
            return sorted(names)
        return []


# Module-level singleton — lazy initialized
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

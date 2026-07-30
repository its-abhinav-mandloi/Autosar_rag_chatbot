"""
Semantic Search Engine.

Performs vector similarity search against the ChromaDB store.
This is the first stage of the retrieval pipeline:
    Query → Embed → Cosine Similarity Search → Top-K Results

Maps to the "Semantic Search" agent in the HLD's Retrieval & Reasoning Engine.

Continuation Note:
    This module is complete. It embeds the query using the same model
    used during ingestion and retrieves top-k nearest chunks from ChromaDB.
"""

from typing import List, Optional

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import metrics
from app.services.ingestion.embedder import embed_single
from app.storage.vector_store import SearchResult, get_vector_store

logger = get_logger("search")


async def semantic_search(
    query: str,
    top_k: Optional[int] = None,
    document_filter: Optional[str] = None,
) -> List[SearchResult]:
    """
    Perform semantic search for a natural language query.

    Pipeline:
        1. Embed the query text using the same embedding model
        2. Search ChromaDB for top-k nearest neighbors (cosine similarity)
        3. Return results with similarity scores and metadata

    Args:
        query: Natural language question
        top_k: Number of results to retrieve (default from config)
        document_filter: Optional filter to search within a specific document

    Returns:
        List of SearchResult objects ordered by similarity (highest first)
    """
    settings = get_settings()
    top_k = top_k or settings.TOP_K_RETRIEVAL

    logger.info("search_started", query=query[:100], top_k=top_k)

    # Step 1: Embed the query
    query_embedding = await embed_single(query)

    # Step 2: Search the vector store
    vector_store = get_vector_store()

    where_filter = None
    if document_filter:
        where_filter = {"document_name": document_filter}

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        where_filter=where_filter,
    )

    # Record top retrieval score for metrics
    if results:
        metrics.record_retrieval_score(results[0].similarity_score)

    logger.info(
        "search_completed",
        query=query[:100],
        results=len(results),
        top_score=results[0].similarity_score if results else 0,
    )

    return results

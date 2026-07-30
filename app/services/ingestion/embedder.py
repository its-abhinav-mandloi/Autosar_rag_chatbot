"""
Embedding Generator via Ollama API.

Converts text chunks into dense vector representations using Ollama's
embedding endpoint. Uses the nomic-embed-text model (768-dim vectors).

This is the core ML component in the ingestion pipeline:
    Text → Embedding Model → Dense Vector → Vector Store

The embedding model is the "learned representation" — it maps natural
language text into a vector space where semantically similar texts
are close together (cosine similarity).

Continuation Note:
    This module is complete. If switching embedding models, update
    EMBEDDING_MODEL in .env. The embedding dimension is auto-detected.
    Retry logic handles temporary Ollama failures.
"""

import asyncio
from typing import List

import httpx

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import metrics

logger = get_logger("embedder")

# ─── Constants ───────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # Exponential backoff: 2s, 4s, 8s
BATCH_SIZE = 50  # Texts per batch to avoid overloading Ollama
REQUEST_TIMEOUT = 60.0  # Seconds per embedding request


# ─── Core Embedding Functions ────────────────────────────────────────────


async def embed_single(text: str, model: str = None) -> List[float]:
    """
    Generate embedding for a single text string.

    Args:
        text: Input text to embed
        model: Embedding model name (default from config)

    Returns:
        List of floats representing the dense vector

    Raises:
        RuntimeError: If embedding fails after all retries
    """
    settings = get_settings()
    model = model or settings.EMBEDDING_MODEL

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", [])

                if not embedding:
                    raise ValueError("Empty embedding returned from Ollama")

                return embedding

        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as e:
            wait_time = RETRY_BACKOFF_BASE**attempt
            logger.warning(
                "embedding_retry",
                attempt=attempt + 1,
                max_retries=MAX_RETRIES,
                error=str(e),
                wait_seconds=wait_time,
            )
            metrics.record_error("embedding_retry")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait_time)
            else:
                metrics.record_error("embedding_failed")
                raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {e}")


async def embed_batch(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.

    Processes texts in sub-batches of BATCH_SIZE to manage memory and
    avoid overloading the Ollama API.

    Args:
        texts: List of text strings to embed
        model: Embedding model name (default from config)

    Returns:
        List of embedding vectors (same order as input texts)
    """
    if not texts:
        return []

    settings = get_settings()
    model = model or settings.EMBEDDING_MODEL
    all_embeddings = []

    logger.info("batch_embedding_started", total_texts=len(texts), model=model)

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(texts))
        batch_texts = texts[batch_start:batch_end]

        logger.info(
            "batch_progress",
            batch=f"{batch_start // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}",
            texts=len(batch_texts),
        )

        # Process each text in the sub-batch
        # Ollama's /api/embeddings only supports single texts,
        # so we parallelize with asyncio.gather
        tasks = [embed_single(text, model) for text in batch_texts]
        batch_embeddings = await asyncio.gather(*tasks)
        all_embeddings.extend(batch_embeddings)

    logger.info(
        "batch_embedding_completed",
        total_texts=len(texts),
        embedding_dim=len(all_embeddings[0]) if all_embeddings else 0,
    )

    return all_embeddings


def embed_single_sync(text: str, model: str = None) -> List[float]:
    """
    Synchronous wrapper for embed_single.

    Used in contexts where async is not available (e.g., tests).
    """
    return asyncio.run(embed_single(text, model))

"""
LLM-Based Re-ranking.

Improves retrieval precision by using the LLM to score the relevance
of each retrieved chunk against the original query. This is the
"Re-ranking" agent in the HLD's Retrieval & Reasoning Engine.

Strategy:
    1. Take the top-K results from semantic search (e.g., K=10)
    2. Ask the LLM to score each chunk's relevance (0-10)
    3. Re-sort by LLM relevance score
    4. Return top-N results (e.g., N=5)

This catches cases where embedding similarity is high but actual
relevance is low (semantic similarity ≠ topical relevance).

Continuation Note:
    This module is complete. The re-ranker uses the same LLM as
    the generation step but with a lightweight scoring prompt.
    If re-ranking is too slow, it can be disabled by setting
    TOP_K_RERANK == TOP_K_RETRIEVAL in config.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import List

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.services.inference.llm import generate_completion
from app.storage.vector_store import SearchResult

logger = get_logger("reranker")


@dataclass
class RankedResult:
    """A search result with an additional LLM-assigned relevance score."""

    search_result: SearchResult
    relevance_score: float  # 0.0 - 1.0 (LLM-assigned)
    original_rank: int  # Position in original search results


RERANK_PROMPT_TEMPLATE = """Rate the relevance of the following text passage to the given question.
Respond with ONLY a number from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = highly relevant and directly answers the question

Question: {question}

Passage:
{passage}

Relevance score (0-10):"""


async def rerank(
    query: str,
    search_results: List[SearchResult],
    top_n: int = None,
) -> List[RankedResult]:
    """
    Re-rank search results using LLM relevance scoring.

    Args:
        query: The original user question
        search_results: Results from semantic search
        top_n: Number of results to keep after re-ranking (default from config)

    Returns:
        List of RankedResult sorted by LLM relevance (highest first)
    """
    settings = get_settings()
    top_n = top_n or settings.TOP_K_RERANK

    if not search_results:
        return []

    # If we already have fewer results than top_n, skip re-ranking
    if len(search_results) <= top_n:
        return [
            RankedResult(
                search_result=r,
                relevance_score=r.similarity_score,
                original_rank=i,
            )
            for i, r in enumerate(search_results)
        ]

    logger.info(
        "reranking_started",
        query=query[:100],
        candidates=len(search_results),
        top_n=top_n,
    )

    # Score each result with the LLM
    async def score_result(index: int, result: SearchResult) -> RankedResult:
        prompt = RERANK_PROMPT_TEMPLATE.format(
            question=query,
            passage=result.text[:1000],  # Limit passage length
        )
        try:
            response = await generate_completion(
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,  # Deterministic scoring
            )
            score = _parse_score(response)
        except Exception as e:
            logger.warning("rerank_score_failed", index=index, error=str(e))
            score = result.similarity_score  # Fallback to embedding score

        return RankedResult(
            search_result=result,
            relevance_score=score,
            original_rank=index,
        )

    # Run scoring in parallel
    tasks = [score_result(i, r) for i, r in enumerate(search_results)]
    ranked_results = await asyncio.gather(*tasks)

    # Sort by relevance score (highest first)
    ranked_results.sort(key=lambda r: r.relevance_score, reverse=True)

    # Keep only top_n
    top_results = ranked_results[:top_n]

    logger.info(
        "reranking_completed",
        candidates=len(search_results),
        kept=len(top_results),
        top_score=top_results[0].relevance_score if top_results else 0,
    )

    return top_results


def _parse_score(response: str) -> float:
    """
    Parse a numeric score from the LLM's response.

    Handles various formats: "7", "7/10", "Score: 7", etc.
    Returns normalized 0.0-1.0 score.
    """
    # Extract first number from response
    numbers = re.findall(r"(\d+(?:\.\d+)?)", response.strip())
    if numbers:
        raw_score = float(numbers[0])
        # Normalize to 0-1 range (assuming 0-10 scale)
        return min(max(raw_score / 10.0, 0.0), 1.0)
    return 0.5  # Default middle score if parsing fails

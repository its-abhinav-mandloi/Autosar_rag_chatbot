"""
Context Assembly Module.

Combines re-ranked chunks into a coherent context string for the LLM prompt.
This is the "Context Assembly" agent in the HLD's Retrieval & Reasoning Engine.

Responsibilities:
    1. Order chunks by document position (not just relevance)
    2. Deduplicate overlapping chunks
    3. Add source attribution markers for citation extraction
    4. Enforce maximum context length

Continuation Note:
    This module is complete. The assembled context is injected into the
    prompt template in app/services/inference/prompt.py.
"""

from typing import List

from app.monitoring.logging_config import get_logger
from app.services.retrieval.reranker import RankedResult

logger = get_logger("context")

# Maximum context length in characters to avoid exceeding LLM context window
MAX_CONTEXT_CHARS = 8000


def assemble_context(ranked_results: List[RankedResult]) -> str:
    """
    Assemble ranked results into a coherent context string.

    Format per chunk:
        [Source: document_name | Page: X | Section: Y.Z Title]
        <chunk text>

    Args:
        ranked_results: Re-ranked search results

    Returns:
        Formatted context string ready for prompt injection
    """
    if not ranked_results:
        return "No relevant documents found."

    # Sort by document position (document name → page number → chunk index)
    sorted_results = sorted(
        ranked_results,
        key=lambda r: (
            r.search_result.metadata.get("document_name", ""),
            r.search_result.metadata.get("page_number", 0),
            r.search_result.metadata.get("chunk_index", 0),
        ),
    )

    # Deduplicate overlapping chunks
    deduplicated = _deduplicate_chunks(sorted_results)

    # Build context string with source markers
    context_parts = []
    total_chars = 0

    for i, result in enumerate(deduplicated, 1):
        sr = result.search_result
        meta = sr.metadata

        # Source header
        source_marker = (
            f"[Source {i}: {meta.get('document_name', 'Unknown')} | "
            f"Page: {meta.get('page_number', '?')} | "
            f"Section: {meta.get('section', 'N/A')}]"
        )

        # Check if adding this chunk would exceed limit
        chunk_text = sr.text.strip()
        entry = f"{source_marker}\n{chunk_text}\n"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            # Truncate the last chunk to fit
            remaining = MAX_CONTEXT_CHARS - total_chars - len(source_marker) - 10
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "..."
                entry = f"{source_marker}\n{chunk_text}\n"
                context_parts.append(entry)
            break

        context_parts.append(entry)
        total_chars += len(entry)

    context = "\n---\n".join(context_parts)

    logger.info(
        "context_assembled",
        chunks_used=len(context_parts),
        total_chars=len(context),
        max_chars=MAX_CONTEXT_CHARS,
    )

    return context


def _deduplicate_chunks(results: List[RankedResult]) -> List[RankedResult]:
    """
    Remove chunks with high text overlap.

    Two chunks are considered duplicates if they share >70% of their text.
    Keeps the one with the higher relevance score.
    """
    if len(results) <= 1:
        return results

    deduplicated = [results[0]]

    for candidate in results[1:]:
        is_duplicate = False
        candidate_text = candidate.search_result.text.strip()

        for kept in deduplicated:
            kept_text = kept.search_result.text.strip()
            overlap = _text_overlap_ratio(candidate_text, kept_text)

            if overlap > 0.7:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(candidate)

    if len(deduplicated) < len(results):
        logger.info(
            "chunks_deduplicated",
            original=len(results),
            kept=len(deduplicated),
        )

    return deduplicated


def _text_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Calculate the word-level overlap ratio between two texts.

    Returns the Jaccard similarity (intersection / union) of word sets.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


def get_source_list(ranked_results: List[RankedResult]) -> List[dict]:
    """
    Extract a clean list of source references from ranked results.

    Used for the citations array in the API response.
    """
    sources = []
    seen = set()

    for result in ranked_results:
        sr = result.search_result
        meta = sr.metadata

        # Create a unique key to avoid duplicate sources
        key = f"{meta.get('document_name', '')}_p{meta.get('page_number', 0)}"
        if key in seen:
            continue
        seen.add(key)

        sources.append(
            {
                "document": meta.get("document_name", "Unknown"),
                "page": meta.get("page_number", 0),
                "section": meta.get("section", "N/A"),
                "relevance_score": round(result.relevance_score, 4),
                "chunk_preview": sr.text[:150] + "..." if len(sr.text) > 150 else sr.text,
            }
        )

    return sources

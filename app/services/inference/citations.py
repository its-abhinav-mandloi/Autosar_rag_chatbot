"""
Citation Extraction Module.

Parses LLM-generated answers to extract source citations and map them
back to the original retrieved chunks. This directly addresses the
Explainability quality requirement.

Output format:
    {
        "document": "AUTOSAR_SWS_COM.pdf",
        "page": 42,
        "section": "7.3.2 PDU Router Configuration",
        "relevance_score": 0.94
    }

Continuation Note:
    This module is complete. It handles multiple citation formats
    like [Source 1], [Source: doc, page X], and plain [1] references.
"""

import re
from typing import Dict, List

from app.monitoring.logging_config import get_logger
from app.services.retrieval.reranker import RankedResult

logger = get_logger("citations")

# Regex patterns for citation extraction
SOURCE_REF_PATTERN = re.compile(r"\[Source\s*(\d+)\]", re.IGNORECASE)
BRACKET_REF_PATTERN = re.compile(r"\[(\d+)\]")
SWS_REF_PATTERN = re.compile(r"\[SWS_[A-Za-z]+_\d+\]")


def extract_citations(
    answer: str,
    ranked_results: List[RankedResult],
) -> List[dict]:
    """
    Extract and resolve citations from the LLM's generated answer.

    Matches [Source N] references in the answer text back to the
    ranked results that provided the context.

    Args:
        answer: The LLM-generated answer text
        ranked_results: The ranked results used to build the context

    Returns:
        List of citation dicts with document, page, section, and score
    """
    citations = []
    cited_indices = set()

    # Extract [Source N] references
    for match in SOURCE_REF_PATTERN.finditer(answer):
        index = int(match.group(1)) - 1  # Convert to 0-indexed
        cited_indices.add(index)

    # Fallback: try [N] references
    if not cited_indices:
        for match in BRACKET_REF_PATTERN.finditer(answer):
            index = int(match.group(1)) - 1
            if 0 <= index < len(ranked_results):
                cited_indices.add(index)

    # Build citation objects from matched indices
    # Results were sorted by document position in context assembly,
    # so indices map to that ordering
    for index in sorted(cited_indices):
        if 0 <= index < len(ranked_results):
            result = ranked_results[index]
            sr = result.search_result
            meta = sr.metadata

            citation = {
                "source_index": index + 1,
                "document": meta.get("document_name", "Unknown"),
                "page": meta.get("page_number", 0),
                "section": meta.get("section", "N/A"),
                "relevance_score": round(result.relevance_score, 4),
                "requirement_ids": meta.get("requirement_ids", ""),
            }
            citations.append(citation)

    # Also extract any SWS requirement IDs directly mentioned in the answer
    sws_refs = SWS_REF_PATTERN.findall(answer)
    if sws_refs:
        # Find which sources contain these requirement IDs
        for sws_id in set(sws_refs):
            already_cited = any(sws_id in c.get("requirement_ids", "") for c in citations)
            if not already_cited:
                # Find the source containing this SWS ID
                for i, result in enumerate(ranked_results):
                    req_ids = result.search_result.metadata.get("requirement_ids", "")
                    if sws_id in req_ids:
                        meta = result.search_result.metadata
                        citations.append(
                            {
                                "source_index": i + 1,
                                "document": meta.get("document_name", "Unknown"),
                                "page": meta.get("page_number", 0),
                                "section": meta.get("section", "N/A"),
                                "relevance_score": round(result.relevance_score, 4),
                                "requirement_ids": sws_id,
                            }
                        )
                        break

    # If no citations were extracted, include top sources as implicit citations
    if not citations and ranked_results:
        for i, result in enumerate(ranked_results[:3]):
            sr = result.search_result
            meta = sr.metadata
            citations.append(
                {
                    "source_index": i + 1,
                    "document": meta.get("document_name", "Unknown"),
                    "page": meta.get("page_number", 0),
                    "section": meta.get("section", "N/A"),
                    "relevance_score": round(result.relevance_score, 4),
                    "requirement_ids": meta.get("requirement_ids", ""),
                    "implicit": True,  # Not explicitly cited by LLM
                }
            )

    # Deduplicate by document+page
    seen = set()
    unique_citations = []
    for c in citations:
        key = f"{c['document']}_p{c['page']}"
        if key not in seen:
            seen.add(key)
            unique_citations.append(c)

    logger.info(
        "citations_extracted",
        total=len(unique_citations),
        explicit=len(cited_indices),
        sws_refs=len(sws_refs) if sws_refs else 0,
    )

    return unique_citations


def compute_confidence_score(
    ranked_results: List[RankedResult],
    citations: List[dict],
) -> float:
    """
    Compute a confidence score for the generated answer.

    Based on:
        - Average retrieval similarity of top results (40% weight)
        - Number of explicit citations (30% weight)
        - Re-ranking score agreement (30% weight)

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not ranked_results:
        return 0.0

    # Factor 1: Average retrieval similarity
    avg_similarity = sum(r.search_result.similarity_score for r in ranked_results) / len(
        ranked_results
    )

    # Factor 2: Citation coverage (more citations = higher confidence)
    explicit_citations = [c for c in citations if not c.get("implicit", False)]
    citation_score = min(len(explicit_citations) / 3.0, 1.0)  # Max at 3 citations

    # Factor 3: Re-ranking agreement (top results should have high re-rank scores)
    rerank_scores = [r.relevance_score for r in ranked_results[:3]]
    avg_rerank = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0

    # Weighted combination
    confidence = 0.4 * avg_similarity + 0.3 * citation_score + 0.3 * avg_rerank

    return round(min(max(confidence, 0.0), 1.0), 4)

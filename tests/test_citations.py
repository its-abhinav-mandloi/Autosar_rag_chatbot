"""
Unit tests for Citation Extraction and Confidence Scoring.
"""

import pytest
from app.services.inference.citations import (
    extract_citations,
    compute_confidence_score,
)
from app.services.retrieval.reranker import RankedResult
from app.storage.vector_store import SearchResult


def _make_ranked_result(doc_name="doc.pdf", page=1, section="1 Intro", score=0.9):
    """Helper to create mock RankedResult objects."""
    sr = SearchResult(
        chunk_id=f"{doc_name}_chunk_0",
        text="Some chunk text here",
        metadata={
            "document_name": doc_name,
            "page_number": page,
            "section": section,
            "requirement_ids": "[SWS_Com_00001]",
        },
        distance=1 - score,
        similarity_score=score,
    )
    return RankedResult(search_result=sr, relevance_score=score, original_rank=0)


class TestCitationExtraction:
    """Test extracting citations from LLM answers."""

    def test_source_n_format(self):
        results = [
            _make_ranked_result("doc1.pdf", 10, "1 Intro", 0.9),
            _make_ranked_result("doc2.pdf", 20, "2 API", 0.8),
        ]
        answer = "According to [Source 1], the API is defined. See also [Source 2]."
        citations = extract_citations(answer, results)
        assert len(citations) >= 2

    def test_bracket_n_format(self):
        results = [
            _make_ranked_result("doc1.pdf", 10, "1 Intro", 0.9),
        ]
        answer = "The spec states [1] that initialization is required."
        citations = extract_citations(answer, results)
        assert len(citations) >= 1

    def test_no_citations_fallback(self):
        """If no explicit citations, top sources should be included as implicit."""
        results = [
            _make_ranked_result("doc1.pdf", 10, "1 Intro", 0.9),
        ]
        answer = "The module initializes correctly."
        citations = extract_citations(answer, results)
        assert len(citations) >= 1
        assert any(c.get("implicit", False) for c in citations)

    def test_empty_results(self):
        citations = extract_citations("No answer", [])
        assert len(citations) == 0


class TestConfidenceScoring:
    """Test confidence score computation."""

    def test_high_confidence(self):
        results = [
            _make_ranked_result(score=0.95),
            _make_ranked_result(score=0.90),
        ]
        citations = [{"source_index": 1, "document": "doc.pdf", "page": 1, "section": "1", "relevance_score": 0.95}]
        score = compute_confidence_score(results, citations)
        assert 0.5 < score <= 1.0

    def test_low_confidence(self):
        results = [
            _make_ranked_result(score=0.3),
        ]
        citations = []
        score = compute_confidence_score(results, citations)
        assert score < 0.5

    def test_empty_results(self):
        score = compute_confidence_score([], [])
        assert score == 0.0

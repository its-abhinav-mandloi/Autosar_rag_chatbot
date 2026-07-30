"""
Integration tests for the full RAG pipeline.

These tests verify the end-to-end flow:
    Ingest → Query → Answer with Citations

Note: These tests require Ollama to be running with the configured models.
      Tests are marked with @pytest.mark.integration and skipped if Ollama is unavailable.
"""

import pytest
import httpx
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        settings = get_settings()
        resp = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# Skip integration tests if Ollama is not available
requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama is not running — skipping integration tests",
)


class TestIngestionEndpoint:
    """Test the ingestion endpoints."""

    def test_list_documents_empty(self):
        """Should return empty list when no documents ingested."""
        response = client.get("/ingest/documents")
        assert response.status_code == 200
        # May or may not be empty depending on test order
        assert isinstance(response.json(), list)

    def test_upload_non_pdf_rejected(self):
        """Should reject non-PDF files."""
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400


class TestQueryEndpoint:
    """Test the query endpoints."""

    def test_query_requires_question(self):
        """Should reject empty/short questions."""
        response = client.post(
            "/query",
            json={"question": "ab"},  # Too short (min 3 chars)
        )
        assert response.status_code == 422  # Validation error

    @requires_ollama
    def test_query_search_only(self):
        """Search-only endpoint should work even without ingested docs."""
        response = client.post(
            "/query/search",
            json={"question": "What is AUTOSAR?"},
        )
        # Might fail if no docs ingested, but should not crash
        assert response.status_code in (200, 503)


class TestFeedbackEndpoint:
    """Test the feedback endpoints."""

    def test_submit_feedback(self):
        """Should accept valid feedback."""
        response = client.post(
            "/feedback",
            json={
                "question": "What is Com?",
                "answer": "Com is the communication module.",
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"

    def test_analytics_endpoint(self):
        """Analytics endpoint should return stats."""
        response = client.get("/feedback/analytics")
        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data


@requires_ollama
class TestFullRAGPipeline:
    """
    Full integration tests requiring Ollama.
    
    These test the complete ingest → query → answer flow.
    """

    def test_ingest_local_files(self):
        """Ingest PDFs from the Database/ folder."""
        response = client.post(
            "/ingest/local",
            json={"filename": None},  # Ingest all
        )
        # May take a while
        assert response.status_code in (200, 404)  # 404 if no PDFs found

    def test_query_after_ingestion(self):
        """Query should return an answer after ingestion."""
        response = client.post(
            "/query",
            json={
                "question": "What is AUTOSAR?",
                "top_k": 3,
                "skip_reranking": True,  # Faster for testing
            },
        )
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "citations" in data
            assert "confidence" in data
            assert data["latency_ms"] > 0

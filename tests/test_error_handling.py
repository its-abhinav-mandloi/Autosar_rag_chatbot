"""
Regression tests for client-facing API errors.

These tests keep backend failures readable for the browser UI instead of
letting object-shaped errors show up as "[object Object]".
"""

from fastapi.testclient import TestClient

from app.main import app
from app.services.inference.hallucination import (
    ClaimEvaluation,
    HallucinationReport,
)
from app.storage.vector_store import SearchResult

client = TestClient(app)


class TestQueryErrorHandling:
    """Query endpoint should return consistent, actionable error details."""

    def test_search_failure_returns_structured_error(self, monkeypatch):
        async def fail_search(*args, **kwargs):
            raise RuntimeError("embedding model is unavailable")

        monkeypatch.setattr("app.services.retrieval.router.semantic_search", fail_search)

        response = client.post(
            "/query",
            json={"question": "What is CAN initialization?", "skip_reranking": True},
        )

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "SEARCH_UNAVAILABLE"
        assert "Unable to search" in detail["message"]
        assert "embedding model" in detail["action"]
        assert "embedding model is unavailable" in detail["technical_detail"]

    def test_llm_failure_returns_structured_error(self, monkeypatch):
        async def fake_search(*args, **kwargs):
            return [
                SearchResult(
                    chunk_id="chunk-1",
                    text="CAN initialization is described here.",
                    metadata={
                        "document_name": "AUTOSAR_CAN.pdf",
                        "page_number": 12,
                        "section": "CAN Init",
                        "chunk_index": 0,
                    },
                    distance=0.1,
                    similarity_score=0.9,
                )
            ]

        async def fail_completion(*args, **kwargs):
            raise RuntimeError("llama3.2 model is not installed")

        monkeypatch.setattr("app.services.retrieval.router.semantic_search", fake_search)
        monkeypatch.setattr("app.services.retrieval.router.generate_completion", fail_completion)

        response = client.post(
            "/query",
            json={"question": "What is CAN initialization?", "skip_reranking": True},
        )

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "LLM_UNAVAILABLE"
        assert "local LLM" in detail["message"]
        assert "Ollama" in detail["action"]
        assert "llama3.2 model is not installed" in detail["technical_detail"]

    def test_query_can_return_hallucination_evaluation(self, monkeypatch):
        async def fake_search(*args, **kwargs):
            return [
                SearchResult(
                    chunk_id="chunk-1",
                    text="Can_Init initializes the CAN driver.",
                    metadata={
                        "document_name": "AUTOSAR_CAN.pdf",
                        "page_number": 12,
                        "section": "CAN Init",
                        "chunk_index": 0,
                    },
                    distance=0.1,
                    similarity_score=0.9,
                )
            ]

        async def fake_completion(*args, **kwargs):
            return "Can_Init initializes the CAN driver. [Source 1]"

        async def fake_grounding(*args, **kwargs):
            return HallucinationReport(
                factual_claims=1,
                supported_claims=1,
                contradicted_claims=0,
                unsupported_claims=0,
                not_factual_claims=0,
                hallucination_rate=0.0,
                faithfulness=1.0,
                verdict="fully_grounded",
                claims=[
                    ClaimEvaluation(
                        claim="Can_Init initializes the CAN driver.",
                        status="supported",
                        source_indices=[1],
                        rationale="Source 1 states this.",
                    )
                ],
            )

        monkeypatch.setattr("app.services.retrieval.router.semantic_search", fake_search)
        monkeypatch.setattr("app.services.retrieval.router.generate_completion", fake_completion)
        monkeypatch.setattr("app.services.retrieval.router.evaluate_answer_grounding", fake_grounding)

        response = client.post(
            "/query",
            json={
                "question": "What does Can_Init do?",
                "skip_reranking": True,
                "evaluate_hallucination": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["hallucination_evaluation"]["hallucination_rate"] == 0.0
        assert data["hallucination_evaluation"]["faithfulness"] == 1.0
        assert data["hallucination_evaluation"]["claims"][0]["status"] == "supported"

"""
ML Component Tests — embedding model and LLM inference behaviour.

These tests verify the ML core of the RAG pipeline:

  Embedding tests  (analogous to "training" checks for a pre-trained model):
    - Output dimension is fixed (shape check)
    - Identical input always produces identical output (determinism)
    - Semantically similar texts are closer than unrelated texts (directional)

  Inference tests  (LLM answer quality):
    - Output is a non-empty string (range check)
    - Answer with context has higher confidence than answer without (directional)
    - Confidence score stays in [0, 1] (invariance)

All tests that require Ollama are automatically skipped when Ollama is not running.
pytest-asyncio (asyncio_mode=auto in pytest.ini) handles the event loop.
"""

import math

import httpx
import pytest

from app.config import get_settings


# ─── Skip guard ─────────────────────────────────────────────────────────

def ollama_available() -> bool:
    try:
        settings = get_settings()
        # Check that the embedding model is actually loaded, not just the API
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/embeddings",
            json={"model": settings.EMBEDDING_MODEL, "prompt": "health"},
            timeout=5.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama is not running — skipping ML component tests",
)


# ─── Helpers ────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Embedding Tests ─────────────────────────────────────────────────────

class TestEmbeddingModel:
    """
    Verify embedding model output shape, determinism, and directional ordering.

    These are "training" analogs for a system that uses a pre-trained embedder:
    we cannot check loss decreasing, but we can verify the model produces
    geometrically consistent representations.
    """

    @requires_ollama
    async def test_embedding_output_has_fixed_dimension(self):
        """Every text must produce a vector of the same non-zero length."""
        from app.services.ingestion.embedder import embed_single
        embedding = await embed_single("AUTOSAR Com module initialization")
        assert isinstance(embedding, list), "Embedding must be a list"
        assert len(embedding) > 0, "Embedding must be non-empty"

    @requires_ollama
    async def test_embedding_dimension_is_consistent(self):
        """Two different texts must produce vectors of identical length."""
        from app.services.ingestion.embedder import embed_single
        emb1 = await embed_single("AUTOSAR initialization")
        emb2 = await embed_single("NvM block configuration")
        assert len(emb1) == len(emb2), (
            f"Dimension mismatch: {len(emb1)} vs {len(emb2)}"
        )

    @requires_ollama
    async def test_embedding_is_deterministic(self):
        """Embedding the same text twice must return identical vectors."""
        from app.services.ingestion.embedder import embed_single
        text = "The Com module handles signal routing."
        emb1 = await embed_single(text)
        emb2 = await embed_single(text)
        assert emb1 == emb2, "Embedding is not deterministic for identical input"

    @requires_ollama
    async def test_similar_texts_score_higher_than_unrelated(self):
        """
        Directional test: two AUTOSAR texts should be more similar to each
        other than either is to an unrelated sentence.
        """
        from app.services.ingestion.embedder import embed_single
        emb_a = await embed_single("AUTOSAR Com module initialization")
        emb_b = await embed_single("Initialize the communication module")
        emb_c = await embed_single("Recipe for baking chocolate cake")

        sim_related = cosine_similarity(emb_a, emb_b)
        sim_unrelated = cosine_similarity(emb_a, emb_c)

        assert sim_related > sim_unrelated, (
            f"Expected sim(related)={sim_related:.3f} > sim(unrelated)={sim_unrelated:.3f}"
        )

    @requires_ollama
    async def test_embedding_values_are_finite_floats(self):
        """Output vector must contain only finite numbers (no NaN or Inf)."""
        from app.services.ingestion.embedder import embed_single
        embedding = await embed_single("Signal mapping in AUTOSAR")
        for i, val in enumerate(embedding):
            assert math.isfinite(val), f"Non-finite value at index {i}: {val}"


# ─── LLM Inference Tests ─────────────────────────────────────────────────

class TestLLMInference:
    """
    Verify LLM output range, directional confidence, and invariance.
    """

    @requires_ollama
    async def test_llm_output_is_non_empty_string(self):
        """Inference must produce at least one non-whitespace character."""
        from app.services.inference.llm import generate_completion
        answer = await generate_completion(
            prompt="What does AUTOSAR stand for?",
            system_prompt="You are a helpful assistant.",
        )
        assert isinstance(answer, str)
        assert answer.strip() != "", "LLM returned an empty response"

    def test_confidence_score_in_valid_range(self):
        """compute_confidence_score must always return a value in [0, 1]."""
        from app.services.inference.citations import compute_confidence_score
        from app.services.retrieval.reranker import RankedResult
        from app.storage.vector_store import SearchResult

        def _make_result(score: float) -> RankedResult:
            sr = SearchResult(
                chunk_id="doc_chunk_0",
                text="AUTOSAR is a standard.",
                metadata={
                    "document_name": "doc.pdf",
                    "page_number": 1,
                    "section": "1 Intro",
                    "requirement_ids": "",
                },
                distance=1 - score,
                similarity_score=score,
            )
            return RankedResult(search_result=sr, relevance_score=score, original_rank=0)

        for score in [0.1, 0.5, 0.9, 1.0]:
            results = [_make_result(score)]
            citations = [{
                "source_index": 1, "document": "doc.pdf",
                "page": 1, "section": "1 Intro", "relevance_score": score,
            }]
            conf = compute_confidence_score(results, citations)
            assert 0.0 <= conf <= 1.0, (
                f"Confidence {conf} out of [0, 1] for retrieval score {score}"
            )

    def test_confidence_higher_with_relevant_context(self):
        """
        Directional test: a high-relevance retrieval result should produce
        a higher confidence score than a low-relevance one.
        """
        from app.services.inference.citations import compute_confidence_score
        from app.services.retrieval.reranker import RankedResult
        from app.storage.vector_store import SearchResult

        def _make_result(score: float) -> RankedResult:
            sr = SearchResult(
                chunk_id="doc_chunk_0",
                text="AUTOSAR Com module.",
                metadata={
                    "document_name": "doc.pdf",
                    "page_number": 1,
                    "section": "1 Intro",
                    "requirement_ids": "",
                },
                distance=1 - score,
                similarity_score=score,
            )
            return RankedResult(search_result=sr, relevance_score=score, original_rank=0)

        high_results = [_make_result(0.95), _make_result(0.90)]
        low_results = [_make_result(0.20), _make_result(0.15)]
        high_citations = [{
            "source_index": 1, "document": "doc.pdf",
            "page": 1, "section": "1 Intro", "relevance_score": 0.95,
        }]
        low_citations = [{
            "source_index": 1, "document": "doc.pdf",
            "page": 1, "section": "1 Intro", "relevance_score": 0.20,
        }]

        high_conf = compute_confidence_score(high_results, high_citations)
        low_conf = compute_confidence_score(low_results, low_citations)

        assert high_conf > low_conf, (
            f"Expected high_conf={high_conf:.3f} > low_conf={low_conf:.3f}"
        )


# ─── Model Quality Metrics Tests ─────────────────────────────────────────

class TestModelQualityMetrics:
    """Verify that model quality counters in MetricsStore work correctly."""

    def test_retrieval_precision_none_when_no_feedback(self):
        from app.monitoring.metrics import MetricsStore
        store = MetricsStore()
        summary = store.get_summary()
        assert summary["model_quality"]["retrieval_precision"] is None

    def test_retrieval_precision_computed_from_feedback(self):
        from app.monitoring.metrics import MetricsStore
        store = MetricsStore()
        store.record_feedback(is_positive=True, confidence=0.9)
        store.record_feedback(is_positive=True, confidence=0.8)
        store.record_feedback(is_positive=False, confidence=0.3)
        summary = store.get_summary()
        precision = summary["model_quality"]["retrieval_precision"]
        assert precision == pytest.approx(2 / 3, rel=1e-3)

    def test_confidence_calibration_split_by_rating(self):
        from app.monitoring.metrics import MetricsStore
        store = MetricsStore()
        store.record_feedback(is_positive=True, confidence=0.9)
        store.record_feedback(is_positive=True, confidence=0.8)
        store.record_feedback(is_positive=False, confidence=0.2)
        summary = store.get_summary()
        mq = summary["model_quality"]
        assert mq["avg_confidence_on_positive_feedback"] == pytest.approx(0.85, rel=1e-3)
        assert mq["avg_confidence_on_negative_feedback"] == pytest.approx(0.2, rel=1e-3)

    def test_model_quality_section_in_summary(self):
        from app.monitoring.metrics import MetricsStore
        store = MetricsStore()
        summary = store.get_summary()
        assert "model_quality" in summary
        assert "retrieval_precision" in summary["model_quality"]
        assert "avg_confidence_on_positive_feedback" in summary["model_quality"]
        assert "avg_confidence_on_negative_feedback" in summary["model_quality"]

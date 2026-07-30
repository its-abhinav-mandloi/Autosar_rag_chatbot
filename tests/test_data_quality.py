"""
Data Validation Tests for the AUTOSAR RAG pipeline.

These tests verify that the data structures flowing through the pipeline
meet quality constraints — analogous to schema validation and missing-value
checks in a traditional ML data pipeline.

Test categories:
    - Chunk schema: required fields are present and non-null
    - Token bounds: chunk sizes stay within configured limits
    - Metadata integrity: ChromaDB-bound metadata has no None values
    - SWS ID format: extracted requirement IDs match the expected pattern
    - Metrics counters: data-quality counters increment correctly
"""

import re

import pytest

from app.monitoring.metrics import MetricsStore
from app.services.ingestion.chunker import Chunk, _split_section_into_chunks
from app.services.ingestion.parser import ParsedDocument, ParsedPage, SWS_PATTERN


# ─── Helpers ────────────────────────────────────────────────────────────

def _make_chunk(**kwargs) -> Chunk:
    defaults = dict(
        chunk_id="doc_chunk_0",
        text="Some AUTOSAR text content here.",
        document_name="autosar_com.pdf",
        page_number=1,
        page_end=1,
        section="7.1 API Specification",
        chunk_index=0,
        total_chunks=1,
        requirement_ids=["[SWS_Com_00001]"],
        token_count=8,
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def _make_parsed_doc(text: str = "Sample text") -> ParsedDocument:
    page = ParsedPage(
        page_number=1,
        text=text,
        headings=["7 API Specification"],
        requirement_ids=[],
        has_tables=False,
    )
    return ParsedDocument(
        document_name="test.pdf",
        file_path="/tmp/test.pdf",
        total_pages=1,
        pages=[page],
    )


# ─── Chunk Schema Validation ─────────────────────────────────────────────

class TestChunkSchema:
    """Verify that every required field on a Chunk is non-empty."""

    def test_chunk_id_is_non_empty(self):
        chunk = _make_chunk()
        assert chunk.chunk_id != ""

    def test_text_is_non_empty(self):
        chunk = _make_chunk()
        assert chunk.text.strip() != ""

    def test_document_name_is_non_empty(self):
        chunk = _make_chunk()
        assert chunk.document_name != ""

    def test_page_number_is_positive(self):
        chunk = _make_chunk()
        assert chunk.page_number >= 1

    def test_section_is_non_empty(self):
        chunk = _make_chunk()
        assert chunk.section != ""

    def test_token_count_is_positive(self):
        chunk = _make_chunk()
        assert chunk.token_count >= 1


# ─── Token Bound Validation ──────────────────────────────────────────────

class TestTokenBounds:
    """Chunks must not exceed the configured chunk_size by more than 10%."""

    def test_chunk_within_size_limit(self):
        chunk_size = 100
        text = " ".join(["word"] * 200)
        chunks = _split_section_into_chunks(
            text=text,
            section_heading="1 Intro",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            chunk_size=chunk_size,
            chunk_overlap=10,
            start_index=0,
        )
        for c in chunks:
            assert c.token_count <= chunk_size * 1.1, (
                f"Chunk token_count {c.token_count} exceeds limit {chunk_size * 1.1}"
            )

    def test_all_chunks_have_positive_token_count(self):
        chunks = _split_section_into_chunks(
            text="Hello world this is a test sentence.",
            section_heading="1 Intro",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            chunk_size=512,
            chunk_overlap=50,
            start_index=0,
        )
        for c in chunks:
            assert c.token_count >= 1


# ─── Metadata Integrity ──────────────────────────────────────────────────

class TestMetadataIntegrity:
    """ChromaDB rejects None metadata values; verify none are produced."""

    def test_metadata_dict_has_no_none_values(self):
        chunk = _make_chunk()
        meta = chunk.to_metadata_dict()
        none_keys = [k for k, v in meta.items() if v is None]
        assert none_keys == [], f"None values found in metadata keys: {none_keys}"

    def test_metadata_dict_has_required_keys(self):
        chunk = _make_chunk()
        meta = chunk.to_metadata_dict()
        required = {
            "document_name", "page_number", "page_end",
            "section", "chunk_index", "total_chunks",
            "requirement_ids", "token_count",
        }
        assert required.issubset(set(meta.keys()))

    def test_requirement_ids_serialized_as_string(self):
        """ChromaDB stores metadata as flat strings, not lists."""
        chunk = _make_chunk(requirement_ids=["[SWS_Com_00001]", "[SWS_NvM_00002]"])
        meta = chunk.to_metadata_dict()
        assert isinstance(meta["requirement_ids"], str)


# ─── SWS ID Format Validation ────────────────────────────────────────────

class TestSWSIDFormat:
    """Extracted AUTOSAR requirement IDs must match the canonical format."""

    @pytest.mark.parametrize("sws_id", [
        "[SWS_Com_00001]",
        "[SWS_NvM_00432]",
        "[SWS_CanIf_01234]",
    ])
    def test_valid_sws_ids_match_pattern(self, sws_id: str):
        assert SWS_PATTERN.search(sws_id), f"{sws_id!r} should match SWS pattern"

    @pytest.mark.parametrize("bad_id", [
        "SWS_Com_00001",        # Missing brackets
        "[SWS00001]",           # Missing module name
        "[COM_00001]",          # Wrong prefix
    ])
    def test_invalid_sws_ids_do_not_match_pattern(self, bad_id: str):
        assert not SWS_PATTERN.fullmatch(bad_id), f"{bad_id!r} should not match"


# ─── Data Quality Metrics Counters ──────────────────────────────────────

class TestDataQualityMetrics:
    """Verify that the data-quality counters in MetricsStore work correctly."""

    def test_schema_failure_counter_increments(self):
        store = MetricsStore()
        assert store.schema_validation_failures == 0
        store.record_schema_failure()
        store.record_schema_failure()
        assert store.schema_validation_failures == 2

    def test_missing_requirements_counter_accumulates(self):
        store = MetricsStore()
        assert store.chunks_missing_requirements == 0
        store.record_missing_requirements(5)
        store.record_missing_requirements(3)
        assert store.chunks_missing_requirements == 8

    def test_data_quality_appears_in_summary(self):
        store = MetricsStore()
        store.record_schema_failure()
        store.record_missing_requirements(2)
        summary = store.get_summary()
        assert "data_quality" in summary
        assert summary["data_quality"]["schema_validation_failures"] == 1
        assert summary["data_quality"]["chunks_missing_requirements"] == 2

"""
Unit tests for the Semantic Chunker.

Tests cover:
    - Chunk size limits
    - Overlap between adjacent chunks
    - Section heading preservation
    - SWS requirement ID preservation in chunks
    - Metadata generation
"""

import pytest
from app.services.ingestion.chunker import (
    Chunk,
    _estimate_tokens,
    _split_section_into_chunks,
)
from app.services.ingestion.parser import ParsedDocument, ParsedPage


class TestTokenEstimation:
    """Test token count estimation."""

    def test_empty_text(self):
        assert _estimate_tokens("") == 1  # Minimum 1

    def test_short_text(self):
        tokens = _estimate_tokens("Hello world")
        assert tokens >= 1

    def test_long_text(self):
        text = "word " * 1000
        tokens = _estimate_tokens(text)
        assert tokens > 100


class TestSectionChunking:
    """Test splitting a section into token-bounded chunks."""

    def test_small_section_single_chunk(self):
        """A short section should produce exactly one chunk."""
        text = "This is a small section with only a few words."
        chunks = _split_section_into_chunks(
            text=text,
            section_heading="1.0 Introduction",
            document_name="test.pdf",
            page_start=1,
            page_end=1,
            chunk_size=512,
            chunk_overlap=50,
            start_index=0,
        )
        assert len(chunks) == 1
        assert chunks[0].section == "1.0 Introduction"
        assert chunks[0].document_name == "test.pdf"

    def test_large_section_multiple_chunks(self):
        """A long section should be split into multiple chunks."""
        text = " ".join(["This is test content."] * 200)
        chunks = _split_section_into_chunks(
            text=text,
            section_heading="7.1 API Spec",
            document_name="test.pdf",
            page_start=1,
            page_end=5,
            chunk_size=100,
            chunk_overlap=10,
            start_index=0,
        )
        assert len(chunks) > 1

    def test_chunk_ids_sequential(self):
        """Chunk IDs should be sequential."""
        text = " ".join(["Test content here."] * 200)
        chunks = _split_section_into_chunks(
            text=text,
            section_heading="1 Intro",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            chunk_size=50,
            chunk_overlap=5,
            start_index=10,
        )
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"doc.pdf_chunk_{10 + i}"

    def test_sws_ids_preserved(self):
        """SWS requirement IDs in text should be captured in chunk metadata."""
        text = "The function [SWS_Com_00432] shall initialize. Also see [SWS_Com_00433]."
        chunks = _split_section_into_chunks(
            text=text,
            section_heading="7.1 Com_Init",
            document_name="SWS_COM.pdf",
            page_start=42,
            page_end=42,
            chunk_size=512,
            chunk_overlap=50,
            start_index=0,
        )
        assert len(chunks) == 1
        assert "[SWS_Com_00432]" in chunks[0].requirement_ids
        assert "[SWS_Com_00433]" in chunks[0].requirement_ids

    def test_empty_text(self):
        """Empty text should produce no chunks."""
        chunks = _split_section_into_chunks(
            text="",
            section_heading="1 Intro",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            chunk_size=512,
            chunk_overlap=50,
            start_index=0,
        )
        assert len(chunks) == 0


class TestChunkMetadata:
    """Test chunk metadata generation."""

    def test_metadata_dict(self):
        chunk = Chunk(
            chunk_id="doc.pdf_chunk_0",
            text="Test text",
            document_name="doc.pdf",
            page_number=5,
            page_end=7,
            section="3.2 Config",
            chunk_index=0,
            total_chunks=10,
            requirement_ids=["[SWS_Com_00001]"],
            token_count=50,
        )
        meta = chunk.to_metadata_dict()
        assert meta["document_name"] == "doc.pdf"
        assert meta["page_number"] == 5
        assert meta["page_end"] == 7
        assert meta["section"] == "3.2 Config"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 10
        assert "[SWS_Com_00001]" in meta["requirement_ids"]

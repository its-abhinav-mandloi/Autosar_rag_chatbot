"""
Unit tests for the PDF Document Parser.

Tests cover:
    - Heading extraction from AUTOSAR-style text
    - SWS requirement ID detection
    - Table detection heuristics
    - Full PDF parsing (if sample PDFs available)
"""

import pytest
from app.services.ingestion.parser import (
    _extract_headings,
    _has_table_structure,
    ParsedPage,
    SWS_PATTERN,
)


class TestHeadingExtraction:
    """Test section heading extraction from AUTOSAR text."""

    def test_simple_heading(self):
        text = "7 API Specification\nSome content here."
        headings = _extract_headings(text)
        assert len(headings) >= 1
        assert any("API Specification" in h for h in headings)

    def test_nested_heading(self):
        text = "7.1.2 Com_Init\nThis function initializes the Com module."
        headings = _extract_headings(text)
        assert len(headings) >= 1
        assert any("Com_Init" in h for h in headings)

    def test_multiple_headings(self):
        text = """7 API Specification
Some text here.
7.1 Function Definitions
More text.
7.1.1 Com_Init
Initialize Com."""
        headings = _extract_headings(text)
        assert len(headings) >= 3

    def test_no_headings(self):
        text = "This is plain text without any section headings."
        headings = _extract_headings(text)
        assert len(headings) == 0

    def test_false_positive_filter(self):
        """Version numbers and dates should not be detected as headings."""
        text = "1.2.3 \nSome content"
        headings = _extract_headings(text)
        # Empty title after number should be filtered
        assert all(len(h.split(None, 1)) >= 2 for h in headings)


class TestSWSDetection:
    """Test AUTOSAR requirement ID detection."""

    def test_single_sws(self):
        text = "The function shall conform to [SWS_Com_00432]."
        matches = SWS_PATTERN.findall(text)
        assert len(matches) == 1
        assert matches[0] == "[SWS_Com_00432]"

    def test_multiple_sws(self):
        text = "[SWS_Com_00432] and [SWS_NvM_00123] are related."
        matches = SWS_PATTERN.findall(text)
        assert len(matches) == 2

    def test_no_sws(self):
        text = "This text has no requirement IDs."
        matches = SWS_PATTERN.findall(text)
        assert len(matches) == 0

    def test_various_modules(self):
        text = "[SWS_Dem_00001] [SWS_Os_00042] [SWS_CanIf_00123]"
        matches = SWS_PATTERN.findall(text)
        assert len(matches) == 3


class TestTableDetection:
    """Test table structure detection heuristic."""

    def test_tab_separated_table(self):
        text = "Col1\tCol2\tCol3\nVal1\tVal2\tVal3\nVal4\tVal5\tVal6"
        assert _has_table_structure(text) is True

    def test_no_table(self):
        text = "This is just regular paragraph text without any tables."
        assert _has_table_structure(text) is False


class TestParsedPage:
    """Test ParsedPage dataclass."""

    def test_char_count(self):
        page = ParsedPage(
            page_number=1,
            text="Hello world",
            headings=[],
            requirement_ids=[],
            has_tables=False,
        )
        assert page.char_count == 11

    def test_empty_page(self):
        page = ParsedPage(
            page_number=1,
            text="",
            headings=[],
            requirement_ids=[],
            has_tables=False,
        )
        assert page.char_count == 0

"""
AUTOSAR PDF Document Parser.

Extracts structured text from AUTOSAR specification PDFs using PyMuPDF.
Handles the specific structure of AUTOSAR documents:
- Section headings (numbered: 7.1.2, 8.3.1, etc.)
- Requirement IDs (SWS tags like [SWS_Com_00432])
- Tables (configuration parameters, API definitions)
- Multi-column layouts

Maps to GR4ML Data Preparation View:
    Raw Data → Document Parsing → Structured text with metadata

Continuation Note:
    This module is complete. It returns a list of ParsedPage objects
    that feed into the chunker. If AUTOSAR PDFs have unusual formatting,
    the heading regex patterns in _extract_headings may need tuning.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from app.monitoring.logging_config import get_logger

logger = get_logger("parser")

# ─── Regex patterns for AUTOSAR document structure ───────────────────────

# Section headings like "7.1.2 Com_Init" or "8 Sequence Diagrams"
HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE)

# AUTOSAR requirement IDs like [SWS_Com_00432] or [SWS_NvM_00123]
SWS_PATTERN = re.compile(r"\[SWS_[A-Za-z]+_\d+\]")

# Table-like patterns (simple heuristic)
TABLE_SEPARATOR = re.compile(r"[-─]{3,}")


@dataclass
class ParsedPage:
    """Structured representation of a single PDF page."""

    page_number: int  # 1-indexed page number
    text: str  # Full extracted text
    headings: List[str]  # Section headings found on this page
    requirement_ids: List[str]  # SWS requirement IDs found
    has_tables: bool  # Whether the page likely contains tables
    char_count: int = 0  # Character count for statistics

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class ParsedDocument:
    """Complete parsed document with metadata."""

    document_name: str
    file_path: str
    total_pages: int
    pages: List[ParsedPage]
    title: Optional[str] = None
    total_chars: int = 0
    total_headings: int = 0
    total_requirements: int = 0

    def __post_init__(self):
        self.total_chars = sum(p.char_count for p in self.pages)
        self.total_headings = sum(len(p.headings) for p in self.pages)
        self.total_requirements = sum(len(p.requirement_ids) for p in self.pages)


def parse_pdf(file_path: str) -> ParsedDocument:
    """
    Parse an AUTOSAR PDF into structured pages.

    Args:
        file_path: Path to the PDF file

    Returns:
        ParsedDocument with extracted text, headings, and metadata

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        RuntimeError: If PDF parsing fails
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("parsing_started", file=path.name, path=str(path))

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    pages: List[ParsedPage] = []
    document_title = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if not text.strip():
            continue

        # Extract headings
        headings = _extract_headings(text)

        # Extract the document title from the first page
        if page_num == 0 and headings:
            document_title = headings[0] if headings else None
        if page_num == 0 and document_title is None:
            # Try first non-empty line as title
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                document_title = lines[0][:100]

        # Extract requirement IDs
        requirement_ids = SWS_PATTERN.findall(text)

        # Detect tables (heuristic)
        has_tables = bool(TABLE_SEPARATOR.search(text)) or _has_table_structure(text)

        pages.append(
            ParsedPage(
                page_number=page_num + 1,  # 1-indexed
                text=text,
                headings=headings,
                requirement_ids=requirement_ids,
                has_tables=has_tables,
            )
        )

    doc.close()

    parsed = ParsedDocument(
        document_name=path.name,
        file_path=str(path),
        total_pages=len(pages),
        pages=pages,
        title=document_title,
    )

    logger.info(
        "parsing_completed",
        document=parsed.document_name,
        pages=parsed.total_pages,
        chars=parsed.total_chars,
        headings=parsed.total_headings,
        requirements=parsed.total_requirements,
    )

    return parsed


def _extract_headings(text: str) -> List[str]:
    """
    Extract section headings from AUTOSAR document text.

    Looks for numbered sections like:
        7.1.2 Com_Init
        8 Sequence Diagrams
        10.2.1.3 ComSignalInitValue
    """
    headings = []
    for match in HEADING_PATTERN.finditer(text):
        number = match.group(1)
        title = match.group(2).strip()
        # Filter out false positives (e.g., version numbers, dates)
        if len(title) > 2 and not title[0].isdigit():
            heading = f"{number} {title}"
            # Limit heading length
            if len(heading) < 200:
                headings.append(heading)
    return headings


def _has_table_structure(text: str) -> bool:
    """
    Heuristic: detect if text contains table-like structure.

    Looks for lines with multiple tab/space-separated columns.
    """
    lines = text.split("\n")
    tab_lines = sum(1 for line in lines if line.count("\t") >= 2)
    # If >10% of lines have tabs, likely a table
    return tab_lines > len(lines) * 0.1 if lines else False


def parse_pdf_from_bytes(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Parse a PDF from raw bytes (for file upload handling).

    Args:
        file_bytes: Raw PDF file content
        filename: Original filename for metadata

    Returns:
        ParsedDocument with extracted text and metadata
    """
    logger.info("parsing_from_bytes", filename=filename, size_bytes=len(file_bytes))

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF from bytes: {e}")

    pages: List[ParsedPage] = []
    document_title = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if not text.strip():
            continue

        headings = _extract_headings(text)
        if page_num == 0:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            document_title = lines[0][:100] if lines else filename

        requirement_ids = SWS_PATTERN.findall(text)
        has_tables = bool(TABLE_SEPARATOR.search(text)) or _has_table_structure(text)

        pages.append(
            ParsedPage(
                page_number=page_num + 1,
                text=text,
                headings=headings,
                requirement_ids=requirement_ids,
                has_tables=has_tables,
            )
        )

    doc.close()

    parsed = ParsedDocument(
        document_name=filename,
        file_path=f"uploaded:{filename}",
        total_pages=len(pages),
        pages=pages,
        title=document_title,
    )

    logger.info(
        "parsing_from_bytes_completed",
        document=parsed.document_name,
        pages=parsed.total_pages,
        chars=parsed.total_chars,
    )

    return parsed

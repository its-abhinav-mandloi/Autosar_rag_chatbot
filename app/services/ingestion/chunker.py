"""
Semantic Chunking Engine for AUTOSAR Documents.

Splits parsed PDF pages into semantic chunks suitable for embedding and retrieval.
The chunking strategy is a critical engineering decision that directly impacts
retrieval quality (GR4ML Data Preparation View).

Strategy:
    1. First, split by AUTOSAR section headings (semantic boundaries)
    2. Then, split large sections by token count with overlap
    3. Preserve [SWS_*] requirement IDs within chunks
    4. Attach rich metadata to each chunk (document, page, section, position)

Continuation Note:
    This module is complete. Adjust CHUNK_SIZE and CHUNK_OVERLAP in config
    to tune retrieval quality. Larger chunks = more context but less precision.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.services.ingestion.parser import ParsedDocument, ParsedPage

logger = get_logger("chunker")

# Regex for section headings (used to split text into semantic sections)
SECTION_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE)

# SWS requirement IDs
SWS_RE = re.compile(r"\[SWS_[A-Za-z]+_\d+\]")


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding.

    Contains the text content plus rich metadata for citation generation.
    """

    chunk_id: str  # Unique ID: "{doc_name}_chunk_{index}"
    text: str  # The actual text content
    document_name: str  # Source document filename
    page_number: int  # Page where this chunk starts
    page_end: int  # Page where this chunk ends
    section: str  # Current section heading
    chunk_index: int  # Position in the document's chunk sequence
    total_chunks: int  # Total chunks in the document (set after all chunks created)
    requirement_ids: List[str]  # SWS IDs found in this chunk
    token_count: int  # Approximate token count
    metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """Convert to flat dict for ChromaDB metadata storage."""
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "requirement_ids": ", ".join(self.requirement_ids),
            "token_count": self.token_count,
        }


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Split a parsed document into semantic chunks.

    Args:
        parsed_doc: Output from the PDF parser
        chunk_size: Max tokens per chunk (default from config)
        chunk_overlap: Overlap tokens between chunks (default from config)

    Returns:
        List of Chunk objects ready for embedding
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    logger.info(
        "chunking_started",
        document=parsed_doc.document_name,
        pages=parsed_doc.total_pages,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    # Step 1: Combine all pages into sections based on headings
    sections = _split_into_sections(parsed_doc)

    # Step 2: Split sections into token-bounded chunks with overlap
    chunks: List[Chunk] = []
    chunk_index = 0

    for section_heading, section_text, page_start, page_end in sections:
        section_chunks = _split_section_into_chunks(
            text=section_text,
            section_heading=section_heading,
            document_name=parsed_doc.document_name,
            page_start=page_start,
            page_end=page_end,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=chunk_index,
        )
        chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    # Set total_chunks on all chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    logger.info(
        "chunking_completed",
        document=parsed_doc.document_name,
        sections=len(sections),
        chunks=len(chunks),
        avg_tokens=sum(c.token_count for c in chunks) // max(len(chunks), 1),
    )

    return chunks


def _split_into_sections(
    parsed_doc: ParsedDocument,
) -> List[tuple]:
    """
    Split document into sections based on headings.

    Returns list of (heading, text, page_start, page_end) tuples.
    """
    sections = []
    current_heading = "Introduction"
    current_text = []
    current_page_start = 1
    current_page_end = 1

    for page in parsed_doc.pages:
        text = page.text

        # Find section headings on this page
        heading_matches = list(SECTION_HEADING_RE.finditer(text))

        if not heading_matches:
            # No headings — append entire page to current section
            current_text.append(text)
            current_page_end = page.page_number
        else:
            # Split page text at heading boundaries
            last_end = 0
            for match in heading_matches:
                # Text before this heading belongs to previous section
                pre_heading_text = text[last_end : match.start()].strip()
                if pre_heading_text:
                    current_text.append(pre_heading_text)

                # Save the previous section
                if current_text:
                    combined = "\n".join(current_text).strip()
                    if combined:
                        sections.append(
                            (
                                current_heading,
                                combined,
                                current_page_start,
                                current_page_end,
                            )
                        )

                # Start new section
                current_heading = match.group(0).strip()
                current_text = []
                current_page_start = page.page_number
                current_page_end = page.page_number
                last_end = match.end()

            # Text after the last heading on this page
            remaining = text[last_end:].strip()
            if remaining:
                current_text.append(remaining)
                current_page_end = page.page_number

    # Don't forget the last section
    if current_text:
        combined = "\n".join(current_text).strip()
        if combined:
            sections.append(
                (
                    current_heading,
                    combined,
                    current_page_start,
                    current_page_end,
                )
            )

    return sections


def _split_section_into_chunks(
    text: str,
    section_heading: str,
    document_name: str,
    page_start: int,
    page_end: int,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int,
) -> List[Chunk]:
    """
    Split a section's text into token-bounded chunks with overlap.

    Uses word-based tokenization (approximate: 1 token ≈ 0.75 words).
    """
    words = text.split()

    if not words:
        return []

    # Approximate tokens: 1 token ≈ 0.75 words (conservative estimate)
    words_per_chunk = int(chunk_size * 0.75)
    words_overlap = int(chunk_overlap * 0.75)

    if len(words) <= words_per_chunk:
        # Section fits in one chunk
        chunk_text = " ".join(words)
        return [
            Chunk(
                chunk_id=f"{document_name}_chunk_{start_index}",
                text=chunk_text,
                document_name=document_name,
                page_number=page_start,
                page_end=page_end,
                section=section_heading[:200],
                chunk_index=start_index,
                total_chunks=0,  # Set later
                requirement_ids=SWS_RE.findall(chunk_text),
                token_count=_estimate_tokens(chunk_text),
            )
        ]

    chunks = []
    position = 0
    local_index = 0

    while position < len(words):
        end = min(position + words_per_chunk, len(words))
        chunk_words = words[position:end]
        chunk_text = " ".join(chunk_words)

        # Try to break at sentence boundaries
        if end < len(words):
            # Look for the last period in the chunk
            last_period = chunk_text.rfind(".")
            if last_period > len(chunk_text) * 0.5:
                chunk_text = chunk_text[: last_period + 1]
                # Recalculate end position based on actual words used
                actual_words = len(chunk_text.split())
                end = position + actual_words

        chunks.append(
            Chunk(
                chunk_id=f"{document_name}_chunk_{start_index + local_index}",
                text=chunk_text,
                document_name=document_name,
                page_number=page_start,
                page_end=page_end,
                section=section_heading[:200],
                chunk_index=start_index + local_index,
                total_chunks=0,  # Set later
                requirement_ids=SWS_RE.findall(chunk_text),
                token_count=_estimate_tokens(chunk_text),
            )
        )

        local_index += 1
        # Move forward by (chunk_size - overlap) words
        position = end - words_overlap if end < len(words) else end

        # Safety: prevent infinite loop
        if position <= (end - words_per_chunk):
            position = end

    return chunks


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses the heuristic: 1 token ≈ 4 characters (for English text).
    More accurate than word count for mixed technical content.
    """
    return max(1, len(text) // 4)

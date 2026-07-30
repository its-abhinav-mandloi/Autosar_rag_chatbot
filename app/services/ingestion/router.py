"""
Ingestion Service API Router.

Provides the POST /ingest endpoint for uploading and processing AUTOSAR PDFs.
Orchestrates the full ingestion pipeline:
    Upload → Parse → Chunk → Embed → Enrich → Index

Also provides endpoints to:
    - List all ingested documents
    - Delete a document from the store
    - Ingest from the local Database/ folder

Continuation Note:
    This module is complete. The /ingest endpoint handles both file uploads
    and local file ingestion. The /ingest/local endpoint processes files
    from the Database/ folder.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import metrics
from app.services.ingestion.chunker import chunk_document
from app.services.ingestion.embedder import embed_batch
from app.services.ingestion.parser import parse_pdf, parse_pdf_from_bytes
from app.storage.metadata_store import get_metadata_store
from app.storage.vector_store import get_vector_store

logger = get_logger("ingestion_router")
router = APIRouter(prefix="/ingest", tags=["Ingestion Service"])


# ─── Response Models ─────────────────────────────────────────────────────


class IngestionResponse(BaseModel):
    status: str
    document: str
    chunks_created: int
    pages_processed: int
    total_characters: int
    processing_time_seconds: float
    sections_found: int
    requirement_ids_found: int


class DocumentInfo(BaseModel):
    document_name: str
    page_count: int
    chunk_count: int
    ingested_at: str
    embedding_model: str


class LocalIngestionRequest(BaseModel):
    filename: Optional[str] = None  # If None, ingest all files in Database/


# ─── Ingestion Pipeline ─────────────────────────────────────────────────


async def _run_ingestion_pipeline(
    parsed_doc,
    file_path_str: str,
) -> IngestionResponse:
    """
    Core ingestion pipeline shared by upload and local ingestion.

    Steps:
        1. Chunk the parsed document
        2. Embed all chunks via Ollama
        3. Store embeddings in ChromaDB
        4. Record metadata
    """
    settings = get_settings()
    start_time = time.time()

    # Step 1: Chunk
    logger.info("pipeline_chunking", document=parsed_doc.document_name)
    chunks = chunk_document(parsed_doc)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No text chunks extracted from {parsed_doc.document_name}. "
            f"The PDF may be image-only or empty.",
        )

    # Step 2: Embed
    logger.info(
        "pipeline_embedding",
        document=parsed_doc.document_name,
        chunks=len(chunks),
    )
    texts = [chunk.text for chunk in chunks]
    embeddings = await embed_batch(texts)

    # Step 3: Store in vector database
    logger.info("pipeline_indexing", document=parsed_doc.document_name)
    vector_store = get_vector_store()

    # Delete existing chunks for this document (idempotent re-ingestion)
    vector_store.delete_document(parsed_doc.document_name)

    # Prepare data for ChromaDB
    ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [chunk.to_metadata_dict() for chunk in chunks]

    vector_store.add_documents(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Step 4: Record metadata
    processing_time = time.time() - start_time
    metadata_store = get_metadata_store()
    metadata_store.record_ingestion(
        document_name=parsed_doc.document_name,
        file_path=file_path_str,
        page_count=parsed_doc.total_pages,
        chunk_count=len(chunks),
        processing_time_seconds=processing_time,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # Record metrics
    metrics.record_ingestion(
        chunks=len(chunks),
        pages=parsed_doc.total_pages,
        time_seconds=processing_time,
    )

    # Data quality: track chunks with no AUTOSAR requirement IDs
    missing_req_count = sum(1 for c in chunks if not c.requirement_ids)
    if missing_req_count > 0:
        metrics.record_missing_requirements(missing_req_count)

    # Count unique sections
    unique_sections = len(set(c.section for c in chunks))
    # Count unique requirement IDs
    all_req_ids = set()
    for chunk in chunks:
        all_req_ids.update(chunk.requirement_ids)

    response = IngestionResponse(
        status="success",
        document=parsed_doc.document_name,
        chunks_created=len(chunks),
        pages_processed=parsed_doc.total_pages,
        total_characters=parsed_doc.total_chars,
        processing_time_seconds=round(processing_time, 2),
        sections_found=unique_sections,
        requirement_ids_found=len(all_req_ids),
    )

    logger.info(
        "ingestion_completed",
        document=parsed_doc.document_name,
        chunks=len(chunks),
        pages=parsed_doc.total_pages,
        time_s=round(processing_time, 2),
    )

    return response


# ─── API Endpoints ───────────────────────────────────────────────────────


@router.post("/upload", response_model=IngestionResponse)
async def ingest_upload(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document.

    Accepts a PDF file via multipart/form-data, processes it through
    the full ingestion pipeline (parse → chunk → embed → index),
    and stores the results in ChromaDB.

    Idempotent: re-uploading the same document replaces old embeddings.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info("upload_received", filename=file.filename, size=file.size)

    # Read file content
    file_bytes = await file.read()

    # Parse PDF from bytes
    try:
        parsed_doc = parse_pdf_from_bytes(file_bytes, file.filename)
    except Exception as e:
        logger.error("parsing_failed", filename=file.filename, error=str(e))
        metrics.record_schema_failure()
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")

    # Run pipeline
    return await _run_ingestion_pipeline(parsed_doc, f"uploaded:{file.filename}")


@router.post("/local", response_model=List[IngestionResponse])
async def ingest_local(request: LocalIngestionRequest = LocalIngestionRequest()):
    """
    Ingest PDF files from the local Database/ folder.

    If filename is specified, ingest only that file.
    If filename is None, ingest all PDF files in the Database/ folder.
    """
    settings = get_settings()
    data_dir = settings.raw_data_absolute_path

    if not data_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Database directory not found: {data_dir}",
        )

    # Find PDF files
    if request.filename:
        pdf_path = data_dir / request.filename
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.filename}",
            )
        pdf_files = [pdf_path]
    else:
        pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(
            status_code=404,
            detail="No PDF files found in Database/ folder",
        )

    logger.info("local_ingestion_started", files=len(pdf_files))

    results = []
    for pdf_path in pdf_files:
        logger.info("processing_local_file", file=pdf_path.name)
        try:
            parsed_doc = parse_pdf(str(pdf_path))
            result = await _run_ingestion_pipeline(parsed_doc, str(pdf_path))
            results.append(result)
        except Exception as e:
            logger.error("local_ingestion_error", file=pdf_path.name, error=str(e))
            results.append(
                IngestionResponse(
                    status=f"error: {str(e)}",
                    document=pdf_path.name,
                    chunks_created=0,
                    pages_processed=0,
                    total_characters=0,
                    processing_time_seconds=0,
                    sections_found=0,
                    requirement_ids_found=0,
                )
            )

    return results


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all ingested documents with their metadata."""
    metadata_store = get_metadata_store()
    docs = metadata_store.get_all_documents()
    return [
        DocumentInfo(
            document_name=d["document_name"],
            page_count=d["page_count"],
            chunk_count=d["chunk_count"],
            ingested_at=d["ingested_at"],
            embedding_model=d["embedding_model"],
        )
        for d in docs
    ]


@router.delete("/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document and all its chunks from the store."""
    vector_store = get_vector_store()
    metadata_store = get_metadata_store()

    chunks_removed = vector_store.delete_document(document_name)
    metadata_store.delete_document(document_name)

    return {
        "status": "deleted",
        "document": document_name,
        "chunks_removed": chunks_removed,
    }

"""
Retrieval Service API Router.

Provides the POST /query endpoint — the core RAG pipeline entry point.
Orchestrates the full retrieval-augmented generation flow:
    Query → Search → Re-rank → Assemble Context → Build Prompt → LLM → Citations

This implements the RAG Architectural Pattern (from M3):
    1. Retrieve relevant knowledge from the vector store
    2. Augment the LLM prompt with retrieved context
    3. Generate a grounded answer with citations

Continuation Note:
    This module is complete. The /query endpoint is the main user-facing
    endpoint. The /query/search endpoint provides raw search results
    without LLM generation (useful for debugging retrieval quality).
"""

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import Timer, metrics
from app.services.inference.citations import compute_confidence_score, extract_citations
from app.services.inference.llm import generate_completion
from app.services.inference.prompt import build_query_prompt, get_system_prompt
from app.services.retrieval.context import assemble_context, get_source_list
from app.services.retrieval.reranker import rerank
from app.services.retrieval.search import semantic_search

logger = get_logger("retrieval_router")
router = APIRouter(prefix="/query", tags=["Retrieval Service"])


# ─── Request / Response Models ───────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=2000, description="Natural language question"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to retrieve")
    document_filter: Optional[str] = Field(default=None, description="Filter to specific document")
    skip_reranking: bool = Field(
        default=False, description="Skip LLM re-ranking for faster results"
    )


class Citation(BaseModel):
    source_index: int
    document: str
    page: int
    section: str
    relevance_score: float
    requirement_ids: str = ""
    implicit: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    latency_ms: float
    chunks_retrieved: int
    chunks_after_rerank: int
    model_used: str


class SearchOnlyResponse(BaseModel):
    results: List[dict]
    query: str
    total_results: int
    latency_ms: float


# ─── API Endpoints ───────────────────────────────────────────────────────


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about ingested AUTOSAR documents.

    Full RAG pipeline:
        1. Semantic search for relevant chunks
        2. LLM-based re-ranking (optional)
        3. Context assembly with source markers
        4. Prompt construction with anti-hallucination guardrails
        5. LLM answer generation
        6. Citation extraction and confidence scoring

    Returns a structured response with:
        - Natural language answer
        - Citations to source documents (page, section)
        - Confidence score (0-1)
        - Latency in milliseconds
    """
    settings = get_settings()

    with Timer() as timer:
        # Step 1: Semantic search
        logger.info("query_started", question=request.question[:100])

        try:
            search_results = await semantic_search(
                query=request.question,
                top_k=settings.TOP_K_RETRIEVAL,
                document_filter=request.document_filter,
            )
        except Exception as e:
            logger.error("search_failed", error=str(e))
            raise HTTPException(status_code=503, detail=f"Search failed: {e}")

        if not search_results:
            return QueryResponse(
                answer=(
                    "No relevant documents found. Please ingest AUTOSAR documents"
                    " first using the /ingest endpoint."
                ),
                citations=[],
                confidence=0.0,
                latency_ms=timer.elapsed_ms,
                chunks_retrieved=0,
                chunks_after_rerank=0,
                model_used=settings.LLM_MODEL,
            )

        chunks_retrieved = len(search_results)

        # Step 2: Re-ranking
        if not request.skip_reranking and len(search_results) > request.top_k:
            try:
                ranked_results = await rerank(
                    query=request.question,
                    search_results=search_results,
                    top_n=request.top_k,
                )
            except Exception as e:
                logger.warning("reranking_failed_fallback", error=str(e))
                # Fallback: use top-k search results directly
                from app.services.retrieval.reranker import RankedResult

                ranked_results = [
                    RankedResult(
                        search_result=r,
                        relevance_score=r.similarity_score,
                        original_rank=i,
                    )
                    for i, r in enumerate(search_results[: request.top_k])
                ]
        else:
            from app.services.retrieval.reranker import RankedResult

            ranked_results = [
                RankedResult(
                    search_result=r,
                    relevance_score=r.similarity_score,
                    original_rank=i,
                )
                for i, r in enumerate(search_results[: request.top_k])
            ]

        chunks_after_rerank = len(ranked_results)

        # Step 3: Context assembly
        context = assemble_context(ranked_results)

        # Step 4: Prompt construction
        system_prompt = get_system_prompt()
        query_prompt = build_query_prompt(
            question=request.question,
            context=context,
        )

        # Step 5: LLM generation
        try:
            answer = await generate_completion(
                prompt=query_prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.1,
            )
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=f"LLM generation failed: {e}. Check Ollama status at /health",
            )

        # Step 6: Citation extraction
        citations = extract_citations(answer, ranked_results)
        confidence = compute_confidence_score(ranked_results, citations)

    # Record metrics
    metrics.record_query(timer.elapsed_ms)

    logger.info(
        "query_completed",
        question=request.question[:100],
        latency_ms=round(timer.elapsed_ms, 1),
        citations=len(citations),
        confidence=confidence,
    )

    return QueryResponse(
        answer=answer,
        citations=[Citation(**c) for c in citations],
        confidence=confidence,
        latency_ms=round(timer.elapsed_ms, 1),
        chunks_retrieved=chunks_retrieved,
        chunks_after_rerank=chunks_after_rerank,
        model_used=settings.LLM_MODEL,
    )


@router.post("/search", response_model=SearchOnlyResponse)
async def search_only(request: QueryRequest):
    """
    Perform semantic search without LLM generation.

    Useful for:
        - Debugging retrieval quality
        - Evaluating search precision
        - Faster lookups when LLM answer isn't needed
    """
    with Timer() as timer:
        results = await semantic_search(
            query=request.question,
            top_k=request.top_k,
            document_filter=request.document_filter,
        )

    return SearchOnlyResponse(
        results=[
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:500],
                "document": r.metadata.get("document_name", "Unknown"),
                "page": r.metadata.get("page_number", 0),
                "section": r.metadata.get("section", "N/A"),
                "similarity_score": r.similarity_score,
                "requirement_ids": r.metadata.get("requirement_ids", ""),
            }
            for r in results
        ],
        query=request.question,
        total_results=len(results),
        latency_ms=round(timer.elapsed_ms, 1),
    )

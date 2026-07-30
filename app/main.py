"""
AUTOSAR Document Intelligence Assistant — Main Application Entry Point.

FastAPI application that serves as the API Gateway for the microservices:
    - Ingestion Service (/ingest)
    - Retrieval Service (/query)
    - Feedback Service (/feedback)
    - Health & Monitoring (/health)

Implements cross-cutting concerns:
    - CORS middleware for frontend access
    - Correlation ID middleware for request tracing
    - Global exception handling
    - Heartbeat background task startup

Group 151 — BITS Pilani WILP
    - Abhinav Mandloi (2025aa05473@wilp.bits-pilani.ac.in)
    - Prajwal Rastogi (2025aa05564@wilp.bits-pilani.ac.in)
    - Pritish Joshi (2025aa05686@wilp.bits-pilani.ac.in)
    - Satwinder Singh (2025aa05553@wilp.bits-pilani.ac.in)
    - Shray Vijay (2025aa05533@wilp.bits-pilani.ac.in)

Continuation Note:
    This module is complete. To start the server:
        python -m uvicorn app.main:app --reload --port 8000
    Or:
        python -m app.main
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.feedback.router import router as feedback_router
from app.monitoring.health import router as health_router
from app.monitoring.health import start_heartbeat
from app.monitoring.logging_config import (
    correlation_id_var,
    generate_correlation_id,
    get_logger,
    setup_logging,
)
from app.services.ingestion.router import router as ingestion_router
from app.services.retrieval.router import router as retrieval_router

logger = get_logger("main")


# ─── Application Lifespan ───────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup:
        - Initialize structured logging
        - Verify Ollama connectivity
        - Start heartbeat background task

    Shutdown:
        - Cancel heartbeat task
        - Clean up resources
    """
    settings = get_settings()

    # --- Startup ---
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "application_starting",
        host=settings.HOST,
        port=settings.PORT,
        ollama_url=settings.OLLAMA_BASE_URL,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL,
    )

    # Ensure data directories exist
    settings.chroma_db_absolute_path.mkdir(parents=True, exist_ok=True)
    settings.metadata_store_absolute_path.mkdir(parents=True, exist_ok=True)

    # Start heartbeat monitoring
    heartbeat_task = start_heartbeat()
    logger.info("heartbeat_task_started")

    yield

    # --- Shutdown ---
    heartbeat_task.cancel()
    logger.info("application_shutdown")


# ─── FastAPI App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="AUTOSAR Document Intelligence Assistant",
    description=(
        "RAG-powered system for intelligent querying of AUTOSAR BSW specifications. "
        "Upload AUTOSAR PDFs, ask natural language questions, and get precise answers "
        "with citations. Built with Ollama (local LLM), ChromaDB (vector store), and FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc at /redoc
)


# ─── Middleware ──────────────────────────────────────────────────────────

# CORS — allow frontend clients to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Inject a unique correlation ID into each request for cross-service tracing.

    The correlation ID is:
        1. Read from X-Correlation-ID header (if provided by client)
        2. Or generated as a new UUID
        3. Set in the context variable for structured logging
        4. Returned in the response headers
    """
    # Get or generate correlation ID
    corr_id = request.headers.get("X-Correlation-ID", generate_correlation_id())
    correlation_id_var.set(corr_id)

    # Log request
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        correlation_id=corr_id,
    )

    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Add correlation ID and timing to response headers
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 1))

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 1),
    )

    return response


# ─── Global Exception Handler ───────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "correlation_id": correlation_id_var.get(),
        },
    )


# ─── Register Routers (Microservice Endpoints) ──────────────────────────

app.include_router(health_router)  # GET /health, GET /health/metrics
app.include_router(ingestion_router)  # POST /ingest/upload, POST /ingest/local
app.include_router(retrieval_router)  # POST /query, POST /query/search
app.include_router(feedback_router)  # POST /feedback, GET /feedback/analytics


# ─── Static Files & Chat UI ──────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["UI"], response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    # Fallback: JSON API overview
    return HTMLResponse(
        content="<h1>AUTOSAR Intelligence Assistant</h1><p>UI not found. Visit <a href='/docs'>/docs</a> for API.</p>"
    )


@app.get("/api", tags=["Root"])
async def api_overview():
    """API overview endpoint."""
    return {
        "name": "AUTOSAR Document Intelligence Assistant",
        "version": "1.0.0",
        "group": "Group 151 — BITS Pilani WILP",
        "description": "RAG-powered AUTOSAR specification query system",
        "ui": "/",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ingest_upload": "POST /ingest/upload",
            "ingest_local": "POST /ingest/local",
            "query": "POST /query",
            "search": "POST /query/search",
            "feedback": "POST /feedback",
            "analytics": "GET /feedback/analytics",
            "metrics": "GET /health/metrics",
        },
    }


# ─── Direct Run Support ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )

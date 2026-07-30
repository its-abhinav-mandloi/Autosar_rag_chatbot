"""
Health Check & Heartbeat Tactic Implementation.

Implements the Heartbeat architectural tactic (from M3 - Architecture):
- Each microservice exposes a /health endpoint
- A background task pings all services every HEARTBEAT_INTERVAL_SECONDS
- Degraded services are detected and reported
- Circuit breaker state is tracked

This directly addresses the Availability quality requirement:
    "The system shall maintain 99% uptime with graceful degradation."

Continuation Note:
    This module is complete. The heartbeat background task is started in main.py
    via the `start_heartbeat` function. Health status is served at GET /health.
"""

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import httpx
from fastapi import APIRouter

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import metrics

logger = get_logger("health")
router = APIRouter(tags=["Health & Monitoring"])

# ─── Service Status Tracking ────────────────────────────────────────────


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ServiceHealth:
    """Tracks health state of an individual service/dependency."""

    def __init__(self, name: str):
        self.name = name
        self.status: ServiceStatus = ServiceStatus.UNAVAILABLE
        self.last_check: Optional[datetime] = None
        self.last_healthy: Optional[datetime] = None
        self.consecutive_failures: int = 0
        self.error_message: Optional[str] = None

    def mark_healthy(self) -> None:
        self.status = ServiceStatus.HEALTHY
        self.last_check = datetime.now(timezone.utc)
        self.last_healthy = self.last_check
        self.consecutive_failures = 0
        self.error_message = None

    def mark_unhealthy(self, error: str) -> None:
        self.consecutive_failures += 1
        self.last_check = datetime.now(timezone.utc)
        self.error_message = error
        # After 3 consecutive failures, mark as unavailable
        if self.consecutive_failures >= 3:
            self.status = ServiceStatus.UNAVAILABLE
        else:
            self.status = ServiceStatus.DEGRADED

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "consecutive_failures": self.consecutive_failures,
            "error": self.error_message,
        }


# ─── Global Health Registry ─────────────────────────────────────────────

_start_time = time.time()
_service_health: Dict[str, ServiceHealth] = {
    "ollama_api": ServiceHealth("ollama_api"),
    "embedding_model": ServiceHealth("embedding_model"),
    "llm_model": ServiceHealth("llm_model"),
    "vector_store": ServiceHealth("vector_store"),
}


# ─── Health Check Functions ──────────────────────────────────────────────


async def check_ollama_api(settings=None) -> bool:
    """Check if Ollama API is reachable."""
    if settings is None:
        settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                _service_health["ollama_api"].mark_healthy()
                return True
            else:
                _service_health["ollama_api"].mark_unhealthy(f"HTTP {resp.status_code}")
                return False
    except Exception as e:
        _service_health["ollama_api"].mark_unhealthy(str(e))
        return False


async def check_embedding_model(settings=None) -> bool:
    """Check if the embedding model is loaded in Ollama."""
    if settings is None:
        settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": settings.EMBEDDING_MODEL, "prompt": "health check"},
            )
            if resp.status_code == 200:
                _service_health["embedding_model"].mark_healthy()
                return True
            else:
                _service_health["embedding_model"].mark_unhealthy(f"HTTP {resp.status_code}")
                return False
    except Exception as e:
        _service_health["embedding_model"].mark_unhealthy(str(e))
        return False


async def check_llm_model(settings=None) -> bool:
    """Check if the LLM model is loaded in Ollama."""
    if settings is None:
        settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": "Say OK",
                    "stream": False,
                    "options": {"num_predict": 5},
                },
            )
            if resp.status_code == 200:
                _service_health["llm_model"].mark_healthy()
                return True
            else:
                _service_health["llm_model"].mark_unhealthy(f"HTTP {resp.status_code}")
                return False
    except Exception as e:
        _service_health["llm_model"].mark_unhealthy(str(e))
        return False


async def check_vector_store() -> bool:
    """Check if ChromaDB is accessible."""
    try:
        import chromadb

        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.chroma_db_absolute_path))
        # Try to list collections — if this works, ChromaDB is healthy
        client.heartbeat()
        _service_health["vector_store"].mark_healthy()
        return True
    except Exception as e:
        _service_health["vector_store"].mark_unhealthy(str(e))
        return False


# ─── Heartbeat Background Task ──────────────────────────────────────────


async def heartbeat_loop() -> None:
    """
    Background task: periodically check all service health.

    This implements the Heartbeat Tactic from the Architecture module (M3).
    Runs every HEARTBEAT_INTERVAL_SECONDS and logs status changes.
    """
    settings = get_settings()
    logger.info("heartbeat_started", interval=settings.HEARTBEAT_INTERVAL_SECONDS)

    while True:
        try:
            await asyncio.gather(
                check_ollama_api(settings),
                check_embedding_model(settings),
                check_llm_model(settings),
                check_vector_store(),
            )

            # Log any unhealthy services
            for name, health in _service_health.items():
                if health.status != ServiceStatus.HEALTHY:
                    logger.warning(
                        "service_unhealthy",
                        service=name,
                        status=health.status.value,
                        failures=health.consecutive_failures,
                        error=health.error_message,
                    )
        except Exception as e:
            logger.error("heartbeat_error", error=str(e))

        await asyncio.sleep(settings.HEARTBEAT_INTERVAL_SECONDS)


def start_heartbeat() -> asyncio.Task:
    """Start the heartbeat background task. Call from main.py lifespan."""
    return asyncio.create_task(heartbeat_loop())


# ─── API Endpoints ───────────────────────────────────────────────────────


@router.get("/health")
async def health_check():
    """
    System health check endpoint.

    Returns overall system status and individual service health.
    Used by monitoring tools and the heartbeat tactic.
    """
    # Run fresh checks
    await asyncio.gather(
        check_ollama_api(),
        check_embedding_model(),
        check_llm_model(),
        check_vector_store(),
    )

    # Determine overall status
    statuses = [h.status for h in _service_health.values()]
    if all(s == ServiceStatus.HEALTHY for s in statuses):
        overall = ServiceStatus.HEALTHY
    elif any(s == ServiceStatus.UNAVAILABLE for s in statuses):
        overall = ServiceStatus.DEGRADED
    else:
        overall = ServiceStatus.DEGRADED

    uptime = time.time() - _start_time

    return {
        "status": overall.value,
        "uptime_seconds": round(uptime, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {name: health.to_dict() for name, health in _service_health.items()},
    }


@router.get("/health/metrics")
async def get_metrics():
    """
    Return operational metrics for observability.

    Includes query latency percentiles, ingestion stats, retrieval scores,
    LLM usage, feedback rates, and error counts.
    """
    return metrics.get_summary()

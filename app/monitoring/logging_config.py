"""
Structured Logging Configuration.

Sets up structlog for JSON-formatted, correlation-ID-tagged logs across all services.
FastAPI middleware injects a unique correlation_id per request so logs can be traced
across the entire RAG pipeline (ingestion → retrieval → inference).

Continuation Note:
    This module is complete. Import `get_logger` anywhere to get a bound logger.
    The correlation ID middleware is added in app/main.py.
"""

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

# Context variable to hold per-request correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="no-correlation-id")


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog with JSON output and stdlib integration.

    Call once at application startup (in main.py).
    """
    # Set stdlib root logger level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_correlation_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_correlation_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Structlog processor: inject correlation_id from context var."""
    event_dict["correlation_id"] = correlation_id_var.get()
    return event_dict


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound with the given name."""
    return structlog.get_logger(name)


def generate_correlation_id() -> str:
    """Generate a new UUID-based correlation ID for a request."""
    return str(uuid.uuid4())[:8]

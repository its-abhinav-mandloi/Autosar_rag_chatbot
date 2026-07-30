"""
Ollama LLM Client.

Handles all communication with the Ollama API for text generation.
Implements:
    - Synchronous generation (for re-ranking, short responses)
    - Streaming generation (for user-facing answers)
    - Retry with exponential backoff
    - Circuit breaker pattern (fails fast after consecutive errors)

This is the "LLM Inference" agent in the HLD's Retrieval & Reasoning Engine.

Continuation Note:
    This module is complete. If switching LLM models, update LLM_MODEL in .env.
    The circuit breaker resets after CIRCUIT_BREAKER_RESET_SECONDS.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

import httpx

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import metrics

logger = get_logger("llm")

# ─── Circuit Breaker State ───────────────────────────────────────────────


class CircuitBreaker:
    """
    Simple circuit breaker for Ollama LLM calls.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Too many failures, requests fail immediately
        HALF_OPEN: Testing if service recovered (after reset timeout)
    """

    def __init__(self, failure_threshold: int = 5, reset_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def can_proceed(self) -> bool:
        """Check if a request should be allowed."""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            # Check if reset timeout has elapsed
            if time.time() - self.last_failure_time > self.reset_seconds:
                self.state = "HALF_OPEN"
                logger.info("circuit_breaker_half_open")
                return True
            return False
        # HALF_OPEN: allow one request to test
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == "HALF_OPEN":
            logger.info("circuit_breaker_closed", reason="successful_test_request")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                "circuit_breaker_open",
                failures=self.failure_count,
                reset_after_s=self.reset_seconds,
            )
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning("circuit_breaker_reopened")


# Global circuit breaker instance
_circuit_breaker = CircuitBreaker()

# ─── Constants ───────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
REQUEST_TIMEOUT = 120.0  # LLM generation can be slow


# ─── Generation Functions ────────────────────────────────────────────────


async def generate_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Generate a text completion from the LLM.

    Args:
        prompt: The user/query prompt
        system_prompt: Optional system instructions
        model: LLM model name (default from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = deterministic)

    Returns:
        Generated text string

    Raises:
        RuntimeError: If generation fails or circuit breaker is open
    """
    settings = get_settings()
    model = model or settings.LLM_MODEL

    # Circuit breaker check
    if not _circuit_breaker.can_proceed():
        raise RuntimeError(
            "LLM circuit breaker is OPEN. Service temporarily unavailable. "
            f"Will retry after {_circuit_breaker.reset_seconds}s."
        )

    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        request_body["system"] = system_prompt

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()

                generated_text = data.get("response", "")
                total_tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

                # Record metrics
                metrics.record_llm_call(tokens=total_tokens)
                _circuit_breaker.record_success()

                logger.info(
                    "llm_generation_completed",
                    model=model,
                    prompt_length=len(prompt),
                    response_length=len(generated_text),
                    tokens=total_tokens,
                )

                return generated_text

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            wait_time = RETRY_BACKOFF_BASE**attempt
            logger.warning(
                "llm_retry",
                attempt=attempt + 1,
                error=str(e),
                wait_s=wait_time,
            )
            metrics.record_error("llm_retry")

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait_time)
            else:
                _circuit_breaker.record_failure()
                metrics.record_error("llm_failed")
                raise RuntimeError(f"LLM generation failed after {MAX_RETRIES} attempts: {e}")


async def generate_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """
    Stream a text completion from the LLM.

    Yields text tokens as they are generated. Used for low-latency
    user-facing responses.

    Args:
        prompt: The user/query prompt
        system_prompt: Optional system instructions
        model: LLM model name (default from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Yields:
        Text chunks as they are generated
    """
    settings = get_settings()
    model = model or settings.LLM_MODEL

    if not _circuit_breaker.can_proceed():
        yield "[Error: LLM service temporarily unavailable. Please retry later.]"
        return

    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        request_body["system"] = system_prompt

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=request_body,
            ) as response:
                response.raise_for_status()
                total_tokens = 0

                async for line in response.aiter_lines():
                    if line:
                        import json

                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                total_tokens = data.get("eval_count", 0)
                                break
                        except json.JSONDecodeError:
                            continue

                metrics.record_llm_call(tokens=total_tokens)
                _circuit_breaker.record_success()

    except Exception as e:
        _circuit_breaker.record_failure()
        logger.error("llm_stream_error", error=str(e))
        yield f"[Error generating response: {str(e)}]"

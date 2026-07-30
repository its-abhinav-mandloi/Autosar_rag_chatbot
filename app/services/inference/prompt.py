"""
Prompt Construction Module.

Builds the final prompts sent to the LLM for answer generation.
This is the "Prompt Construction" agent in the HLD.

Key design decisions:
    - System prompt establishes AUTOSAR domain expertise
    - Context is injected with source markers for citation extraction
    - Anti-hallucination guardrails are embedded in the prompt
    - Prompts are loaded from versioned template files in prompts/

Continuation Note:
    This module is complete. Prompt templates are in prompts/ directory.
    To update prompts, edit the template files and bump version suffix.
"""

from pathlib import Path
from typing import Optional

from app.monitoring.logging_config import get_logger

logger = get_logger("prompt")

# ─── Default Prompts (fallback if template files not found) ──────────────

DEFAULT_SYSTEM_PROMPT = """You are an expert AUTOSAR (AUTomotive Open System ARchitecture) technical assistant.
Your role is to answer questions about AUTOSAR specifications accurately and precisely.

CRITICAL RULES:
1. ONLY answer based on the provided context passages. Do NOT use prior knowledge.
2. If the context does not contain enough information to answer, say "I cannot find sufficient information in the provided documents to answer this question."
3. Always cite your sources using the format [Source N] where N corresponds to the source number in the context.
4. Be precise and technical — AUTOSAR engineers rely on exact parameter names, API signatures, and requirement IDs.
5. If you reference a specific AUTOSAR requirement, include the [SWS_*] ID.
6. Structure your answer clearly with bullet points or numbered lists when appropriate.
7. Do NOT hallucinate or fabricate information not present in the context."""

DEFAULT_QUERY_PROMPT = """Based on the following AUTOSAR documentation excerpts, answer the user's question.

CONTEXT:
{context}

---

USER QUESTION: {question}

Provide a detailed, accurate answer with citations to the source documents. Use [Source N] notation to reference the context passages above."""


# ─── Template Loading ────────────────────────────────────────────────────


def _load_template(filename: str, default: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    # Try multiple paths (relative to project root)
    search_paths = [
        Path("prompts") / filename,
        Path(__file__).parent.parent.parent.parent / "prompts" / filename,
    ]

    for path in search_paths:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    logger.info("template_loaded", file=str(path))
                    return content
            except IOError as e:
                logger.warning("template_load_error", file=str(path), error=str(e))

    logger.info("using_default_template", filename=filename)
    return default


# ─── Public Functions ────────────────────────────────────────────────────


def get_system_prompt() -> str:
    """Get the system prompt for AUTOSAR document Q&A."""
    return _load_template("system_prompt_v1.txt", DEFAULT_SYSTEM_PROMPT)


def build_query_prompt(question: str, context: str) -> str:
    """
    Build the complete query prompt with injected context.

    Args:
        question: The user's natural language question
        context: Assembled context string with source markers

    Returns:
        Complete prompt ready for LLM generation
    """
    template = _load_template("query_prompt_v1.txt", DEFAULT_QUERY_PROMPT)

    prompt = template.format(
        question=question,
        context=context,
    )

    logger.info(
        "prompt_built",
        question_length=len(question),
        context_length=len(context),
        prompt_length=len(prompt),
    )

    return prompt

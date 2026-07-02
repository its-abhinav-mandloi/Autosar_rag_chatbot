"""
Hallucination / Grounding Evaluation.

Evaluates whether an answer's factual claims are supported by the retrieved
RAG context. This is intentionally separate from the confidence score:
confidence estimates retrieval/citation quality, while this module measures
claim-level grounding against source excerpts.
"""

import json
import re
from dataclasses import asdict, dataclass
from typing import List

from app.monitoring.logging_config import get_logger
from app.services.inference.llm import generate_completion

logger = get_logger("hallucination")

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNSUPPORTED = "unsupported"
NOT_FACTUAL = "not_factual"
VALID_STATUSES = {SUPPORTED, CONTRADICTED, UNSUPPORTED, NOT_FACTUAL}


@dataclass
class ClaimEvaluation:
    """One answer claim and its grounding status."""

    claim: str
    status: str
    source_indices: List[int]
    rationale: str


@dataclass
class HallucinationReport:
    """Aggregate hallucination metrics plus claim-level evidence."""

    factual_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    not_factual_claims: int
    hallucination_rate: float
    faithfulness: float
    verdict: str
    claims: List[ClaimEvaluation]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["claims"] = [asdict(claim) for claim in self.claims]
        return payload


EVALUATION_PROMPT_TEMPLATE = """You are a strict factual grounding evaluator for an AUTOSAR RAG system.

Your task:
1. Split the ANSWER into atomic factual claims.
2. Compare each claim ONLY against the supplied CONTEXT.
3. Label every claim with exactly one status:
   - supported: the context directly supports the claim.
   - contradicted: the context conflicts with the claim.
   - unsupported: the context does not contain enough evidence for the claim.
   - not_factual: opinion, formatting, caveat, or non-checkable text.

Rules:
- Do not use prior knowledge.
- Treat missing evidence as unsupported, not supported.
- Source indices must refer to the [Source N] markers in CONTEXT.
- Return only valid JSON. Do not include markdown.

JSON schema:
{{
  "claims": [
    {{
      "claim": "single atomic claim",
      "status": "supported|contradicted|unsupported|not_factual",
      "source_indices": [1],
      "rationale": "brief reason"
    }}
  ]
}}

CONTEXT:
{context}

ANSWER:
{answer}
"""


async def evaluate_answer_grounding(
    *,
    answer: str,
    context: str,
    max_tokens: int = 1536,
) -> HallucinationReport:
    """
    Evaluate answer hallucination against retrieved context.

    Returns claim-level labels and aggregate metrics:
        hallucination_rate =
            (contradicted_claims + unsupported_claims) / factual_claims
        faithfulness =
            supported_claims / factual_claims
    """
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        context=context,
        answer=answer,
    )

    response = await generate_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    claims = _parse_claim_evaluations(response)
    report = build_hallucination_report(claims)

    logger.info(
        "hallucination_evaluated",
        factual_claims=report.factual_claims,
        supported_claims=report.supported_claims,
        contradicted_claims=report.contradicted_claims,
        unsupported_claims=report.unsupported_claims,
        hallucination_rate=report.hallucination_rate,
    )

    return report


def build_hallucination_report(claims: List[ClaimEvaluation]) -> HallucinationReport:
    """Compute aggregate metrics from claim-level labels."""
    supported = sum(1 for claim in claims if claim.status == SUPPORTED)
    contradicted = sum(1 for claim in claims if claim.status == CONTRADICTED)
    unsupported = sum(1 for claim in claims if claim.status == UNSUPPORTED)
    not_factual = sum(1 for claim in claims if claim.status == NOT_FACTUAL)
    factual = supported + contradicted + unsupported

    hallucination_rate = (
        (contradicted + unsupported) / factual
        if factual
        else 0.0
    )
    faithfulness = supported / factual if factual else 0.0

    if factual == 0:
        verdict = "no_factual_claims"
    elif hallucination_rate == 0:
        verdict = "fully_grounded"
    elif hallucination_rate <= 0.25:
        verdict = "mostly_grounded"
    elif hallucination_rate <= 0.5:
        verdict = "partially_grounded"
    else:
        verdict = "high_hallucination_risk"

    return HallucinationReport(
        factual_claims=factual,
        supported_claims=supported,
        contradicted_claims=contradicted,
        unsupported_claims=unsupported,
        not_factual_claims=not_factual,
        hallucination_rate=round(hallucination_rate, 4),
        faithfulness=round(faithfulness, 4),
        verdict=verdict,
        claims=claims,
    )


def _parse_claim_evaluations(raw_response: str) -> List[ClaimEvaluation]:
    """Parse and normalize the evaluator's JSON response."""
    payload = _load_json_object(raw_response)
    raw_claims = payload.get("claims", [])

    if not isinstance(raw_claims, list):
        raise ValueError("Evaluator response field 'claims' must be a list")

    claims: List[ClaimEvaluation] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue

        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            continue

        status = str(item.get("status", UNSUPPORTED)).strip().lower()
        if status not in VALID_STATUSES:
            status = UNSUPPORTED

        source_indices = _normalize_source_indices(item.get("source_indices", []))
        rationale = str(item.get("rationale", "")).strip()

        claims.append(
            ClaimEvaluation(
                claim=claim_text,
                status=status,
                source_indices=source_indices,
                rationale=rationale,
            )
        )

    return claims


def _load_json_object(raw_response: str) -> dict:
    """Load JSON from plain text or a fenced code block."""
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Evaluator response must be a JSON object")
    return payload


def _normalize_source_indices(value) -> List[int]:
    """Return positive integer source indices from arbitrary evaluator output."""
    if not isinstance(value, list):
        return []

    indices = []
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index > 0 and index not in indices:
            indices.append(index)
    return indices

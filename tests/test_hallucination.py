"""
Unit tests for hallucination / grounding evaluation.
"""

from app.services.inference.hallucination import (
    CONTRADICTED,
    NOT_FACTUAL,
    SUPPORTED,
    UNSUPPORTED,
    ClaimEvaluation,
    _parse_claim_evaluations,
    build_hallucination_report,
)


class TestHallucinationMetrics:
    """Test aggregate hallucination metric calculation."""

    def test_build_report_counts_factual_hallucinations(self):
        claims = [
            ClaimEvaluation("A is defined in the spec.", SUPPORTED, [1], "Found in source 1."),
            ClaimEvaluation("B has default value 10.", UNSUPPORTED, [], "No source states this."),
            ClaimEvaluation("C is mandatory.", CONTRADICTED, [2], "Source 2 says optional."),
            ClaimEvaluation("This is helpful.", NOT_FACTUAL, [], "Opinion."),
        ]

        report = build_hallucination_report(claims)

        assert report.factual_claims == 3
        assert report.supported_claims == 1
        assert report.unsupported_claims == 1
        assert report.contradicted_claims == 1
        assert report.not_factual_claims == 1
        assert report.hallucination_rate == 0.6667
        assert report.faithfulness == 0.3333
        assert report.verdict == "high_hallucination_risk"

    def test_no_factual_claims(self):
        report = build_hallucination_report([
            ClaimEvaluation("I cannot answer from the context.", NOT_FACTUAL, [], "Caveat.")
        ])

        assert report.factual_claims == 0
        assert report.hallucination_rate == 0.0
        assert report.faithfulness == 0.0
        assert report.verdict == "no_factual_claims"


class TestEvaluatorParsing:
    """Test robust parsing of evaluator JSON."""

    def test_parse_plain_json(self):
        raw = """
        {
          "claims": [
            {
              "claim": "Can_Init initializes the CAN driver.",
              "status": "supported",
              "source_indices": [1, "2", 0, "bad"],
              "rationale": "The context says so."
            }
          ]
        }
        """

        claims = _parse_claim_evaluations(raw)

        assert len(claims) == 1
        assert claims[0].claim == "Can_Init initializes the CAN driver."
        assert claims[0].status == SUPPORTED
        assert claims[0].source_indices == [1, 2]

    def test_parse_fenced_json_and_normalize_unknown_status(self):
        raw = """```json
        {
          "claims": [
            {
              "claim": "Made-up claim.",
              "status": "maybe",
              "source_indices": [],
              "rationale": "Unknown label."
            }
          ]
        }
        ```"""

        claims = _parse_claim_evaluations(raw)

        assert len(claims) == 1
        assert claims[0].status == UNSUPPORTED

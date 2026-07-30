"""
Feedback & Analytics Module.

Implements the Feedback & Learning Loop (Step 5 in the HLD):
    - User feedback collection (thumbs up/down)
    - Query-answer-context triples storage
    - Usage analytics and accuracy tracking
    - Retrieval optimization signals

This maps to the measurable goals from M2:
    - Track retrieval precision via feedback
    - Monitor hallucination rate
    - Identify queries needing re-indexing

Continuation Note:
    This module is complete. Feedback is stored in a JSON file at
    data/feedback/feedback_log.json. Analytics are served via GET /feedback/analytics.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.monitoring.logging_config import get_logger
from app.monitoring.metrics import metrics

logger = get_logger("analytics")


class FeedbackStore:
    """JSON-file-backed feedback storage."""

    def __init__(self):
        settings = get_settings()
        self._store_dir = settings.metadata_store_absolute_path / "feedback"
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._store_dir / "feedback_log.json"
        self._data: List[dict] = self._load()

    def _load(self) -> List[dict]:
        if self._file_path.exists():
            try:
                with open(self._file_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self) -> None:
        with open(self._file_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def record_feedback(
        self,
        query_id: str,
        question: str,
        answer: str,
        rating: str,
        comment: Optional[str] = None,
        citations: Optional[List[dict]] = None,
        confidence: float = 0.0,
    ) -> dict:
        """Record user feedback on a query-answer pair."""
        entry = {
            "feedback_id": str(uuid.uuid4())[:8],
            "query_id": query_id,
            "question": question,
            "answer": answer[:500],  # Truncate for storage
            "rating": rating,
            "comment": comment,
            "citations_count": len(citations) if citations else 0,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._data.append(entry)
        self._save()

        # Update metrics
        is_positive = rating in ("thumbs_up", "positive", "good", "5", "4")
        metrics.record_feedback(is_positive, confidence=confidence)

        logger.info(
            "feedback_recorded",
            query_id=query_id,
            rating=rating,
            feedback_id=entry["feedback_id"],
        )

        return entry

    def get_analytics(self) -> dict:
        """Compute analytics from all stored feedback."""
        if not self._data:
            return {
                "total_feedback": 0,
                "positive_rate": None,
                "avg_confidence": None,
                "common_issues": [],
                "recent_feedback": [],
            }

        total = len(self._data)
        positive = sum(
            1 for f in self._data if f.get("rating") in ("thumbs_up", "positive", "good", "5", "4")
        )
        negative = sum(
            1 for f in self._data if f.get("rating") in ("thumbs_down", "negative", "bad", "1", "2")
        )

        confidences = [f.get("confidence", 0) for f in self._data if f.get("confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Find low-confidence queries (potential retrieval issues)
        low_confidence = [f for f in self._data if f.get("confidence", 1.0) < 0.5]

        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total, 4) if total > 0 else None,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
            "low_confidence_queries": len(low_confidence),
            "recent_feedback": self._data[-10:],  # Last 10 entries
        }

    def get_all(self) -> List[dict]:
        return self._data


# Module-level singleton
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store

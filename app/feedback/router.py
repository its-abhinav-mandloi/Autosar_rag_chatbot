"""
Feedback Service API Router.

Provides endpoints for the Feedback & Learning Loop (Step 5 in HLD):
    - POST /feedback: Submit feedback on a query-answer pair
    - GET /feedback/analytics: View aggregated analytics

Continuation Note:
    This module is complete.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.feedback.analytics import get_feedback_store
from app.monitoring.logging_config import get_logger

logger = get_logger("feedback_router")
router = APIRouter(prefix="/feedback", tags=["Feedback & Learning Loop"])


class FeedbackRequest(BaseModel):
    query_id: str = Field(default="unknown", description="ID of the query (from query response)")
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer that was generated")
    rating: str = Field(..., description="Rating: 'thumbs_up' or 'thumbs_down'")
    comment: Optional[str] = Field(default=None, description="Optional comment")
    confidence: float = Field(default=0.0, description="Confidence score from the query response")


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message: str


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a query-answer pair.

    Used to track answer quality and identify retrieval issues.
    Feeds into the continuous improvement loop.
    """
    store = get_feedback_store()
    entry = store.record_feedback(
        query_id=request.query_id,
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        comment=request.comment,
        confidence=request.confidence,
    )

    return FeedbackResponse(
        status="recorded",
        feedback_id=entry["feedback_id"],
        message=f"Feedback recorded. Thank you! (Rating: {request.rating})",
    )


@router.get("/analytics")
async def get_analytics():
    """
    View aggregated feedback analytics.

    Returns:
        - Total feedback count
        - Positive/negative rates
        - Average confidence
        - Low-confidence queries (potential retrieval issues)
        - Recent feedback entries
    """
    store = get_feedback_store()
    return store.get_analytics()

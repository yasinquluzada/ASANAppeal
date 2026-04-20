from __future__ import annotations

from app.models.domain import (
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
)
from app.providers.base import AIProvider


class ExplanationService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def explain(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        human_review: HumanReviewTask,
        verification: VerificationDecision | None = None,
    ) -> ExplanationNote:
        return self.provider.explain_case(
            structured_issue, routing, priority, human_review, verification
        )

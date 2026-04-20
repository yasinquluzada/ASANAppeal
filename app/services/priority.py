from __future__ import annotations

from app.models.api import SubmissionInput
from app.models.domain import PriorityDecision, RoutingDecision, StructuredIssue
from app.providers.base import AIProvider


class PriorityService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def assess(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
    ) -> PriorityDecision:
        return self.provider.assess_priority(submission, structured_issue, routing)

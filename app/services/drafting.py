from __future__ import annotations

from app.models.api import SubmissionInput
from app.models.domain import DraftAppeal, PriorityDecision, RoutingDecision, StructuredIssue
from app.providers.base import AIProvider


class DraftingService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def build_draft(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
    ) -> DraftAppeal:
        return self.provider.draft_appeal(submission, structured_issue, routing, priority)

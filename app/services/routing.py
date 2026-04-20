from __future__ import annotations

from app.models.api import SubmissionInput
from app.models.domain import RoutingDecision, StructuredIssue
from app.providers.base import AIProvider


class RoutingService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def route(self, submission: SubmissionInput, structured_issue: StructuredIssue) -> RoutingDecision:
        return self.provider.route_issue(submission, structured_issue)

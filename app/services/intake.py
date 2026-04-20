from __future__ import annotations

from app.models.api import SubmissionInput
from app.models.domain import StructuredIssue
from app.providers.base import AIProvider


class IntakeService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def analyze(self, submission: SubmissionInput) -> StructuredIssue:
        return self.provider.analyze_submission(submission)

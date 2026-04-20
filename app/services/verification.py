from __future__ import annotations

from app.models.api import InstitutionResponseInput, SubmissionInput
from app.models.domain import StructuredIssue, VerificationDecision
from app.providers.base import AIProvider


class VerificationService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def verify(
        self,
        original_submission: SubmissionInput,
        structured_issue: StructuredIssue,
        institution_response: InstitutionResponseInput,
    ) -> VerificationDecision:
        return self.provider.verify_resolution(
            original_submission, structured_issue, institution_response
        )

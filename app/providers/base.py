from __future__ import annotations

from typing import Protocol

from app.models.api import InstitutionResponseInput, SubmissionInput
from app.models.domain import (
    DecisionProvenance,
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
)


class AIProvider(Protocol):
    def analyze_submission(self, submission: SubmissionInput) -> StructuredIssue:
        ...

    def route_issue(
        self, submission: SubmissionInput, structured_issue: StructuredIssue
    ) -> RoutingDecision:
        ...

    def assess_priority(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
    ) -> PriorityDecision:
        ...

    def draft_appeal(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
    ) -> DraftAppeal:
        ...

    def verify_resolution(
        self,
        original_submission: SubmissionInput,
        structured_issue: StructuredIssue,
        institution_response: InstitutionResponseInput,
    ) -> VerificationDecision:
        ...

    def explain_case(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        human_review: HumanReviewTask,
        verification: VerificationDecision | None = None,
    ) -> ExplanationNote:
        ...

    def clear_stage_provenance(self) -> None:
        ...

    def get_stage_provenance(self) -> dict[str, DecisionProvenance]:
        ...

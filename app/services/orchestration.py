from __future__ import annotations

from uuid import uuid4

from app.lifecycle import build_transition_entry, default_final_disposition
from app.models.api import ProcessCaseRequest
from app.models.domain import (
    AccountRole,
    CasePrivacyState,
    CaseOperationalState,
    CaseRecord,
    CaseStatus,
    CaseTransition,
)
from app.provenance import build_case_model_context
from app.repository import CaseRepository
from app.services.drafting import DraftingService
from app.services.explanation import ExplanationService
from app.services.intake import IntakeService
from app.services.priority import PriorityService
from app.services.review import ReviewService
from app.services.routing import RoutingService
from app.services.verification import VerificationService


class CaseOrchestrator:
    def __init__(
        self,
        repository: CaseRepository,
        intake_service: IntakeService,
        routing_service: RoutingService,
        priority_service: PriorityService,
        drafting_service: DraftingService,
        verification_service: VerificationService,
        explanation_service: ExplanationService,
        review_service: ReviewService,
        *,
        provider_name: str = "unknown",
        model_name: str = "unknown",
    ) -> None:
        self.repository = repository
        self.intake_service = intake_service
        self.routing_service = routing_service
        self.priority_service = priority_service
        self.drafting_service = drafting_service
        self.verification_service = verification_service
        self.explanation_service = explanation_service
        self.review_service = review_service
        self.provider_name = provider_name
        self.model_name = model_name

    def process_case(
        self,
        request: ProcessCaseRequest,
        *,
        submitted_by_user_id: str | None = None,
        submitted_by_role: AccountRole | None = None,
        privacy_state: CasePrivacyState | None = None,
        stored_request_payload: dict[str, object] | None = None,
        case_redactor=None,
    ) -> CaseRecord:
        self.intake_service.provider.clear_stage_provenance()
        structured_issue = self.intake_service.analyze(request.submission)
        routing = self.routing_service.route(request.submission, structured_issue)
        priority = self.priority_service.assess(request.submission, structured_issue, routing)
        draft = self.drafting_service.build_draft(
            request.submission, structured_issue, routing, priority
        )
        verification = None
        if request.institution_response:
            verification = self.verification_service.verify(
                request.submission, structured_issue, request.institution_response
            )
        human_review = self.review_service.evaluate(
            structured_issue, routing, priority, verification
        )
        explanation = self.explanation_service.explain(
            structured_issue, routing, priority, human_review, verification
        )

        case_id = uuid4().hex[:12]
        submission_excerpt = (request.submission.citizen_text or structured_issue.summary)[:160]
        bootstrap_transition = (
            CaseTransition.submit_for_review
            if human_review.needed
            else CaseTransition.mark_dispatch_ready
        )
        bootstrap_entry = build_transition_entry(
            from_status=CaseStatus.drafted,
            transition=bootstrap_transition,
            actor_id="system",
            note="Initial lifecycle routing from AI orchestration.",
        )
        status = bootstrap_entry.to_status
        stage_provenance = self.intake_service.provider.get_stage_provenance()
        stage_provenance.update(self.review_service.get_stage_provenance())
        case = CaseRecord(
            case_id=case_id,
            status=status,
            submission_excerpt=submission_excerpt,
            structured_issue=structured_issue,
            routing=routing,
            priority=priority,
            draft=draft,
            verification=verification,
            explanation=explanation,
            human_review=human_review,
            privacy=privacy_state or CasePrivacyState(),
            model_context=build_case_model_context(
                provider=self.provider_name,
                model_name=self.model_name,
                model_version=self.model_name,
                stage_provenance=stage_provenance,
            ),
            operations=CaseOperationalState(
                submitted_by_user_id=submitted_by_user_id,
                submitted_by_role=submitted_by_role,
                final_disposition=default_final_disposition(status),
                status_updated_at=bootstrap_entry.acted_at,
                disposition_updated_at=bootstrap_entry.acted_at,
                transition_history=[bootstrap_entry],
            ),
        )
        if callable(case_redactor):
            case = case_redactor(case)
        return self.repository.save_case(
            case,
            stored_request_payload
            or request.model_dump(mode="json", exclude_unset=True, exclude_none=True),
        )

from __future__ import annotations

from datetime import datetime, timezone

from app.lifecycle import build_transition_entry, can_transition, default_final_disposition
from app.models.api import (
    CaseWorkflowActionRequest,
    InstitutionResponseInput,
    ProcessCaseRequest,
)
from app.models.domain import (
    CaseOperationalState,
    CaseRecord,
    CaseStatus,
    CaseTransition,
    CaseWorkflowAction,
    CaseWorkflowActionEntry,
    ExplanationNote,
    HumanReviewTask,
    VerificationLabel,
)
from app.provenance import merge_case_model_context
from app.repository import CaseRepository
from app.services.explanation import ExplanationService
from app.services.review import ReviewService
from app.services.verification import VerificationService


class CaseWorkflowService:
    def __init__(
        self,
        repository: CaseRepository,
        verification_service: VerificationService,
        review_service: ReviewService,
        explanation_service: ExplanationService,
    ) -> None:
        self.repository = repository
        self.verification_service = verification_service
        self.review_service = review_service
        self.explanation_service = explanation_service

    def apply_action(
        self,
        case_id: str,
        payload: CaseWorkflowActionRequest,
    ) -> CaseRecord | None:
        case = self.repository.get_case(case_id)
        if case is None:
            return None
        request_payload = self.repository.get_case_request_payload(case_id)
        updated_case = case

        if payload.action == CaseWorkflowAction.comment:
            if not payload.note:
                raise RuntimeError("comment action requires a note.")
            updated_case = self._append_action(
                updated_case,
                action=payload.action,
                actor_id=payload.actor_id,
                note=payload.note,
                assignee_id=payload.assignee_id,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.claim:
            if not payload.actor_id:
                raise RuntimeError("claim action requires actor_id.")
            updated_case = self._assign_case(
                updated_case,
                assignee_id=payload.actor_id,
                actor_id=payload.actor_id,
                note=payload.note or f"Case claimed by {payload.actor_id}.",
                action=payload.action,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.assign:
            assignee_id = payload.assignee_id or payload.actor_id
            if not assignee_id:
                raise RuntimeError("assign action requires assignee_id or actor_id.")
            updated_case = self._assign_case(
                updated_case,
                assignee_id=assignee_id,
                actor_id=payload.actor_id,
                note=payload.note or f"Case assigned to {assignee_id}.",
                action=payload.action,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.approve:
            updated_case = self._ensure_transition(
                updated_case,
                transition=CaseTransition.mark_dispatch_ready,
                actor_id=payload.actor_id,
                note=payload.note or "Case approved for dispatch.",
            )
            updated_case = self._clear_manual_review(
                updated_case,
                final_disposition="approved",
                explanation_summary="Case approved during manual review.",
                next_action="Dispatch the case to an assigned owner.",
                disposition_reason=payload.note,
            )
            updated_case = self._append_action(
                updated_case,
                action=payload.action,
                actor_id=payload.actor_id,
                note=payload.note,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.reject:
            updated_case = self._ensure_transition(
                updated_case,
                transition=CaseTransition.reject,
                actor_id=payload.actor_id,
                note=payload.note or "Case rejected during review.",
            )
            updated_case = self._clear_manual_review(
                updated_case,
                final_disposition="rejected",
                explanation_summary="Case rejected during manual review.",
                next_action="Case can be closed or reopened with new information.",
                disposition_reason=payload.note,
            )
            updated_case = self._append_action(
                updated_case,
                action=payload.action,
                actor_id=payload.actor_id,
                note=payload.note,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.dispatch:
            assignee_id = payload.assignee_id or updated_case.operations.reviewer_id or payload.actor_id
            if updated_case.status in {
                CaseStatus.drafted,
                CaseStatus.needs_review,
                CaseStatus.reopened,
            }:
                updated_case = self._ensure_transition(
                    updated_case,
                    transition=CaseTransition.mark_dispatch_ready,
                    actor_id=payload.actor_id,
                    note="Case approved for dispatch as part of dispatch action.",
                )
                updated_case = self._clear_manual_review(
                    updated_case,
                    final_disposition="approved",
                    explanation_summary="Case approved during dispatch.",
                    next_action="Dispatch the case to an assigned owner.",
                    disposition_reason=payload.note,
                )
            if updated_case.status == CaseStatus.ready_for_dispatch:
                if not assignee_id:
                    raise RuntimeError(
                        "dispatch action requires assignee_id, actor_id, or an existing reviewer."
                    )
                updated_case = self._assign_case(
                    updated_case,
                    assignee_id=assignee_id,
                    actor_id=payload.actor_id,
                    note=payload.note or f"Case dispatched to {assignee_id}.",
                    action=payload.action,
                    final_disposition="dispatched",
                    next_action="Assigned owner can begin processing the case.",
                )
            elif updated_case.status not in {CaseStatus.assigned, CaseStatus.in_progress}:
                raise RuntimeError(
                    f"dispatch action is not valid from status {updated_case.status.value}."
                )
            else:
                updated_case = self._update_case(
                    updated_case,
                    operations=updated_case.operations.model_copy(
                        update={
                            "reviewer_id": assignee_id or updated_case.operations.reviewer_id,
                            "final_disposition": "dispatched",
                            "final_disposition_reason": payload.note
                            or updated_case.operations.final_disposition_reason,
                            "disposition_updated_at": datetime.now(timezone.utc),
                        }
                    ),
                    explanation=ExplanationNote(
                        summary="Case dispatch recorded.",
                        next_action="Assigned owner can continue the case workflow.",
                        detailed_rationale=["Workflow dispatch action was recorded."],
                        risk_flags=[],
                    ),
                )
                updated_case = self._append_action(
                    updated_case,
                    action=payload.action,
                    actor_id=payload.actor_id,
                    note=payload.note,
                    assignee_id=assignee_id,
                )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.close:
            updated_case = self._ensure_transition(
                updated_case,
                transition=CaseTransition.close,
                actor_id=payload.actor_id,
                note=payload.note or "Case closed.",
            )
            updated_case = self._clear_manual_review(
                updated_case,
                final_disposition="closed",
                explanation_summary="Case closed.",
                next_action="No further action is required unless the case is reopened.",
                disposition_reason=payload.note,
            )
            updated_case = self._append_action(
                updated_case,
                action=payload.action,
                actor_id=payload.actor_id,
                note=payload.note,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.reopen:
            updated_case = self._ensure_transition(
                updated_case,
                transition=CaseTransition.reopen,
                actor_id=payload.actor_id,
                note=payload.note or "Case reopened for additional work.",
            )
            reopened_review = self.review_service.evaluate(
                updated_case.structured_issue,
                updated_case.routing,
                updated_case.priority,
                updated_case.verification,
                manual_reasons=["reopened_case"],
            )
            updated_case = self._update_case(
                updated_case,
                human_review=reopened_review,
                model_context=merge_case_model_context(
                    updated_case.model_context,
                    stage_updates=self.review_service.get_stage_provenance(),
                ),
                operations=updated_case.operations.model_copy(
                    update={
                        "final_disposition": "reopened",
                        "final_disposition_reason": payload.note,
                        "disposition_updated_at": datetime.now(timezone.utc),
                    }
                ),
                explanation=ExplanationNote(
                    summary="Case reopened.",
                    next_action="Review the case and decide whether to dispatch or reject it.",
                    detailed_rationale=["Workflow reopen action moved the case back into review."],
                    risk_flags=["reopened_case"],
                ),
            )
            updated_case = self._append_action(
                updated_case,
                action=payload.action,
                actor_id=payload.actor_id,
                note=payload.note,
            )
            return self.repository.save_case(updated_case, request_payload)

        if payload.action == CaseWorkflowAction.verify:
            if payload.institution_response is None:
                raise RuntimeError("verify action requires institution_response.")
            if request_payload is None or "submission" not in request_payload:
                raise RuntimeError("original submission is unavailable for verification.")

            original_request = ProcessCaseRequest.model_validate(request_payload)
            self.verification_service.provider.clear_stage_provenance()
            verification = self.verification_service.verify(
                original_request.submission,
                updated_case.structured_issue,
                payload.institution_response,
            )
            human_review = self.review_service.evaluate(
                updated_case.structured_issue,
                updated_case.routing,
                updated_case.priority,
                verification,
            )
            explanation = self.explanation_service.explain(
                updated_case.structured_issue,
                updated_case.routing,
                updated_case.priority,
                human_review,
                verification,
            )
            provider = self.verification_service.provider
            updated_case = self._update_case(
                updated_case,
                verification=verification,
                human_review=human_review,
                explanation=explanation,
                model_context=merge_case_model_context(
                    updated_case.model_context,
                    provider=type(provider).__name__,
                    model_name=getattr(provider, "_model", updated_case.model_context.model_name),
                    model_version=getattr(provider, "_model", updated_case.model_context.model_version),
                    stage_updates={
                        **provider.get_stage_provenance(),
                        **self.review_service.get_stage_provenance(),
                    },
                ),
                operations=updated_case.operations.model_copy(
                    update={
                        "reviewer_id": payload.actor_id or updated_case.operations.reviewer_id,
                        "final_disposition": (
                            "verified_resolved"
                            if verification.issue_resolved == VerificationLabel.yes
                            else "verified"
                        ),
                        "final_disposition_reason": verification.summary,
                        "disposition_updated_at": datetime.now(timezone.utc),
                    }
                ),
            )

            if (
                verification.issue_resolved == VerificationLabel.yes
                and updated_case.status in {CaseStatus.assigned, CaseStatus.in_progress}
                and can_transition(updated_case.status, CaseTransition.resolve)
            ):
                updated_case = self._ensure_transition(
                    updated_case,
                    transition=CaseTransition.resolve,
                    actor_id=payload.actor_id,
                    note="Case resolved by verification evidence.",
                )
            elif (
                verification.issue_resolved == VerificationLabel.no
                and updated_case.status in {CaseStatus.resolved, CaseStatus.closed}
                and can_transition(updated_case.status, CaseTransition.reopen)
            ):
                updated_case = self._ensure_transition(
                    updated_case,
                    transition=CaseTransition.reopen,
                    actor_id=payload.actor_id,
                    note="Verification evidence shows the case is not resolved.",
                )

            updated_case = self._append_action(
                updated_case,
                action=payload.action,
                actor_id=payload.actor_id,
                note=payload.note or verification.summary,
            )
            return self.repository.save_case(updated_case, request_payload)

        raise RuntimeError(f"Unsupported workflow action: {payload.action.value}.")

    def verify_case(
        self,
        case_id: str,
        *,
        institution_response: InstitutionResponseInput,
        actor_id: str | None = None,
        note: str | None = None,
    ) -> CaseRecord | None:
        return self.apply_action(
            case_id,
            CaseWorkflowActionRequest(
                action=CaseWorkflowAction.verify,
                actor_id=actor_id,
                note=note,
                institution_response=institution_response,
            ),
        )

    def _assign_case(
        self,
        case: CaseRecord,
        *,
        assignee_id: str,
        actor_id: str | None,
        note: str,
        action: CaseWorkflowAction,
        final_disposition: str = "assigned",
        next_action: str = "Assigned owner can begin processing the case.",
    ) -> CaseRecord:
        updated_case = case
        if updated_case.status in {
            CaseStatus.drafted,
            CaseStatus.needs_review,
            CaseStatus.ready_for_dispatch,
            CaseStatus.reopened,
        }:
            updated_case = self._ensure_transition(
                updated_case,
                transition=CaseTransition.assign,
                actor_id=actor_id,
                note=note,
            )
        elif updated_case.status not in {CaseStatus.assigned, CaseStatus.in_progress}:
            raise RuntimeError(f"assign action is not valid from status {updated_case.status.value}.")

        updated_case = self._clear_manual_review(
            updated_case,
            final_disposition=final_disposition,
            explanation_summary=f"Case assigned to {assignee_id}.",
            next_action=next_action,
            disposition_reason=note,
            reviewer_id=assignee_id,
        )
        updated_case = self._append_action(
            updated_case,
            action=action,
            actor_id=actor_id,
            note=note,
            assignee_id=assignee_id,
        )
        return updated_case

    def _ensure_transition(
        self,
        case: CaseRecord,
        *,
        transition: CaseTransition,
        actor_id: str | None,
        note: str,
    ) -> CaseRecord:
        if not can_transition(case.status, transition):
            raise RuntimeError(
                f"{transition.value} is not valid from status {case.status.value}."
            )
        entry = build_transition_entry(
            from_status=case.status,
            transition=transition,
            actor_id=actor_id,
            note=note,
        )
        operations = case.operations.model_copy(
            update={
                "status_updated_at": entry.acted_at,
                "disposition_updated_at": entry.acted_at,
                "transition_history": [*case.operations.transition_history, entry],
                "final_disposition": default_final_disposition(entry.to_status),
            }
        )
        return case.model_copy(
            update={
                "status": entry.to_status,
                "operations": operations,
            }
        )

    def _clear_manual_review(
        self,
        case: CaseRecord,
        *,
        final_disposition: str,
        explanation_summary: str,
        next_action: str,
        disposition_reason: str | None,
        reviewer_id: str | None = None,
    ) -> CaseRecord:
        return self._update_case(
            case,
            human_review=HumanReviewTask(
                needed=False,
                queue=case.human_review.queue,
                reasons=[],
                confidence=1.0,
            ),
            operations=case.operations.model_copy(
                update={
                    "reviewer_id": reviewer_id if reviewer_id is not None else case.operations.reviewer_id,
                    "final_disposition": final_disposition,
                    "final_disposition_reason": disposition_reason,
                    "disposition_updated_at": datetime.now(timezone.utc),
                }
            ),
            explanation=ExplanationNote(
                summary=explanation_summary,
                next_action=next_action,
                detailed_rationale=["Manual workflow action updated the case state."],
                risk_flags=[],
            ),
        )

    def _append_action(
        self,
        case: CaseRecord,
        *,
        action: CaseWorkflowAction,
        actor_id: str | None,
        note: str | None,
        assignee_id: str | None = None,
    ) -> CaseRecord:
        entry = CaseWorkflowActionEntry(
            action=action,
            resulting_status=case.status,
            actor_id=actor_id,
            note=note,
            assignee_id=assignee_id,
        )
        operations = case.operations.model_copy(
            update={
                "workflow_history": [*case.operations.workflow_history, entry],
            }
        )
        return case.model_copy(update={"operations": operations})

    def _update_case(self, case: CaseRecord, **updates) -> CaseRecord:
        return case.model_copy(update=updates)

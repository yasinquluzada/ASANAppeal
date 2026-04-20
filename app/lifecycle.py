from __future__ import annotations

from datetime import datetime, timezone

from app.models.domain import CaseStatus, CaseTransition, CaseTransitionEntry


CASE_TRANSITION_TARGETS: dict[CaseTransition, CaseStatus] = {
    CaseTransition.submit_for_review: CaseStatus.needs_review,
    CaseTransition.mark_dispatch_ready: CaseStatus.ready_for_dispatch,
    CaseTransition.assign: CaseStatus.assigned,
    CaseTransition.start_progress: CaseStatus.in_progress,
    CaseTransition.resolve: CaseStatus.resolved,
    CaseTransition.reopen: CaseStatus.reopened,
    CaseTransition.reject: CaseStatus.rejected,
    CaseTransition.close: CaseStatus.closed,
}


ALLOWED_TRANSITIONS: dict[CaseStatus, tuple[CaseTransition, ...]] = {
    CaseStatus.drafted: (
        CaseTransition.submit_for_review,
        CaseTransition.mark_dispatch_ready,
        CaseTransition.reject,
    ),
    CaseStatus.needs_review: (
        CaseTransition.mark_dispatch_ready,
        CaseTransition.assign,
        CaseTransition.reject,
    ),
    CaseStatus.ready_for_dispatch: (
        CaseTransition.assign,
        CaseTransition.reject,
    ),
    CaseStatus.assigned: (
        CaseTransition.start_progress,
        CaseTransition.resolve,
        CaseTransition.reject,
    ),
    CaseStatus.in_progress: (
        CaseTransition.resolve,
        CaseTransition.reject,
    ),
    CaseStatus.resolved: (
        CaseTransition.close,
        CaseTransition.reopen,
    ),
    CaseStatus.reopened: (
        CaseTransition.submit_for_review,
        CaseTransition.mark_dispatch_ready,
        CaseTransition.assign,
        CaseTransition.reject,
    ),
    CaseStatus.rejected: (
        CaseTransition.reopen,
        CaseTransition.close,
    ),
    CaseStatus.closed: (CaseTransition.reopen,),
}


def allowed_transitions(status: CaseStatus) -> tuple[CaseTransition, ...]:
    return ALLOWED_TRANSITIONS.get(status, ())


def can_transition(status: CaseStatus, transition: CaseTransition) -> bool:
    return transition in allowed_transitions(status)


def apply_transition(status: CaseStatus, transition: CaseTransition) -> CaseStatus:
    if not can_transition(status, transition):
        allowed = ", ".join(item.value for item in allowed_transitions(status)) or "none"
        raise ValueError(
            f"Invalid lifecycle transition {transition.value} from {status.value}. "
            f"Allowed transitions: {allowed}."
        )
    return CASE_TRANSITION_TARGETS[transition]


def default_final_disposition(status: CaseStatus) -> str:
    return {
        CaseStatus.drafted: "drafted",
        CaseStatus.needs_review: "pending_review",
        CaseStatus.ready_for_dispatch: "dispatch_ready",
        CaseStatus.assigned: "assigned",
        CaseStatus.in_progress: "in_progress",
        CaseStatus.resolved: "resolved",
        CaseStatus.reopened: "reopened",
        CaseStatus.rejected: "rejected",
        CaseStatus.closed: "closed",
    }[status]


def build_transition_entry(
    *,
    from_status: CaseStatus,
    transition: CaseTransition,
    actor_id: str | None = None,
    note: str | None = None,
) -> CaseTransitionEntry:
    to_status = apply_transition(from_status, transition)
    return CaseTransitionEntry(
        transition=transition,
        from_status=from_status,
        to_status=to_status,
        acted_at=datetime.now(timezone.utc),
        actor_id=actor_id,
        note=note,
    )

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import uuid
from typing import Protocol

from app.models.domain import (
    AuditEventDraft,
    AuditEventType,
    AuditLogEvent,
    AuthenticatedUser,
    CaseRecord,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_ai_snapshot(case: CaseRecord) -> dict[str, object]:
    return {
        "structured_issue": case.structured_issue.model_dump(mode="json"),
        "routing": case.routing.model_dump(mode="json"),
        "priority": case.priority.model_dump(mode="json"),
        "draft": case.draft.model_dump(mode="json"),
        "verification": case.verification.model_dump(mode="json") if case.verification else None,
        "explanation": case.explanation.model_dump(mode="json"),
        "human_review": case.human_review.model_dump(mode="json"),
        "model_context": case.model_context.model_dump(mode="json"),
    }


def build_override_snapshot(before_case: CaseRecord, after_case: CaseRecord) -> dict[str, object]:
    before_operations = before_case.operations.model_dump(mode="json")
    after_operations = after_case.operations.model_dump(mode="json")
    changed_fields: list[str] = []
    for field_name in sorted(
        set(before_operations.keys()) | set(after_operations.keys()) | {"status"}
    ):
        before_value = before_case.status.value if field_name == "status" else before_operations.get(field_name)
        after_value = after_case.status.value if field_name == "status" else after_operations.get(field_name)
        if before_value != after_value:
            changed_fields.append(field_name)
    return {
        "before_status": before_case.status.value,
        "after_status": after_case.status.value,
        "before_operations": before_operations,
        "after_operations": after_operations,
        "changed_fields": changed_fields,
    }


def finalize_audit_event(
    draft: AuditEventDraft,
    *,
    previous_event_id: str | None,
    previous_event_hash: str | None,
    occurred_at: datetime | None = None,
) -> AuditLogEvent:
    resolved_occurred_at = occurred_at or _utc_now()
    event_id = uuid.uuid4().hex
    signable_payload = {
        "event_id": event_id,
        "case_id": draft.case_id,
        "occurred_at": resolved_occurred_at.isoformat(),
        "event_type": draft.event_type.value,
        "event_source": draft.event_source,
        "summary": draft.summary,
        "actor_id": draft.actor_id,
        "actor_role": draft.actor_role.value if draft.actor_role is not None else None,
        "actor_username": draft.actor_username,
        "status_before": draft.status_before,
        "status_after": draft.status_after,
        "human_override": draft.human_override,
        "ai_snapshot_hash": draft.ai_snapshot_hash,
        "override_snapshot_hash": draft.override_snapshot_hash,
        "previous_event_id": previous_event_id,
        "previous_event_hash": previous_event_hash,
        "payload": draft.payload,
    }
    event_hash = stable_hash(signable_payload)
    return AuditLogEvent(
        event_id=event_id,
        case_id=draft.case_id,
        occurred_at=resolved_occurred_at,
        event_type=draft.event_type,
        event_source=draft.event_source,
        summary=draft.summary,
        actor_id=draft.actor_id,
        actor_role=draft.actor_role,
        actor_username=draft.actor_username,
        status_before=draft.status_before,
        status_after=draft.status_after,
        human_override=draft.human_override,
        ai_snapshot_hash=draft.ai_snapshot_hash,
        override_snapshot_hash=draft.override_snapshot_hash,
        previous_event_id=previous_event_id,
        previous_event_hash=previous_event_hash,
        event_hash=event_hash,
        payload=draft.payload,
    )


def expected_event_hash(
    event: AuditLogEvent,
    *,
    previous_event_id: str | None,
    previous_event_hash: str | None,
) -> str:
    return stable_hash(
        {
            "event_id": event.event_id,
            "case_id": event.case_id,
            "occurred_at": event.occurred_at.isoformat(),
            "event_type": event.event_type.value,
            "event_source": event.event_source,
            "summary": event.summary,
            "actor_id": event.actor_id,
            "actor_role": event.actor_role.value if event.actor_role is not None else None,
            "actor_username": event.actor_username,
            "status_before": event.status_before,
            "status_after": event.status_after,
            "human_override": event.human_override,
            "ai_snapshot_hash": event.ai_snapshot_hash,
            "override_snapshot_hash": event.override_snapshot_hash,
            "previous_event_id": previous_event_id,
            "previous_event_hash": previous_event_hash,
            "payload": event.payload,
        }
    )


class AuditRepository(Protocol):
    def append_audit_event(self, event: AuditEventDraft) -> AuditLogEvent:
        ...


class AuditService:
    def __init__(self, repository: "AuditRepository") -> None:
        self.repository = repository

    def _build_actor_fields(self, actor: AuthenticatedUser) -> dict[str, object]:
        return {
            "actor_id": actor.user_id,
            "actor_role": actor.role,
            "actor_username": actor.username,
        }

    def record_case_created(
        self,
        *,
        case: CaseRecord,
        actor: AuthenticatedUser,
        original_request: dict[str, object] | None,
    ) -> AuditLogEvent:
        ai_snapshot = build_ai_snapshot(case)
        return self.repository.append_audit_event(
            AuditEventDraft(
                case_id=case.case_id,
                event_type=AuditEventType.case_created,
                event_source="api.cases.process",
                summary="Case created from submitted evidence.",
                status_before=None,
                status_after=case.status.value,
                human_override=False,
                ai_snapshot_hash=stable_hash(ai_snapshot),
                payload={
                    "original_request": original_request or {},
                    "ai_outputs": ai_snapshot,
                },
                **self._build_actor_fields(actor),
            )
        )

    def record_operations_update(
        self,
        *,
        before_case: CaseRecord,
        after_case: CaseRecord,
        actor: AuthenticatedUser,
        reviewer_id: str | None,
        final_disposition: str | None,
        final_disposition_reason: str | None,
    ) -> AuditLogEvent:
        ai_snapshot = build_ai_snapshot(after_case)
        override_snapshot = build_override_snapshot(before_case, after_case)
        override_snapshot["requested_update"] = {
            "reviewer_id": reviewer_id,
            "final_disposition": final_disposition,
            "final_disposition_reason": final_disposition_reason,
        }
        return self.repository.append_audit_event(
            AuditEventDraft(
                case_id=after_case.case_id,
                event_type=AuditEventType.operations_updated,
                event_source="api.cases.operations",
                summary="Case operational metadata was updated.",
                status_before=before_case.status.value,
                status_after=after_case.status.value,
                human_override=True,
                ai_snapshot_hash=stable_hash(ai_snapshot),
                override_snapshot_hash=stable_hash(override_snapshot),
                payload={
                    "ai_outputs": ai_snapshot,
                    "human_override": override_snapshot,
                },
                **self._build_actor_fields(actor),
            )
        )

    def record_transition(
        self,
        *,
        before_case: CaseRecord,
        after_case: CaseRecord,
        actor: AuthenticatedUser,
        transition: str,
        note: str | None,
    ) -> AuditLogEvent:
        ai_snapshot = build_ai_snapshot(after_case)
        override_snapshot = build_override_snapshot(before_case, after_case)
        override_snapshot["transition"] = transition
        override_snapshot["note"] = note
        override_snapshot["transition_entry"] = (
            after_case.operations.transition_history[-1].model_dump(mode="json")
            if after_case.operations.transition_history
            else None
        )
        return self.repository.append_audit_event(
            AuditEventDraft(
                case_id=after_case.case_id,
                event_type=AuditEventType.lifecycle_transition,
                event_source="api.cases.transition",
                summary=f"Lifecycle transition {transition} was applied.",
                status_before=before_case.status.value,
                status_after=after_case.status.value,
                human_override=True,
                ai_snapshot_hash=stable_hash(ai_snapshot),
                override_snapshot_hash=stable_hash(override_snapshot),
                payload={
                    "ai_outputs": ai_snapshot,
                    "human_override": override_snapshot,
                },
                **self._build_actor_fields(actor),
            )
        )

    def record_workflow_action(
        self,
        *,
        before_case: CaseRecord,
        after_case: CaseRecord,
        actor: AuthenticatedUser,
        action: str,
        note: str | None,
        assignee_id: str | None = None,
        institution_response: dict[str, object] | None = None,
        source: str = "api.cases.workflow_actions",
    ) -> AuditLogEvent:
        ai_snapshot = build_ai_snapshot(after_case)
        override_snapshot = build_override_snapshot(before_case, after_case)
        override_snapshot["action"] = action
        override_snapshot["note"] = note
        override_snapshot["assignee_id"] = assignee_id
        override_snapshot["workflow_entry"] = (
            after_case.operations.workflow_history[-1].model_dump(mode="json")
            if after_case.operations.workflow_history
            else None
        )
        if institution_response is not None:
            override_snapshot["institution_response"] = institution_response
        event_type = (
            AuditEventType.verification_action
            if action == "verify"
            else AuditEventType.workflow_action
        )
        summary = (
            "Verification workflow action was applied."
            if action == "verify"
            else f"Workflow action {action} was applied."
        )
        return self.repository.append_audit_event(
            AuditEventDraft(
                case_id=after_case.case_id,
                event_type=event_type,
                event_source=source,
                summary=summary,
                status_before=before_case.status.value,
                status_after=after_case.status.value,
                human_override=True,
                ai_snapshot_hash=stable_hash(ai_snapshot),
                override_snapshot_hash=stable_hash(override_snapshot),
                payload={
                    "ai_outputs": ai_snapshot,
                    "human_override": override_snapshot,
                },
                **self._build_actor_fields(actor),
            )
        )

    def record_privacy_export(
        self,
        *,
        case: CaseRecord,
        actor: AuthenticatedUser,
        export_path: str,
        evidence_count: int,
        audit_event_count: int,
    ) -> AuditLogEvent:
        ai_snapshot = build_ai_snapshot(case)
        return self.repository.append_audit_event(
            AuditEventDraft(
                case_id=case.case_id,
                event_type=AuditEventType.privacy_export,
                event_source="api.cases.privacy_export",
                summary="Privacy export package was generated.",
                status_before=case.status.value,
                status_after=case.status.value,
                human_override=True,
                ai_snapshot_hash=stable_hash(ai_snapshot),
                payload={
                    "ai_outputs": ai_snapshot,
                    "privacy_export": {
                        "export_path": export_path,
                        "evidence_count": evidence_count,
                        "audit_event_count": audit_event_count,
                    },
                },
                **self._build_actor_fields(actor),
            )
        )

    def record_privacy_delete(
        self,
        *,
        before_case: CaseRecord,
        after_case: CaseRecord,
        actor: AuthenticatedUser,
        deleted_evidence_ids: list[str],
        note: str | None,
        source: str = "api.cases.privacy_delete",
    ) -> AuditLogEvent:
        ai_snapshot = build_ai_snapshot(after_case)
        override_snapshot = build_override_snapshot(before_case, after_case)
        override_snapshot["deleted_evidence_ids"] = deleted_evidence_ids
        override_snapshot["note"] = note
        return self.repository.append_audit_event(
            AuditEventDraft(
                case_id=after_case.case_id,
                event_type=AuditEventType.privacy_delete,
                event_source=source,
                summary="Privacy deletion workflow removed personal data and evidence.",
                status_before=before_case.status.value,
                status_after=after_case.status.value,
                human_override=True,
                ai_snapshot_hash=stable_hash(ai_snapshot),
                override_snapshot_hash=stable_hash(override_snapshot),
                payload={
                    "ai_outputs": ai_snapshot,
                    "human_override": override_snapshot,
                },
                **self._build_actor_fields(actor),
            )
        )

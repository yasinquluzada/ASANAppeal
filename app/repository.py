from __future__ import annotations

import json
import logging
import math
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from app.audit import expected_event_hash, finalize_audit_event
from app.lifecycle import (
    allowed_transitions,
    build_transition_entry,
    can_transition,
    default_final_disposition,
)
from app.models.domain import (
    AuditChainVerification,
    AuditEventDraft,
    AuditLogEvent,
    CaseOperationalState,
    CaseRecord,
    CaseStatus,
    CaseTransition,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_SCHEMA_VERSION = 7
T = TypeVar("T")
PRIORITY_SLA_HOURS = {
    "critical": 4,
    "high": 24,
    "medium": 72,
    "low": 168,
}


class CaseRepository(Protocol):
    def save_case(self, case: CaseRecord, request_payload: dict | None = None) -> CaseRecord:
        ...

    def get_case(self, case_id: str) -> CaseRecord | None:
        ...

    def get_case_request_payload(self, case_id: str) -> dict | None:
        ...

    def append_audit_event(self, event: AuditEventDraft) -> AuditLogEvent:
        ...

    def list_case_audit_events(self, case_id: str, limit: int = 200) -> list[AuditLogEvent]:
        ...

    def verify_audit_chain(self) -> AuditChainVerification:
        ...

    def list_review_queue(self) -> list[CaseRecord]:
        ...

    def query_review_queue(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        review_queue: str | None = None,
        priority_level: str | None = None,
        status: str | None = None,
        assignee_id: str | None = None,
        assignment_state: str | None = None,
        sort_by: str = "sla",
    ) -> dict[str, object]:
        ...

    def list_cases(
        self,
        *,
        category: str | None = None,
        institution: str | None = None,
        priority_level: str | None = None,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        ...

    def summarize_cases(self) -> dict[str, object]:
        ...

    def update_case_operational_fields(
        self,
        case_id: str,
        *,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        final_disposition_reason: str | None = None,
    ) -> CaseRecord | None:
        ...

    def transition_case(
        self,
        case_id: str,
        *,
        transition: CaseTransition,
        actor_id: str | None = None,
        note: str | None = None,
        reviewer_id: str | None = None,
    ) -> CaseRecord | None:
        ...

    def list_backups(self, limit: int = 20) -> list[dict[str, object]]:
        ...

    def backup_database(
        self,
        destination_path: str | None = None,
        *,
        label: str | None = None,
        backup_type: str = "manual_backup",
    ) -> dict[str, object]:
        ...

    def restore_database(
        self,
        source_path: str,
        *,
        create_pre_restore_backup: bool = True,
    ) -> dict[str, object]:
        ...

    def diagnostics(self) -> dict[str, object]:
        ...


class InMemoryCaseRepository(CaseRepository):
    def __init__(self) -> None:
        self._cases: dict[str, CaseRecord] = {}
        self._review_queue: list[str] = []
        self._request_payloads: dict[str, dict] = {}
        self._audit_events: list[AuditLogEvent] = []

    def save_case(self, case: CaseRecord, request_payload: dict | None = None) -> CaseRecord:
        self._cases[case.case_id] = case
        if request_payload is not None:
            self._request_payloads[case.case_id] = request_payload
        if case.human_review.needed:
            if case.case_id not in self._review_queue:
                self._review_queue.append(case.case_id)
        elif case.case_id in self._review_queue:
            self._review_queue.remove(case.case_id)
        return case

    def get_case(self, case_id: str) -> CaseRecord | None:
        return self._cases.get(case_id)

    def get_case_request_payload(self, case_id: str) -> dict | None:
        return self._request_payloads.get(case_id)

    def append_audit_event(self, event: AuditEventDraft) -> AuditLogEvent:
        previous = self._audit_events[-1] if self._audit_events else None
        finalized = finalize_audit_event(
            event,
            previous_event_id=previous.event_id if previous is not None else None,
            previous_event_hash=previous.event_hash if previous is not None else None,
        )
        self._audit_events.append(finalized)
        return finalized

    def list_case_audit_events(self, case_id: str, limit: int = 200) -> list[AuditLogEvent]:
        case_events = [event for event in self._audit_events if event.case_id == case_id]
        return case_events[-max(1, limit) :]

    def verify_audit_chain(self) -> AuditChainVerification:
        previous_event_id: str | None = None
        previous_event_hash: str | None = None
        checked_events = 0
        for event in self._audit_events:
            expected_hash = expected_event_hash(
                event,
                previous_event_id=previous_event_id,
                previous_event_hash=previous_event_hash,
            )
            if event.previous_event_id != previous_event_id or event.previous_event_hash != previous_event_hash:
                return AuditChainVerification(
                    verified=False,
                    checked_events=checked_events,
                    latest_event_id=previous_event_id,
                    latest_event_hash=previous_event_hash,
                    failure_event_id=event.event_id,
                    failure_reason="previous audit link does not match the stored chain.",
                )
            if event.event_hash != expected_hash:
                return AuditChainVerification(
                    verified=False,
                    checked_events=checked_events,
                    latest_event_id=previous_event_id,
                    latest_event_hash=previous_event_hash,
                    failure_event_id=event.event_id,
                    failure_reason="stored audit event hash does not match the canonical payload.",
                )
            previous_event_id = event.event_id
            previous_event_hash = event.event_hash
            checked_events += 1
        latest_event = self._audit_events[-1] if self._audit_events else None
        return AuditChainVerification(
            verified=True,
            checked_events=checked_events,
            latest_event_id=latest_event.event_id if latest_event is not None else None,
            latest_event_hash=latest_event.event_hash if latest_event is not None else None,
        )

    def list_review_queue(self) -> list[CaseRecord]:
        queue = self.query_review_queue(page=1, page_size=max(1, len(self._review_queue) or 1))
        return [
            self._cases[str(item["case_id"])]
            for item in queue["items"]
            if str(item["case_id"]) in self._cases
        ]

    def query_review_queue(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        review_queue: str | None = None,
        priority_level: str | None = None,
        status: str | None = None,
        assignee_id: str | None = None,
        assignment_state: str | None = None,
        sort_by: str = "sla",
    ) -> dict[str, object]:
        items = [
            _review_queue_item_from_projection(_case_projection(case))
            for case_id, case in self._cases.items()
            if case_id in self._review_queue
        ]
        filtered: list[dict[str, object]] = []
        for item in items:
            if review_queue and item["review_queue"] != review_queue:
                continue
            if priority_level and item["priority_level"] != priority_level:
                continue
            if status and item["status"] != status:
                continue
            if assignee_id and item["assignee_id"] != assignee_id:
                continue
            if assignment_state and item["assignment_state"] != assignment_state:
                continue
            filtered.append(item)
        ordered = _sort_review_queue_items(filtered, sort_by)
        safe_page = max(1, page)
        safe_page_size = max(1, page_size)
        total_items = len(ordered)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        total_pages = max(1, math.ceil(total_items / safe_page_size)) if total_items else 1
        return {
            "items": ordered[start:end],
            "meta": {
                "page": safe_page,
                "page_size": safe_page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "sort_by": sort_by,
                "assignment_state": assignment_state,
                "review_queue": review_queue,
                "priority_level": priority_level,
                "assignee_id": assignee_id,
                "status": status,
            },
        }

    def list_cases(
        self,
        *,
        category: str | None = None,
        institution: str | None = None,
        priority_level: str | None = None,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        items = [_case_projection(case) for case in self._cases.values()]
        filtered: list[dict[str, object]] = []
        for item in items:
            if category and item["category"] != category:
                continue
            if institution and item["institution"] != institution:
                continue
            if priority_level and item["priority_level"] != priority_level:
                continue
            if reviewer_id and item["operations"]["reviewer_id"] != reviewer_id:
                continue
            if final_disposition and item["operations"]["final_disposition"] != final_disposition:
                continue
            if status and item["status"] != status:
                continue
            filtered.append(item)
        filtered.sort(key=lambda item: item["created_at"], reverse=True)
        return filtered[: max(1, limit)]

    def summarize_cases(self) -> dict[str, object]:
        items = self.list_cases(limit=max(1, len(self._cases) or 1))

        def _count_by(key: str, *, nested: bool = False) -> dict[str, int]:
            counts: dict[str, int] = {}
            for item in items:
                value = item[key] if not nested else item["operations"][key]
                bucket = str(value or "(none)")
                counts[bucket] = counts.get(bucket, 0) + 1
            return counts

        return {
            "total_cases": len(items),
            "review_needed_cases": sum(1 for item in items if item["review_needed"]),
            "counts_by_status": _count_by("status"),
            "counts_by_category": _count_by("category"),
            "counts_by_institution": _count_by("institution"),
            "counts_by_priority": _count_by("priority_level"),
            "counts_by_final_disposition": _count_by("final_disposition", nested=True),
            "counts_by_reviewer": _count_by("reviewer_id", nested=True),
        }

    def update_case_operational_fields(
        self,
        case_id: str,
        *,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        final_disposition_reason: str | None = None,
    ) -> CaseRecord | None:
        case = self._cases.get(case_id)
        if case is None:
            return None
        operations = case.operations.model_copy(
            update={
                "reviewer_id": reviewer_id if reviewer_id is not None else case.operations.reviewer_id,
                "final_disposition": (
                    final_disposition
                    if final_disposition is not None
                    else case.operations.final_disposition or default_final_disposition(case.status)
                ),
                "final_disposition_reason": (
                    final_disposition_reason
                    if final_disposition_reason is not None
                    else case.operations.final_disposition_reason
                ),
                "disposition_updated_at": (
                    datetime.now(timezone.utc)
                    if final_disposition is not None or final_disposition_reason is not None
                    else case.operations.disposition_updated_at
                ),
            }
        )
        updated_case = case.model_copy(
            update={
                "operations": operations,
            }
        )
        self._cases[case_id] = updated_case
        if updated_case.human_review.needed and case_id not in self._review_queue:
            self._review_queue.append(case_id)
        return updated_case

    def transition_case(
        self,
        case_id: str,
        *,
        transition: CaseTransition,
        actor_id: str | None = None,
        note: str | None = None,
        reviewer_id: str | None = None,
    ) -> CaseRecord | None:
        case = self._cases.get(case_id)
        if case is None:
            return None
        entry = build_transition_entry(
            from_status=case.status,
            transition=transition,
            actor_id=actor_id,
            note=note,
        )
        operations = case.operations.model_copy(
            update={
                "reviewer_id": reviewer_id if reviewer_id is not None else case.operations.reviewer_id,
                "final_disposition": default_final_disposition(entry.to_status),
                "status_updated_at": entry.acted_at,
                "disposition_updated_at": entry.acted_at,
                "transition_history": [*case.operations.transition_history, entry],
            }
        )
        updated_case = case.model_copy(update={"status": entry.to_status, "operations": operations})
        self._cases[case_id] = updated_case
        return updated_case

    def list_backups(self, limit: int = 20) -> list[dict[str, object]]:
        return []

    def backup_database(
        self,
        destination_path: str | None = None,
        *,
        label: str | None = None,
        backup_type: str = "manual_backup",
    ) -> dict[str, object]:
        raise RuntimeError("Backup workflow requires the SQLite repository backend.")

    def restore_database(
        self,
        source_path: str,
        *,
        create_pre_restore_backup: bool = True,
    ) -> dict[str, object]:
        raise RuntimeError("Restore workflow requires the SQLite repository backend.")

    def diagnostics(self) -> dict[str, object]:
        audit_verification = self.verify_audit_chain()
        return {
            "sqlite_mode": "disabled",
            "sqlite_db_path": None,
            "sqlite_schema_version": None,
            "sqlite_backup_count": 0,
            "sqlite_restore_count": 0,
            "audit_event_count": len(self._audit_events),
            "audit_chain_verified": audit_verification.verified,
            "audit_chain_latest_hash": audit_verification.latest_event_hash,
        }


@dataclass(frozen=True)
class SQLiteLocationChoice:
    requested_path: str
    resolved_path: str
    used_fallback: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class SQLiteBackupDirChoice:
    requested_dir: str | None
    resolved_dir: str
    used_fallback: bool
    fallback_reason: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _priority_rank(priority_level: str | None) -> int:
    return {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }.get((priority_level or "").lower(), 0)


def _assignment_state(assignee_id: str | None) -> str:
    return "assigned" if assignee_id else "unassigned"


def _sla_deadline_iso(created_at: str, priority_level: str | None) -> str:
    created = _parse_iso_datetime(created_at)
    hours = PRIORITY_SLA_HOURS.get((priority_level or "").lower(), PRIORITY_SLA_HOURS["medium"])
    return (created + timedelta(hours=hours)).isoformat()


def _sla_breached(created_at: str, priority_level: str | None, *, now: datetime | None = None) -> bool:
    deadline = _parse_iso_datetime(_sla_deadline_iso(created_at, priority_level))
    reference = now or datetime.now(timezone.utc)
    return reference >= deadline


def _review_queue_item_from_projection(item: dict[str, object]) -> dict[str, object]:
    assignee_id = item["operations"].get("reviewer_id") if isinstance(item["operations"], dict) else None
    created_at = str(item["created_at"])
    priority_level = str(item["priority_level"])
    return {
        "case_id": item["case_id"],
        "created_at": created_at,
        "status": item["status"],
        "submission_excerpt": item["submission_excerpt"],
        "category": item["category"],
        "issue_type": item["issue_type"],
        "institution": item["institution"],
        "department": item["department"],
        "priority_level": priority_level,
        "priority_score": int(item["priority_score"]),
        "review_queue": item["review_queue"],
        "review_confidence": float(item["review_confidence"]),
        "assignee_id": assignee_id,
        "assignment_state": _assignment_state(assignee_id),
        "sla_deadline_at": _sla_deadline_iso(created_at, priority_level),
        "sla_breached": _sla_breached(created_at, priority_level),
        "model_context": item["model_context"],
        "operations": item["operations"],
        "allowed_transitions": item["allowed_transitions"],
    }


def _sort_review_queue_items(items: list[dict[str, object]], sort_by: str) -> list[dict[str, object]]:
    if sort_by == "created_at":
        return sorted(items, key=lambda item: item["created_at"], reverse=True)
    if sort_by == "priority":
        return sorted(
            items,
            key=lambda item: (
                -_priority_rank(str(item["priority_level"])),
                -int(item["priority_score"]),
                item["created_at"],
            ),
        )
    if sort_by == "assignee":
        return sorted(
            items,
            key=lambda item: (
                0 if item["assignee_id"] is None else 1,
                str(item["assignee_id"] or ""),
                item["created_at"],
            ),
        )
    return sorted(
        items,
        key=lambda item: (
            0 if item["sla_breached"] else 1,
            item["sla_deadline_at"],
            -_priority_rank(str(item["priority_level"])),
            -int(item["priority_score"]),
            item["created_at"],
        ),
    )


def _case_projection(case: CaseRecord) -> dict[str, object]:
    operations = case.operations or CaseOperationalState()
    final_disposition = operations.final_disposition or default_final_disposition(case.status)
    return {
        "case_id": case.case_id,
        "created_at": case.created_at.isoformat(),
        "status": case.status.value,
        "submission_excerpt": case.submission_excerpt,
        "category": case.structured_issue.category,
        "issue_type": case.structured_issue.issue_type,
        "institution": case.routing.institution,
        "department": case.routing.department,
        "priority_level": getattr(case.priority.level, "value", str(case.priority.level)),
        "priority_score": case.priority.score,
        "review_needed": bool(case.human_review.needed),
        "review_queue": case.human_review.queue,
        "review_confidence": case.human_review.confidence,
        "issue_confidence": case.structured_issue.confidence,
        "routing_confidence": case.routing.confidence,
        "priority_confidence": case.priority.confidence,
        "verification_same_place": (
            getattr(case.verification.same_place, "value", str(case.verification.same_place))
            if case.verification
            else None
        ),
        "verification_issue_resolved": (
            getattr(case.verification.issue_resolved, "value", str(case.verification.issue_resolved))
            if case.verification
            else None
        ),
        "provider_name": case.model_context.provider,
        "model_name": case.model_context.model_name,
        "model_version": case.model_context.model_version,
        "provenance_schema_version": case.model_context.provenance_schema_version,
        "prompt_bundle_version": case.model_context.prompt_bundle_version,
        "classifier_bundle_version": case.model_context.classifier_bundle_version,
        "threshold_set_version": case.model_context.threshold_set_version,
        "reviewer_id": operations.reviewer_id,
        "final_disposition": final_disposition,
        "final_disposition_reason": operations.final_disposition_reason,
        "status_updated_at": (
            operations.status_updated_at.isoformat()
            if operations.status_updated_at is not None
            else None
        ),
        "disposition_updated_at": (
            operations.disposition_updated_at.isoformat()
            if operations.disposition_updated_at is not None
            else None
        ),
        "model_context": case.model_context.model_dump(mode="json"),
        "operations": {
            **operations.model_dump(mode="json"),
            "final_disposition": final_disposition,
        },
        "allowed_transitions": [
            transition.value for transition in allowed_transitions(case.status)
        ],
    }


def _normalize_db_path(db_path: str) -> Path | str:
    if db_path == ":memory:":
        return db_path
    path = Path(db_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _normalize_dir_path(dir_path: str) -> Path:
    path = Path(dir_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _candidate_locations(requested_path: str) -> list[SQLiteLocationChoice]:
    normalized_requested = requested_path.strip() or "./asanappeal.db"
    explicit = SQLiteLocationChoice(
        requested_path=normalized_requested,
        resolved_path=str(_normalize_db_path(normalized_requested)),
        used_fallback=False,
        fallback_reason=None,
    )
    candidates: list[SQLiteLocationChoice] = [explicit]

    fallback_specs = [
        ("./.data/asanappeal.db", "project_data_dir"),
        (str(Path.home() / ".local" / "share" / "asanappeal" / "asanappeal.db"), "user_data_dir"),
        (str(Path(tempfile.gettempdir()) / "asanappeal" / "asanappeal.db"), "temp_dir"),
        (":memory:", "in_memory_fallback"),
    ]

    seen = {explicit.resolved_path}
    for candidate_path, reason in fallback_specs:
        resolved = str(_normalize_db_path(candidate_path))
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(
            SQLiteLocationChoice(
                requested_path=normalized_requested,
                resolved_path=resolved,
                used_fallback=True,
                fallback_reason=reason,
            )
        )
    return candidates


def _default_backup_dir_for_db(db_path: Path | str) -> Path:
    if isinstance(db_path, Path):
        return db_path.parent / "backups"
    return Path(tempfile.gettempdir()) / "asanappeal" / "backups"


def _candidate_backup_directories(
    requested_dir: str | None,
    db_path: Path | str,
) -> list[SQLiteBackupDirChoice]:
    candidates: list[SQLiteBackupDirChoice] = []
    seen: set[str] = set()

    def _add(path: Path, used_fallback: bool, reason: str | None) -> None:
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(
            SQLiteBackupDirChoice(
                requested_dir=requested_dir,
                resolved_dir=resolved,
                used_fallback=used_fallback,
                fallback_reason=reason,
            )
        )

    if requested_dir:
        _add(_normalize_dir_path(requested_dir), False, None)

    _add(_default_backup_dir_for_db(db_path), requested_dir is not None, "database_sibling_backup_dir")
    _add(
        Path(tempfile.gettempdir()) / "asanappeal" / "backups",
        True,
        "temp_backup_dir",
    )
    return candidates


def _choose_backup_directory(requested_dir: str | None, db_path: Path | str) -> SQLiteBackupDirChoice:
    failures: list[str] = []
    for candidate in _candidate_backup_directories(requested_dir, db_path):
        directory = Path(candidate.resolved_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as exc:
            failures.append(f"{candidate.resolved_dir}: {exc}")
    failure_summary = "; ".join(failures) if failures else "no backup directories tried"
    raise OSError(f"SQLite backup directory startup failed for all candidate paths: {failure_summary}")


class SQLiteCaseRepository(CaseRepository):
    def __init__(
        self,
        db_path: str,
        *,
        requested_path: str | None = None,
        timeout_seconds: float = 5.0,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
        max_write_retries: int = 5,
        write_retry_backoff_ms: int = 50,
        backup_dir: str = "./.data/backups",
        backup_pages_per_step: int = 128,
        used_fallback: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        normalized_path = _normalize_db_path(db_path)
        self.db_path = normalized_path
        self.requested_path = requested_path or db_path
        self.timeout_seconds = timeout_seconds
        self.busy_timeout_ms = busy_timeout_ms
        self.journal_mode = journal_mode.upper()
        self.synchronous = synchronous.upper()
        self.max_write_retries = max_write_retries
        self.write_retry_backoff_ms = write_retry_backoff_ms
        self.backup_pages_per_step = backup_pages_per_step
        backup_dir_choice = _choose_backup_directory(backup_dir, normalized_path)
        self.backup_dir = Path(backup_dir_choice.resolved_dir)
        self.requested_backup_dir = backup_dir
        self.backup_dir_used_fallback = backup_dir_choice.used_fallback
        self.backup_dir_fallback_reason = backup_dir_choice.fallback_reason
        if self.backup_dir_used_fallback:
            logger.warning(
                "SQLite backup directory used fallback path %s because %s was not usable.",
                self.backup_dir,
                backup_dir or "<default>",
            )
        self.used_fallback = used_fallback
        self.fallback_reason = fallback_reason
        self.effective_journal_mode = "unknown"
        self.effective_synchronous = "unknown"
        self.schema_version = 0
        self.migration_count = 0
        self.backup_count = 0
        self.restore_count = 0
        self.last_backup_at: str | None = None
        self.audit_event_count = 0
        self.audit_latest_event_hash: str | None = None
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def build_hardened(
        cls,
        db_path: str,
        *,
        timeout_seconds: float = 5.0,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
        max_write_retries: int = 5,
        write_retry_backoff_ms: int = 50,
        backup_dir: str = "./.data/backups",
        backup_pages_per_step: int = 128,
    ) -> SQLiteCaseRepository:
        failures: list[str] = []
        for candidate in _candidate_locations(db_path):
            try:
                repository = cls(
                    candidate.resolved_path,
                    requested_path=candidate.requested_path,
                    timeout_seconds=timeout_seconds,
                    busy_timeout_ms=busy_timeout_ms,
                    journal_mode=journal_mode,
                    synchronous=synchronous,
                    max_write_retries=max_write_retries,
                    write_retry_backoff_ms=write_retry_backoff_ms,
                    backup_dir=backup_dir,
                    backup_pages_per_step=backup_pages_per_step,
                    used_fallback=candidate.used_fallback,
                    fallback_reason=candidate.fallback_reason,
                )
                if candidate.used_fallback:
                    logger.warning(
                        "SQLite startup used fallback path %s because the requested path %s was not usable.",
                        candidate.resolved_path,
                        candidate.requested_path,
                    )
                return repository
            except (OSError, sqlite3.Error) as exc:
                failures.append(f"{candidate.resolved_path}: {exc}")

        failure_summary = "; ".join(failures) if failures else "no candidates tried"
        raise RuntimeError(f"SQLite startup failed for all candidate paths: {failure_summary}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout_seconds, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        if self.db_path != ":memory:":
            journal_row = connection.execute(f"PRAGMA journal_mode={self.journal_mode}").fetchone()
            if journal_row and journal_row[0]:
                self.effective_journal_mode = str(journal_row[0]).lower()
        else:
            journal_row = connection.execute("PRAGMA journal_mode").fetchone()
            if journal_row and journal_row[0]:
                self.effective_journal_mode = str(journal_row[0]).lower()
        connection.execute(f"PRAGMA synchronous={self.synchronous}")
        sync_row = connection.execute("PRAGMA synchronous").fetchone()
        if sync_row and sync_row[0] is not None:
            self.effective_synchronous = str(sync_row[0])
        return connection

    def _initialize(self) -> None:
        self._execute_write("initialize_schema", self._initialize_schema)
        with self._connect() as connection:
            self._refresh_storage_metrics(connection)

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        self._ensure_migration_table(connection)
        self._migration_1_create_cases(connection)
        self._migration_2_create_storage_backups(connection)
        self._migration_3_create_storage_restores(connection)
        current_version = self._read_user_version(connection)

        migrations: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
            (1, "create_cases_table", self._migration_1_create_cases),
            (2, "create_storage_backups_table", self._migration_2_create_storage_backups),
            (3, "create_storage_restores_table", self._migration_3_create_storage_restores),
            (4, "case_operational_projection", self._migration_4_case_operational_projection),
            (5, "case_lifecycle_state_machine", self._migration_5_case_lifecycle_state_machine),
            (6, "case_model_provenance", self._migration_6_case_model_provenance),
            (7, "case_audit_log", self._migration_7_case_audit_log),
        ]

        for version, name, migration in migrations:
            if version <= current_version:
                continue
            migration(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO schema_migrations (version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (version, name, _utc_now_iso()),
            )
            connection.execute(f"PRAGMA user_version={version}")
            current_version = version

        self._reconcile_schema(connection)
        self.schema_version = current_version
        self.migration_count = current_version

    def _reconcile_schema(self, connection: sqlite3.Connection) -> None:
        # Older database files may already advertise a later user_version while
        # still missing newer projection columns or refreshed lifecycle triggers.
        self._migration_4_case_operational_projection(connection)
        self._migration_5_case_lifecycle_state_machine(connection)
        self._migration_6_case_model_provenance(connection)
        self._migration_7_case_audit_log(connection)

    def _ensure_migration_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )

    def _read_user_version(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("PRAGMA user_version").fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])

    def _migration_1_create_cases(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                review_needed INTEGER NOT NULL,
                case_json TEXT NOT NULL,
                request_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cases_review_needed_created
            ON cases(review_needed, created_at DESC);
            """
        )

    def _migration_2_create_storage_backups(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS storage_backups (
                backup_id TEXT PRIMARY KEY,
                backup_path TEXT NOT NULL,
                label TEXT,
                backup_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_path TEXT NOT NULL,
                integrity_check TEXT NOT NULL,
                size_bytes INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_storage_backups_created
            ON storage_backups(created_at DESC);
            """
        )

    def _migration_3_create_storage_restores(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS storage_restores (
                restore_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                restored_at TEXT NOT NULL,
                source_integrity TEXT NOT NULL,
                pre_restore_backup_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_storage_restores_restored
            ON storage_restores(restored_at DESC);
            """
        )

    def _migration_4_case_operational_projection(self, connection: sqlite3.Connection) -> None:
        columns_to_add = [
            ("submission_excerpt", "TEXT"),
            ("category", "TEXT"),
            ("issue_type", "TEXT"),
            ("institution", "TEXT"),
            ("department", "TEXT"),
            ("priority_level", "TEXT"),
            ("priority_score", "INTEGER"),
            ("issue_confidence", "REAL"),
            ("routing_confidence", "REAL"),
            ("priority_confidence", "REAL"),
            ("review_queue", "TEXT"),
            ("review_confidence", "REAL"),
            ("verification_same_place", "TEXT"),
            ("verification_issue_resolved", "TEXT"),
            ("provider_name", "TEXT"),
            ("model_name", "TEXT"),
            ("model_version", "TEXT"),
            ("reviewer_id", "TEXT"),
            ("final_disposition", "TEXT"),
            ("final_disposition_reason", "TEXT"),
            ("status_updated_at", "TEXT"),
            ("disposition_updated_at", "TEXT"),
        ]
        existing_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(cases)").fetchall()
        }
        for name, sql_type in columns_to_add:
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE cases ADD COLUMN {name} {sql_type}")

        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_cases_category_status_created
            ON cases(category, status, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_cases_institution_priority_created
            ON cases(institution, priority_level, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_cases_final_disposition_created
            ON cases(final_disposition, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_cases_provider_model
            ON cases(provider_name, model_version);

            CREATE INDEX IF NOT EXISTS idx_cases_reviewer_created
            ON cases(reviewer_id, created_at DESC)
            WHERE reviewer_id IS NOT NULL;
            """
        )
        self._backfill_case_projection(connection)

    def _migration_5_case_lifecycle_state_machine(self, connection: sqlite3.Connection) -> None:
        valid_statuses = ", ".join(f"'{status.value}'" for status in CaseStatus)
        transition_conditions: list[str] = []
        for status in CaseStatus:
            transitions = allowed_transitions(status)
            if not transitions:
                continue
            target_statuses = ", ".join(
                f"'{build_transition_entry(from_status=status, transition=transition).to_status.value}'"
                for transition in transitions
            )
            transition_conditions.append(
                f"(OLD.status = '{status.value}' AND NEW.status IN ({target_statuses}))"
            )
        lifecycle_expression = " OR ".join(transition_conditions) or "0"
        connection.executescript(
            f"""
            DROP TRIGGER IF EXISTS trg_cases_status_insert_guard;
            CREATE TRIGGER trg_cases_status_insert_guard
            BEFORE INSERT ON cases
            FOR EACH ROW
            WHEN NEW.status NOT IN ({valid_statuses})
            BEGIN
                SELECT RAISE(ABORT, 'invalid case lifecycle status');
            END;

            DROP TRIGGER IF EXISTS trg_cases_status_update_guard;
            CREATE TRIGGER trg_cases_status_update_guard
            BEFORE UPDATE OF status ON cases
            FOR EACH ROW
            WHEN NEW.status NOT IN ({valid_statuses})
            BEGIN
                SELECT RAISE(ABORT, 'invalid case lifecycle status');
            END;

            DROP TRIGGER IF EXISTS trg_cases_status_transition_guard;
            CREATE TRIGGER trg_cases_status_transition_guard
            BEFORE UPDATE OF status ON cases
            FOR EACH ROW
            WHEN OLD.status <> NEW.status
             AND NOT ({lifecycle_expression})
            BEGIN
                SELECT RAISE(ABORT, 'invalid case lifecycle transition');
            END;
            """
        )

    def _migration_6_case_model_provenance(self, connection: sqlite3.Connection) -> None:
        columns_to_add = [
            ("provenance_schema_version", "TEXT"),
            ("prompt_bundle_version", "TEXT"),
            ("classifier_bundle_version", "TEXT"),
            ("threshold_set_version", "TEXT"),
        ]
        existing_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(cases)").fetchall()
        }
        for name, sql_type in columns_to_add:
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE cases ADD COLUMN {name} {sql_type}")

        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_cases_provenance_provider_prompt
            ON cases(provider_name, prompt_bundle_version, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_cases_provenance_threshold_bundle
            ON cases(threshold_set_version, created_at DESC);
            """
        )
        self._backfill_case_projection(connection)

    def _migration_7_case_audit_log(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS case_audit_log (
                event_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_source TEXT NOT NULL,
                summary TEXT NOT NULL,
                actor_id TEXT,
                actor_role TEXT,
                actor_username TEXT,
                status_before TEXT,
                status_after TEXT,
                human_override INTEGER NOT NULL,
                ai_snapshot_hash TEXT,
                override_snapshot_hash TEXT,
                previous_event_id TEXT,
                previous_event_hash TEXT,
                event_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id)
            );

            CREATE INDEX IF NOT EXISTS idx_case_audit_log_case_time
            ON case_audit_log(case_id, occurred_at ASC);

            CREATE INDEX IF NOT EXISTS idx_case_audit_log_time
            ON case_audit_log(occurred_at ASC);

            DROP TRIGGER IF EXISTS trg_case_audit_log_no_update;
            CREATE TRIGGER trg_case_audit_log_no_update
            BEFORE UPDATE ON case_audit_log
            FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'case audit log is append-only');
            END;

            DROP TRIGGER IF EXISTS trg_case_audit_log_no_delete;
            CREATE TRIGGER trg_case_audit_log_no_delete
            BEFORE DELETE ON case_audit_log
            FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'case audit log is append-only');
            END;
            """
        )

    def _refresh_storage_metrics(self, connection: sqlite3.Connection) -> None:
        self.schema_version = self._read_user_version(connection)
        migration_row = connection.execute("SELECT COUNT(*) AS count FROM schema_migrations").fetchone()
        self.migration_count = int(migration_row["count"]) if migration_row else 0

        backup_row = connection.execute(
            "SELECT COUNT(*) AS count, MAX(created_at) AS last_created FROM storage_backups"
        ).fetchone()
        if backup_row:
            self.backup_count = int(backup_row["count"])
            self.last_backup_at = str(backup_row["last_created"]) if backup_row["last_created"] else None
        else:
            self.backup_count = 0
            self.last_backup_at = None

        restore_row = connection.execute("SELECT COUNT(*) AS count FROM storage_restores").fetchone()
        self.restore_count = int(restore_row["count"]) if restore_row else 0

        audit_row = connection.execute(
            """
            SELECT COUNT(*) AS count, MAX(occurred_at) AS last_occurred
            FROM case_audit_log
            """
        ).fetchone()
        self.audit_event_count = int(audit_row["count"]) if audit_row else 0
        latest_hash_row = connection.execute(
            """
            SELECT event_hash
            FROM case_audit_log
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT 1
            """
        ).fetchone()
        self.audit_latest_event_hash = (
            str(latest_hash_row["event_hash"]) if latest_hash_row is not None else None
        )

    def _projection_columns(self) -> tuple[str, ...]:
        return (
            "submission_excerpt",
            "category",
            "issue_type",
            "institution",
            "department",
            "priority_level",
            "priority_score",
            "issue_confidence",
            "routing_confidence",
            "priority_confidence",
            "review_queue",
            "review_confidence",
            "verification_same_place",
            "verification_issue_resolved",
            "provider_name",
            "model_name",
            "model_version",
            "provenance_schema_version",
            "prompt_bundle_version",
            "classifier_bundle_version",
            "threshold_set_version",
            "reviewer_id",
            "final_disposition",
            "final_disposition_reason",
            "status_updated_at",
            "disposition_updated_at",
        )

    def _projection_values(self, case: CaseRecord) -> tuple[object, ...]:
        projection = _case_projection(case)
        return tuple(projection[column] for column in self._projection_columns())

    def _backfill_case_projection(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT case_id, case_json FROM cases").fetchall()
        if not rows:
            return
        existing_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(cases)").fetchall()
        }
        projection_columns = tuple(
            column for column in self._projection_columns() if column in existing_columns
        )
        assignments = ", ".join(f"{column} = ?" for column in projection_columns)
        for row in rows:
            case = CaseRecord.model_validate_json(row["case_json"])
            connection.execute(
                f"UPDATE cases SET {assignments} WHERE case_id = ?",
                (*(_case_projection(case)[column] for column in projection_columns), row["case_id"]),
            )

    def _case_list_select(self) -> str:
        return """
            SELECT
                case_id,
                created_at,
                status,
                submission_excerpt,
                category,
                issue_type,
                institution,
                department,
                priority_level,
                priority_score,
                review_needed,
                review_queue,
                review_confidence,
                issue_confidence,
                routing_confidence,
                priority_confidence,
                verification_same_place,
                verification_issue_resolved,
                provider_name,
                model_name,
                model_version,
                provenance_schema_version,
                prompt_bundle_version,
                classifier_bundle_version,
                threshold_set_version,
                reviewer_id,
                final_disposition,
                final_disposition_reason,
                status_updated_at,
                disposition_updated_at
            FROM cases
        """

    def _row_to_case_list_item(self, row: sqlite3.Row) -> dict[str, object]:
        return {
            "case_id": row["case_id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "submission_excerpt": row["submission_excerpt"],
            "category": row["category"],
            "issue_type": row["issue_type"],
            "institution": row["institution"],
            "department": row["department"],
            "priority_level": row["priority_level"],
            "priority_score": int(row["priority_score"] or 0),
            "review_needed": bool(row["review_needed"]),
            "review_queue": row["review_queue"],
            "review_confidence": float(row["review_confidence"] or 0.0),
            "issue_confidence": float(row["issue_confidence"] or 0.0),
            "routing_confidence": float(row["routing_confidence"] or 0.0),
            "priority_confidence": float(row["priority_confidence"] or 0.0),
            "verification_same_place": row["verification_same_place"],
            "verification_issue_resolved": row["verification_issue_resolved"],
            "model_context": {
                "provider": row["provider_name"] or "unknown",
                "model_name": row["model_name"] or "unknown",
                "model_version": row["model_version"] or "unknown",
                "provenance_schema_version": row["provenance_schema_version"] or "case-provenance.v1",
                "prompt_bundle_version": row["prompt_bundle_version"] or "not_applicable",
                "classifier_bundle_version": row["classifier_bundle_version"] or "not_applicable",
                "threshold_set_version": row["threshold_set_version"] or "not_applicable",
                "stage_provenance": {},
            },
            "operations": {
                "reviewer_id": row["reviewer_id"],
                "final_disposition": row["final_disposition"] or "open",
                "final_disposition_reason": row["final_disposition_reason"],
                "status_updated_at": row["status_updated_at"],
                "disposition_updated_at": row["disposition_updated_at"],
                "transition_history": [],
            },
            "allowed_transitions": [
                transition.value for transition in allowed_transitions(CaseStatus(row["status"]))
            ],
        }

    def _row_to_audit_event(self, row: sqlite3.Row) -> AuditLogEvent:
        payload = json.loads(str(row["payload_json"])) if row["payload_json"] else {}
        return AuditLogEvent(
            event_id=str(row["event_id"]),
            case_id=str(row["case_id"]),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            event_type=str(row["event_type"]),
            event_source=str(row["event_source"]),
            summary=str(row["summary"]),
            actor_id=str(row["actor_id"]) if row["actor_id"] is not None else None,
            actor_role=str(row["actor_role"]) if row["actor_role"] is not None else None,
            actor_username=str(row["actor_username"]) if row["actor_username"] is not None else None,
            status_before=str(row["status_before"]) if row["status_before"] is not None else None,
            status_after=str(row["status_after"]) if row["status_after"] is not None else None,
            human_override=bool(row["human_override"]),
            ai_snapshot_hash=(
                str(row["ai_snapshot_hash"]) if row["ai_snapshot_hash"] is not None else None
            ),
            override_snapshot_hash=(
                str(row["override_snapshot_hash"])
                if row["override_snapshot_hash"] is not None
                else None
            ),
            previous_event_id=(
                str(row["previous_event_id"]) if row["previous_event_id"] is not None else None
            ),
            previous_event_hash=(
                str(row["previous_event_hash"])
                if row["previous_event_hash"] is not None
                else None
            ),
            event_hash=str(row["event_hash"]),
            payload=payload,
        )

    def _is_retryable_lock_error(self, exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message or "locked" in message

    def _execute_write(self, operation_name: str, callback: Callable[[sqlite3.Connection], T]) -> T:
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(self.max_write_retries + 1):
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                result = callback(connection)
                connection.commit()
                return result
            except sqlite3.OperationalError as exc:
                if connection.in_transaction:
                    connection.rollback()
                if not self._is_retryable_lock_error(exc) or attempt >= self.max_write_retries:
                    raise
                last_error = exc
                sleep_seconds = (self.write_retry_backoff_ms / 1000.0) * (2**attempt)
                logger.warning(
                    "SQLite write operation %s retried after lock contention (attempt %s/%s).",
                    operation_name,
                    attempt + 1,
                    self.max_write_retries + 1,
                )
                time.sleep(sleep_seconds)
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"SQLite write operation {operation_name} failed without an error.")

    def _next_backup_path(self, label: str | None = None, *, directory: Path | None = None) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = (label or "backup").strip().lower().replace(" ", "-")
        slug = "".join(ch for ch in slug if ch.isalnum() or ch in {"-", "_"})
        slug = slug or "backup"
        filename = f"{timestamp}-{slug}-{uuid.uuid4().hex[:10]}.sqlite3"
        return (directory or self.backup_dir) / filename

    def _resolve_backup_destination(self, destination_path: str | None, label: str | None) -> Path:
        if destination_path:
            destination = _normalize_dir_path(str(Path(destination_path).expanduser()))
            if destination.suffix == "":
                destination = destination / self._next_backup_path(label).name
        else:
            destination = self._next_backup_path(label)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _candidate_backup_destinations(
        self,
        destination_path: str | None,
        label: str | None,
    ) -> list[Path]:
        primary = self._resolve_backup_destination(destination_path, label)
        candidates = [primary]
        if destination_path is None:
            temp_dir = Path(tempfile.gettempdir()) / "asanappeal" / "backups"
            temp_dir.mkdir(parents=True, exist_ok=True)
            fallback = self._next_backup_path(label, directory=temp_dir)
            if fallback != primary:
                candidates.append(fallback)
        return candidates

    def _resolve_existing_path(self, source_path: str) -> Path:
        candidate = Path(source_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"SQLite restore source does not exist: {resolved}")
        return resolved

    def _integrity_check(self, path: Path) -> str:
        with sqlite3.connect(str(path)) as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row and row[0] else "unknown"

    def _record_backup(
        self,
        *,
        backup_id: str,
        backup_path: Path,
        label: str | None,
        backup_type: str,
        created_at: str,
        source_path: str,
        integrity_check: str,
        size_bytes: int,
    ) -> dict[str, object]:
        resolved_backup_path = backup_path.resolve()
        payload = {
            "backup_id": backup_id,
            "backup_path": str(resolved_backup_path),
            "label": label,
            "backup_type": backup_type,
            "created_at": created_at,
            "source_path": source_path,
            "integrity_check": integrity_check,
            "size_bytes": size_bytes,
        }

        def _write(connection: sqlite3.Connection) -> dict[str, object]:
            connection.execute(
                """
                INSERT INTO storage_backups (
                    backup_id,
                    backup_path,
                    label,
                    backup_type,
                    created_at,
                    source_path,
                    integrity_check,
                    size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    backup_id,
                    str(resolved_backup_path),
                    label,
                    backup_type,
                    created_at,
                    source_path,
                    integrity_check,
                    size_bytes,
                ),
            )
            self._refresh_storage_metrics(connection)
            return payload

        return self._execute_write("record_backup", _write)

    def _lookup_backup_by_path(self, backup_path: Path) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    backup_id,
                    backup_path,
                    label,
                    backup_type,
                    created_at,
                    source_path,
                    integrity_check,
                    size_bytes
                FROM storage_backups
                WHERE backup_path = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(backup_path),),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def save_case(self, case: CaseRecord, request_payload: dict | None = None) -> CaseRecord:
        case_json = case.model_dump_json()
        request_json = json.dumps(request_payload or {}, ensure_ascii=False)
        projection_columns = ",\n                    ".join(self._projection_columns())
        projection_placeholders = ", ".join("?" for _ in self._projection_columns())
        upsert_assignments = ",\n                    ".join(
            f"{column} = excluded.{column}" for column in self._projection_columns()
        )

        def _write(connection: sqlite3.Connection) -> CaseRecord:
            connection.execute(
                """
                INSERT INTO cases (
                    case_id,
                    created_at,
                    status,
                    review_needed,
                    """
                + projection_columns
                + """
                    ,
                    case_json,
                    request_json
                ) VALUES (?, ?, ?, ?, """
                + projection_placeholders
                + """, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    status = excluded.status,
                    review_needed = excluded.review_needed,
                    """
                + upsert_assignments
                + """,
                    case_json = excluded.case_json,
                    request_json = excluded.request_json
                """,
                (
                    case.case_id,
                    case.created_at.isoformat(),
                    case.status,
                    1 if case.human_review.needed else 0,
                    *self._projection_values(case),
                    case_json,
                    request_json,
                ),
            )
            return case

        return self._execute_write("save_case", _write)

    def get_case(self, case_id: str) -> CaseRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT case_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return CaseRecord.model_validate_json(row["case_json"])

    def get_case_request_payload(self, case_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT request_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None or not row["request_json"]:
            return None
        return json.loads(str(row["request_json"]))

    def append_audit_event(self, event: AuditEventDraft) -> AuditLogEvent:
        def _write(connection: sqlite3.Connection) -> AuditLogEvent:
            previous_row = connection.execute(
                """
                SELECT event_id, event_hash
                FROM case_audit_log
                ORDER BY occurred_at DESC, event_id DESC
                LIMIT 1
                """
            ).fetchone()
            finalized = finalize_audit_event(
                event,
                previous_event_id=str(previous_row["event_id"]) if previous_row is not None else None,
                previous_event_hash=str(previous_row["event_hash"]) if previous_row is not None else None,
            )
            connection.execute(
                """
                INSERT INTO case_audit_log (
                    event_id,
                    case_id,
                    occurred_at,
                    event_type,
                    event_source,
                    summary,
                    actor_id,
                    actor_role,
                    actor_username,
                    status_before,
                    status_after,
                    human_override,
                    ai_snapshot_hash,
                    override_snapshot_hash,
                    previous_event_id,
                    previous_event_hash,
                    event_hash,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finalized.event_id,
                    finalized.case_id,
                    finalized.occurred_at.isoformat(),
                    finalized.event_type.value,
                    finalized.event_source,
                    finalized.summary,
                    finalized.actor_id,
                    finalized.actor_role.value if finalized.actor_role is not None else None,
                    finalized.actor_username,
                    finalized.status_before,
                    finalized.status_after,
                    1 if finalized.human_override else 0,
                    finalized.ai_snapshot_hash,
                    finalized.override_snapshot_hash,
                    finalized.previous_event_id,
                    finalized.previous_event_hash,
                    finalized.event_hash,
                    json.dumps(finalized.payload, ensure_ascii=False),
                ),
            )
            self._refresh_storage_metrics(connection)
            return finalized

        return self._execute_write("append_audit_event", _write)

    def list_case_audit_events(self, case_id: str, limit: int = 200) -> list[AuditLogEvent]:
        safe_limit = max(1, limit)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    event_id,
                    case_id,
                    occurred_at,
                    event_type,
                    event_source,
                    summary,
                    actor_id,
                    actor_role,
                    actor_username,
                    status_before,
                    status_after,
                    human_override,
                    ai_snapshot_hash,
                    override_snapshot_hash,
                    previous_event_id,
                    previous_event_hash,
                    event_hash,
                    payload_json
                FROM case_audit_log
                WHERE case_id = ?
                ORDER BY occurred_at ASC, event_id ASC
                LIMIT ?
                """,
                (case_id, safe_limit),
            ).fetchall()
        return [self._row_to_audit_event(row) for row in rows]

    def verify_audit_chain(self) -> AuditChainVerification:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    event_id,
                    case_id,
                    occurred_at,
                    event_type,
                    event_source,
                    summary,
                    actor_id,
                    actor_role,
                    actor_username,
                    status_before,
                    status_after,
                    human_override,
                    ai_snapshot_hash,
                    override_snapshot_hash,
                    previous_event_id,
                    previous_event_hash,
                    event_hash,
                    payload_json
                FROM case_audit_log
                ORDER BY occurred_at ASC, event_id ASC
                """
            ).fetchall()

        previous_event_id: str | None = None
        previous_event_hash: str | None = None
        checked_events = 0
        latest_event: AuditLogEvent | None = None
        for row in rows:
            event = self._row_to_audit_event(row)
            if event.previous_event_id != previous_event_id or event.previous_event_hash != previous_event_hash:
                return AuditChainVerification(
                    verified=False,
                    checked_events=checked_events,
                    latest_event_id=previous_event_id,
                    latest_event_hash=previous_event_hash,
                    failure_event_id=event.event_id,
                    failure_reason="previous audit link does not match the stored chain.",
                )
            expected_hash = expected_event_hash(
                event,
                previous_event_id=previous_event_id,
                previous_event_hash=previous_event_hash,
            )
            if event.event_hash != expected_hash:
                return AuditChainVerification(
                    verified=False,
                    checked_events=checked_events,
                    latest_event_id=previous_event_id,
                    latest_event_hash=previous_event_hash,
                    failure_event_id=event.event_id,
                    failure_reason="stored audit event hash does not match the canonical payload.",
                )
            previous_event_id = event.event_id
            previous_event_hash = event.event_hash
            latest_event = event
            checked_events += 1
        return AuditChainVerification(
            verified=True,
            checked_events=checked_events,
            latest_event_id=latest_event.event_id if latest_event is not None else None,
            latest_event_hash=latest_event.event_hash if latest_event is not None else None,
        )

    def list_review_queue(self) -> list[CaseRecord]:
        queue = self.query_review_queue(page=1, page_size=1000)
        case_ids = [str(item["case_id"]) for item in queue["items"]]
        if not case_ids:
            return []
        records = {case_id: self.get_case(case_id) for case_id in case_ids}
        return [records[case_id] for case_id in case_ids if records.get(case_id) is not None]

    def query_review_queue(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        review_queue: str | None = None,
        priority_level: str | None = None,
        status: str | None = None,
        assignee_id: str | None = None,
        assignment_state: str | None = None,
        sort_by: str = "sla",
    ) -> dict[str, object]:
        where_clauses = ["review_needed = 1"]
        params: list[object] = []
        if review_queue:
            where_clauses.append("review_queue = ?")
            params.append(review_queue)
        if priority_level:
            where_clauses.append("priority_level = ?")
            params.append(priority_level)
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if assignee_id:
            where_clauses.append("reviewer_id = ?")
            params.append(assignee_id)
        if assignment_state == "assigned":
            where_clauses.append("reviewer_id IS NOT NULL")
        elif assignment_state == "unassigned":
            where_clauses.append("reviewer_id IS NULL")

        where_sql = " WHERE " + " AND ".join(where_clauses)
        priority_rank_sql = (
            "CASE priority_level "
            "WHEN 'critical' THEN 4 "
            "WHEN 'high' THEN 3 "
            "WHEN 'medium' THEN 2 "
            "WHEN 'low' THEN 1 "
            "ELSE 0 END"
        )
        sla_due_sql = (
            "julianday(created_at) + CASE priority_level "
            "WHEN 'critical' THEN 4.0/24.0 "
            "WHEN 'high' THEN 24.0/24.0 "
            "WHEN 'medium' THEN 72.0/24.0 "
            "WHEN 'low' THEN 168.0/24.0 "
            "ELSE 72.0/24.0 END"
        )
        order_by_sql = {
            "created_at": "created_at DESC",
            "priority": f"{priority_rank_sql} DESC, priority_score DESC, created_at ASC",
            "assignee": "CASE WHEN reviewer_id IS NULL THEN 0 ELSE 1 END ASC, reviewer_id ASC, created_at ASC",
            "sla": (
                f"CASE WHEN julianday('now') >= {sla_due_sql} THEN 0 ELSE 1 END ASC, "
                f"{sla_due_sql} ASC, {priority_rank_sql} DESC, priority_score DESC, created_at ASC"
            ),
        }.get(sort_by, (
            f"CASE WHEN julianday('now') >= {sla_due_sql} THEN 0 ELSE 1 END ASC, "
            f"{sla_due_sql} ASC, {priority_rank_sql} DESC, priority_score DESC, created_at ASC"
        ))

        safe_page = max(1, page)
        safe_page_size = max(1, page_size)
        offset = (safe_page - 1) * safe_page_size

        with self._connect() as connection:
            total_row = connection.execute(
                "SELECT COUNT(*) AS total FROM cases" + where_sql,
                tuple(params),
            ).fetchone()
            rows = connection.execute(
                self._case_list_select()
                + where_sql
                + f" ORDER BY {order_by_sql} LIMIT ? OFFSET ?",
                (*params, safe_page_size, offset),
            ).fetchall()

        items = [_review_queue_item_from_projection(self._row_to_case_list_item(row)) for row in rows]
        total_items = int(total_row["total"]) if total_row else 0
        total_pages = max(1, math.ceil(total_items / safe_page_size)) if total_items else 1
        return {
            "items": items,
            "meta": {
                "page": safe_page,
                "page_size": safe_page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "sort_by": sort_by,
                "assignment_state": assignment_state,
                "review_queue": review_queue,
                "priority_level": priority_level,
                "assignee_id": assignee_id,
                "status": status,
            },
        }

    def list_cases(
        self,
        *,
        category: str | None = None,
        institution: str | None = None,
        priority_level: str | None = None,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        where_clauses: list[str] = []
        params: list[object] = []
        if category:
            where_clauses.append("category = ?")
            params.append(category)
        if institution:
            where_clauses.append("institution = ?")
            params.append(institution)
        if priority_level:
            where_clauses.append("priority_level = ?")
            params.append(priority_level)
        if reviewer_id:
            where_clauses.append("reviewer_id = ?")
            params.append(reviewer_id)
        if final_disposition:
            where_clauses.append("final_disposition = ?")
            params.append(final_disposition)
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        sql = self._case_list_select()
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))

        with self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        return [self._row_to_case_list_item(row) for row in rows]

    def summarize_cases(self) -> dict[str, object]:
        with self._connect() as connection:
            total_row = connection.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(review_needed), 0) AS review_needed FROM cases"
            ).fetchone()

            def _group_counts(column: str) -> dict[str, int]:
                rows = connection.execute(
                    f"""
                    SELECT COALESCE({column}, '(none)') AS bucket, COUNT(*) AS count
                    FROM cases
                    GROUP BY COALESCE({column}, '(none)')
                    ORDER BY count DESC, bucket ASC
                    """
                ).fetchall()
                return {str(row["bucket"]): int(row["count"]) for row in rows}

            return {
                "total_cases": int(total_row["total"]) if total_row else 0,
                "review_needed_cases": int(total_row["review_needed"]) if total_row else 0,
                "counts_by_status": _group_counts("status"),
                "counts_by_category": _group_counts("category"),
                "counts_by_institution": _group_counts("institution"),
                "counts_by_priority": _group_counts("priority_level"),
                "counts_by_final_disposition": _group_counts("final_disposition"),
                "counts_by_reviewer": _group_counts("reviewer_id"),
            }

    def update_case_operational_fields(
        self,
        case_id: str,
        *,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        final_disposition_reason: str | None = None,
    ) -> CaseRecord | None:
        updated_at = datetime.now(timezone.utc)

        def _write(connection: sqlite3.Connection) -> CaseRecord | None:
            row = connection.execute(
                "SELECT case_json, request_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                return None

            case = CaseRecord.model_validate_json(row["case_json"])
            operations = case.operations.model_copy(
                update={
                    "reviewer_id": reviewer_id if reviewer_id is not None else case.operations.reviewer_id,
                    "final_disposition": (
                        final_disposition
                        if final_disposition is not None
                        else case.operations.final_disposition or default_final_disposition(case.status)
                    ),
                    "final_disposition_reason": (
                        final_disposition_reason
                        if final_disposition_reason is not None
                        else case.operations.final_disposition_reason
                    ),
                    "disposition_updated_at": (
                        updated_at
                        if final_disposition is not None or final_disposition_reason is not None
                        else case.operations.disposition_updated_at
                    ),
                }
            )
            updated_case = case.model_copy(
                update={
                    "operations": operations,
                }
            )
            projection_assignments = ", ".join(f"{column} = ?" for column in self._projection_columns())
            connection.execute(
                """
                UPDATE cases
                SET
                    status = ?,
                    review_needed = ?,
                    """
                + projection_assignments
                + """,
                    case_json = ?,
                    request_json = ?
                WHERE case_id = ?
                """,
                (
                    updated_case.status,
                    1 if updated_case.human_review.needed else 0,
                    *self._projection_values(updated_case),
                    updated_case.model_dump_json(),
                    row["request_json"],
                    case_id,
                ),
            )
            return updated_case

        return self._execute_write("update_case_operational_fields", _write)

    def transition_case(
        self,
        case_id: str,
        *,
        transition: CaseTransition,
        actor_id: str | None = None,
        note: str | None = None,
        reviewer_id: str | None = None,
    ) -> CaseRecord | None:
        def _write(connection: sqlite3.Connection) -> CaseRecord | None:
            row = connection.execute(
                "SELECT case_json, request_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                return None

            case = CaseRecord.model_validate_json(row["case_json"])
            if not can_transition(case.status, transition):
                allowed = ", ".join(item.value for item in allowed_transitions(case.status)) or "none"
                raise RuntimeError(
                    f"Invalid lifecycle transition {transition.value} from {case.status.value}. "
                    f"Allowed transitions: {allowed}."
                )

            entry = build_transition_entry(
                from_status=case.status,
                transition=transition,
                actor_id=actor_id,
                note=note,
            )
            updated_operations = case.operations.model_copy(
                update={
                    "reviewer_id": reviewer_id if reviewer_id is not None else case.operations.reviewer_id,
                    "final_disposition": default_final_disposition(entry.to_status),
                    "status_updated_at": entry.acted_at,
                    "disposition_updated_at": entry.acted_at,
                    "transition_history": [*case.operations.transition_history, entry],
                }
            )
            updated_case = case.model_copy(
                update={
                    "status": entry.to_status,
                    "operations": updated_operations,
                }
            )
            projection_assignments = ", ".join(f"{column} = ?" for column in self._projection_columns())
            connection.execute(
                """
                UPDATE cases
                SET
                    status = ?,
                    review_needed = ?,
                    """
                + projection_assignments
                + """,
                    case_json = ?,
                    request_json = ?
                WHERE case_id = ?
                """,
                (
                    updated_case.status,
                    1 if updated_case.human_review.needed else 0,
                    *self._projection_values(updated_case),
                    updated_case.model_dump_json(),
                    row["request_json"],
                    case_id,
                ),
            )
            return updated_case

        return self._execute_write("transition_case", _write)

    def list_backups(self, limit: int = 20) -> list[dict[str, object]]:
        safe_limit = max(1, limit)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    backup_id,
                    backup_path,
                    label,
                    backup_type,
                    created_at,
                    source_path,
                    integrity_check,
                    size_bytes
                FROM storage_backups
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def backup_database(
        self,
        destination_path: str | None = None,
        *,
        label: str | None = None,
        backup_type: str = "manual_backup",
    ) -> dict[str, object]:
        backup_id = uuid.uuid4().hex
        created_at = _utc_now_iso()
        source_path = str(self.db_path)
        destination: Path | None = None
        last_error: sqlite3.OperationalError | None = None
        for candidate in self._candidate_backup_destinations(destination_path, label):
            try:
                with self._connect() as source_connection, sqlite3.connect(
                    str(candidate)
                ) as destination_connection:
                    source_connection.backup(
                        destination_connection,
                        pages=self.backup_pages_per_step,
                        sleep=0.05,
                    )
                destination = candidate
                break
            except sqlite3.OperationalError as exc:
                last_error = exc
                continue
        if destination is None:
            raise last_error or RuntimeError("Unable to create SQLite backup destination.")

        integrity_check = self._integrity_check(destination)
        size_bytes = destination.stat().st_size
        return self._record_backup(
            backup_id=backup_id,
            backup_path=destination,
            label=label,
            backup_type=backup_type,
            created_at=created_at,
            source_path=source_path,
            integrity_check=integrity_check,
            size_bytes=size_bytes,
        )

    def restore_database(
        self,
        source_path: str,
        *,
        create_pre_restore_backup: bool = True,
    ) -> dict[str, object]:
        source = self._resolve_existing_path(source_path)
        source_integrity = self._integrity_check(source)
        if source_integrity != "ok":
            raise RuntimeError(
                f"SQLite restore source failed integrity_check with status: {source_integrity}"
            )

        source_backup_record = self._lookup_backup_by_path(source)
        pre_restore_backup: dict[str, object] | None = None
        if create_pre_restore_backup:
            pre_restore_backup = self.backup_database(
                label="pre-restore",
                backup_type="pre_restore_backup",
            )

        restore_id = uuid.uuid4().hex
        restored_at = _utc_now_iso()

        with sqlite3.connect(str(source)) as source_connection, self._connect() as destination_connection:
            source_connection.backup(
                destination_connection,
                pages=self.backup_pages_per_step,
                sleep=0.05,
            )
            self._ensure_migration_table(destination_connection)
            self._initialize_schema(destination_connection)

        def _record_restore(connection: sqlite3.Connection) -> dict[str, object]:
            if source_backup_record is not None:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO storage_backups (
                        backup_id,
                        backup_path,
                        label,
                        backup_type,
                        created_at,
                        source_path,
                        integrity_check,
                        size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_backup_record["backup_id"],
                        source_backup_record["backup_path"],
                        source_backup_record["label"],
                        source_backup_record["backup_type"],
                        source_backup_record["created_at"],
                        source_backup_record["source_path"],
                        source_backup_record["integrity_check"],
                        source_backup_record["size_bytes"],
                    ),
                )
            if pre_restore_backup is not None:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO storage_backups (
                        backup_id,
                        backup_path,
                        label,
                        backup_type,
                        created_at,
                        source_path,
                        integrity_check,
                        size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pre_restore_backup["backup_id"],
                        pre_restore_backup["backup_path"],
                        pre_restore_backup["label"],
                        pre_restore_backup["backup_type"],
                        pre_restore_backup["created_at"],
                        pre_restore_backup["source_path"],
                        pre_restore_backup["integrity_check"],
                        pre_restore_backup["size_bytes"],
                    ),
                )
            connection.execute(
                """
                INSERT INTO storage_restores (
                    restore_id,
                    source_path,
                    restored_at,
                    source_integrity,
                    pre_restore_backup_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    restore_id,
                    str(source),
                    restored_at,
                    source_integrity,
                    pre_restore_backup["backup_id"] if pre_restore_backup else None,
                ),
            )
            self._refresh_storage_metrics(connection)
            return {
                "restore_id": restore_id,
                "source_path": str(source),
                "restored_at": restored_at,
                "source_integrity": source_integrity,
                "pre_restore_backup": pre_restore_backup,
                "schema_version": self.schema_version,
            }

        return self._execute_write("record_restore", _record_restore)

    def diagnostics(self) -> dict[str, object]:
        with self._connect() as connection:
            self._refresh_storage_metrics(connection)
        audit_verification = self.verify_audit_chain()
        return {
            "sqlite_mode": "enabled",
            "sqlite_db_path": str(self.db_path),
            "sqlite_requested_path": self.requested_path,
            "sqlite_fallback_used": self.used_fallback,
            "sqlite_fallback_reason": self.fallback_reason,
            "sqlite_journal_mode": self.effective_journal_mode,
            "sqlite_synchronous": self.effective_synchronous,
            "sqlite_busy_timeout_ms": self.busy_timeout_ms,
            "sqlite_schema_version": self.schema_version,
            "sqlite_schema_target_version": CURRENT_SCHEMA_VERSION,
            "sqlite_migration_count": self.migration_count,
            "sqlite_case_projection_ready": self.schema_version >= 4,
            "sqlite_case_projection_columns": list(self._projection_columns()),
            "sqlite_case_lifecycle_ready": self.schema_version >= 5,
            "sqlite_case_provenance_ready": self.schema_version >= 6,
            "sqlite_case_audit_ready": self.schema_version >= 7,
            "sqlite_case_lifecycle_states": [status.value for status in CaseStatus],
            "sqlite_write_strategy": "begin_immediate_retry",
            "sqlite_max_write_retries": self.max_write_retries,
            "sqlite_write_retry_backoff_ms": self.write_retry_backoff_ms,
            "sqlite_backup_dir": str(self.backup_dir),
            "sqlite_backup_dir_requested": self.requested_backup_dir,
            "sqlite_backup_dir_fallback_used": self.backup_dir_used_fallback,
            "sqlite_backup_dir_fallback_reason": self.backup_dir_fallback_reason,
            "sqlite_backup_pages_per_step": self.backup_pages_per_step,
            "sqlite_backup_count": self.backup_count,
            "sqlite_restore_count": self.restore_count,
            "sqlite_last_backup_at": self.last_backup_at,
            "audit_event_count": self.audit_event_count,
            "audit_chain_verified": audit_verification.verified,
            "audit_chain_checked_events": audit_verification.checked_events,
            "audit_chain_latest_hash": audit_verification.latest_event_hash,
            "audit_chain_failure_reason": audit_verification.failure_reason,
        }

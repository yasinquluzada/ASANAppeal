from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from app.models.domain import (
    AuditEventDraft,
    AuditEventType,
    CaseModelContext,
    CaseOperationalState,
    CaseRecord,
    CaseStatus,
    CaseTransition,
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    PriorityLevel,
    RoutingDecision,
    StructuredIssue,
)
from app.repository import CURRENT_SCHEMA_VERSION, SQLiteCaseRepository


def _build_case(case_id: str, *, review_needed: bool = True) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        status="needs_review" if review_needed else "ready_for_dispatch",
        submission_excerpt="Problem near the school road.",
        structured_issue=StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary=f"Large pothole for {case_id}.",
            extracted_signals=["pothole"],
            missing_information=[],
            confidence=0.81,
        ),
        routing=RoutingDecision(
            institution="ASAN Road Maintenance Agency",
            department="Road Surface Response",
            category="road_damage",
            rationale="Seed rationale.",
            confidence=0.84,
        ),
        priority=PriorityDecision(
            level=PriorityLevel.high,
            score=76,
            reasons=["Safety risk on a public road."],
            confidence=0.79,
            requires_human_review=False,
        ),
        draft=DraftAppeal(
            title="Road Damage reported near school",
            body="Please repair the pothole.",
            citizen_review_checklist=[],
            confidence=0.8,
        ),
        explanation=ExplanationNote(
            summary="Road damage case routed for repair.",
            next_action="Send to human-review for manual review.",
            detailed_rationale=["Confidence was below dispatch threshold."],
            risk_flags=["location_needs_check"],
        ),
        human_review=HumanReviewTask(
            needed=review_needed,
            queue="human-review",
            reasons=["Need clearer verification before dispatch."] if review_needed else [],
            confidence=0.58 if review_needed else 0.91,
        ),
    )


def test_sqlite_repository_persists_cases(tmp_path: Path) -> None:
    repository = SQLiteCaseRepository(str(tmp_path / "asanappeal.db"))
    case = _build_case("case123")

    request_payload = {"submission": {"citizen_text": "Need help"}}
    repository.save_case(case, request_payload=request_payload)
    loaded = repository.get_case("case123")

    assert loaded is not None
    assert loaded.case_id == "case123"
    assert repository.list_review_queue()[0].case_id == "case123"
    assert repository.get_case_request_payload("case123") == request_payload


def test_sqlite_repository_hardens_to_writable_fallback() -> None:
    repository = SQLiteCaseRepository.build_hardened("/dev/null/asanappeal.db")

    diagnostics = repository.diagnostics()
    assert diagnostics["sqlite_mode"] == "enabled"
    assert diagnostics["sqlite_fallback_used"] is True
    assert diagnostics["sqlite_db_path"] != "/dev/null/asanappeal.db"


def test_sqlite_repository_tracks_schema_and_backup_restore(tmp_path: Path) -> None:
    repository = SQLiteCaseRepository(str(tmp_path / "asanappeal.db"))
    repository.save_case(_build_case("before-restore"))

    backup = repository.backup_database(label="baseline")
    assert Path(str(backup["backup_path"])).exists()
    assert backup["integrity_check"] == "ok"

    repository.save_case(_build_case("after-backup"))
    assert repository.get_case("after-backup") is not None

    restore = repository.restore_database(str(backup["backup_path"]))

    assert repository.get_case("before-restore") is not None
    assert repository.get_case("after-backup") is None
    assert restore["schema_version"] == CURRENT_SCHEMA_VERSION
    assert restore["source_integrity"] == "ok"
    assert restore["pre_restore_backup"] is not None

    diagnostics = repository.diagnostics()
    assert diagnostics["sqlite_schema_version"] == CURRENT_SCHEMA_VERSION
    assert diagnostics["sqlite_backup_count"] >= 2
    assert diagnostics["sqlite_restore_count"] == 1
    assert diagnostics["sqlite_write_strategy"] == "begin_immediate_retry"
    assert diagnostics["sqlite_migration_count"] == CURRENT_SCHEMA_VERSION
    assert len(repository.list_backups()) >= 2


def test_sqlite_repository_retries_locked_write(tmp_path: Path) -> None:
    database_path = tmp_path / "asanappeal.db"
    repository = SQLiteCaseRepository(
        str(database_path),
        timeout_seconds=0.05,
        busy_timeout_ms=25,
        max_write_retries=6,
        write_retry_backoff_ms=25,
    )
    locker = sqlite3.connect(database_path, timeout=0.05, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")

    errors: list[BaseException] = []

    def writer() -> None:
        try:
            repository.save_case(_build_case("lock-retry-case", review_needed=False))
        except BaseException as exc:  # pragma: no cover - assertion below surfaces it
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.15)
    locker.rollback()
    locker.close()
    thread.join(timeout=3)

    assert not errors
    assert repository.get_case("lock-retry-case") is not None


def test_sqlite_repository_queries_review_queue_with_filters_and_paging(tmp_path: Path) -> None:
    repository = SQLiteCaseRepository(str(tmp_path / "asanappeal.db"))
    high_case = _build_case("queue-high", review_needed=True).model_copy(
        update={
            "operations": CaseOperationalState(
                reviewer_id="reviewer-queue-1",
                final_disposition="pending_review",
            )
        }
    )
    low_case = _build_case("queue-low", review_needed=True).model_copy(
        update={
            "priority": PriorityDecision(
                level=PriorityLevel.low,
                score=18,
                reasons=["Low clarity."],
                confidence=0.41,
                requires_human_review=True,
            )
        }
    )

    repository.save_case(high_case)
    repository.save_case(low_case)

    queue = repository.query_review_queue(page=1, page_size=1, sort_by="sla")
    assigned_only = repository.query_review_queue(
        assignee_id="reviewer-queue-1",
        assignment_state="assigned",
    )

    assert queue["meta"]["page"] == 1
    assert queue["meta"]["page_size"] == 1
    assert queue["meta"]["total_items"] == 2
    assert queue["items"][0]["case_id"] == "queue-high"
    assert queue["items"][0]["sla_deadline_at"]
    assert isinstance(queue["items"][0]["sla_breached"], bool)

    assert assigned_only["meta"]["assignment_state"] == "assigned"
    assert len(assigned_only["items"]) == 1
    assert assigned_only["items"][0]["assignee_id"] == "reviewer-queue-1"


def test_sqlite_repository_exposes_queryable_case_projection(tmp_path: Path) -> None:
    repository = SQLiteCaseRepository(str(tmp_path / "asanappeal.db"))
    case = _build_case("analytics-case", review_needed=False).model_copy(
        update={
            "model_context": CaseModelContext(
                provider="LocalFreeProvider",
                model_name="localfree+heuristic",
                model_version="localfree+heuristic",
            ),
            "operations": CaseOperationalState(
                reviewer_id="reviewer-42",
                final_disposition="resolved",
            ),
        }
    )
    repository.save_case(case)

    items = repository.list_cases(
        category="road_damage",
        reviewer_id="reviewer-42",
        final_disposition="resolved",
    )
    assert len(items) == 1
    item = items[0]
    assert item["institution"] == "ASAN Road Maintenance Agency"
    assert item["priority_level"] == "high"
    assert item["model_context"]["provider"] == "LocalFreeProvider"
    assert item["model_context"]["provenance_schema_version"] == "case-provenance.v1"
    assert item["operations"]["final_disposition"] == "resolved"

    updated = repository.update_case_operational_fields(
        "analytics-case",
        reviewer_id="reviewer-99",
        final_disposition="closed",
        final_disposition_reason="Manually verified.",
    )
    assert updated is not None
    assert updated.status == CaseStatus.ready_for_dispatch
    transitioned = repository.transition_case(
        "analytics-case",
        transition=CaseTransition.assign,
        actor_id="reviewer-99",
        reviewer_id="reviewer-99",
    )
    assert transitioned is not None
    progressed = repository.transition_case(
        "analytics-case",
        transition=CaseTransition.resolve,
        actor_id="reviewer-99",
    )
    assert progressed is not None
    closed = repository.transition_case(
        "analytics-case",
        transition=CaseTransition.close,
        actor_id="reviewer-99",
    )
    assert closed is not None
    assert closed.status == CaseStatus.closed
    assert len(closed.operations.transition_history) >= 3
    assert updated.operations.reviewer_id == "reviewer-99"
    assert closed.operations.final_disposition == "closed"

    summary = repository.summarize_cases()
    assert summary["counts_by_status"]["closed"] == 1
    assert summary["counts_by_final_disposition"]["closed"] == 1
    assert summary["counts_by_reviewer"]["reviewer-99"] == 1


def test_sqlite_repository_rejects_invalid_lifecycle_transition(tmp_path: Path) -> None:
    repository = SQLiteCaseRepository(str(tmp_path / "asanappeal.db"))
    repository.save_case(_build_case("invalid-transition", review_needed=True))

    try:
        repository.transition_case(
            "invalid-transition",
            transition=CaseTransition.resolve,
            actor_id="reviewer-1",
        )
    except RuntimeError as exc:
        assert "Invalid lifecycle transition" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid lifecycle transition to fail.")


def test_sqlite_repository_migrates_legacy_blob_rows_into_projection(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    legacy_case = _build_case("legacy-case").model_copy(
        update={
            "model_context": CaseModelContext(
                provider="LegacyProvider",
                model_name="legacy-model",
                model_version="legacy-v1",
            ),
            "operations": CaseOperationalState(
                reviewer_id="legacy-reviewer",
                final_disposition="pending_review",
            ),
        }
    )

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                review_needed INTEGER NOT NULL,
                case_json TEXT NOT NULL,
                request_json TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO cases (
                case_id,
                created_at,
                status,
                review_needed,
                case_json,
                request_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_case.case_id,
                legacy_case.created_at.isoformat(),
                legacy_case.status,
                1,
                legacy_case.model_dump_json(),
                "{}",
            ),
        )
        connection.execute("PRAGMA user_version = 3")

    repository = SQLiteCaseRepository(str(database_path))
    diagnostics = repository.diagnostics()
    items = repository.list_cases(category="road_damage")

    assert diagnostics["sqlite_schema_version"] == CURRENT_SCHEMA_VERSION
    assert diagnostics["sqlite_case_projection_ready"] is True
    assert diagnostics["sqlite_case_provenance_ready"] is True
    assert len(items) == 1
    assert items[0]["model_context"]["model_version"] == "legacy-v1"
    assert items[0]["model_context"]["provenance_schema_version"] == "case-provenance.v1"
    assert items[0]["operations"]["reviewer_id"] == "legacy-reviewer"


def test_sqlite_repository_reconciles_existing_latest_version_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "reconcile.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                review_needed INTEGER NOT NULL,
                case_json TEXT NOT NULL,
                request_json TEXT,
                submission_excerpt TEXT,
                category TEXT,
                issue_type TEXT,
                institution TEXT,
                department TEXT,
                priority_level TEXT,
                priority_score INTEGER,
                issue_confidence REAL,
                routing_confidence REAL,
                priority_confidence REAL,
                review_queue TEXT,
                review_confidence REAL,
                verification_same_place TEXT,
                verification_issue_resolved TEXT,
                provider_name TEXT,
                model_name TEXT,
                model_version TEXT,
                provenance_schema_version TEXT,
                prompt_bundle_version TEXT,
                classifier_bundle_version TEXT,
                threshold_set_version TEXT,
                reviewer_id TEXT,
                final_disposition TEXT,
                final_disposition_reason TEXT,
                disposition_updated_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        for version, name in (
            (1, "create_cases_table"),
            (2, "create_storage_backups_table"),
            (3, "create_storage_restores_table"),
            (4, "case_operational_projection"),
            (5, "case_lifecycle_state_machine"),
            (6, "case_model_provenance"),
        ):
            connection.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, "2026-04-19T00:00:00+00:00"),
            )
        connection.execute("PRAGMA user_version = 6")

    repository = SQLiteCaseRepository(str(database_path))
    repository.save_case(_build_case("reconciled-case", review_needed=False))

    with sqlite3.connect(database_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(cases)")]

    diagnostics = repository.diagnostics()
    loaded = repository.get_case("reconciled-case")

    assert "status_updated_at" in columns
    assert "provenance_schema_version" in columns
    assert diagnostics["sqlite_schema_version"] == CURRENT_SCHEMA_VERSION
    assert diagnostics["sqlite_case_lifecycle_ready"] is True
    assert diagnostics["sqlite_case_provenance_ready"] is True
    assert loaded is not None


def test_sqlite_repository_stores_append_only_tamper_evident_audit_log(tmp_path: Path) -> None:
    repository = SQLiteCaseRepository(str(tmp_path / "asanappeal.db"))
    case = _build_case("audit-case", review_needed=True)
    repository.save_case(case, request_payload={"submission": {"citizen_text": "Audit me"}})

    created_event = repository.append_audit_event(
        AuditEventDraft(
            case_id=case.case_id,
            event_type=AuditEventType.case_created,
            event_source="test.case_created",
            summary="Initial case creation.",
            actor_id="citizen-1",
            actor_role="citizen",
            actor_username="citizen.demo",
            status_after=case.status.value,
            human_override=False,
            ai_snapshot_hash="ai-hash-1",
            payload={"ai_outputs": {"routing": "ASAN Road Maintenance Agency"}},
        )
    )
    updated_event = repository.append_audit_event(
        AuditEventDraft(
            case_id=case.case_id,
            event_type=AuditEventType.workflow_action,
            event_source="test.workflow",
            summary="Case approved by reviewer.",
            actor_id="reviewer-1",
            actor_role="reviewer",
            actor_username="reviewer.demo",
            status_before=case.status.value,
            status_after="ready_for_dispatch",
            human_override=True,
            ai_snapshot_hash="ai-hash-1",
            override_snapshot_hash="override-hash-1",
            payload={"human_override": {"action": "approve"}},
        )
    )

    events = repository.list_case_audit_events(case.case_id)
    verification = repository.verify_audit_chain()

    assert len(events) == 2
    assert events[0].event_id == created_event.event_id
    assert events[1].previous_event_id == created_event.event_id
    assert events[1].previous_event_hash == created_event.event_hash
    assert updated_event.human_override is True
    assert verification.verified is True

    with sqlite3.connect(repository.db_path) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "UPDATE case_audit_log SET summary = ? WHERE event_id = ?",
                ("tampered", created_event.event_id),
            )

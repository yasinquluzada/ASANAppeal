from __future__ import annotations

from app.config import Settings
from app.local_ml import LocalPriorityModel, LocalRoutingModel, export_reviewed_feedback
from app.models.api import SubmissionInput
from app.models.domain import (
    CaseModelContext,
    CaseOperationalState,
    CaseRecord,
    CaseStatus,
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    PriorityLevel,
    RoutingDecision,
    StructuredIssue,
)
from app.repository import SQLiteCaseRepository


def _build_reviewed_case(case_id: str) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        status=CaseStatus.closed,
        submission_excerpt="Retaining wall and carriageway collapse beside library underpass.",
        structured_issue=StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary="Retaining wall failure damaged the carriageway beside the library underpass.",
            extracted_signals=["collapse", "carriageway", "underpass"],
            missing_information=[],
            confidence=0.94,
        ),
        routing=RoutingDecision(
            institution="ASAN Road Maintenance Agency",
            department="Road Surface Response",
            category="road_damage",
            rationale="Reviewed and confirmed as a roadway failure case.",
            confidence=0.93,
        ),
        priority=PriorityDecision(
            level=PriorityLevel.high,
            score=88,
            reasons=["High public safety risk on an active carriageway."],
            confidence=0.91,
            requires_human_review=False,
        ),
        draft=DraftAppeal(
            title="Road Damage near library underpass",
            body="Please dispatch a repair crew for the carriageway collapse.",
            citizen_review_checklist=[],
            confidence=0.9,
        ),
        explanation=ExplanationNote(
            summary="Reviewed road damage case.",
            next_action="Keep with road maintenance operations.",
            detailed_rationale=["Human reviewer validated the category and urgency."],
            risk_flags=[],
        ),
        human_review=HumanReviewTask(
            needed=False,
            queue="triage-review",
            reasons=[],
            confidence=0.95,
        ),
        model_context=CaseModelContext(
            provider="LocalFreeProvider",
            model_name="localfree+naive-bayes-corpus",
            model_version="localfree+naive-bayes-corpus",
        ),
        operations=CaseOperationalState(
            reviewer_id="reviewer-17",
            final_disposition="closed",
        ),
    )


def test_local_routing_model_loads_bootstrap_corpus() -> None:
    model = LocalRoutingModel(Settings(repository_backend="memory"))

    assert model.training_examples_count >= 60
    assert model.training_sources == ["bootstrap_corpus"]
    assert model.label_counts["road_damage"] >= 8
    assert model.label_counts["general_public_service"] >= 8


def test_local_priority_model_loads_bootstrap_corpus() -> None:
    model = LocalPriorityModel(Settings(repository_backend="memory"))

    assert model.training_examples_count >= 30
    assert model.training_sources == ["bootstrap_corpus"]
    assert model.label_counts["critical"] >= 8
    assert model.label_counts["low"] >= 8


def test_local_models_include_reviewed_sqlite_feedback(tmp_path) -> None:
    database_path = tmp_path / "asan-local-ml.db"
    repository = SQLiteCaseRepository(str(database_path))
    repository.save_case(
        _build_reviewed_case("sqlite-feedback-case"),
        request_payload={
            "submission": {
                "citizen_text": "Retaining wall and carriageway collapse beside library underpass.",
                "location_hint": "Library underpass",
                "time_hint": "2026-04-19 13:00",
                "evidence": [],
            }
        },
    )

    bootstrap_routing = LocalRoutingModel(Settings(repository_backend="memory"))
    bootstrap_priority = LocalPriorityModel(Settings(repository_backend="memory"))
    feedback_settings = Settings(
        repository_backend="sqlite",
        sqlite_path=str(database_path),
        local_ml_include_sqlite_feedback=True,
    )
    routing_model = LocalRoutingModel(feedback_settings)
    priority_model = LocalPriorityModel(feedback_settings)

    assert "sqlite_reviewed_cases" in routing_model.training_sources
    assert "sqlite_reviewed_cases" in priority_model.training_sources
    assert routing_model.training_examples_count > bootstrap_routing.training_examples_count
    assert priority_model.training_examples_count > bootstrap_priority.training_examples_count
    assert routing_model.predict(
        submission=SubmissionInput(
            citizen_text="Library underpass carriageway collapse with dangerous road break.",
            location_hint="Library underpass",
            time_hint="today",
            language="en",
            evidence=[],
        ),
        structured_issue=StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary="Underpass carriageway collapse.",
            extracted_signals=[],
            missing_information=[],
            confidence=0.8,
        ),
    ).label == "road_damage"
    assert priority_model.predict(
        submission=SubmissionInput(
            citizen_text="Library underpass carriageway collapse with dangerous road break.",
            location_hint="Library underpass",
            time_hint="today",
            language="en",
            evidence=[],
        ),
        structured_issue=StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary="Underpass carriageway collapse.",
            extracted_signals=[],
            missing_information=[],
            confidence=0.8,
        ),
        routing=RoutingDecision(
            institution="ASAN Road Maintenance Agency",
            department="Road Surface Response",
            category="road_damage",
            rationale="",
            confidence=0.8,
        ),
    ).label == "high"


def test_export_reviewed_feedback_writes_jsonl_artifacts(tmp_path) -> None:
    database_path = tmp_path / "asan-local-ml.db"
    feedback_dir = tmp_path / "feedback_exports"
    repository = SQLiteCaseRepository(str(database_path))
    repository.save_case(
        _build_reviewed_case("sqlite-feedback-export"),
        request_payload={
            "submission": {
                "citizen_text": "Retaining wall and carriageway collapse beside library underpass.",
                "location_hint": "Library underpass",
                "time_hint": "2026-04-19 13:00",
                "evidence": [],
            }
        },
    )

    result = export_reviewed_feedback(
        Settings(
            repository_backend="sqlite",
            sqlite_path=str(database_path),
            local_ml_feedback_dir=str(feedback_dir),
        )
    )

    assert result.routing.exported_examples >= 1
    assert result.priority.exported_examples >= 1
    assert (feedback_dir / "routing_feedback.jsonl").exists()
    assert (feedback_dir / "priority_feedback.jsonl").exists()
    assert (feedback_dir / "retrain_report.json").exists()
    assert "exported_sqlite_feedback" in (feedback_dir / "routing_feedback.jsonl").read_text(encoding="utf-8")

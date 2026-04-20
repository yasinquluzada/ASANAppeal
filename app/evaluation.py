from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from app.config import Settings
from app.models.api import InstitutionResponseInput, SubmissionInput
from app.models.domain import PriorityDecision, RoutingDecision, StructuredIssue, VerificationDecision
from app.providers import create_provider
from app.providers.base import AIProvider


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATASET_DIR = PROJECT_ROOT / "data" / "evals"
DEFAULT_EVAL_ARTIFACT_DIR = PROJECT_ROOT / ".data" / "evals"
LATEST_REPORT_FILENAME = "latest_evaluation_report.json"
LATEST_CALIBRATION_FILENAME = "latest_threshold_calibration.json"
THRESHOLD_CANDIDATES = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
TModel = TypeVar("TModel", bound=BaseModel)


class IntakeEvalExample(BaseModel):
    example_id: str
    submission: SubmissionInput
    expected_category: str
    expected_issue_type: str


class RoutingEvalExample(BaseModel):
    example_id: str
    submission: SubmissionInput
    structured_issue: StructuredIssue
    expected_institution: str
    expected_department: str


class PriorityEvalExample(BaseModel):
    example_id: str
    submission: SubmissionInput
    structured_issue: StructuredIssue
    routing: RoutingDecision
    expected_level: str


class VerificationEvalExample(BaseModel):
    example_id: str
    original_submission: SubmissionInput
    structured_issue: StructuredIssue
    institution_response: InstitutionResponseInput
    expected_same_place: str
    expected_issue_resolved: str


def _normalize_project_path(value: str | None, *, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_jsonl_records(path: Path, model: type[TModel]) -> list[TModel]:
    records: list[TModel] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        records.append(model.model_validate(json.loads(line)))
    return records


def _round_metric(value: float) -> float:
    return round(value, 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _build_calibration_rows(records: list[dict[str, object]]) -> tuple[list[dict[str, float | int]], float]:
    if not records:
        return [], 0.0

    rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLD_CANDIDATES:
        qualified = [record for record in records if float(record["confidence"]) >= threshold]
        qualified_cases = len(qualified)
        correct_cases = sum(1 for record in qualified if bool(record["correct"]))
        rows.append(
            {
                "threshold": threshold,
                "qualified_cases": qualified_cases,
                "coverage": _round_metric(_ratio(qualified_cases, len(records))),
                "accuracy": _round_metric(_ratio(correct_cases, qualified_cases)),
            }
        )

    best_row = max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["coverage"]),
            -float(row["threshold"]),
        ),
    )
    return rows, float(best_row["threshold"])


def _build_acceptance_gate(
    *,
    task: str,
    metric: str,
    minimum: float,
    observed: float,
) -> dict[str, object]:
    return {
        "task": task,
        "metric": metric,
        "minimum": _round_metric(minimum),
        "observed": _round_metric(observed),
        "passed": observed >= minimum,
    }


def _intake_dataset_path(settings: Settings) -> Path:
    dataset_dir = _normalize_project_path(settings.eval_dataset_dir, default=DEFAULT_EVAL_DATASET_DIR)
    return dataset_dir / "intake_gold.jsonl"


def _routing_dataset_path(settings: Settings) -> Path:
    dataset_dir = _normalize_project_path(settings.eval_dataset_dir, default=DEFAULT_EVAL_DATASET_DIR)
    return dataset_dir / "routing_gold.jsonl"


def _priority_dataset_path(settings: Settings) -> Path:
    dataset_dir = _normalize_project_path(settings.eval_dataset_dir, default=DEFAULT_EVAL_DATASET_DIR)
    return dataset_dir / "priority_gold.jsonl"


def _verification_dataset_path(settings: Settings) -> Path:
    dataset_dir = _normalize_project_path(settings.eval_dataset_dir, default=DEFAULT_EVAL_DATASET_DIR)
    return dataset_dir / "verification_gold.jsonl"


def _evaluate_intake(provider: AIProvider, settings: Settings) -> dict[str, object]:
    dataset_path = _intake_dataset_path(settings)
    examples = _load_jsonl_records(dataset_path, IntakeEvalExample)
    records: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    issue_type_correct = 0

    for example in examples:
        prediction = provider.analyze_submission(example.submission)
        category_match = prediction.category == example.expected_category
        issue_type_match = prediction.issue_type == example.expected_issue_type
        if issue_type_match:
            issue_type_correct += 1
        record = {
            "example_id": example.example_id,
            "correct": category_match and issue_type_match,
            "confidence": prediction.confidence,
        }
        records.append(record)
        if not record["correct"]:
            mismatches.append(
                {
                    "example_id": example.example_id,
                    "expected_category": example.expected_category,
                    "predicted_category": prediction.category,
                    "expected_issue_type": example.expected_issue_type,
                    "predicted_issue_type": prediction.issue_type,
                    "confidence": prediction.confidence,
                }
            )

    accuracy = _ratio(sum(1 for record in records if bool(record["correct"])), len(records))
    calibration_rows, recommended_threshold = _build_calibration_rows(records)
    return {
        "task": "intake",
        "dataset_path": str(dataset_path),
        "total_cases": len(records),
        "correct_cases": sum(1 for record in records if bool(record["correct"])),
        "accuracy": _round_metric(accuracy),
        "average_confidence": _round_metric(
            sum(float(record["confidence"]) for record in records) / max(1, len(records))
        ),
        "component_metrics": {
            "category_accuracy": _round_metric(accuracy),
            "issue_type_accuracy": _round_metric(_ratio(issue_type_correct, len(records))),
        },
        "calibration": calibration_rows,
        "recommended_threshold": recommended_threshold,
        "acceptance_gate": _build_acceptance_gate(
            task="intake",
            metric="accuracy",
            minimum=settings.eval_intake_min_accuracy,
            observed=accuracy,
        ),
        "mismatches": mismatches[:10],
    }


def _evaluate_routing(provider: AIProvider, settings: Settings) -> dict[str, object]:
    dataset_path = _routing_dataset_path(settings)
    examples = _load_jsonl_records(dataset_path, RoutingEvalExample)
    records: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    institution_correct = 0
    department_correct = 0

    for example in examples:
        prediction = provider.route_issue(example.submission, example.structured_issue)
        institution_match = prediction.institution == example.expected_institution
        department_match = prediction.department == example.expected_department
        if institution_match:
            institution_correct += 1
        if department_match:
            department_correct += 1
        record = {
            "example_id": example.example_id,
            "correct": institution_match and department_match,
            "confidence": prediction.confidence,
        }
        records.append(record)
        if not record["correct"]:
            mismatches.append(
                {
                    "example_id": example.example_id,
                    "expected_institution": example.expected_institution,
                    "predicted_institution": prediction.institution,
                    "expected_department": example.expected_department,
                    "predicted_department": prediction.department,
                    "confidence": prediction.confidence,
                }
            )

    accuracy = _ratio(sum(1 for record in records if bool(record["correct"])), len(records))
    calibration_rows, recommended_threshold = _build_calibration_rows(records)
    return {
        "task": "routing",
        "dataset_path": str(dataset_path),
        "total_cases": len(records),
        "correct_cases": sum(1 for record in records if bool(record["correct"])),
        "accuracy": _round_metric(accuracy),
        "average_confidence": _round_metric(
            sum(float(record["confidence"]) for record in records) / max(1, len(records))
        ),
        "component_metrics": {
            "institution_accuracy": _round_metric(_ratio(institution_correct, len(records))),
            "department_accuracy": _round_metric(_ratio(department_correct, len(records))),
        },
        "calibration": calibration_rows,
        "recommended_threshold": recommended_threshold,
        "acceptance_gate": _build_acceptance_gate(
            task="routing",
            metric="accuracy",
            minimum=settings.eval_routing_min_accuracy,
            observed=accuracy,
        ),
        "mismatches": mismatches[:10],
    }


def _evaluate_priority(provider: AIProvider, settings: Settings) -> dict[str, object]:
    dataset_path = _priority_dataset_path(settings)
    examples = _load_jsonl_records(dataset_path, PriorityEvalExample)
    records: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    requires_human_review_correct = 0

    for example in examples:
        prediction = provider.assess_priority(
            example.submission,
            example.structured_issue,
            example.routing,
        )
        level_match = prediction.level.value == example.expected_level
        expected_human_review = example.expected_level == "critical"
        if prediction.requires_human_review == expected_human_review:
            requires_human_review_correct += 1
        record = {
            "example_id": example.example_id,
            "correct": level_match,
            "confidence": prediction.confidence,
        }
        records.append(record)
        if not level_match:
            mismatches.append(
                {
                    "example_id": example.example_id,
                    "expected_level": example.expected_level,
                    "predicted_level": prediction.level.value,
                    "predicted_score": prediction.score,
                    "confidence": prediction.confidence,
                }
            )

    accuracy = _ratio(sum(1 for record in records if bool(record["correct"])), len(records))
    calibration_rows, recommended_threshold = _build_calibration_rows(records)
    return {
        "task": "priority",
        "dataset_path": str(dataset_path),
        "total_cases": len(records),
        "correct_cases": sum(1 for record in records if bool(record["correct"])),
        "accuracy": _round_metric(accuracy),
        "average_confidence": _round_metric(
            sum(float(record["confidence"]) for record in records) / max(1, len(records))
        ),
        "component_metrics": {
            "level_accuracy": _round_metric(accuracy),
            "human_review_flag_accuracy": _round_metric(
                _ratio(requires_human_review_correct, len(records))
            ),
        },
        "calibration": calibration_rows,
        "recommended_threshold": recommended_threshold,
        "acceptance_gate": _build_acceptance_gate(
            task="priority",
            metric="accuracy",
            minimum=settings.eval_priority_min_accuracy,
            observed=accuracy,
        ),
        "mismatches": mismatches[:10],
    }


def _evaluate_verification(provider: AIProvider, settings: Settings) -> dict[str, object]:
    dataset_path = _verification_dataset_path(settings)
    examples = _load_jsonl_records(dataset_path, VerificationEvalExample)
    records: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    same_place_correct = 0
    issue_resolved_correct = 0

    for example in examples:
        prediction = provider.verify_resolution(
            example.original_submission,
            example.structured_issue,
            example.institution_response,
        )
        same_place_match = prediction.same_place.value == example.expected_same_place
        issue_resolved_match = prediction.issue_resolved.value == example.expected_issue_resolved
        if same_place_match:
            same_place_correct += 1
        if issue_resolved_match:
            issue_resolved_correct += 1
        record = {
            "example_id": example.example_id,
            "correct": same_place_match and issue_resolved_match,
            "confidence": prediction.confidence,
        }
        records.append(record)
        if not record["correct"]:
            mismatches.append(
                {
                    "example_id": example.example_id,
                    "expected_same_place": example.expected_same_place,
                    "predicted_same_place": prediction.same_place.value,
                    "expected_issue_resolved": example.expected_issue_resolved,
                    "predicted_issue_resolved": prediction.issue_resolved.value,
                    "confidence": prediction.confidence,
                }
            )

    accuracy = _ratio(sum(1 for record in records if bool(record["correct"])), len(records))
    calibration_rows, recommended_threshold = _build_calibration_rows(records)
    return {
        "task": "verification",
        "dataset_path": str(dataset_path),
        "total_cases": len(records),
        "correct_cases": sum(1 for record in records if bool(record["correct"])),
        "accuracy": _round_metric(accuracy),
        "average_confidence": _round_metric(
            sum(float(record["confidence"]) for record in records) / max(1, len(records))
        ),
        "component_metrics": {
            "same_place_accuracy": _round_metric(_ratio(same_place_correct, len(records))),
            "issue_resolved_accuracy": _round_metric(
                _ratio(issue_resolved_correct, len(records))
            ),
        },
        "calibration": calibration_rows,
        "recommended_threshold": recommended_threshold,
        "acceptance_gate": _build_acceptance_gate(
            task="verification",
            metric="accuracy",
            minimum=settings.eval_verification_min_accuracy,
            observed=accuracy,
        ),
        "mismatches": mismatches[:10],
    }


def _artifact_paths(settings: Settings, *, ensure_exists: bool) -> tuple[Path, Path]:
    artifact_dir = _normalize_project_path(
        settings.eval_artifact_dir,
        default=DEFAULT_EVAL_ARTIFACT_DIR,
    )
    if ensure_exists:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    return (
        artifact_dir / LATEST_REPORT_FILENAME,
        artifact_dir / LATEST_CALIBRATION_FILENAME,
    )


def run_evaluation_suite(
    settings: Settings | None = None,
    *,
    provider: AIProvider | None = None,
) -> dict[str, object]:
    settings = settings or Settings()
    provider = provider or create_provider(settings)
    dataset_dir = _normalize_project_path(settings.eval_dataset_dir, default=DEFAULT_EVAL_DATASET_DIR)
    report_path, calibration_path = _artifact_paths(settings, ensure_exists=True)

    tasks = {
        "intake": _evaluate_intake(provider, settings),
        "routing": _evaluate_routing(provider, settings),
        "priority": _evaluate_priority(provider, settings),
        "verification": _evaluate_verification(provider, settings),
    }
    overall_passed = all(bool(task["acceptance_gate"]["passed"]) for task in tasks.values())
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "benchmark_name": "asanappeal-regression-suite",
        "generated_at": generated_at,
        "provider": type(provider).__name__,
        "model": getattr(provider, "_model", "unknown"),
        "dataset_dir": str(dataset_dir),
        "artifact_dir": str(report_path.parent),
        "overall_passed": overall_passed,
        "tasks": tasks,
        "report_path": str(report_path),
        "calibration_report_path": str(calibration_path),
    }
    calibration_report = {
        "generated_at": generated_at,
        "provider": type(provider).__name__,
        "model": getattr(provider, "_model", "unknown"),
        "tasks": {
            name: {
                "recommended_threshold": task["recommended_threshold"],
                "calibration": task["calibration"],
            }
            for name, task in tasks.items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration_report, indent=2), encoding="utf-8")
    return report


def load_latest_evaluation_report(settings: Settings | None = None) -> dict[str, object] | None:
    settings = settings or Settings()
    report_path, _ = _artifact_paths(settings, ensure_exists=False)
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None

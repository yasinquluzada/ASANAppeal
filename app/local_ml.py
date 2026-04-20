from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.models.api import SubmissionInput
from app.models.domain import PriorityLevel, RoutingDecision, StructuredIssue


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_BOOTSTRAP_DIR = PROJECT_ROOT / "data" / "local_ml"
DEFAULT_FEEDBACK_DIR = PROJECT_ROOT / ".data" / "local_ml_feedback"
ROUTING_BOOTSTRAP_FILENAME = "routing_examples.jsonl"
PRIORITY_BOOTSTRAP_FILENAME = "priority_examples.jsonl"
ROUTING_FEEDBACK_FILENAME = "routing_feedback.jsonl"
PRIORITY_FEEDBACK_FILENAME = "priority_feedback.jsonl"
RETRAIN_REPORT_FILENAME = "retrain_report.json"
ROUTING_LABELS = {
    "road_damage",
    "street_lighting",
    "water_infrastructure",
    "waste_management",
    "tree_maintenance",
    "signage_safety",
    "public_transport",
    "general_public_service",
}
PRIORITY_LABELS = {level.value for level in PriorityLevel}


@dataclass(frozen=True)
class TrainingExample:
    text: str
    label: str
    source: str
    record_id: str | None = None


@dataclass(frozen=True)
class FeedbackExportArtifact:
    task: str
    output_path: str
    exported_examples: int


@dataclass(frozen=True)
class FeedbackExportResult:
    exported_at: str
    sqlite_path: str
    feedback_dir: str
    routing: FeedbackExportArtifact
    priority: FeedbackExportArtifact


def _clip_confidence(value: float) -> float:
    return round(max(0.05, min(value, 0.99)), 2)


def _normalize_parts(parts: list[str | None]) -> str:
    return " ".join(part.strip().lower() for part in parts if part and part.strip())


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) >= 3]


def _submission_text(
    submission: SubmissionInput,
    structured_issue: StructuredIssue | None = None,
    routing: RoutingDecision | None = None,
) -> str:
    evidence_text = " ".join(
        filter(None, [item.description or item.filename or item.uri or "" for item in submission.evidence])
    )
    return _normalize_parts(
        [
            submission.citizen_text,
            submission.location_hint,
            submission.time_hint,
            evidence_text,
            structured_issue.summary if structured_issue else None,
            structured_issue.category if structured_issue else None,
            routing.category if routing else None,
        ]
    )


def _normalize_project_path(value: str | None, *, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_jsonl_examples(path: Path, *, allowed_labels: set[str]) -> list[TrainingExample]:
    if not path.exists():
        logger.warning("Local ML corpus file %s was not found.", path)
        return []
    examples: list[TrainingExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping invalid JSONL row %s in %s: %s", line_number, path, exc)
            continue
        if not isinstance(payload, dict):
            logger.warning("Skipping non-object JSONL row %s in %s.", line_number, path)
            continue
        text = str(payload.get("text", "")).strip()
        label = str(payload.get("label", "")).strip()
        if not text or label not in allowed_labels:
            continue
        examples.append(
            TrainingExample(
                text=text,
                label=label,
                source=str(payload.get("source") or "bootstrap_corpus"),
                record_id=str(payload["record_id"]) if "record_id" in payload else None,
            )
        )
    return examples


def _write_jsonl_examples(path: Path, examples: list[TrainingExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_rows = [
        json.dumps(
            {
                "text": example.text,
                "label": example.label,
                "source": example.source,
                "record_id": example.record_id,
            },
            ensure_ascii=False,
        )
        for example in _dedupe_examples(examples)
    ]
    payload = "\n".join(serialized_rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
    return len(serialized_rows)


def _dedupe_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    seen: set[tuple[str, str]] = set()
    deduped: list[TrainingExample] = []
    for example in examples:
        normalized_text = _normalize_parts([example.text])
        key = (normalized_text, example.label)
        if not normalized_text or key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def _safe_json_load(raw_value: str | None) -> dict[str, object]:
    if not raw_value:
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_training_text_from_case_payload(
    *,
    request_payload: dict[str, object],
    case_payload: dict[str, object],
) -> str:
    submission = request_payload.get("submission")
    if not isinstance(submission, dict):
        submission = {}
    evidence = submission.get("evidence")
    evidence_items = evidence if isinstance(evidence, list) else []
    evidence_text = " ".join(
        str(item.get("description") or item.get("filename") or item.get("uri") or "")
        for item in evidence_items
        if isinstance(item, dict)
    )

    structured_issue = case_payload.get("structured_issue")
    if not isinstance(structured_issue, dict):
        structured_issue = {}
    routing = case_payload.get("routing")
    if not isinstance(routing, dict):
        routing = {}
    priority = case_payload.get("priority")
    if not isinstance(priority, dict):
        priority = {}

    return _normalize_parts(
        [
            str(submission.get("citizen_text") or ""),
            str(submission.get("location_hint") or ""),
            str(submission.get("time_hint") or ""),
            evidence_text,
            str(structured_issue.get("summary") or ""),
            str(structured_issue.get("category") or ""),
            str(routing.get("category") or ""),
            str(priority.get("level") or ""),
        ]
    )


def _sqlite_feedback_examples(
    settings: Settings,
    *,
    task: str,
    allowed_labels: set[str],
    force_export: bool = False,
) -> list[TrainingExample]:
    if not force_export and not settings.local_ml_include_sqlite_feedback:
        return []
    if settings.repository_backend != "sqlite":
        return []

    sqlite_path_raw = settings.sqlite_path.strip()
    if not sqlite_path_raw or sqlite_path_raw == ":memory:":
        return []

    sqlite_path = _normalize_project_path(sqlite_path_raw, default=PROJECT_ROOT / "asanappeal.db")
    if not sqlite_path.exists():
        return []

    try:
        connection = sqlite3.connect(sqlite_path)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        logger.warning("Unable to open local ML feedback database %s: %s", sqlite_path, exc)
        return []

    if task == "routing":
        label_column = "category"
    else:
        label_column = "priority_level"

    query = f"""
        SELECT
            case_id,
            status,
            reviewer_id,
            final_disposition,
            {label_column} AS label_value,
            case_json,
            request_json
        FROM cases
        WHERE {label_column} IS NOT NULL
          AND (
              reviewer_id IS NOT NULL
              OR final_disposition IN ('dispatched', 'resolved', 'closed', 'completed')
              OR status IN ('assigned', 'in_progress', 'resolved', 'closed')
          )
        ORDER BY created_at DESC
    """
    try:
        rows = connection.execute(query).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Unable to query local ML feedback rows from %s: %s", sqlite_path, exc)
        connection.close()
        return []
    finally:
        connection.close()

    examples: list[TrainingExample] = []
    for row in rows:
        label = str(row["label_value"] or "").strip()
        if label not in allowed_labels:
            continue
        request_payload = _safe_json_load(row["request_json"])
        case_payload = _safe_json_load(row["case_json"])
        text = _build_training_text_from_case_payload(
            request_payload=request_payload,
            case_payload=case_payload,
        )
        if not text:
            continue
        examples.append(
            TrainingExample(
                text=text,
                label=label,
                source="sqlite_reviewed_cases",
                record_id=str(row["case_id"]),
            )
        )
    return examples


def _feedback_export_examples(
    settings: Settings,
    *,
    task: str,
) -> list[TrainingExample]:
    feedback_dir = _normalize_project_path(
        settings.local_ml_feedback_dir,
        default=DEFAULT_FEEDBACK_DIR,
    )
    if task == "routing":
        feedback_file = feedback_dir / ROUTING_FEEDBACK_FILENAME
        allowed_labels = ROUTING_LABELS
    else:
        feedback_file = feedback_dir / PRIORITY_FEEDBACK_FILENAME
        allowed_labels = PRIORITY_LABELS
    return _load_jsonl_examples(feedback_file, allowed_labels=allowed_labels)


def _training_examples_for_task(
    settings: Settings,
    *,
    task: str,
) -> list[TrainingExample]:
    bootstrap_dir = _normalize_project_path(
        settings.local_ml_bootstrap_dir,
        default=DEFAULT_BOOTSTRAP_DIR,
    )
    if task == "routing":
        bootstrap_file = bootstrap_dir / ROUTING_BOOTSTRAP_FILENAME
        allowed_labels = ROUTING_LABELS
    else:
        bootstrap_file = bootstrap_dir / PRIORITY_BOOTSTRAP_FILENAME
        allowed_labels = PRIORITY_LABELS
    examples = _load_jsonl_examples(bootstrap_file, allowed_labels=allowed_labels)
    examples.extend(_feedback_export_examples(settings, task=task))
    examples.extend(
        _sqlite_feedback_examples(
            settings,
            task=task,
            allowed_labels=allowed_labels,
        )
    )
    return _dedupe_examples(examples)


def load_training_examples(settings: Settings, *, task: str) -> list[TrainingExample]:
    return list(_training_examples_for_task(settings, task=task))


def export_reviewed_feedback(settings: Settings) -> FeedbackExportResult:
    if settings.repository_backend != "sqlite":
        raise RuntimeError("Local ML retraining from stored cases requires the SQLite repository backend.")

    feedback_dir = _normalize_project_path(
        settings.local_ml_feedback_dir,
        default=DEFAULT_FEEDBACK_DIR,
    )
    feedback_dir.mkdir(parents=True, exist_ok=True)

    routing_examples = [
        TrainingExample(
            text=example.text,
            label=example.label,
            source="exported_sqlite_feedback",
            record_id=example.record_id,
        )
        for example in _sqlite_feedback_examples(
            settings,
            task="routing",
            allowed_labels=ROUTING_LABELS,
            force_export=True,
        )
    ]
    priority_examples = [
        TrainingExample(
            text=example.text,
            label=example.label,
            source="exported_sqlite_feedback",
            record_id=example.record_id,
        )
        for example in _sqlite_feedback_examples(
            settings,
            task="priority",
            allowed_labels=PRIORITY_LABELS,
            force_export=True,
        )
    ]

    routing_path = feedback_dir / ROUTING_FEEDBACK_FILENAME
    priority_path = feedback_dir / PRIORITY_FEEDBACK_FILENAME
    routing_count = _write_jsonl_examples(routing_path, routing_examples)
    priority_count = _write_jsonl_examples(priority_path, priority_examples)

    sqlite_path = _normalize_project_path(
        settings.sqlite_path.strip(),
        default=PROJECT_ROOT / "asanappeal.db",
    )
    exported_at = datetime.now(timezone.utc).isoformat()
    result = FeedbackExportResult(
        exported_at=exported_at,
        sqlite_path=str(sqlite_path),
        feedback_dir=str(feedback_dir),
        routing=FeedbackExportArtifact(
            task="routing",
            output_path=str(routing_path),
            exported_examples=routing_count,
        ),
        priority=FeedbackExportArtifact(
            task="priority",
            output_path=str(priority_path),
            exported_examples=priority_count,
        ),
    )
    (feedback_dir / RETRAIN_REPORT_FILENAME).write_text(
        json.dumps(
            {
                "exported_at": result.exported_at,
                "sqlite_path": result.sqlite_path,
                "feedback_dir": result.feedback_dir,
                "routing": {
                    "task": result.routing.task,
                    "output_path": result.routing.output_path,
                    "exported_examples": result.routing.exported_examples,
                },
                "priority": {
                    "task": result.priority.task,
                    "output_path": result.priority.output_path,
                    "exported_examples": result.priority.exported_examples,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


@dataclass(frozen=True)
class TextPrediction:
    label: str
    confidence: float
    scorecard: dict[str, float]


class NaiveBayesTextClassifier:
    def __init__(self, examples: list[TrainingExample]) -> None:
        self.examples = examples
        self.labels: set[str] = set()
        self.label_doc_counts: Counter[str] = Counter()
        self.label_token_totals: Counter[str] = Counter()
        self.label_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()
        self.total_documents = 0
        self._fit()

    def _fit(self) -> None:
        for example in self.examples:
            tokens = _tokenize(example.text)
            if not tokens:
                continue
            label = example.label
            self.labels.add(label)
            self.label_doc_counts[label] += 1
            self.total_documents += 1
            self.label_token_counts[label].update(tokens)
            self.label_token_totals[label] += len(tokens)
            self.vocabulary.update(tokens)

    def predict(self, text: str) -> TextPrediction | None:
        tokens = _tokenize(text)
        if not tokens or not self.labels:
            return None

        vocabulary_size = max(len(self.vocabulary), 1)
        log_scores: dict[str, float] = {}
        for label in self.labels:
            doc_prior = self.label_doc_counts[label] / max(self.total_documents, 1)
            score = math.log(doc_prior or 1e-9)
            token_counts = self.label_token_counts[label]
            total_tokens = self.label_token_totals[label]
            denominator = total_tokens + vocabulary_size
            for token in tokens:
                score += math.log((token_counts[token] + 1) / denominator)
            log_scores[label] = score

        max_log_score = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_log_score) for label, score in log_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {label: value / total for label, value in exp_scores.items()}
        best_label = max(probabilities, key=probabilities.get)

        return TextPrediction(
            label=best_label,
            confidence=_clip_confidence(probabilities[best_label]),
            scorecard={label: round(score, 4) for label, score in probabilities.items()},
        )


class LocalRoutingModel:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.examples = _training_examples_for_task(self.settings, task="routing")
        self.training_examples_count = len(self.examples)
        self.training_sources = sorted({example.source for example in self.examples})
        self.label_counts = dict(sorted(Counter(example.label for example in self.examples).items()))
        self.classifier = NaiveBayesTextClassifier(self.examples)

    def predict(
        self, submission: SubmissionInput, structured_issue: StructuredIssue
    ) -> TextPrediction | None:
        return self.classifier.predict(_submission_text(submission, structured_issue=structured_issue))


class LocalPriorityModel:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.examples = _training_examples_for_task(self.settings, task="priority")
        self.training_examples_count = len(self.examples)
        self.training_sources = sorted({example.source for example in self.examples})
        self.label_counts = dict(sorted(Counter(example.label for example in self.examples).items()))
        self.classifier = NaiveBayesTextClassifier(self.examples)

    def predict(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
    ) -> TextPrediction | None:
        return self.classifier.predict(
            _submission_text(
                submission,
                structured_issue=structured_issue,
                routing=routing,
            )
        )

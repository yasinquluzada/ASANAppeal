from __future__ import annotations

import base64
import json
import logging
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import unquote_to_bytes

from pydantic import BaseModel

from app.models.api import InstitutionResponseInput, SubmissionInput
from app.models.domain import (
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
)
from app.provenance import (
    DRAFT_PROMPT_VERSION,
    EXPLANATION_PROMPT_VERSION,
    INTAKE_PROMPT_VERSION,
    PRIORITY_PROMPT_VERSION,
    ROUTING_PROMPT_VERSION,
    STRUCTURED_OUTPUT_SCHEMA_VERSION,
    THRESHOLD_VERSION_NOT_APPLICABLE,
    VERIFICATION_PROMPT_VERSION,
)
from app.providers.heuristic import CATEGORY_KEYWORDS, ROUTING_TABLE, HeuristicAIProvider
from app.video_processing import video_context_from_item

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredProviderBase(HeuristicAIProvider, ABC):
    provider_label = "LLM"

    def __init__(self) -> None:
        self._heuristic = HeuristicAIProvider()

    def analyze_submission(self, submission: SubmissionInput) -> StructuredIssue:
        sections = [
            (
                "Submission JSON",
                self._json_block(
                    {
                        "citizen_text": submission.citizen_text,
                        "language": submission.language,
                        "location_hint": submission.location_hint,
                        "time_hint": submission.time_hint,
                    }
                ),
            ),
            (
                "Allowed categories",
                self._json_block({key: list(value) for key, value in CATEGORY_KEYWORDS.items()}),
            ),
        ]
        return self._call_or_fallback(
            stage="intake",
            schema=StructuredIssue,
            system_prompt=(
                "You are the intake-understanding engine for ASANAppeal, a civic appeals platform. "
                "Analyze the citizen submission and any image evidence. "
                "Return a StructuredIssue object only. "
                "Choose the closest category from the allowed categories. "
                "If the evidence is ambiguous, use category='general_public_service'. "
                "Keep summary concise, extracted_signals concrete, missing_information actionable, "
                "and confidence between 0.05 and 0.99."
            ),
            sections=sections,
            evidence=submission.evidence,
            fallback=lambda: self._heuristic.analyze_submission(submission),
        )

    def route_issue(
        self, submission: SubmissionInput, structured_issue: StructuredIssue
    ) -> RoutingDecision:
        sections = [
            ("Submission JSON", self._json_block(submission.model_dump(mode="json"))),
            (
                "Structured issue JSON",
                self._json_block(structured_issue.model_dump(mode="json")),
            ),
            (
                "Routing table",
                self._json_block(
                    {
                        category: {
                            "institution": institution,
                            "department": department,
                        }
                        for category, (institution, department) in ROUTING_TABLE.items()
                    }
                ),
            ),
        ]
        return self._call_or_fallback(
            stage="routing",
            schema=RoutingDecision,
            system_prompt=(
                "You are the routing engine for ASANAppeal. "
                "Use the structured issue and submission context to select the most appropriate "
                "institution and department from the routing table. "
                "Return a RoutingDecision object only. "
                "The chosen category must match the structured issue category unless there is a "
                "clear reason to fall back to general_public_service."
            ),
            sections=sections,
            fallback=lambda: self._heuristic.route_issue(submission, structured_issue),
        )

    def assess_priority(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
    ) -> PriorityDecision:
        sections = [
            ("Submission JSON", self._json_block(submission.model_dump(mode="json"))),
            (
                "Structured issue JSON",
                self._json_block(structured_issue.model_dump(mode="json")),
            ),
            ("Routing JSON", self._json_block(routing.model_dump(mode="json"))),
            (
                "Priority policy",
                self._json_block(
                    {
                        "low": "Minor service issue with low immediate risk.",
                        "medium": "Meaningful public inconvenience or moderate safety implications.",
                        "high": "Clear safety risk or major disruption requiring fast response.",
                        "critical": "Severe immediate danger, injury risk, or urgent escalation.",
                    }
                ),
            ),
        ]
        return self._call_or_fallback(
            stage="priority",
            schema=PriorityDecision,
            system_prompt=(
                "You are the priority assessment engine for ASANAppeal. "
                "Score urgency from 0 to 100 and map it to low, medium, high, or critical. "
                "Use critical only for severe immediate danger. "
                "Set requires_human_review=true whenever the case is critical, the evidence is weak, "
                "or the classification is materially uncertain. "
                "Return a PriorityDecision object only."
            ),
            sections=sections,
            evidence=submission.evidence,
            fallback=lambda: self._heuristic.assess_priority(
                submission, structured_issue, routing
            ),
        )

    def draft_appeal(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
    ) -> DraftAppeal:
        sections = [
            ("Submission JSON", self._json_block(submission.model_dump(mode="json"))),
            (
                "Structured issue JSON",
                self._json_block(structured_issue.model_dump(mode="json")),
            ),
            ("Routing JSON", self._json_block(routing.model_dump(mode="json"))),
            ("Priority JSON", self._json_block(priority.model_dump(mode="json"))),
        ]
        return self._call_or_fallback(
            stage="draft",
            schema=DraftAppeal,
            system_prompt=(
                "You write high-quality editable civic appeal drafts for ASANAppeal. "
                "Return a DraftAppeal object only. "
                "Keep the title concise and concrete. "
                "Write a professional but plain-language body the citizen can review before submission. "
                "The citizen_review_checklist should contain 3 to 5 actionable checks."
            ),
            sections=sections,
            evidence=submission.evidence,
            fallback=lambda: self._heuristic.draft_appeal(
                submission, structured_issue, routing, priority
            ),
        )

    def verify_resolution(
        self,
        original_submission: SubmissionInput,
        structured_issue: StructuredIssue,
        institution_response: InstitutionResponseInput,
    ) -> VerificationDecision:
        sections = [
            (
                "Original submission JSON",
                self._json_block(original_submission.model_dump(mode="json")),
            ),
            (
                "Structured issue JSON",
                self._json_block(structured_issue.model_dump(mode="json")),
            ),
            (
                "Institution response JSON",
                self._json_block(institution_response.model_dump(mode="json")),
            ),
        ]
        evidence = list(original_submission.evidence) + list(institution_response.evidence)
        return self._call_or_fallback(
            stage="verification",
            schema=VerificationDecision,
            system_prompt=(
                "You are the verification engine for ASANAppeal. "
                "Compare the citizen's original evidence with the institution's response evidence. "
                "Return a VerificationDecision object only. "
                "same_place answers whether the response appears to show the same location. "
                "issue_resolved answers whether the reported issue appears resolved. "
                "Use uncertain when the evidence is incomplete or conflicting. "
                "Keep mismatch_flags in short snake_case tokens."
            ),
            sections=sections,
            evidence=evidence,
            fallback=lambda: self._heuristic.verify_resolution(
                original_submission, structured_issue, institution_response
            ),
        )

    def explain_case(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        human_review: HumanReviewTask,
        verification: VerificationDecision | None = None,
    ) -> ExplanationNote:
        sections = [
            (
                "Structured issue JSON",
                self._json_block(structured_issue.model_dump(mode="json")),
            ),
            ("Routing JSON", self._json_block(routing.model_dump(mode="json"))),
            ("Priority JSON", self._json_block(priority.model_dump(mode="json"))),
            ("Human review JSON", self._json_block(human_review.model_dump(mode="json"))),
        ]
        if verification is not None:
            sections.append(
                ("Verification JSON", self._json_block(verification.model_dump(mode="json")))
            )
        return self._call_or_fallback(
            stage="explanation",
            schema=ExplanationNote,
            system_prompt=(
                "You are the explanation engine for ASANAppeal. "
                "Summarize the case outcome clearly for operators. "
                "Return an ExplanationNote object only. "
                "The summary must be one short paragraph. "
                "next_action must be a concrete operational step."
            ),
            sections=sections,
            fallback=lambda: self._heuristic.explain_case(
                structured_issue, routing, priority, human_review, verification
            ),
        )

    def _call_or_fallback(
        self,
        *,
        stage: str,
        schema: type[SchemaT],
        system_prompt: str,
        sections: list[tuple[str, str]],
        fallback: Callable[[], SchemaT],
        evidence: list[Any] | None = None,
    ) -> SchemaT:
        try:
            result = self._structured_completion(
                schema=schema,
                system_prompt=system_prompt,
                sections=sections,
                evidence=evidence or [],
            )
            self._record_stage_provenance(
                stage,
                provider=type(self).__name__,
                engine="structured-llm",
                model_name=getattr(self, "_model", "unknown"),
                model_version=getattr(self, "_model", "unknown"),
                prompt_version={
                    "intake": INTAKE_PROMPT_VERSION,
                    "routing": ROUTING_PROMPT_VERSION,
                    "priority": PRIORITY_PROMPT_VERSION,
                    "draft": DRAFT_PROMPT_VERSION,
                    "verification": VERIFICATION_PROMPT_VERSION,
                    "explanation": EXPLANATION_PROMPT_VERSION,
                }[stage],
                classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
                threshold_set_version=THRESHOLD_VERSION_NOT_APPLICABLE,
                notes=[f"provider_label={self.provider_label}"],
            )
            return result
        except Exception as exc:  # pragma: no cover - network/sdk failures are environment-specific
            logger.warning("%s call failed, using heuristic fallback: %s", self.provider_label, exc)
            observer = getattr(self, "_provider_error_observer", None)
            if callable(observer):
                observer(stage=stage, provider_name=type(self).__name__, error=exc)
            result = fallback()
            fallback_provenance = self._heuristic.get_stage_provenance().get(stage)
            if fallback_provenance is not None:
                self._record_stage_provenance(
                    stage,
                    provider=fallback_provenance.provider,
                    engine=fallback_provenance.engine,
                    model_name=fallback_provenance.model_name,
                    model_version=fallback_provenance.model_version,
                    prompt_version=fallback_provenance.prompt_version,
                    classifier_version=fallback_provenance.classifier_version,
                    threshold_set_version=fallback_provenance.threshold_set_version,
                    thresholds=fallback_provenance.thresholds,
                    notes=[*fallback_provenance.notes, f"fallback_from={type(self).__name__}"],
                )
            return result

    @abstractmethod
    def _structured_completion(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        sections: list[tuple[str, str]],
        evidence: list[Any],
    ) -> SchemaT:
        raise NotImplementedError

    def _describe_evidence(self, item: Any, index: int) -> str:
        parts = [
            f"Evidence item {index}",
            f"kind={getattr(item, 'kind', 'unknown')}",
        ]
        if getattr(item, "filename", None):
            parts.append(f"filename={item.filename}")
        if getattr(item, "description", None):
            parts.append(f"description={item.description}")
        if getattr(item, "uri", None):
            parts.append(f"uri={item.uri}")
        metadata = getattr(item, "metadata", None) or {}
        if metadata:
            parts.append(f"metadata={json.dumps(metadata, ensure_ascii=True)}")
        video_context = video_context_from_item(item)
        if video_context is not None:
            parts.extend(video_context.summary_lines())
        return "\n".join(parts)

    def _iter_text_blocks(
        self, sections: list[tuple[str, str]], evidence: list[Any]
    ) -> list[str]:
        content: list[str] = []
        for title, body in sections:
            if body:
                content.append(f"{title}:\n{body}")

        for index, item in enumerate(evidence, start=1):
            descriptor = self._describe_evidence(item, index)
            if descriptor:
                content.append(descriptor)

        if not content:
            return ["No additional context provided."]
        return content

    def _iter_image_payloads(self, evidence: list[Any]) -> list[tuple[bytes, str]]:
        payloads: list[tuple[bytes, str]] = []
        for item in evidence:
            image_payload = self._image_bytes(item)
            if image_payload is not None:
                payloads.append(image_payload)
                continue
            video_context = video_context_from_item(item)
            if video_context is not None:
                payloads.extend(
                    (frame.image_bytes, frame.mime_type) for frame in video_context.frame_samples
                )
        return payloads

    def _image_bytes(self, item: Any) -> tuple[bytes, str] | None:
        kind = getattr(item, "kind", None)
        kind_value = getattr(kind, "value", kind)
        if kind_value != "image":
            return None

        uri = getattr(item, "uri", None)
        filename = getattr(item, "filename", None)
        if uri:
            if uri.startswith("data:image/"):
                return self._decode_data_url(uri)
            if uri.startswith(("http://", "https://")):
                return None
            path = self._resolve_local_path(uri)
            if path:
                return self._read_image_path(path)

        if filename:
            path = self._resolve_local_path(filename)
            if path:
                return self._read_image_path(path)
        return None

    def _resolve_local_path(self, value: str) -> Path | None:
        raw_value = value.replace("file://", "", 1)
        path = Path(raw_value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() and path.is_file():
            return path
        return None

    def _read_image_path(self, path: Path) -> tuple[bytes, str]:
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None or not mime_type.startswith("image/"):
            raise RuntimeError(f"Unsupported image type for {path}.")
        return path.read_bytes(), mime_type

    def _decode_data_url(self, data_url: str) -> tuple[bytes, str]:
        header, encoded = data_url.split(",", 1)
        mime_type = header.split(";", 1)[0].split(":", 1)[1]
        if ";base64" in header:
            return base64.b64decode(encoded), mime_type
        return unquote_to_bytes(encoded), mime_type

    def _path_to_data_url(self, path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None or not mime_type.startswith("image/"):
            raise RuntimeError(f"Unsupported image type for {path}.")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _json_block(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=True)


def has_placeholder_secret(value: str | None) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    return (
        normalized == ""
        or "your_" in normalized
        or normalized.endswith("_here")
        or "api_key_here" in normalized
    )

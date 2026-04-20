from __future__ import annotations

import re

from app.models.domain import (
    DecisionProvenance,
    HumanReviewTask,
    PriorityDecision,
    PriorityLevel,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
    VerificationLabel,
)
from app.provenance import (
    PROMPT_VERSION_NOT_APPLICABLE,
    REVIEW_POLICY_VERSION,
    REVIEW_THRESHOLD_SET_VERSION,
)


class ReviewService:
    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold
        self._last_provenance: DecisionProvenance | None = None

    def evaluate(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        verification: VerificationDecision | None = None,
        manual_reasons: list[str] | None = None,
    ) -> HumanReviewTask:
        reasons: list[str] = []

        if structured_issue.confidence < self.confidence_threshold:
            reasons.append("low_issue_confidence")
        if routing.confidence < self.confidence_threshold:
            reasons.append("low_routing_confidence")
        if priority.requires_human_review:
            reasons.append("priority_requires_review")
        if structured_issue.missing_information:
            reasons.append("missing_case_information")

        if verification:
            if verification.same_place == verification.same_place.uncertain:
                reasons.append("same_place_uncertain")
            if verification.issue_resolved == verification.issue_resolved.uncertain:
                reasons.append("resolution_uncertain")
            reasons.extend(verification.mismatch_flags)

        if manual_reasons:
            reasons.extend(manual_reasons)

        needed = bool(reasons)
        confidence = min(
            structured_issue.confidence,
            routing.confidence,
            priority.confidence,
            verification.confidence if verification else 0.99,
        )
        queue = "triage-review"
        secondary_queues: list[str] = []
        candidate_groups: list[str] = []
        institution_queue = self._institution_queue_name(routing.institution)
        if needed:
            queue, secondary_queues = self._route_review_queue(
                structured_issue=structured_issue,
                routing=routing,
                priority=priority,
                verification=verification,
                reasons=sorted(set(reasons)),
                institution_queue=institution_queue,
            )
            candidate_groups = self._candidate_groups(queue, secondary_queues, institution_queue)

        review_task = HumanReviewTask(
            needed=needed,
            queue=queue,
            reasons=sorted(set(reasons)),
            confidence=round(confidence, 2),
            secondary_queues=secondary_queues,
            candidate_groups=candidate_groups,
            institution_queue=institution_queue,
        )
        self._last_provenance = DecisionProvenance(
            stage="review",
            provider=type(self).__name__,
            engine="rule-policy",
            model_name="human-review-router",
            model_version=REVIEW_POLICY_VERSION,
            prompt_version=PROMPT_VERSION_NOT_APPLICABLE,
            classifier_version=REVIEW_POLICY_VERSION,
            threshold_set_version=REVIEW_THRESHOLD_SET_VERSION,
            thresholds={
                "human_review_confidence_threshold": self.confidence_threshold,
            },
            notes=[
                f"primary_queue={queue}",
                f"needed={needed}",
            ],
        )
        return review_task

    def get_stage_provenance(self) -> dict[str, DecisionProvenance]:
        if self._last_provenance is None:
            return {}
        return {"review": self._last_provenance.model_copy(deep=True)}

    def _route_review_queue(
        self,
        *,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        verification: VerificationDecision | None,
        reasons: list[str],
        institution_queue: str,
    ) -> tuple[str, list[str]]:
        safety_categories = {
            "road_damage",
            "street_lighting",
            "water_infrastructure",
            "tree_maintenance",
            "signage_safety",
        }
        evidence_reasons = {
            "missing_case_information",
            "same_place_uncertain",
            "resolution_uncertain",
            "weak_location_match",
            "missing_response_evidence",
            "weak_resolution_language",
        }
        legal_reasons = {
            "location_mismatch",
        }
        triage_reasons = {
            "low_issue_confidence",
            "low_routing_confidence",
            "reopened_case",
        }

        primary = institution_queue
        if verification and (
            verification.same_place == VerificationLabel.no
            or any(reason in legal_reasons for reason in reasons)
        ):
            primary = "legal-review"
        elif (
            priority.level in {PriorityLevel.high, PriorityLevel.critical}
            and structured_issue.category in safety_categories
        ) or priority.level == PriorityLevel.critical:
            primary = "urgent-safety-review"
        elif any(reason in evidence_reasons for reason in reasons):
            primary = "evidence-quality-review"
        elif (
            structured_issue.category == "general_public_service"
            or routing.category == "general_public_service"
            or any(reason in triage_reasons for reason in reasons)
        ):
            primary = "triage-review"

        secondary: list[str] = []
        for candidate in (
            "legal-review"
            if verification and (
                verification.same_place == VerificationLabel.no
                or any(reason in legal_reasons for reason in reasons)
            )
            else None,
            "urgent-safety-review"
            if (
                priority.level in {PriorityLevel.high, PriorityLevel.critical}
                and structured_issue.category in safety_categories
            ) or priority.level == PriorityLevel.critical
            else None,
            "evidence-quality-review"
            if any(reason in evidence_reasons for reason in reasons)
            else None,
            "triage-review"
            if (
                structured_issue.category == "general_public_service"
                or routing.category == "general_public_service"
                or any(reason in triage_reasons for reason in reasons)
            )
            else None,
            institution_queue,
        ):
            if candidate and candidate != primary and candidate not in secondary:
                secondary.append(candidate)

        return primary, secondary

    def _candidate_groups(
        self,
        primary_queue: str,
        secondary_queues: list[str],
        institution_queue: str,
    ) -> list[str]:
        groups: list[str] = []
        for queue in [primary_queue, *secondary_queues]:
            if queue == "triage-review":
                groups.append("review-triage")
            elif queue == "legal-review":
                groups.append("review-legal")
            elif queue == "evidence-quality-review":
                groups.append("review-evidence")
            elif queue == "urgent-safety-review":
                groups.append("review-urgent-safety")
            elif queue == institution_queue:
                slug = institution_queue.removeprefix("institution-review:")
                groups.extend([f"review-institution-{slug}", f"institution-{slug}"])
        return sorted(dict.fromkeys(groups))

    def _institution_queue_name(self, institution: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", institution.lower()).strip("-") or "general"
        return f"institution-review:{slug}"

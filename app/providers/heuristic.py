from __future__ import annotations

import math
import re
from collections import defaultdict

from app.config import Settings
from app.models.api import InstitutionResponseInput, SubmissionInput
from app.models.domain import (
    DecisionProvenance,
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    PriorityLevel,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
    VerificationLabel,
)
from app.provenance import (
    HEURISTIC_RULESET_VERSION,
    HEURISTIC_THRESHOLD_SET_VERSION,
    PROMPT_VERSION_NOT_APPLICABLE,
    THRESHOLD_VERSION_NOT_APPLICABLE,
    VERIFICATION_POLICY_VERSION,
    VERIFICATION_THRESHOLD_SET_VERSION,
)
from app.verification_reasoning import POLICY_BY_CATEGORY, verify_resolution_advanced


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "road_damage": (
        "pothole",
        "crack",
        "asphalt",
        "road",
        "sidewalk",
        "pavement",
        "hole",
        "street damage",
    ),
    "street_lighting": (
        "streetlight",
        "street light",
        "lamp",
        "lighting",
        "dark",
        "pole light",
    ),
    "water_infrastructure": (
        "water",
        "pipe",
        "drain",
        "sewer",
        "leak",
        "flood",
        "overflow",
    ),
    "waste_management": (
        "trash",
        "garbage",
        "waste",
        "dump",
        "overflowing bin",
        "unclean",
    ),
    "tree_maintenance": (
        "tree",
        "branch",
        "fallen tree",
        "park",
        "greenery",
        "bush",
    ),
    "signage_safety": (
        "traffic light",
        "signal",
        "sign",
        "crosswalk",
        "zebra crossing",
        "barrier",
    ),
    "public_transport": (
        "bus",
        "stop",
        "metro",
        "tram",
        "public transport",
        "station",
    ),
}

ROUTING_TABLE: dict[str, tuple[str, str]] = {
    "road_damage": ("ASAN Road Maintenance Agency", "Road Surface Response"),
    "street_lighting": ("City Lighting Department", "Lighting Fault Response"),
    "water_infrastructure": ("Water and Sewer Authority", "Leak and Drainage Desk"),
    "waste_management": ("Municipal Sanitation Department", "Cleanliness Response"),
    "tree_maintenance": ("Parks and Greenery Department", "Tree Safety Unit"),
    "signage_safety": ("Traffic Management Authority", "Road Safety Operations"),
    "public_transport": ("Transit Operations Agency", "Public Transport Support"),
    "general_public_service": ("ASAN Operations Triage Desk", "Manual Classification"),
}

CRITICAL_KEYWORDS = {
    "collapse",
    "sinkhole",
    "injury",
    "injured",
    "fire",
    "electrical",
    "exposed wire",
    "accident",
    "ambulance",
}
HIGH_PRIORITY_KEYWORDS = {
    "danger",
    "dangerous",
    "deep",
    "large",
    "night",
    "school",
    "hospital",
    "bridge",
    "main road",
    "blocked",
}
MEDIUM_PRIORITY_KEYWORDS = {
    "busy",
    "traffic",
    "crowded",
    "children",
    "elderly",
    "rain",
    "slippery",
}

RESOLVED_WORDS = {
    "fixed",
    "repaired",
    "resolved",
    "completed",
    "cleaned",
    "restored",
    "removed",
    "replaced",
    "patched",
}
UNRESOLVED_WORDS = {
    "pending",
    "scheduled",
    "awaiting",
    "not fixed",
    "inspection",
    "will review",
    "will inspect",
}


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def _slug_to_title(value: str) -> str:
    return value.replace("_", " ").title()


def _clip_confidence(value: float) -> float:
    return round(max(0.05, min(value, 0.99)), 2)


LOCATION_STOPWORDS = {
    "the",
    "and",
    "near",
    "next",
    "beside",
    "street",
    "road",
    "area",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in LOCATION_STOPWORDS
    }


def _first_sentence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return parts[0][:220]


class HeuristicAIProvider:
    """Offline-friendly AI behavior using deterministic civic-case heuristics."""

    def _provenance_state(self) -> dict[str, DecisionProvenance]:
        state = getattr(self, "_stage_provenance", None)
        if state is None:
            state = {}
            self._stage_provenance = state
        return state

    def clear_stage_provenance(self) -> None:
        self._stage_provenance = {}

    def get_stage_provenance(self) -> dict[str, DecisionProvenance]:
        return {
            stage: provenance.model_copy(deep=True)
            for stage, provenance in self._provenance_state().items()
        }

    def _record_stage_provenance(
        self,
        stage: str,
        *,
        provider: str | None = None,
        engine: str,
        model_name: str,
        model_version: str,
        prompt_version: str = PROMPT_VERSION_NOT_APPLICABLE,
        classifier_version: str | None = None,
        threshold_set_version: str = THRESHOLD_VERSION_NOT_APPLICABLE,
        thresholds: dict[str, float | int | str | bool] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self._provenance_state()[stage] = DecisionProvenance(
            stage=stage,
            provider=provider or type(self).__name__,
            engine=engine,
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
            classifier_version=classifier_version,
            threshold_set_version=threshold_set_version,
            thresholds=thresholds or {},
            notes=notes or [],
        )

    def analyze_submission(self, submission: SubmissionInput) -> StructuredIssue:
        evidence_text = " ".join(
            filter(None, [item.description or item.filename or "" for item in submission.evidence])
        )
        combined = _normalize(
            " ".join(
                filter(
                    None,
                    [
                        submission.citizen_text,
                        submission.location_hint,
                        submission.time_hint,
                        evidence_text,
                    ],
                )
            )
        )

        category_scores: dict[str, int] = defaultdict(int)
        matched_signals: list[str] = []
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined:
                    category_scores[category] += 1
                    matched_signals.append(keyword)

        if category_scores:
            category = max(category_scores, key=category_scores.get)
            top_score = category_scores[category]
        else:
            category = "general_public_service"
            top_score = 0

        summary = _first_sentence(submission.citizen_text)
        if not summary:
            summary = (
                f"Citizen reported {_slug_to_title(category)}"
                if category != "general_public_service"
                else "Citizen submitted a public-service appeal"
            )

        issue_type = _slug_to_title(category)
        missing_information: list[str] = []
        if len(submission.citizen_text.strip()) < 12:
            missing_information.append("Add a clearer natural-language description of the problem.")
        if not submission.location_hint:
            missing_information.append("Provide a precise location or landmark.")
        if not submission.time_hint:
            missing_information.append("Provide when the issue was observed.")
        if not submission.evidence:
            missing_information.append("Attach at least one photo or video of the issue.")

        confidence = 0.42
        confidence += min(top_score, 4) * 0.09
        confidence += min(len(submission.evidence), 3) * 0.06
        confidence -= len(missing_information) * 0.05
        if category == "general_public_service":
            confidence -= 0.12

        issue = StructuredIssue(
            category=category,
            issue_type=issue_type,
            summary=summary,
            extracted_signals=sorted(set(matched_signals))[:8],
            missing_information=missing_information,
            confidence=_clip_confidence(confidence),
        )
        self._record_stage_provenance(
            "intake",
            engine="heuristic-rules",
            model_name="heuristic-intake",
            model_version=HEURISTIC_RULESET_VERSION,
            threshold_set_version=HEURISTIC_THRESHOLD_SET_VERSION,
            thresholds={
                "keyword_confidence_bonus": 0.09,
                "evidence_confidence_bonus": 0.06,
                "missing_information_penalty": 0.05,
                "general_public_service_penalty": 0.12,
            },
            notes=["keyword-category-matching"],
        )
        return issue

    def route_issue(
        self, submission: SubmissionInput, structured_issue: StructuredIssue
    ) -> RoutingDecision:
        institution, department = ROUTING_TABLE.get(
            structured_issue.category, ROUTING_TABLE["general_public_service"]
        )

        confidence = 0.58
        if structured_issue.category != "general_public_service":
            confidence += 0.14
        confidence += (structured_issue.confidence - 0.5) * 0.35
        if submission.location_hint:
            confidence += 0.05

        if not submission.location_hint and structured_issue.category == "general_public_service":
            institution, department = ROUTING_TABLE["general_public_service"]

        rationale = (
            f"Matched the case to {_slug_to_title(structured_issue.category)} and selected "
            f"{institution} based on issue semantics and civic-service ownership."
        )
        routing = RoutingDecision(
            institution=institution,
            department=department,
            category=structured_issue.category,
            rationale=rationale,
            confidence=_clip_confidence(confidence),
        )
        self._record_stage_provenance(
            "routing",
            engine="heuristic-rules",
            model_name="heuristic-routing",
            model_version=HEURISTIC_RULESET_VERSION,
            threshold_set_version=HEURISTIC_THRESHOLD_SET_VERSION,
            thresholds={
                "base_confidence": 0.58,
                "known_category_bonus": 0.14,
                "location_hint_bonus": 0.05,
            },
            notes=["routing-table-ownership-match"],
        )
        return routing

    def assess_priority(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
    ) -> PriorityDecision:
        combined = _normalize(
            " ".join(
                filter(
                    None,
                    [submission.citizen_text, submission.location_hint, structured_issue.summary],
                )
            )
        )

        score = 18
        reasons: list[str] = []

        if structured_issue.category == "road_damage":
            score += 22
            reasons.append("Road damage can create direct safety risk for vehicles and pedestrians.")

        critical_hits = sorted({word for word in CRITICAL_KEYWORDS if word in combined})
        high_hits = sorted({word for word in HIGH_PRIORITY_KEYWORDS if word in combined})
        medium_hits = sorted({word for word in MEDIUM_PRIORITY_KEYWORDS if word in combined})

        if critical_hits:
            score += 48
            reasons.append(f"Critical risk language detected: {', '.join(critical_hits)}.")
        if high_hits:
            score += 22
            reasons.append(f"High-priority context detected: {', '.join(high_hits)}.")
        if medium_hits:
            score += 10
            reasons.append(f"Operational urgency context detected: {', '.join(medium_hits)}.")
        if len(submission.evidence) >= 2:
            score += 5
            reasons.append("Multiple evidence items were attached.")
        if submission.location_hint and any(
            marker in submission.location_hint.lower()
            for marker in ("school", "hospital", "bridge", "intersection")
        ):
            score += 10
            reasons.append("Location hint suggests higher public impact.")

        score = max(0, min(score, 100))
        if score >= 85:
            level = PriorityLevel.critical
        elif score >= 65:
            level = PriorityLevel.high
        elif score >= 40:
            level = PriorityLevel.medium
        else:
            level = PriorityLevel.low

        confidence = 0.52
        confidence += 0.08 if structured_issue.category != "general_public_service" else -0.05
        confidence += 0.08 if reasons else 0.0
        confidence += 0.06 if submission.location_hint else 0.0
        confidence += (routing.confidence - 0.5) * 0.2
        requires_human_review = level == PriorityLevel.critical or confidence < 0.6

        if not reasons:
            reasons.append("No explicit danger language was detected, so default service urgency was used.")

        priority = PriorityDecision(
            level=level,
            score=score,
            reasons=reasons,
            confidence=_clip_confidence(confidence),
            requires_human_review=requires_human_review,
        )
        self._record_stage_provenance(
            "priority",
            engine="heuristic-rules",
            model_name="heuristic-priority",
            model_version=HEURISTIC_RULESET_VERSION,
            threshold_set_version=HEURISTIC_THRESHOLD_SET_VERSION,
            thresholds={
                "critical_threshold": 85,
                "high_threshold": 65,
                "medium_threshold": 40,
                "human_review_confidence_floor": 0.6,
            },
            notes=["keyword-urgency-scoring"],
        )
        return priority

    def draft_appeal(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
    ) -> DraftAppeal:
        location = submission.location_hint or "the reported location"
        title = f"{_slug_to_title(structured_issue.category)} reported near {location}"
        body_lines = [
            f"I am reporting {structured_issue.summary.lower()}.",
            f"The issue appears to fall under {_slug_to_title(structured_issue.category)}.",
            f"The location provided is {location}.",
        ]
        if submission.time_hint:
            body_lines.append(f"The issue was observed at or around {submission.time_hint}.")
        if submission.evidence:
            body_lines.append(
                f"I attached {len(submission.evidence)} evidence item(s) to support this appeal."
            )
        body_lines.append(
            f"Please review and assign this case to {routing.institution}. "
            f"The current operational priority is {priority.level.value}."
        )

        checklist = [
            "Verify the location text is precise enough for dispatch.",
            "Confirm the generated category matches the actual problem.",
            "Check that sensitive or private details are not included unnecessarily.",
        ]
        if structured_issue.missing_information:
            checklist.extend(structured_issue.missing_information[:2])

        confidence = (structured_issue.confidence + routing.confidence + priority.confidence) / 3
        draft = DraftAppeal(
            title=title,
            body="\n\n".join(body_lines),
            citizen_review_checklist=checklist,
            confidence=_clip_confidence(confidence),
        )
        self._record_stage_provenance(
            "draft",
            engine="heuristic-rules",
            model_name="heuristic-draft",
            model_version=HEURISTIC_RULESET_VERSION,
            notes=["template-composition"],
        )
        return draft

    def verify_resolution(
        self,
        original_submission: SubmissionInput,
        structured_issue: StructuredIssue,
        institution_response: InstitutionResponseInput,
    ) -> VerificationDecision:
        verification = verify_resolution_advanced(
            original_submission,
            structured_issue,
            institution_response,
            settings=Settings(),
        )
        policy = POLICY_BY_CATEGORY.get(
            structured_issue.category,
            POLICY_BY_CATEGORY["general_public_service"],
        )
        self._record_stage_provenance(
            "verification",
            engine="geo-visual-policy",
            model_name="geo-visual-verifier",
            model_version=VERIFICATION_POLICY_VERSION,
            threshold_set_version=VERIFICATION_THRESHOLD_SET_VERSION,
            thresholds={
                "same_place_yes_score": 0.66,
                "same_place_no_score": 0.28,
                "coordinate_exact_match_m": 150,
                "coordinate_near_match_m": 400,
                "policy_visual_threshold": float(policy["visual_threshold"]),
            },
            notes=[f"policy_owner={policy['institution']}"],
        )
        return verification

    def explain_case(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        human_review: HumanReviewTask,
        verification: VerificationDecision | None = None,
    ) -> ExplanationNote:
        summary = (
            f"{_slug_to_title(structured_issue.category)} case routed to {routing.institution} "
            f"with {priority.level.value} priority."
        )

        detailed_rationale = [
            f"Issue understanding confidence: {structured_issue.confidence:.2f}.",
            f"Routing confidence: {routing.confidence:.2f}.",
            f"Priority confidence: {priority.confidence:.2f}.",
        ]
        risk_flags = list(human_review.reasons)
        next_action = f"Dispatch to {routing.department}."

        if verification:
            detailed_rationale.append(verification.summary)
            risk_flags.extend(verification.mismatch_flags)
        if human_review.needed:
            next_action = f"Send to {human_review.queue} for manual review."

        explanation = ExplanationNote(
            summary=summary,
            next_action=next_action,
            detailed_rationale=detailed_rationale,
            risk_flags=sorted(set(risk_flags)),
        )
        self._record_stage_provenance(
            "explanation",
            engine="heuristic-rules",
            model_name="heuristic-explanation",
            model_version=HEURISTIC_RULESET_VERSION,
            notes=["summary-from-issue-routing-priority"],
        )
        return explanation

from app.models.domain import (
    PriorityDecision,
    PriorityLevel,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
    VerificationLabel,
)
from app.services.review import ReviewService


def _review_service() -> ReviewService:
    return ReviewService(confidence_threshold=0.7)


def test_review_service_routes_to_triage_queue() -> None:
    task = _review_service().evaluate(
        StructuredIssue(
            category="general_public_service",
            issue_type="General Public Service",
            summary="Unclear public service issue.",
            extracted_signals=[],
            missing_information=[],
            confidence=0.42,
        ),
        RoutingDecision(
            institution="ASAN Operations Triage Desk",
            department="Manual Classification",
            category="general_public_service",
            rationale="Low-confidence generic issue.",
            confidence=0.46,
        ),
        PriorityDecision(
            level=PriorityLevel.low,
            score=20,
            reasons=["Low confidence intake."],
            confidence=0.74,
            requires_human_review=False,
        ),
    )

    assert task.needed is True
    assert task.queue == "triage-review"
    assert "review-triage" in task.candidate_groups


def test_review_service_routes_to_urgent_safety_queue() -> None:
    task = _review_service().evaluate(
        StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary="Deep pothole creating direct vehicle hazard.",
            extracted_signals=["pothole", "hospital"],
            missing_information=[],
            confidence=0.82,
        ),
        RoutingDecision(
            institution="ASAN Road Maintenance Agency",
            department="Road Surface Response",
            category="road_damage",
            rationale="Clear road-damage classification.",
            confidence=0.85,
        ),
        PriorityDecision(
            level=PriorityLevel.critical,
            score=95,
            reasons=["Direct traffic safety risk."],
            confidence=0.91,
            requires_human_review=True,
        ),
    )

    assert task.queue == "urgent-safety-review"
    assert "review-urgent-safety" in task.candidate_groups
    assert task.institution_queue == "institution-review:asan-road-maintenance-agency"


def test_review_service_routes_to_legal_queue_for_verification_conflict() -> None:
    task = _review_service().evaluate(
        StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary="Road hazard dispute.",
            extracted_signals=["pothole"],
            missing_information=[],
            confidence=0.84,
        ),
        RoutingDecision(
            institution="ASAN Road Maintenance Agency",
            department="Road Surface Response",
            category="road_damage",
            rationale="Clear route.",
            confidence=0.88,
        ),
        PriorityDecision(
            level=PriorityLevel.medium,
            score=52,
            reasons=["Needs verification."],
            confidence=0.8,
            requires_human_review=False,
        ),
        VerificationDecision(
            same_place=VerificationLabel.no,
            issue_resolved=VerificationLabel.no,
            mismatch_flags=["location_mismatch"],
            summary="Institution response appears to address a different location.",
            confidence=0.86,
        ),
    )

    assert task.queue == "legal-review"
    assert "review-legal" in task.candidate_groups
    assert task.institution_queue == "institution-review:asan-road-maintenance-agency"


def test_review_service_routes_to_evidence_quality_queue() -> None:
    task = _review_service().evaluate(
        StructuredIssue(
            category="street_lighting",
            issue_type="Street Lighting",
            summary="Street light outage needs more evidence.",
            extracted_signals=["lamp"],
            missing_information=["clear_photo_of_fixture"],
            confidence=0.77,
        ),
        RoutingDecision(
            institution="City Lighting Department",
            department="Lighting Fault Response",
            category="street_lighting",
            rationale="Likely lighting issue.",
            confidence=0.8,
        ),
        PriorityDecision(
            level=PriorityLevel.medium,
            score=46,
            reasons=["Moderate visibility risk."],
            confidence=0.75,
            requires_human_review=False,
        ),
    )

    assert task.queue == "evidence-quality-review"
    assert "review-evidence" in task.candidate_groups


def test_review_service_routes_to_institution_specific_queue() -> None:
    task = _review_service().evaluate(
        StructuredIssue(
            category="waste_management",
            issue_type="Waste Management",
            summary="Overflowing bin needs institution follow-up.",
            extracted_signals=["overflowing bin"],
            missing_information=[],
            confidence=0.88,
        ),
        RoutingDecision(
            institution="Municipal Sanitation Department",
            department="Cleanliness Response",
            category="waste_management",
            rationale="Clear sanitation route.",
            confidence=0.9,
        ),
        PriorityDecision(
            level=PriorityLevel.medium,
            score=54,
            reasons=["Manual escalation requested."],
            confidence=0.84,
            requires_human_review=True,
        ),
    )

    assert task.queue == "institution-review:municipal-sanitation-department"
    assert "review-institution-municipal-sanitation-department" in task.candidate_groups

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceKind(str, Enum):
    image = "image"
    video = "video"
    text = "text"


class PriorityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class CaseStatus(str, Enum):
    drafted = "drafted"
    needs_review = "needs_review"
    ready_for_dispatch = "ready_for_dispatch"
    assigned = "assigned"
    in_progress = "in_progress"
    resolved = "resolved"
    reopened = "reopened"
    rejected = "rejected"
    closed = "closed"


class CaseTransition(str, Enum):
    submit_for_review = "submit_for_review"
    mark_dispatch_ready = "mark_dispatch_ready"
    assign = "assign"
    start_progress = "start_progress"
    resolve = "resolve"
    reopen = "reopen"
    reject = "reject"
    close = "close"


class CaseWorkflowAction(str, Enum):
    claim = "claim"
    assign = "assign"
    comment = "comment"
    approve = "approve"
    reject = "reject"
    dispatch = "dispatch"
    close = "close"
    reopen = "reopen"
    verify = "verify"


class VerificationLabel(str, Enum):
    yes = "yes"
    no = "no"
    uncertain = "uncertain"


class AccountRole(str, Enum):
    citizen = "citizen"
    operator = "operator"
    reviewer = "reviewer"
    institution = "institution"
    admin = "admin"


class AuditEventType(str, Enum):
    case_created = "case_created"
    operations_updated = "operations_updated"
    lifecycle_transition = "lifecycle_transition"
    workflow_action = "workflow_action"
    verification_action = "verification_action"
    privacy_export = "privacy_export"
    privacy_delete = "privacy_delete"


class EvidenceItem(BaseModel):
    kind: EvidenceKind = EvidenceKind.image
    evidence_id: str | None = None
    uri: str | None = None
    filename: str | None = None
    description: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    thumbnail_available: bool = False
    width: int | None = None
    height: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    privacy: EvidencePrivacyState = Field(default_factory=lambda: EvidencePrivacyState())


class StoredEvidence(BaseModel):
    evidence_id: str
    kind: EvidenceKind
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filename: str
    description: str | None = None
    mime_type: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    object_path: str
    thumbnail_path: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    privacy: EvidencePrivacyState = Field(default_factory=lambda: EvidencePrivacyState())


class PrivacyTextFinding(BaseModel):
    pii_type: str
    field_path: str
    match_preview: str
    replacement: str


class PrivacyImageRegion(BaseModel):
    pii_type: str
    left: int
    top: int
    width: int
    height: int
    detector: str = "heuristic"
    confidence: float = 1.0


class EvidencePrivacyState(BaseModel):
    pii_detected: bool = False
    redaction_applied: bool = False
    address_minimized: bool = False
    text_findings: list[PrivacyTextFinding] = Field(default_factory=list)
    image_regions: list[PrivacyImageRegion] = Field(default_factory=list)
    retention_delete_after: datetime | None = None
    exported_at: datetime | None = None
    deleted_at: datetime | None = None
    source_sha256: str | None = None


class CasePrivacyState(BaseModel):
    pii_detected: bool = False
    redaction_applied: bool = False
    address_minimized: bool = False
    redacted_field_paths: list[str] = Field(default_factory=list)
    text_findings: list[PrivacyTextFinding] = Field(default_factory=list)
    masked_evidence_ids: list[str] = Field(default_factory=list)
    deleted_evidence_ids: list[str] = Field(default_factory=list)
    case_delete_after: datetime | None = None
    evidence_delete_after: datetime | None = None
    exported_at: datetime | None = None
    last_export_path: str | None = None
    deleted_at: datetime | None = None


class StructuredIssue(BaseModel):
    category: str
    issue_type: str
    summary: str
    extracted_signals: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float


class RoutingDecision(BaseModel):
    institution: str
    department: str
    category: str
    rationale: str
    confidence: float


class PriorityDecision(BaseModel):
    level: PriorityLevel
    score: int
    reasons: list[str] = Field(default_factory=list)
    confidence: float
    requires_human_review: bool = False


class DraftAppeal(BaseModel):
    title: str
    body: str
    citizen_review_checklist: list[str] = Field(default_factory=list)
    confidence: float


class VerificationDecision(BaseModel):
    same_place: VerificationLabel
    issue_resolved: VerificationLabel
    mismatch_flags: list[str] = Field(default_factory=list)
    summary: str
    confidence: float


class ExplanationNote(BaseModel):
    summary: str
    next_action: str
    detailed_rationale: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class HumanReviewTask(BaseModel):
    needed: bool
    queue: str
    reasons: list[str] = Field(default_factory=list)
    confidence: float
    secondary_queues: list[str] = Field(default_factory=list)
    candidate_groups: list[str] = Field(default_factory=list)
    institution_queue: str | None = None


class CaseAnnotation(BaseModel):
    annotated_by: str | None = None
    annotated_at: datetime | None = None
    correct_category: str | None = None
    correct_institution: str | None = None
    correct_department: str | None = None
    correct_priority_level: str | None = None
    correct_same_place: VerificationLabel | None = None
    correct_issue_resolved: VerificationLabel | None = None
    note: str | None = None
    source: str = "reviewer"


class DecisionProvenance(BaseModel):
    stage: str
    provider: str = "unknown"
    engine: str = "unknown"
    model_name: str = "unknown"
    model_version: str = "unknown"
    prompt_version: str = "not_applicable"
    classifier_version: str | None = None
    threshold_set_version: str = "not_applicable"
    thresholds: dict[str, float | int | str | bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CaseModelContext(BaseModel):
    provider: str = "unknown"
    model_name: str = "unknown"
    model_version: str = "unknown"
    provenance_schema_version: str = "case-provenance.v1"
    prompt_bundle_version: str = "not_applicable"
    classifier_bundle_version: str = "not_applicable"
    threshold_set_version: str = "not_applicable"
    stage_provenance: dict[str, DecisionProvenance] = Field(default_factory=dict)


class CaseTransitionEntry(BaseModel):
    transition: CaseTransition
    from_status: CaseStatus
    to_status: CaseStatus
    acted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str | None = None
    note: str | None = None


class CaseWorkflowActionEntry(BaseModel):
    action: CaseWorkflowAction
    resulting_status: CaseStatus
    acted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str | None = None
    note: str | None = None
    assignee_id: str | None = None


class CaseOperationalState(BaseModel):
    submitted_by_user_id: str | None = None
    submitted_by_role: AccountRole | None = None
    reviewer_id: str | None = None
    final_disposition: str = "open"
    final_disposition_reason: str | None = None
    status_updated_at: datetime | None = None
    disposition_updated_at: datetime | None = None
    transition_history: list[CaseTransitionEntry] = Field(default_factory=list)
    workflow_history: list[CaseWorkflowActionEntry] = Field(default_factory=list)


class CaseRecord(BaseModel):
    case_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: CaseStatus
    submission_excerpt: str
    structured_issue: StructuredIssue
    routing: RoutingDecision
    priority: PriorityDecision
    draft: DraftAppeal
    last_institution_response: dict[str, object] | None = None
    verification: VerificationDecision | None = None
    explanation: ExplanationNote
    annotation: CaseAnnotation | None = None
    human_review: HumanReviewTask
    privacy: CasePrivacyState = Field(default_factory=lambda: CasePrivacyState())
    model_context: CaseModelContext = Field(default_factory=CaseModelContext)
    operations: CaseOperationalState = Field(default_factory=CaseOperationalState)


class AuthenticatedUser(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: AccountRole
    institution_slug: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditEventDraft(BaseModel):
    case_id: str
    event_type: AuditEventType
    event_source: str
    summary: str
    actor_id: str | None = None
    actor_role: AccountRole | None = None
    actor_username: str | None = None
    status_before: str | None = None
    status_after: str | None = None
    human_override: bool = False
    ai_snapshot_hash: str | None = None
    override_snapshot_hash: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class AuditLogEvent(BaseModel):
    event_id: str
    case_id: str
    occurred_at: datetime
    event_type: AuditEventType
    event_source: str
    summary: str
    actor_id: str | None = None
    actor_role: AccountRole | None = None
    actor_username: str | None = None
    status_before: str | None = None
    status_after: str | None = None
    human_override: bool = False
    ai_snapshot_hash: str | None = None
    override_snapshot_hash: str | None = None
    previous_event_id: str | None = None
    previous_event_hash: str | None = None
    event_hash: str
    payload: dict[str, object] = Field(default_factory=dict)


class AuditChainVerification(BaseModel):
    verified: bool
    checked_events: int
    latest_event_id: str | None = None
    latest_event_hash: str | None = None
    scope: str = "global"
    failure_event_id: str | None = None
    failure_reason: str | None = None

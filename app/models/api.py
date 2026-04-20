from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.domain import (
    AccountRole,
    AuditChainVerification,
    AuditLogEvent,
    AuthenticatedUser,
    CaseAnnotation,
    CaseRecord,
    CaseModelContext,
    CaseOperationalState,
    CasePrivacyState,
    CaseTransition,
    CaseWorkflowAction,
    DraftAppeal,
    EvidenceItem,
    ExplanationNote,
    HumanReviewTask,
    PrivacyTextFinding,
    PriorityDecision,
    RoutingDecision,
    StoredEvidence,
    StructuredIssue,
    VerificationDecision,
)


class SubmissionInput(BaseModel):
    citizen_text: str = ""
    language: str = "en"
    location_hint: str | None = None
    time_hint: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)


class InstitutionResponseInput(BaseModel):
    response_text: str = ""
    location_hint: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)


class RoutingInput(BaseModel):
    submission: SubmissionInput
    structured_issue: StructuredIssue


class PriorityInput(BaseModel):
    submission: SubmissionInput
    structured_issue: StructuredIssue
    routing: RoutingDecision


class DraftRequest(BaseModel):
    submission: SubmissionInput
    structured_issue: StructuredIssue
    routing: RoutingDecision
    priority: PriorityDecision


class VerificationRequest(BaseModel):
    original_submission: SubmissionInput
    structured_issue: StructuredIssue
    institution_response: InstitutionResponseInput


class CaseVerificationRequest(BaseModel):
    institution_response: InstitutionResponseInput
    actor_id: str | None = None
    note: str | None = None


class PrivacyDeleteRequest(BaseModel):
    actor_id: str | None = None
    note: str | None = None


class ExplanationRequest(BaseModel):
    structured_issue: StructuredIssue
    routing: RoutingDecision
    priority: PriorityDecision
    human_review: HumanReviewTask
    verification: VerificationDecision | None = None


class ProcessCaseRequest(BaseModel):
    submission: SubmissionInput
    institution_response: InstitutionResponseInput | None = None


class AuthRegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class AuthAccountCreateRequest(BaseModel):
    username: str
    password: str
    display_name: str
    role: AccountRole
    institution_slug: str | None = None


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    user: AuthenticatedUser


class AuthLogoutResponse(BaseModel):
    revoked: bool


class AuthAccountsResponse(BaseModel):
    items: list[AuthenticatedUser] = Field(default_factory=list)


class StorageBackupRequest(BaseModel):
    label: str | None = None
    destination_path: str | None = None


class StorageRestoreRequest(BaseModel):
    source_path: str
    create_pre_restore_backup: bool = True


class LocalMLRetrainResponse(BaseModel):
    retrained: bool
    exported_at: str
    sqlite_path: str
    feedback_dir: str
    routing_export_path: str
    routing_exported_examples: int
    priority_export_path: str
    priority_exported_examples: int
    routing_training_examples: int
    priority_training_examples: int
    routing_training_sources: list[str] = Field(default_factory=list)
    priority_training_sources: list[str] = Field(default_factory=list)


class EvaluationCalibrationPoint(BaseModel):
    threshold: float
    qualified_cases: int
    coverage: float
    accuracy: float


class EvaluationAcceptanceGate(BaseModel):
    task: str
    metric: str
    minimum: float
    observed: float
    passed: bool


class EvaluationTaskReport(BaseModel):
    task: str
    dataset_path: str
    total_cases: int
    correct_cases: int
    accuracy: float
    average_confidence: float
    component_metrics: dict[str, float] = Field(default_factory=dict)
    calibration: list[EvaluationCalibrationPoint] = Field(default_factory=list)
    recommended_threshold: float
    acceptance_gate: EvaluationAcceptanceGate
    mismatches: list[dict[str, object]] = Field(default_factory=list)


class EvaluationSuiteResponse(BaseModel):
    benchmark_name: str
    generated_at: str
    provider: str
    model: str
    dataset_dir: str
    artifact_dir: str
    overall_passed: bool
    tasks: dict[str, EvaluationTaskReport] = Field(default_factory=dict)
    report_path: str
    calibration_report_path: str


class CaseOperationsUpdateRequest(BaseModel):
    reviewer_id: str | None = None
    final_disposition: str | None = None
    final_disposition_reason: str | None = None


class CaseLifecycleTransitionRequest(BaseModel):
    transition: CaseTransition
    actor_id: str | None = None
    note: str | None = None
    reviewer_id: str | None = None


class CaseWorkflowActionRequest(BaseModel):
    action: CaseWorkflowAction
    actor_id: str | None = None
    note: str | None = None
    assignee_id: str | None = None
    institution_response: InstitutionResponseInput | None = None


class CaseAnnotationUpdateRequest(BaseModel):
    annotated_by: str | None = None
    correct_category: str | None = None
    correct_institution: str | None = None
    correct_department: str | None = None
    correct_priority_level: str | None = None
    correct_same_place: str | None = None
    correct_issue_resolved: str | None = None
    note: str | None = None
    source: str = "reviewer"


class AnnotationExportRequest(BaseModel):
    destination_dir: str | None = None


class IntakeResponse(BaseModel):
    structured_issue: StructuredIssue


class RoutingResponse(BaseModel):
    routing: RoutingDecision


class PriorityResponse(BaseModel):
    priority: PriorityDecision


class DraftResponse(BaseModel):
    draft: DraftAppeal


class VerificationResponse(BaseModel):
    verification: VerificationDecision


class ExplanationResponse(BaseModel):
    explanation: ExplanationNote


class ProcessCaseResponse(BaseModel):
    case: CaseRecord
    original_request: dict[str, object] | None = None


class PrivacyExportResponse(BaseModel):
    case_id: str
    exported_at: str
    export_path: str
    archive_size_bytes: int
    evidence_count: int
    audit_event_count: int


class PrivacyDeleteResponse(BaseModel):
    case: CaseRecord
    original_request: dict[str, object] | None = None
    deleted_evidence_ids: list[str] = Field(default_factory=list)
    deleted_evidence_count: int = 0


class PrivacyRetentionResponse(BaseModel):
    executed_at: str
    cases_scanned: int
    cases_privacy_deleted: int
    evidence_deleted: int
    affected_case_ids: list[str] = Field(default_factory=list)


class ObservabilityProviderMetric(BaseModel):
    calls: int
    errors: int
    fallback_errors: int
    avg_latency_ms: float
    max_latency_ms: float


class ObservabilityRequestPathMetric(BaseModel):
    count: int
    avg_latency_ms: float
    max_latency_ms: float


class ObservabilityMetricsResponse(BaseModel):
    generated_at: str
    request: dict[str, object] = Field(default_factory=dict)
    provider: dict[str, object] = Field(default_factory=dict)
    queues: dict[str, object] = Field(default_factory=dict)
    reviews: dict[str, object] = Field(default_factory=dict)


class CaseAuditLogResponse(BaseModel):
    case_id: str
    items: list[AuditLogEvent] = Field(default_factory=list)


class AuditChainVerificationResponse(BaseModel):
    verification: AuditChainVerification


class OriginalRequestResponse(BaseModel):
    original_request: dict[str, object]


class CaseAnnotationResponse(BaseModel):
    case_id: str
    annotation: CaseAnnotation | None = None


class AnnotationExportArtifact(BaseModel):
    task: str
    output_path: str
    exported_examples: int


class AnnotationExportResponse(BaseModel):
    exported_at: str
    export_dir: str
    routing: AnnotationExportArtifact
    priority: AnnotationExportArtifact
    verification: AnnotationExportArtifact
    report_path: str


class EvidenceUploadResponse(BaseModel):
    evidence: StoredEvidence
    evidence_item: EvidenceItem
    download_url: str
    thumbnail_url: str | None = None
    download_expires_at: str


class EvidenceMetadataResponse(BaseModel):
    evidence: StoredEvidence
    evidence_item: EvidenceItem
    download_url: str
    thumbnail_url: str | None = None
    download_expires_at: str


class ReviewQueueItem(BaseModel):
    case_id: str
    created_at: str
    status: str
    submission_excerpt: str
    category: str
    issue_type: str
    institution: str
    department: str
    priority_level: str
    priority_score: int
    review_queue: str
    review_confidence: float
    assignee_id: str | None = None
    assignment_state: str
    sla_deadline_at: str
    sla_breached: bool
    model_context: CaseModelContext
    operations: CaseOperationalState
    allowed_transitions: list[str] = Field(default_factory=list)


class ReviewQueueMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int
    sort_by: str
    assignment_state: str | None = None
    review_queue: str | None = None
    priority_level: str | None = None
    assignee_id: str | None = None
    status: str | None = None


class ReviewQueuePageResponse(BaseModel):
    items: list[ReviewQueueItem] = Field(default_factory=list)
    meta: ReviewQueueMeta


class CaseListItem(BaseModel):
    case_id: str
    created_at: str
    status: str
    submission_excerpt: str
    category: str
    issue_type: str
    institution: str
    department: str
    priority_level: str
    priority_score: int
    review_needed: bool
    review_queue: str
    review_confidence: float
    issue_confidence: float
    routing_confidence: float
    priority_confidence: float
    verification_same_place: str | None = None
    verification_issue_resolved: str | None = None
    model_context: CaseModelContext
    operations: CaseOperationalState
    allowed_transitions: list[str] = Field(default_factory=list)


class CaseListResponse(BaseModel):
    items: list[CaseListItem] = Field(default_factory=list)


class CaseAnalyticsSummary(BaseModel):
    total_cases: int
    review_needed_cases: int
    counts_by_status: dict[str, int] = Field(default_factory=dict)
    counts_by_category: dict[str, int] = Field(default_factory=dict)
    counts_by_institution: dict[str, int] = Field(default_factory=dict)
    counts_by_priority: dict[str, int] = Field(default_factory=dict)
    counts_by_final_disposition: dict[str, int] = Field(default_factory=dict)
    counts_by_reviewer: dict[str, int] = Field(default_factory=dict)


class CasePrivacySummaryResponse(BaseModel):
    case_id: str
    privacy: CasePrivacyState
    findings: list[PrivacyTextFinding] = Field(default_factory=list)


class CaseAnalyticsSummaryResponse(BaseModel):
    summary: CaseAnalyticsSummary


class StorageBackupInfo(BaseModel):
    backup_id: str
    backup_path: str
    label: str | None = None
    backup_type: str
    created_at: str
    source_path: str
    integrity_check: str
    size_bytes: int


class StorageRestoreInfo(BaseModel):
    restore_id: str
    source_path: str
    restored_at: str
    source_integrity: str
    pre_restore_backup: StorageBackupInfo | None = None
    schema_version: int


class StorageBackupResponse(BaseModel):
    operation: StorageBackupInfo


class StorageRestoreResponse(BaseModel):
    operation: StorageRestoreInfo


class StorageBackupsResponse(BaseModel):
    items: list[StorageBackupInfo] = Field(default_factory=list)

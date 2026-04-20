from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.audit import AuditService
from app.abuse_protection import AbuseProtectionService
from app.auth import AuthService, build_auth_service, slugify_name
from app.config import Settings, get_settings
from app.evaluation import load_latest_evaluation_report, run_evaluation_suite
from app.evidence_store import LocalEvidenceStore
from app.observability import InstrumentedProvider, ObservabilityService, RequestControlMiddleware
from app.privacy import PrivacyService
from app.models.api import (
    AuthAccountCreateRequest,
    AuthAccountsResponse,
    AuditChainVerificationResponse,
    AuthLoginRequest,
    AuthLogoutResponse,
    AuthRegisterRequest,
    AuthTokenResponse,
    CaseAuditLogResponse,
    CaseAnalyticsSummaryResponse,
    CaseListResponse,
    CaseLifecycleTransitionRequest,
    CaseOperationsUpdateRequest,
    CasePrivacySummaryResponse,
    CaseVerificationRequest,
    CaseWorkflowActionRequest,
    DraftRequest,
    DraftResponse,
    EvidenceMetadataResponse,
    EvidenceUploadResponse,
    EvaluationSuiteResponse,
    ExplanationRequest,
    ExplanationResponse,
    IntakeResponse,
    LocalMLRetrainResponse,
    ObservabilityMetricsResponse,
    OriginalRequestResponse,
    PriorityInput,
    PriorityResponse,
    PrivacyDeleteRequest,
    PrivacyDeleteResponse,
    PrivacyExportResponse,
    PrivacyRetentionResponse,
    ProcessCaseRequest,
    ProcessCaseResponse,
    ReviewQueuePageResponse,
    StorageBackupRequest,
    StorageBackupResponse,
    StorageBackupsResponse,
    StorageRestoreRequest,
    StorageRestoreResponse,
    RoutingInput,
    RoutingResponse,
    SubmissionInput,
    VerificationRequest,
    VerificationResponse,
)
from app.providers.base import AIProvider
from app.providers import create_provider
from app.providers.local_provider import LocalFreeProvider
from app.repository import CaseRepository, InMemoryCaseRepository, SQLiteCaseRepository
from app.models.domain import AccountRole, AuthenticatedUser, EvidenceKind
from app.services.drafting import DraftingService
from app.services.explanation import ExplanationService
from app.services.intake import IntakeService
from app.services.orchestration import CaseOrchestrator
from app.services.priority import PriorityService
from app.services.review import ReviewService
from app.services.routing import RoutingService
from app.services.verification import VerificationService
from app.services.workflow import CaseWorkflowService


DEFAULT_APP_NAME = "ASANAppeal AI"
UI_ROOT = Path(__file__).resolve().parent / "ui"
bearer_scheme = HTTPBearer(auto_error=False)
STAFF_ROLES = {AccountRole.operator, AccountRole.reviewer, AccountRole.admin}
REVIEW_ROLES = {AccountRole.operator, AccountRole.reviewer, AccountRole.admin}
INSTITUTION_ACCESS_ROLES = {
    AccountRole.operator,
    AccountRole.reviewer,
    AccountRole.institution,
    AccountRole.admin,
}
ALL_ACCOUNT_ROLES = {
    AccountRole.citizen,
    AccountRole.operator,
    AccountRole.reviewer,
    AccountRole.institution,
    AccountRole.admin,
}


@dataclass
class AppRuntime:
    settings: Settings
    auth_service: AuthService
    audit_service: AuditService
    abuse_service: AbuseProtectionService
    observability_service: ObservabilityService
    provider: AIProvider
    repository: CaseRepository
    evidence_store: LocalEvidenceStore
    privacy_service: PrivacyService
    intake_service: IntakeService
    routing_service: RoutingService
    priority_service: PriorityService
    drafting_service: DraftingService
    verification_service: VerificationService
    review_service: ReviewService
    explanation_service: ExplanationService
    orchestrator: CaseOrchestrator
    workflow_service: CaseWorkflowService


def _build_repository(settings: Settings) -> CaseRepository:
    if settings.repository_backend == "memory":
        return InMemoryCaseRepository()
    return SQLiteCaseRepository.build_hardened(
        settings.sqlite_path,
        timeout_seconds=settings.sqlite_timeout_seconds,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        journal_mode=settings.sqlite_journal_mode,
        synchronous=settings.sqlite_synchronous,
        max_write_retries=settings.sqlite_max_write_retries,
        write_retry_backoff_ms=settings.sqlite_write_retry_backoff_ms,
        backup_dir=settings.sqlite_backup_dir,
        backup_pages_per_step=settings.sqlite_backup_pages_per_step,
    )


def build_runtime(settings: Settings | None = None) -> AppRuntime:
    settings = settings or get_settings()
    repository = _build_repository(settings)
    auth_service = build_auth_service(settings, repository)
    audit_service = AuditService(repository)
    abuse_service = AbuseProtectionService(settings)
    observability_service = ObservabilityService()
    base_provider = create_provider(settings)
    provider = InstrumentedProvider(base_provider, observability_service)
    evidence_store = LocalEvidenceStore.build_hardened(
        settings.evidence_root,
        max_bytes=settings.evidence_max_bytes,
        signed_url_ttl_seconds=settings.evidence_signed_url_ttl_seconds,
        signing_secret=settings.evidence_signing_secret,
        thumbnail_max_size=settings.evidence_thumbnail_max_size,
    )
    privacy_service = PrivacyService(settings, evidence_store)
    intake_service = IntakeService(provider)
    routing_service = RoutingService(provider)
    priority_service = PriorityService(provider)
    drafting_service = DraftingService(provider)
    verification_service = VerificationService(provider)
    review_service = ReviewService(settings.human_review_confidence_threshold)
    explanation_service = ExplanationService(provider)
    workflow_service = CaseWorkflowService(
        repository=repository,
        verification_service=verification_service,
        review_service=review_service,
        explanation_service=explanation_service,
    )
    orchestrator = CaseOrchestrator(
        repository=repository,
        intake_service=intake_service,
        routing_service=routing_service,
        priority_service=priority_service,
        drafting_service=drafting_service,
        verification_service=verification_service,
        explanation_service=explanation_service,
        review_service=review_service,
        provider_name=type(base_provider).__name__,
        model_name=getattr(base_provider, "_model", "unknown"),
    )
    return AppRuntime(
        settings=settings,
        auth_service=auth_service,
        audit_service=audit_service,
        abuse_service=abuse_service,
        observability_service=observability_service,
        provider=provider,
        repository=repository,
        evidence_store=evidence_store,
        privacy_service=privacy_service,
        intake_service=intake_service,
        routing_service=routing_service,
        priority_service=priority_service,
        drafting_service=drafting_service,
        verification_service=verification_service,
        review_service=review_service,
        explanation_service=explanation_service,
        orchestrator=orchestrator,
        workflow_service=workflow_service,
    )


def _resolve_settings(app: FastAPI) -> Settings:
    settings_override = getattr(app.state, "settings_override", None)
    if settings_override is not None:
        return settings_override
    return get_settings()


def _ensure_runtime(app: FastAPI) -> AppRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime = build_runtime(_resolve_settings(app))
        app.state.runtime = runtime
        app.title = runtime.settings.app_name
    return runtime


def _runtime_from_request(request: Request) -> AppRuntime:
    return _ensure_runtime(request.app)


def _sqlite_repository_from_runtime(runtime: AppRuntime) -> SQLiteCaseRepository:
    if isinstance(runtime.repository, SQLiteCaseRepository):
        return runtime.repository
    raise HTTPException(status_code=409, detail="SQLite storage workflow is not enabled.")


def _unwrap_provider(provider: AIProvider):
    return getattr(provider, "_provider", provider)


def _evidence_store_from_runtime(runtime: AppRuntime) -> LocalEvidenceStore:
    return runtime.evidence_store


def _auth_service_from_runtime(runtime: AppRuntime) -> AuthService:
    return runtime.auth_service


def _audit_service_from_runtime(runtime: AppRuntime) -> AuditService:
    return runtime.audit_service


def _privacy_service_from_runtime(runtime: AppRuntime) -> PrivacyService:
    return runtime.privacy_service


def _localfree_provider_from_runtime(runtime: AppRuntime) -> LocalFreeProvider:
    provider = _unwrap_provider(runtime.provider)
    if isinstance(provider, LocalFreeProvider):
        return provider
    raise HTTPException(
        status_code=409,
        detail="Local ML retraining is only available when the LocalFreeProvider is active.",
    )


def _original_request_from_repository(
    repository: CaseRepository,
    case_id: str,
) -> dict[str, object] | None:
    payload = repository.get_case_request_payload(case_id)
    if payload is None:
        return None
    return payload


def _build_case_response(repository: CaseRepository, case) -> ProcessCaseResponse:
    return ProcessCaseResponse(
        case=case,
        original_request=_original_request_from_repository(repository, case.case_id),
    )


def _build_evidence_response(
    request: Request,
    evidence_store: LocalEvidenceStore,
    evidence,
) -> EvidenceUploadResponse:
    metadata_url = str(request.url_for("get_evidence_metadata", evidence_id=evidence.evidence_id))
    download_params = evidence_store.build_signed_params(evidence, variant="download")
    download_url = str(
        request.url_for("download_evidence", evidence_id=evidence.evidence_id).include_query_params(
            expires=download_params["expires"],
            signature=download_params["signature"],
        )
    )
    thumbnail_url: str | None = None
    if evidence.thumbnail_path is not None:
        thumbnail_params = evidence_store.build_signed_params(evidence, variant="thumbnail")
        thumbnail_url = str(
            request.url_for(
                "download_evidence_thumbnail",
                evidence_id=evidence.evidence_id,
            ).include_query_params(
                expires=thumbnail_params["expires"],
                signature=thumbnail_params["signature"],
            )
        )
    return EvidenceUploadResponse(
        evidence=evidence,
        evidence_item=evidence_store.build_evidence_item(evidence, metadata_url=metadata_url),
        download_url=download_url,
        thumbnail_url=thumbnail_url,
        download_expires_at=download_params["expires_at"],
    )


def _auth_disabled_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id="auth-disabled",
        username="auth-disabled",
        display_name="Auth Disabled",
        role=AccountRole.admin,
    )


def _optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser | None:
    runtime = _runtime_from_request(request)
    if not runtime.settings.auth_enabled:
        return _auth_disabled_user()
    if credentials is None:
        return None
    user = runtime.auth_service.resolve_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def _require_roles(*allowed_roles: AccountRole):
    allowed = set(allowed_roles)

    def dependency(
        current_user: AuthenticatedUser | None = Depends(_optional_current_user),
    ) -> AuthenticatedUser:
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if allowed and current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Authenticated account does not have permission for this route.",
            )
        return current_user

    return dependency


def _case_institution_slug(case) -> str | None:
    return slugify_name(case.routing.institution)


def _assert_case_visible_to_user(current_user: AuthenticatedUser, case) -> None:
    if current_user.role in STAFF_ROLES:
        return
    if current_user.role == AccountRole.admin:
        return
    if current_user.role == AccountRole.citizen:
        if case.operations.submitted_by_user_id == current_user.user_id:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Citizen accounts can only access their own cases.",
        )
    if current_user.role == AccountRole.institution:
        if current_user.institution_slug and _case_institution_slug(case) == current_user.institution_slug:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Institution accounts can only access cases routed to their institution.",
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Authenticated account does not have permission for this case.",
    )


def _scoped_case_list(
    runtime: AppRuntime,
    current_user: AuthenticatedUser,
    *,
    category: str | None,
    institution: str | None,
    priority_level: str | None,
    reviewer_id: str | None,
    final_disposition: str | None,
    status: str | None,
    limit: int,
) -> list[dict[str, object]]:
    if current_user.role in STAFF_ROLES or current_user.role == AccountRole.admin:
        return runtime.repository.list_cases(
            category=category,
            institution=institution,
            priority_level=priority_level,
            reviewer_id=reviewer_id,
            final_disposition=final_disposition,
            status=status,
            limit=limit,
        )

    fetch_limit = max(limit * 10, 200)
    items = runtime.repository.list_cases(
        category=category,
        institution=institution,
        priority_level=priority_level,
        reviewer_id=reviewer_id,
        final_disposition=final_disposition,
        status=status,
        limit=fetch_limit,
    )
    scoped_items: list[dict[str, object]] = []
    for item in items:
        case = runtime.repository.get_case(str(item["case_id"]))
        if case is None:
            continue
        try:
            _assert_case_visible_to_user(current_user, case)
        except HTTPException:
            continue
        scoped_items.append(item)
        if len(scoped_items) >= limit:
            break
    return scoped_items
    thumbnail_url: str | None = None
    if evidence.thumbnail_path is not None:
        thumbnail_params = evidence_store.build_signed_params(evidence, variant="thumbnail")
        thumbnail_url = str(
            request.url_for(
                "download_evidence_thumbnail",
                evidence_id=evidence.evidence_id,
            ).include_query_params(
                expires=thumbnail_params["expires"],
                signature=thumbnail_params["signature"],
            )
        )
    return EvidenceUploadResponse(
        evidence=evidence,
        evidence_item=evidence_store.build_evidence_item(evidence, metadata_url=metadata_url),
        download_url=download_url,
        thumbnail_url=thumbnail_url,
        download_expires_at=download_params["expires_at"],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_settings = _resolve_settings(app)
        app.state.runtime = build_runtime(resolved_settings)
        app.title = resolved_settings.app_name
        yield

    app = FastAPI(
        title=settings.app_name if settings is not None else DEFAULT_APP_NAME,
        version="0.1.0",
        description="A runnable backend foundation for the ASANAppeal AI workflow.",
        lifespan=lifespan,
    )
    app.state.settings_override = settings
    app.add_middleware(RequestControlMiddleware)
    app.mount("/app-assets", StaticFiles(directory=str(UI_ROOT)), name="app_assets")

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(
            "<html><body style=\"font-family: Georgia, serif; padding: 24px; background: #f4efe7; color: #11283a;\">"
            "<h1>ASANAppeal AI</h1>"
            "<p>Open <a href=\"/app\">/app</a> for the citizen and operator portal, or <a href=\"/docs\">/docs</a> for the API.</p>"
            "</body></html>"
        )

    @app.get("/app", response_class=HTMLResponse)
    def portal() -> FileResponse:
        return FileResponse(UI_ROOT / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        runtime = _runtime_from_request(request)
        active_provider = _unwrap_provider(runtime.provider)
        payload: dict[str, object] = {
            "status": "ok",
            "requested_provider": runtime.settings.provider,
            "active_provider": type(active_provider).__name__,
            "model": getattr(active_provider, "_model", "n/a"),
            "repository_backend": runtime.settings.repository_backend,
            "repository": type(runtime.repository).__name__,
        }
        provider_diagnostics = getattr(active_provider, "diagnostics", None)
        if callable(provider_diagnostics):
            payload.update(provider_diagnostics(force_refresh=True))
        repository_diagnostics = getattr(runtime.repository, "diagnostics", None)
        if callable(repository_diagnostics):
            payload.update(repository_diagnostics())
        evidence_diagnostics = getattr(runtime.evidence_store, "diagnostics", None)
        if callable(evidence_diagnostics):
            payload.update(evidence_diagnostics())
        privacy_diagnostics = getattr(runtime.privacy_service, "diagnostics", None)
        if callable(privacy_diagnostics):
            payload.update(privacy_diagnostics())
        abuse_diagnostics = getattr(runtime.abuse_service, "diagnostics", None)
        if callable(abuse_diagnostics):
            payload.update(abuse_diagnostics())
        payload.update(runtime.auth_service.diagnostics())
        payload.update(runtime.observability_service.diagnostics(runtime.repository))
        latest_evaluation = load_latest_evaluation_report(runtime.settings)
        if latest_evaluation is not None:
            payload["evaluation_latest_passed"] = latest_evaluation.get("overall_passed")
            payload["evaluation_latest_generated_at"] = latest_evaluation.get("generated_at")
            payload["evaluation_latest_report_path"] = latest_evaluation.get("report_path")
        degraded_dependencies: list[str] = []
        if payload.get("local_llm_dependency_required") and not payload.get(
            "local_llm_dependency_ok", True
        ):
            degraded_dependencies.append("local_llm")
        if degraded_dependencies:
            payload["status"] = "degraded"
            payload["degraded_dependencies"] = degraded_dependencies
        return payload

    @app.get("/v1/observability/metrics", response_model=ObservabilityMetricsResponse)
    def get_observability_metrics(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> ObservabilityMetricsResponse:
        runtime = _runtime_from_request(request)
        return ObservabilityMetricsResponse(
            **runtime.observability_service.snapshot(runtime.repository)
        )

    @app.post("/v1/auth/register", response_model=AuthTokenResponse)
    def register_citizen_account(
        payload: AuthRegisterRequest,
        request: Request,
    ) -> AuthTokenResponse:
        runtime = _runtime_from_request(request)
        try:
            runtime.auth_service.register_citizen(
                username=payload.username,
                password=payload.password,
                display_name=payload.display_name,
            )
            issued = runtime.auth_service.login(
                username=payload.username,
                password=payload.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AuthTokenResponse(
            access_token=issued.access_token,
            expires_at=issued.expires_at,
            user=issued.user,
        )

    @app.post("/v1/auth/login", response_model=AuthTokenResponse)
    def login_account(payload: AuthLoginRequest, request: Request) -> AuthTokenResponse:
        runtime = _runtime_from_request(request)
        try:
            issued = runtime.auth_service.login(
                username=payload.username,
                password=payload.password,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        return AuthTokenResponse(
            access_token=issued.access_token,
            expires_at=issued.expires_at,
            user=issued.user,
        )

    @app.post("/v1/auth/logout", response_model=AuthLogoutResponse)
    def logout_account(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> AuthLogoutResponse:
        runtime = _runtime_from_request(request)
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return AuthLogoutResponse(
            revoked=runtime.auth_service.revoke_token(credentials.credentials)
        )

    @app.get("/v1/auth/me", response_model=AuthenticatedUser)
    def get_current_account(
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> AuthenticatedUser:
        return current_user

    @app.get("/v1/auth/accounts", response_model=AuthAccountsResponse)
    def list_auth_accounts(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> AuthAccountsResponse:
        runtime = _runtime_from_request(request)
        return AuthAccountsResponse(items=runtime.auth_service.list_accounts())

    @app.post("/v1/auth/accounts", response_model=AuthenticatedUser)
    def create_auth_account(
        payload: AuthAccountCreateRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> AuthenticatedUser:
        runtime = _runtime_from_request(request)
        try:
            return runtime.auth_service.create_account(
                username=payload.username,
                password=payload.password,
                display_name=payload.display_name,
                role=payload.role,
                institution_slug=payload.institution_slug,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/evidence/upload", response_model=EvidenceUploadResponse)
    async def upload_evidence(
        request: Request,
        kind: EvidenceKind = Query(EvidenceKind.image),
        filename: str | None = Query(None),
        description: str | None = Query(None),
        metadata_json: str | None = Query(None),
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> EvidenceUploadResponse:
        runtime = _runtime_from_request(request)
        raw_body = await request.body()
        metadata: dict[str, str] = {}
        if metadata_json:
            try:
                parsed_metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="metadata_json must be valid JSON.") from exc
            if not isinstance(parsed_metadata, dict):
                raise HTTPException(status_code=400, detail="metadata_json must be a JSON object.")
            metadata = {str(key): str(value) for key, value in parsed_metadata.items()}
        try:
            prepared_upload = runtime.privacy_service.prepare_evidence_upload(
                data=raw_body,
                kind=kind,
                filename=filename,
                description=description,
                content_type=request.headers.get("content-type"),
            )
            evidence = runtime.evidence_store.ingest_bytes(
                data=prepared_upload.data,
                kind=kind,
                filename=prepared_upload.filename,
                description=prepared_upload.description,
                content_type=prepared_upload.content_type,
                metadata=metadata,
                privacy=prepared_upload.privacy,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _build_evidence_response(request, runtime.evidence_store, evidence)

    @app.get("/v1/evidence/{evidence_id}", response_model=EvidenceMetadataResponse)
    def get_evidence_metadata(
        evidence_id: str,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> EvidenceMetadataResponse:
        runtime = _runtime_from_request(request)
        evidence = runtime.evidence_store.get(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Evidence not found.")
        response = _build_evidence_response(request, runtime.evidence_store, evidence)
        return EvidenceMetadataResponse(**response.model_dump())

    @app.get("/v1/evidence/{evidence_id}/download", name="download_evidence")
    def download_evidence(
        evidence_id: str,
        request: Request,
        expires: str,
        signature: str,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> FileResponse:
        runtime = _runtime_from_request(request)
        evidence_store = _evidence_store_from_runtime(runtime)
        evidence = evidence_store.get(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Evidence not found.")
        try:
            evidence_store.verify_signed_params(
                evidence,
                variant="download",
                expires=expires,
                signature=signature,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return FileResponse(
            evidence_store.object_file(evidence),
            media_type=evidence.mime_type,
            filename=evidence.filename,
        )

    @app.get("/v1/evidence/{evidence_id}/thumbnail", name="download_evidence_thumbnail")
    def download_evidence_thumbnail(
        evidence_id: str,
        request: Request,
        expires: str,
        signature: str,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> FileResponse:
        runtime = _runtime_from_request(request)
        evidence_store = _evidence_store_from_runtime(runtime)
        evidence = evidence_store.get(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Evidence not found.")
        try:
            evidence_store.verify_signed_params(
                evidence,
                variant="thumbnail",
                expires=expires,
                signature=signature,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        thumbnail_file = evidence_store.thumbnail_file(evidence)
        if thumbnail_file is None:
            raise HTTPException(status_code=404, detail="Thumbnail not available.")
        return FileResponse(
            thumbnail_file,
            media_type="image/png",
            filename=f"{evidence_id}-thumbnail.png",
        )

    @app.post("/v1/intake", response_model=IntakeResponse)
    def intake(
        submission: SubmissionInput,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*STAFF_ROLES)),
    ) -> IntakeResponse:
        runtime = _runtime_from_request(request)
        return IntakeResponse(structured_issue=runtime.intake_service.analyze(submission))

    @app.post("/v1/route", response_model=RoutingResponse)
    def route_case(
        payload: RoutingInput,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*STAFF_ROLES)),
    ) -> RoutingResponse:
        runtime = _runtime_from_request(request)
        return RoutingResponse(
            routing=runtime.routing_service.route(payload.submission, payload.structured_issue)
        )

    @app.post("/v1/priority", response_model=PriorityResponse)
    def assess_priority(
        payload: PriorityInput,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*STAFF_ROLES)),
    ) -> PriorityResponse:
        runtime = _runtime_from_request(request)
        return PriorityResponse(
            priority=runtime.priority_service.assess(
                payload.submission, payload.structured_issue, payload.routing
            )
        )

    @app.post("/v1/draft", response_model=DraftResponse)
    def draft_case(
        payload: DraftRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*STAFF_ROLES)),
    ) -> DraftResponse:
        runtime = _runtime_from_request(request)
        return DraftResponse(
            draft=runtime.drafting_service.build_draft(
                payload.submission, payload.structured_issue, payload.routing, payload.priority
            )
        )

    @app.post("/v1/verify", response_model=VerificationResponse)
    def verify_case(
        payload: VerificationRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*INSTITUTION_ACCESS_ROLES)),
    ) -> VerificationResponse:
        runtime = _runtime_from_request(request)
        return VerificationResponse(
            verification=runtime.verification_service.verify(
                payload.original_submission,
                payload.structured_issue,
                payload.institution_response,
            )
        )

    @app.post("/v1/explain", response_model=ExplanationResponse)
    def explain_case(
        payload: ExplanationRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*STAFF_ROLES)),
    ) -> ExplanationResponse:
        runtime = _runtime_from_request(request)
        return ExplanationResponse(
            explanation=runtime.explanation_service.explain(
                payload.structured_issue,
                payload.routing,
                payload.priority,
                payload.human_review,
                payload.verification,
            )
        )

    @app.post("/v1/local-ml/retrain", response_model=LocalMLRetrainResponse)
    def retrain_local_ml(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> LocalMLRetrainResponse:
        runtime = _runtime_from_request(request)
        _sqlite_repository_from_runtime(runtime)
        provider = _localfree_provider_from_runtime(runtime)
        try:
            summary = provider.retrain_local_models()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return LocalMLRetrainResponse(**summary)

    @app.post("/v1/evals/run", response_model=EvaluationSuiteResponse)
    def run_evals(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> EvaluationSuiteResponse:
        runtime = _runtime_from_request(request)
        report = run_evaluation_suite(runtime.settings, provider=runtime.provider)
        return EvaluationSuiteResponse(**report)

    @app.get("/v1/evals/latest", response_model=EvaluationSuiteResponse)
    def get_latest_evals(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> EvaluationSuiteResponse:
        runtime = _runtime_from_request(request)
        report = load_latest_evaluation_report(runtime.settings)
        if report is None:
            raise HTTPException(status_code=404, detail="No evaluation report has been generated yet.")
        return EvaluationSuiteResponse(**report)

    @app.post("/v1/privacy/retention/enforce", response_model=PrivacyRetentionResponse)
    def enforce_privacy_retention(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> PrivacyRetentionResponse:
        runtime = _runtime_from_request(request)
        result = runtime.privacy_service.enforce_retention(repository=runtime.repository)
        for record in result.records:
            runtime.audit_service.record_privacy_delete(
                before_case=record.before_case,
                after_case=record.after_case,
                actor=current_user,
                deleted_evidence_ids=record.deleted_evidence_ids,
                note="Retention policy executed.",
                source="api.privacy.retention",
            )
        return PrivacyRetentionResponse(
            executed_at=result.executed_at,
            cases_scanned=result.cases_scanned,
            cases_privacy_deleted=result.cases_privacy_deleted,
            evidence_deleted=result.evidence_deleted,
            affected_case_ids=result.affected_case_ids,
        )

    @app.post("/v1/cases/process", response_model=ProcessCaseResponse)
    def process_case(
        payload: ProcessCaseRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(
            _require_roles(
                AccountRole.citizen,
                AccountRole.operator,
                AccountRole.reviewer,
                AccountRole.admin,
            )
        ),
    ) -> ProcessCaseResponse:
        runtime = _runtime_from_request(request)
        abuse_decision = runtime.abuse_service.validate_process_case(
            payload,
            current_user=current_user,
        )
        if not abuse_decision.allowed:
            raise HTTPException(
                status_code=abuse_decision.status_code,
                detail=abuse_decision.reason,
            )
        stored_request_payload, privacy_state = runtime.privacy_service.sanitize_process_request(payload)
        case = runtime.orchestrator.process_case(
            payload,
            submitted_by_user_id=current_user.user_id,
            submitted_by_role=current_user.role,
            privacy_state=privacy_state,
            stored_request_payload=stored_request_payload,
            case_redactor=runtime.privacy_service.redact_case_record,
        )
        runtime.audit_service.record_case_created(
            case=case,
            actor=current_user,
            original_request=stored_request_payload,
        )
        runtime.abuse_service.register_case_submission(payload, current_user=current_user)
        return _build_case_response(runtime.repository, case)

    @app.get("/v1/cases/{case_id}", response_model=ProcessCaseResponse)
    def get_case(
        case_id: str,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> ProcessCaseResponse:
        runtime = _runtime_from_request(request)
        case = runtime.repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, case)
        return _build_case_response(runtime.repository, case)

    @app.get("/v1/cases/{case_id}/original-request", response_model=OriginalRequestResponse)
    def get_case_original_request(
        case_id: str,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> OriginalRequestResponse:
        runtime = _runtime_from_request(request)
        case = runtime.repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, case)
        original_request = _original_request_from_repository(runtime.repository, case_id)
        if original_request is None:
            raise HTTPException(status_code=404, detail="Original request payload not found.")
        return OriginalRequestResponse(original_request=original_request)

    @app.get("/v1/cases/{case_id}/privacy", response_model=CasePrivacySummaryResponse)
    def get_case_privacy_summary(
        case_id: str,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> CasePrivacySummaryResponse:
        runtime = _runtime_from_request(request)
        case = runtime.repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, case)
        return CasePrivacySummaryResponse(
            case_id=case.case_id,
            privacy=case.privacy,
            findings=case.privacy.text_findings,
        )

    @app.get("/v1/cases/{case_id}/audit-log", response_model=CaseAuditLogResponse)
    def get_case_audit_log(
        case_id: str,
        request: Request,
        limit: int = 200,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> CaseAuditLogResponse:
        runtime = _runtime_from_request(request)
        case = runtime.repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, case)
        return CaseAuditLogResponse(
            case_id=case_id,
            items=runtime.repository.list_case_audit_events(case_id, limit=limit),
        )

    @app.post("/v1/cases/{case_id}/privacy-export", response_model=PrivacyExportResponse)
    def export_case_privacy_package(
        case_id: str,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> PrivacyExportResponse:
        runtime = _runtime_from_request(request)
        case = runtime.repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, case)
        export_artifact = runtime.privacy_service.export_case_bundle(
            case=case,
            request_payload=_original_request_from_repository(runtime.repository, case_id),
            audit_events=runtime.repository.list_case_audit_events(case_id, limit=1000),
        )
        updated_case = case.model_copy(
            update={
                "privacy": case.privacy.model_copy(
                    update={
                        "exported_at": datetime.fromisoformat(export_artifact.exported_at),
                        "last_export_path": export_artifact.export_path,
                    }
                )
            }
        )
        runtime.repository.save_case(
            updated_case,
            runtime.repository.get_case_request_payload(case_id),
        )
        runtime.audit_service.record_privacy_export(
            case=updated_case,
            actor=current_user,
            export_path=export_artifact.export_path,
            evidence_count=export_artifact.evidence_count,
            audit_event_count=export_artifact.audit_event_count,
        )
        return PrivacyExportResponse(**export_artifact.__dict__)

    @app.post("/v1/cases/{case_id}/privacy-delete", response_model=PrivacyDeleteResponse)
    def privacy_delete_case(
        case_id: str,
        payload: PrivacyDeleteRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> PrivacyDeleteResponse:
        runtime = _runtime_from_request(request)
        case = runtime.repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        tombstone_case, tombstone_request, deleted_ids = runtime.privacy_service.privacy_delete_case(
            case=case,
            request_payload=_original_request_from_repository(runtime.repository, case_id),
            note=payload.note,
        )
        saved_case = runtime.repository.save_case(tombstone_case, tombstone_request)
        runtime.audit_service.record_privacy_delete(
            before_case=case,
            after_case=saved_case,
            actor=current_user,
            deleted_evidence_ids=deleted_ids,
            note=payload.note,
        )
        return PrivacyDeleteResponse(
            case=saved_case,
            original_request=tombstone_request,
            deleted_evidence_ids=deleted_ids,
            deleted_evidence_count=len(deleted_ids),
        )

    @app.get("/v1/cases", response_model=CaseListResponse)
    def list_cases(
        request: Request,
        category: str | None = None,
        institution: str | None = None,
        priority_level: str | None = None,
        reviewer_id: str | None = None,
        final_disposition: str | None = None,
        status: str | None = None,
        limit: int = 50,
        current_user: AuthenticatedUser = Depends(_require_roles(*ALL_ACCOUNT_ROLES)),
    ) -> CaseListResponse:
        runtime = _runtime_from_request(request)
        items = _scoped_case_list(
            runtime,
            current_user,
            category=category,
            institution=institution,
            priority_level=priority_level,
            reviewer_id=reviewer_id,
            final_disposition=final_disposition,
            status=status,
            limit=limit,
        )
        return CaseListResponse(items=items)

    @app.post("/v1/cases/{case_id}/operations", response_model=ProcessCaseResponse)
    def update_case_operations(
        case_id: str,
        payload: CaseOperationsUpdateRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*REVIEW_ROLES)),
    ) -> ProcessCaseResponse:
        runtime = _runtime_from_request(request)
        existing_case = runtime.repository.get_case(case_id)
        if existing_case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, existing_case)
        case = runtime.repository.update_case_operational_fields(
            case_id,
            reviewer_id=payload.reviewer_id,
            final_disposition=payload.final_disposition,
            final_disposition_reason=payload.final_disposition_reason,
        )
        runtime.audit_service.record_operations_update(
            before_case=existing_case,
            after_case=case,
            actor=current_user,
            reviewer_id=payload.reviewer_id,
            final_disposition=payload.final_disposition,
            final_disposition_reason=payload.final_disposition_reason,
        )
        return _build_case_response(runtime.repository, case)

    @app.post("/v1/cases/{case_id}/transition", response_model=ProcessCaseResponse)
    def transition_case(
        case_id: str,
        payload: CaseLifecycleTransitionRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*REVIEW_ROLES)),
    ) -> ProcessCaseResponse:
        runtime = _runtime_from_request(request)
        repository = _sqlite_repository_from_runtime(runtime)
        existing_case = runtime.repository.get_case(case_id)
        if existing_case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, existing_case)
        try:
            case = repository.transition_case(
                case_id,
                transition=payload.transition,
                actor_id=payload.actor_id or current_user.user_id,
                note=payload.note,
                reviewer_id=payload.reviewer_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        runtime.audit_service.record_transition(
            before_case=existing_case,
            after_case=case,
            actor=current_user,
            transition=payload.transition.value,
            note=payload.note,
        )
        return _build_case_response(runtime.repository, case)

    @app.post("/v1/cases/{case_id}/verify", response_model=ProcessCaseResponse)
    def verify_existing_case(
        case_id: str,
        payload: CaseVerificationRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*INSTITUTION_ACCESS_ROLES)),
    ) -> ProcessCaseResponse:
        runtime = _runtime_from_request(request)
        existing_case = runtime.repository.get_case(case_id)
        if existing_case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, existing_case)
        try:
            case = runtime.workflow_service.verify_case(
                case_id,
                institution_response=payload.institution_response,
                actor_id=payload.actor_id or current_user.user_id,
                note=payload.note,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        case = runtime.repository.save_case(
            runtime.privacy_service.redact_case_record(case),
            runtime.repository.get_case_request_payload(case_id),
        )
        runtime.audit_service.record_workflow_action(
            before_case=existing_case,
            after_case=case,
            actor=current_user,
            action="verify",
            note=payload.note,
            institution_response=runtime.privacy_service.sanitize_audit_payload(
                payload.institution_response.model_dump(mode="json", exclude_none=True)
            ),
            source="api.cases.verify",
        )
        return _build_case_response(runtime.repository, case)

    @app.post("/v1/cases/{case_id}/workflow-actions", response_model=ProcessCaseResponse)
    def apply_workflow_action(
        case_id: str,
        payload: CaseWorkflowActionRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*REVIEW_ROLES)),
    ) -> ProcessCaseResponse:
        runtime = _runtime_from_request(request)
        existing_case = runtime.repository.get_case(case_id)
        if existing_case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        _assert_case_visible_to_user(current_user, existing_case)
        if payload.actor_id is None:
            payload = payload.model_copy(update={"actor_id": current_user.user_id})
        try:
            case = runtime.workflow_service.apply_action(case_id, payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        case = runtime.repository.save_case(
            runtime.privacy_service.redact_case_record(case),
            runtime.repository.get_case_request_payload(case_id),
        )
        runtime.audit_service.record_workflow_action(
            before_case=existing_case,
            after_case=case,
            actor=current_user,
            action=payload.action.value,
            note=payload.note,
            assignee_id=payload.assignee_id,
            institution_response=(
                runtime.privacy_service.sanitize_audit_payload(
                    payload.institution_response.model_dump(mode="json", exclude_none=True)
                )
                if payload.institution_response is not None
                else None
            ),
        )
        return _build_case_response(runtime.repository, case)

    @app.get("/v1/review-queue", response_model=ReviewQueuePageResponse)
    def get_review_queue(
        request: Request,
        page: int = 1,
        page_size: int = 20,
        review_queue: str | None = None,
        priority_level: str | None = None,
        status: str | None = None,
        assignee_id: str | None = None,
        assignment_state: str | None = None,
        sort_by: str = "sla",
        current_user: AuthenticatedUser = Depends(_require_roles(*REVIEW_ROLES)),
    ) -> ReviewQueuePageResponse:
        runtime = _runtime_from_request(request)
        queue = runtime.repository.query_review_queue(
            page=page,
            page_size=page_size,
            review_queue=review_queue,
            priority_level=priority_level,
            status=status,
            assignee_id=assignee_id,
            assignment_state=assignment_state,
            sort_by=sort_by,
        )
        return ReviewQueuePageResponse(**queue)

    @app.get("/v1/analytics/summary", response_model=CaseAnalyticsSummaryResponse)
    def get_case_analytics(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(*STAFF_ROLES)),
    ) -> CaseAnalyticsSummaryResponse:
        runtime = _runtime_from_request(request)
        return CaseAnalyticsSummaryResponse(summary=runtime.repository.summarize_cases())

    @app.get("/v1/audit/verify", response_model=AuditChainVerificationResponse)
    def verify_audit_chain(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> AuditChainVerificationResponse:
        runtime = _runtime_from_request(request)
        return AuditChainVerificationResponse(verification=runtime.repository.verify_audit_chain())

    @app.get("/v1/storage/backups", response_model=StorageBackupsResponse)
    def get_storage_backups(
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> StorageBackupsResponse:
        runtime = _runtime_from_request(request)
        repository = _sqlite_repository_from_runtime(runtime)
        return StorageBackupsResponse(items=repository.list_backups())

    @app.post("/v1/storage/backup", response_model=StorageBackupResponse)
    def backup_storage(
        payload: StorageBackupRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> StorageBackupResponse:
        runtime = _runtime_from_request(request)
        repository = _sqlite_repository_from_runtime(runtime)
        try:
            operation = repository.backup_database(
                payload.destination_path,
                label=payload.label,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StorageBackupResponse(operation=operation)

    @app.post("/v1/storage/restore", response_model=StorageRestoreResponse)
    def restore_storage(
        payload: StorageRestoreRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(_require_roles(AccountRole.admin)),
    ) -> StorageRestoreResponse:
        runtime = _runtime_from_request(request)
        repository = _sqlite_repository_from_runtime(runtime)
        try:
            operation = repository.restore_database(
                payload.source_path,
                create_pre_restore_backup=payload.create_pre_restore_backup,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StorageRestoreResponse(operation=operation)

    return app


app = create_app()

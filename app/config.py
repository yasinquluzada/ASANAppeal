from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("ASAN_APP_NAME", "ASANAppeal AI"))
    provider: str = field(default_factory=lambda: os.getenv("ASAN_PROVIDER", "localfree"))
    default_language: str = field(
        default_factory=lambda: os.getenv("ASAN_DEFAULT_LANGUAGE", "en")
    )
    repository_backend: str = field(
        default_factory=lambda: os.getenv("ASAN_REPOSITORY_BACKEND", "sqlite")
    )
    sqlite_path: str = field(
        default_factory=lambda: os.getenv("ASAN_SQLITE_PATH", "./asanappeal.db")
    )
    sqlite_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ASAN_SQLITE_TIMEOUT_SECONDS", "5"))
    )
    sqlite_busy_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("ASAN_SQLITE_BUSY_TIMEOUT_MS", "5000"))
    )
    sqlite_journal_mode: str = field(
        default_factory=lambda: os.getenv("ASAN_SQLITE_JOURNAL_MODE", "WAL").upper()
    )
    sqlite_synchronous: str = field(
        default_factory=lambda: os.getenv("ASAN_SQLITE_SYNCHRONOUS", "NORMAL").upper()
    )
    sqlite_max_write_retries: int = field(
        default_factory=lambda: int(os.getenv("ASAN_SQLITE_MAX_WRITE_RETRIES", "5"))
    )
    sqlite_write_retry_backoff_ms: int = field(
        default_factory=lambda: int(os.getenv("ASAN_SQLITE_WRITE_RETRY_BACKOFF_MS", "50"))
    )
    sqlite_backup_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_SQLITE_BACKUP_DIR", "./.data/backups")
    )
    sqlite_backup_pages_per_step: int = field(
        default_factory=lambda: int(os.getenv("ASAN_SQLITE_BACKUP_PAGES_PER_STEP", "128"))
    )
    evidence_root: str = field(
        default_factory=lambda: os.getenv("ASAN_EVIDENCE_ROOT", "./.data/evidence")
    )
    evidence_max_bytes: int = field(
        default_factory=lambda: int(os.getenv("ASAN_EVIDENCE_MAX_BYTES", str(25 * 1024 * 1024)))
    )
    evidence_signed_url_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("ASAN_EVIDENCE_SIGNED_URL_TTL_SECONDS", "900"))
    )
    evidence_signing_secret: str = field(
        default_factory=lambda: os.getenv(
            "ASAN_EVIDENCE_SIGNING_SECRET",
            "asanappeal-local-evidence-signing-secret",
        )
    )
    evidence_thumbnail_max_size: int = field(
        default_factory=lambda: int(os.getenv("ASAN_EVIDENCE_THUMBNAIL_MAX_SIZE", "512"))
    )
    privacy_case_retention_days: int = field(
        default_factory=lambda: int(os.getenv("ASAN_PRIVACY_CASE_RETENTION_DAYS", "365"))
    )
    privacy_evidence_retention_days: int = field(
        default_factory=lambda: int(os.getenv("ASAN_PRIVACY_EVIDENCE_RETENTION_DAYS", "180"))
    )
    privacy_export_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_PRIVACY_EXPORT_DIR", "./.data/privacy_exports")
    )
    request_max_body_bytes: int = field(
        default_factory=lambda: int(os.getenv("ASAN_REQUEST_MAX_BODY_BYTES", str(512 * 1024)))
    )
    request_rate_limit_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("ASAN_REQUEST_RATE_LIMIT_WINDOW_SECONDS", "60"))
    )
    request_rate_limit_max_requests: int = field(
        default_factory=lambda: int(os.getenv("ASAN_REQUEST_RATE_LIMIT_MAX_REQUESTS", "120"))
    )
    auth_rate_limit_max_requests: int = field(
        default_factory=lambda: int(os.getenv("ASAN_AUTH_RATE_LIMIT_MAX_REQUESTS", "20"))
    )
    abuse_max_evidence_items: int = field(
        default_factory=lambda: int(os.getenv("ASAN_ABUSE_MAX_EVIDENCE_ITEMS", "8"))
    )
    abuse_max_text_chars: int = field(
        default_factory=lambda: int(os.getenv("ASAN_ABUSE_MAX_TEXT_CHARS", "6000"))
    )
    abuse_spam_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("ASAN_ABUSE_SPAM_WINDOW_SECONDS", "3600"))
    )
    abuse_duplicate_threshold: int = field(
        default_factory=lambda: int(os.getenv("ASAN_ABUSE_DUPLICATE_THRESHOLD", "2"))
    )
    local_llm_backend: str = field(
        default_factory=lambda: os.getenv("ASAN_LOCAL_LLM_BACKEND", "heuristic")
    )
    ollama_url: str = field(
        default_factory=lambda: os.getenv("ASAN_OLLAMA_URL", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("ASAN_OLLAMA_MODEL", "gemma3:4b")
    )
    ollama_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ASAN_OLLAMA_TIMEOUT_SECONDS", "45"))
    )
    local_ml_enabled: bool = field(
        default_factory=lambda: _env_bool("ASAN_LOCAL_ML_ENABLED", True)
    )
    local_ml_bootstrap_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_LOCAL_ML_BOOTSTRAP_DIR", "./data/local_ml")
    )
    local_ml_feedback_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_LOCAL_ML_FEEDBACK_DIR", "./.data/local_ml_feedback")
    )
    local_ml_include_sqlite_feedback: bool = field(
        default_factory=lambda: _env_bool("ASAN_LOCAL_ML_INCLUDE_SQLITE_FEEDBACK", True)
    )
    local_ml_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("ASAN_LOCAL_ML_MIN_CONFIDENCE", "0.58"))
    )
    eval_dataset_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_EVAL_DATASET_DIR", "./data/evals")
    )
    eval_artifact_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_EVAL_ARTIFACT_DIR", "./.data/evals")
    )
    annotation_export_dir: str = field(
        default_factory=lambda: os.getenv("ASAN_ANNOTATION_EXPORT_DIR", "./.data/annotations")
    )
    eval_intake_min_accuracy: float = field(
        default_factory=lambda: float(os.getenv("ASAN_EVAL_INTAKE_MIN_ACCURACY", "0.85"))
    )
    eval_routing_min_accuracy: float = field(
        default_factory=lambda: float(os.getenv("ASAN_EVAL_ROUTING_MIN_ACCURACY", "0.9"))
    )
    eval_priority_min_accuracy: float = field(
        default_factory=lambda: float(os.getenv("ASAN_EVAL_PRIORITY_MIN_ACCURACY", "0.85"))
    )
    eval_verification_min_accuracy: float = field(
        default_factory=lambda: float(os.getenv("ASAN_EVAL_VERIFICATION_MIN_ACCURACY", "0.8"))
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("ASAN_GEMINI_MODEL", "gemini-2.5-flash")
    )
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    openai_model: str = field(
        default_factory=lambda: os.getenv("ASAN_OPENAI_MODEL", "gpt-5.4")
    )
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("ASAN_OPENAI_REASONING_EFFORT", "high")
    )
    openai_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ASAN_OPENAI_TIMEOUT_SECONDS", "90"))
    )
    openai_max_retries: int = field(
        default_factory=lambda: int(os.getenv("ASAN_OPENAI_MAX_RETRIES", "2"))
    )
    human_review_confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ASAN_HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "0.62")
        )
    )
    auth_enabled: bool = field(default_factory=lambda: _env_bool("ASAN_AUTH_ENABLED", True))
    auth_token_ttl_hours: int = field(
        default_factory=lambda: int(os.getenv("ASAN_AUTH_TOKEN_TTL_HOURS", "12"))
    )
    auth_password_iterations: int = field(
        default_factory=lambda: int(os.getenv("ASAN_AUTH_PASSWORD_ITERATIONS", "600000"))
    )
    auth_seed_demo_accounts: bool = field(
        default_factory=lambda: _env_bool("ASAN_AUTH_SEED_DEMO_ACCOUNTS", True)
    )
    auth_demo_citizen_username: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_CITIZEN_USERNAME", "citizen.demo")
    )
    auth_demo_citizen_password: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_CITIZEN_PASSWORD", "citizen-demo-pass")
    )
    auth_demo_operator_username: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_OPERATOR_USERNAME", "operator.demo")
    )
    auth_demo_operator_password: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_OPERATOR_PASSWORD", "operator-demo-pass")
    )
    auth_demo_reviewer_username: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_REVIEWER_USERNAME", "reviewer.demo")
    )
    auth_demo_reviewer_password: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_REVIEWER_PASSWORD", "reviewer-demo-pass")
    )
    auth_demo_institution_username: str = field(
        default_factory=lambda: os.getenv(
            "ASAN_AUTH_DEMO_INSTITUTION_USERNAME",
            "institution.roads",
        )
    )
    auth_demo_institution_password: str = field(
        default_factory=lambda: os.getenv(
            "ASAN_AUTH_DEMO_INSTITUTION_PASSWORD",
            "institution-demo-pass",
        )
    )
    auth_demo_institution_slug: str = field(
        default_factory=lambda: os.getenv(
            "ASAN_AUTH_DEMO_INSTITUTION_SLUG",
            "asan-road-maintenance-agency",
        )
    )
    auth_demo_admin_username: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_ADMIN_USERNAME", "admin.demo")
    )
    auth_demo_admin_password: str = field(
        default_factory=lambda: os.getenv("ASAN_AUTH_DEMO_ADMIN_PASSWORD", "admin-demo-pass")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

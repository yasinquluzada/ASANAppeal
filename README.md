# ASANAppeal AI

This project is a runnable backend foundation for the AI workflow we discussed for ASANAppeal. It turns the high-level concept into an actual service with:

- intake understanding
- routing
- priority scoring
- appeal draft generation
- response verification
- explanation generation
- human-review fallback

The app now supports:

- `localfree` mode for fully local, zero-key, zero-cost usage
- a built-in browser portal for citizen intake, reviewer operations, analytics, and institution verification
- local SQLite persistence for case history and future training data capture
- local evidence-object storage with upload, checksums, MIME validation, thumbnails, and signed downloads
- local authentication and authorization with citizen, operator, reviewer, institution, and admin roles
- corpus-backed local route and priority classifiers that also learn from reviewed SQLite cases
- corpus-backed local intake, draft, and explanation enrichment without any API key or Ollama dependency
- zero-key image understanding for uploaded evidence through local Pillow-based visual analysis
- case-level model provenance with per-stage provider, engine, prompt, classifier, and threshold metadata
- append-only tamper-evident audit logging for case creation, review actions, transitions, verification, and human overrides
- privacy controls with text PII redaction, address minimization, image masking for face/license-plate candidates, retention scheduling, and privacy export/deletion workflows
- abuse protection with request rate limiting, payload-size caps, evidence-count caps, spam checks, and basic content moderation
- observability with structured request logs, request/trace IDs, latency metrics, provider error counters, queue metrics, and review-turnaround metrics
- optional Ollama integration for richer local intake, draft, and explanation generation
- `gemini` mode for live structured multimodal calls through the Gemini Developer API free tier
- `openai` mode as an optional paid-provider path

## Architecture

The backend is organized around the same AI capabilities we identified:

1. `IntakeService`
   - converts citizen text and evidence metadata into a structured issue
2. `RoutingService`
   - selects the likely institution and department
3. `PriorityService`
   - scores urgency and marks cases that need manual review
4. `DraftingService`
   - generates an editable appeal draft
5. `VerificationService`
   - checks same-place and issue-resolved signals against institution responses
6. `ExplanationService`
   - creates a human-readable explanation of what the system decided
7. `ReviewService`
   - pushes uncertain or risky cases into a manual-review queue

## Project Layout

```text
app/
  config.py
  main.py
  models/
  providers/
  services/
  repository.py
tests/
```

## Run Locally

```bash
uvicorn app.main:app --reload
```

The OpenAPI docs will be available at `http://127.0.0.1:8000/docs`.
The browser portal is available at `http://127.0.0.1:8000/app`.

## Authentication And Authorization

The API now enforces local bearer-token authentication for secured routes. Five account roles are built in:

- `citizen`
- `operator`
- `reviewer`
- `institution`
- `admin`

By default the local stack seeds demo accounts for each role:

- `citizen.demo` / `citizen-demo-pass`
- `operator.demo` / `operator-demo-pass`
- `reviewer.demo` / `reviewer-demo-pass`
- `institution.roads` / `institution-demo-pass`
- `admin.demo` / `admin-demo-pass`

Use the auth endpoints to sign in:

- `POST /v1/auth/register` for new citizen accounts
- `POST /v1/auth/login` to get a bearer token
- `GET /v1/auth/me` to inspect the active account
- `POST /v1/auth/logout` to revoke the current token
- `GET /v1/auth/accounts` and `POST /v1/auth/accounts` for admin account management

Route permissions now work like this:

- citizens can submit evidence and cases, then read only their own cases
- reviewers and operators can work the queue, transitions, analytics, and workflow routes
- institution accounts can read and verify only cases routed to their institution
- admin accounts can access all routes plus storage, evaluation, retraining, and account-management endpoints

## Audit Ledger

The case workflow now writes an append-only audit ledger for:

- case creation
- lifecycle transitions
- workflow actions
- verification actions
- manual operational overrides

Each audit event stores:

- actor identity and role
- before/after lifecycle status
- human-override marker
- canonical hashes of AI decision snapshots and override snapshots
- previous-event linkage and a SHA-256 event hash

SQLite now enforces append-only behavior for the audit table with mutation-blocking triggers, and the app can verify the hash chain through:

- `GET /v1/cases/{case_id}/audit-log`
- `GET /v1/audit/verify`

## Privacy Controls

The case and evidence pipeline now applies privacy controls before data is persisted or exported:

- text PII redaction for emails, phone numbers, and identifier-like tokens
- address minimization for exact location strings and high-precision coordinate pairs
- local image masking for face and license-plate candidates in uploaded evidence
- case and evidence retention scheduling
- privacy export packaging and privacy deletion workflows

New privacy endpoints:

- `GET /v1/cases/{case_id}/privacy`
- `POST /v1/cases/{case_id}/privacy-export`
- `POST /v1/cases/{case_id}/privacy-delete`
- `POST /v1/privacy/retention/enforce`

`GET /health` now also exposes privacy diagnostics such as:

- `privacy_enabled`
- `privacy_redaction_backend`
- `privacy_case_retention_days`
- `privacy_evidence_retention_days`
- `privacy_export_dir`

## Abuse Protection And Observability

The runtime now enforces abuse controls before requests reach the business routes:

- in-memory client rate limiting
- request body size caps
- evidence-count caps on case submissions
- local spam/promotion detection
- duplicate-submission flood blocking

The runtime now also exposes observability features:

- structured JSON request/provider logs
- `X-Request-ID` and `X-Trace-ID` response headers
- request latency and status metrics
- provider-call latency/error/fallback counters
- queue depth and SLA-breach metrics
- review-turnaround and resolution-turnaround metrics

New admin endpoint:

- `GET /v1/observability/metrics`

`GET /health` now also reports abuse/observability diagnostics such as:

- `abuse_protection_enabled`
- `abuse_blocked_total`
- `request_total_count`
- `request_rejection_count`
- `provider_error_total`
- `review_queue_depth`
- `review_sla_breached_count`
- `review_turnaround_avg_hours`

## Test

```bash
pytest
```

## Fully Free No-Key Mode

Copy [.env.example](/Users/yasinspc/Documents/Intake/.env.example) to `.env` or export the variables directly:

```bash
cp .env.example .env
```

Then run:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Then check health:

```bash
curl http://127.0.0.1:8000/health
```

You should see:

- `requested_provider`: `localfree`
- `active_provider`: `LocalFreeProvider`
- `repository_backend`: `sqlite`

This mode does not need any API key. By default it now uses:

- SQLite for durable local case storage
- corpus-backed local text classifiers for routing and priority
- local retrieval-plus-template reasoning for intake, draft, and explanation
- local image feature extraction for uploaded evidence, including visual category hints
- geo-visual verification with Unicode-normalized place matching, coordinate distance scoring, before/after image comparison, and category-specific resolution policy
- hardened SQLite startup with automatic writable fallbacks

## Phase 1 Local Production Path

Default local Phase 1 settings:

```bash
export ASAN_PROVIDER=localfree
export ASAN_REPOSITORY_BACKEND=sqlite
export ASAN_SQLITE_PATH=./asanappeal.db
export ASAN_SQLITE_TIMEOUT_SECONDS=5
export ASAN_SQLITE_BUSY_TIMEOUT_MS=5000
export ASAN_SQLITE_JOURNAL_MODE=WAL
export ASAN_SQLITE_SYNCHRONOUS=NORMAL
export ASAN_LOCAL_ML_ENABLED=true
export ASAN_LOCAL_ML_BOOTSTRAP_DIR=./data/local_ml
export ASAN_LOCAL_ML_FEEDBACK_DIR=./.data/local_ml_feedback
export ASAN_LOCAL_ML_INCLUDE_SQLITE_FEEDBACK=true
export ASAN_LOCAL_LLM_BACKEND=heuristic
.venv/bin/uvicorn app.main:app --reload
```

This gives you a zero-cost local stack with:

- durable case history
- durable local evidence object storage
- review queue persistence
- schema-versioned SQLite migrations
- lock-aware writes with `BEGIN IMMEDIATE` retry handling
- backup and restore operations backed by SQLite's online backup API
- bootstrap-corpus routing and priority models
- automatic reviewed-case replay from SQLite into local model training
- retrain/export pipeline that writes reviewed feedback corpora and reloads the active local models
- no network dependency
- automatic fallback from an unusable SQLite path to a writable local path

`GET /health` now also exposes local ML diagnostics such as:

- `routing_model_training_examples`
- `routing_model_training_sources`
- `routing_model_label_counts`
- `priority_model_training_examples`
- `priority_model_training_sources`
- `priority_model_label_counts`
- `local_reasoning_backend`
- `local_reasoning_routing_examples`
- `local_reasoning_priority_examples`
- `local_image_reasoning_backend`
- `local_ml_feedback_dir`
- `local_ml_last_retrain`
- `auth_enabled`
- `auth_backend`
- `auth_account_count`
- `auth_accounts_by_role`
- `audit_event_count`
- `audit_chain_verified`
- `audit_chain_latest_hash`
- `evaluation_latest_passed`
- `evaluation_latest_generated_at`
- `evaluation_latest_report_path`

`GET /health` now also shows the resolved SQLite path and whether a fallback path was used.
When Ollama is configured, `GET /health` now performs a real dependency probe against the Ollama API and reports:

- `status: "ok"` when Ollama is reachable and the requested model is available
- `status: "degraded"` when Ollama is unreachable or the requested model is missing
- detailed fields such as `local_llm_status`, `local_llm_server_reachable`, and `local_llm_model_available`

## Optional Ollama Upgrade

If you want a richer local LLM without paying for any API, install Ollama and pull a local model first, for example:

```bash
ollama pull gemma3:4b
```

Then switch the backend:

```bash
export ASAN_PROVIDER=ollama
export ASAN_OLLAMA_MODEL=gemma3:4b
export ASAN_OLLAMA_URL=http://127.0.0.1:11434
.venv/bin/uvicorn app.main:app --reload
```

That keeps everything local and zero-cost while further improving intake, draft, and explanation quality beyond the built-in corpus-backed local reasoner.

## Optional Gemini Mode

If you later want a hosted multimodal model, set:

```bash
export ASAN_PROVIDER=gemini
export GEMINI_API_KEY=your_gemini_api_key_here
export ASAN_GEMINI_MODEL=gemini-2.5-flash
.venv/bin/uvicorn app.main:app --reload
```

## Optional OpenAI Mode

If you ever want to switch back to OpenAI later, set:

```bash
export ASAN_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key_here
export ASAN_OPENAI_MODEL=gpt-5.4
export ASAN_OPENAI_REASONING_EFFORT=high
.venv/bin/uvicorn app.main:app --reload
```

## API Endpoints

- `GET /app`
- `GET /health`
- `POST /v1/auth/register`
- `POST /v1/auth/login`
- `POST /v1/auth/logout`
- `GET /v1/auth/me`
- `GET /v1/auth/accounts`
- `POST /v1/auth/accounts`
- `POST /v1/local-ml/retrain`
- `POST /v1/evals/run`
- `GET /v1/evals/latest`
- `POST /v1/intake`
- `POST /v1/route`
- `POST /v1/priority`
- `POST /v1/draft`
- `POST /v1/verify`
- `POST /v1/explain`
- `POST /v1/evidence/upload`
- `GET /v1/evidence/{evidence_id}`
- `GET /v1/evidence/{evidence_id}/download`
- `GET /v1/evidence/{evidence_id}/thumbnail`
- `POST /v1/cases/process`
- `GET /v1/cases/{case_id}`
- `GET /v1/cases/{case_id}/original-request`
- `GET /v1/cases/{case_id}/audit-log`
- `POST /v1/cases/{case_id}/verify`
- `GET /v1/cases`
- `POST /v1/cases/{case_id}/operations`
- `POST /v1/cases/{case_id}/transition`
- `POST /v1/cases/{case_id}/workflow-actions`
- `GET /v1/review-queue`
- `GET /v1/analytics/summary`
- `GET /v1/audit/verify`
- `GET /v1/storage/backups`
- `POST /v1/storage/backup`
- `POST /v1/storage/restore`

## Built-In Portal

The app now includes a real browser UI at:

- `GET /app`

The portal includes:

- a built-in login card that authenticates against the secured API
- a citizen intake screen that uploads evidence and calls `POST /v1/cases/process`
- a reviewer desk that refreshes `GET /v1/review-queue`, opens cases, and executes workflow actions
- an analytics screen backed by `GET /v1/analytics/summary`
- an institution response screen that calls `POST /v1/cases/{case_id}/verify`

Static assets are served from:

- `GET /app-assets/styles.css`
- `GET /app-assets/app.js`

## Local ML Retraining

The local ML stack now includes a real retraining pipeline from reviewed SQLite cases.

The retraining endpoint is:

- `POST /v1/local-ml/retrain`

What it does:

- reads reviewed/operationally confirmed cases from SQLite
- exports routing and priority feedback corpora into `ASAN_LOCAL_ML_FEEDBACK_DIR`
- reloads the active `LocalFreeProvider` route and priority models in-process
- exposes the new counts and sources through `GET /health`

Feedback artifacts are written as:

- `routing_feedback.jsonl`
- `priority_feedback.jsonl`
- `retrain_report.json`

## Evaluation And Acceptance Gates

The repo now includes a built-in regression benchmark suite with gold datasets for:

- intake
- routing
- priority
- verification

Run the suite through the API:

```bash
curl -X POST http://127.0.0.1:8000/v1/evals/run
```

Inspect the latest benchmark report:

```bash
curl http://127.0.0.1:8000/v1/evals/latest
```

Evaluation artifacts are written into `ASAN_EVAL_ARTIFACT_DIR`:

- `latest_evaluation_report.json`
- `latest_threshold_calibration.json`

Each benchmark run now includes:

- task-level quality metrics
- per-task acceptance gates
- confidence threshold calibration sweeps
- mismatch samples for regression debugging

## Case Lifecycle

Case records now use a finite lifecycle instead of ad hoc status strings. The supported states are:

- `drafted`
- `needs_review`
- `ready_for_dispatch`
- `assigned`
- `in_progress`
- `resolved`
- `reopened`
- `rejected`
- `closed`

The dedicated lifecycle endpoint is:

- `POST /v1/cases/{case_id}/transition`

Use lifecycle transitions for status changes and use `POST /v1/cases/{case_id}/operations` only for operational metadata such as reviewer assignment notes and disposition reason updates. The repository enforces valid lifecycle transitions both in Python and in SQLite triggers, so invalid status jumps are rejected consistently.

## Workflow Actions

The review workflow now has a dedicated action endpoint:

- `POST /v1/cases/{case_id}/workflow-actions`

Supported actions:

- `claim`
- `assign`
- `comment`
- `approve`
- `reject`
- `dispatch`
- `close`
- `reopen`
- `verify`

These actions let operators move a case through the real review workflow instead of editing raw status fields. Assignment and claim actions set the responsible reviewer, approval clears manual review and moves the case to dispatch readiness, dispatch records handoff to the assigned owner, verify runs the verification AI against a stored original submission plus an institution response, and close or reopen complete the human review loop.

## Original Request Retrieval

The API now exposes the stored original case request in two operator-facing ways:

- `GET /v1/cases/{case_id}` now includes `original_request`
- `GET /v1/cases/{case_id}/original-request` returns only the stored request payload

This allows operators to reconstruct the exact submission payload that was originally processed, including nested submission fields and any institution-response payload that was part of the stored request.

## Evidence Ingestion

Evidence ingestion is now media-first instead of metadata-only.

The local evidence subsystem now provides:

- binary upload API
- local object storage under `ASAN_EVIDENCE_ROOT`
- SHA-256 checksums for stored files
- MIME validation based on decoded content, not only filenames
- image thumbnail generation
- signed download and thumbnail URLs

Evidence endpoints:

- `POST /v1/evidence/upload`
- `GET /v1/evidence/{evidence_id}`
- `GET /v1/evidence/{evidence_id}/download`
- `GET /v1/evidence/{evidence_id}/thumbnail`

Upload an image with raw request bytes:

```bash
curl -X POST "http://127.0.0.1:8000/v1/evidence/upload?kind=image&filename=pothole.png&description=Citizen%20photo" \
  -H "Content-Type: image/png" \
  --data-binary "@pothole.png"
```

The response includes:

- persisted evidence metadata
- a case-ready `evidence_item` payload
- a signed `download_url`
- a signed `thumbnail_url` when the upload is an image

Additional evidence settings:

```bash
export ASAN_EVIDENCE_ROOT=./.data/evidence
export ASAN_EVIDENCE_MAX_BYTES=26214400
export ASAN_EVIDENCE_SIGNED_URL_TTL_SECONDS=900
export ASAN_EVIDENCE_SIGNING_SECRET=change-me-for-production
export ASAN_EVIDENCE_THUMBNAIL_MAX_SIZE=512
```

`GET /health` now also exposes evidence-storage diagnostics such as:

- `evidence_root`
- `evidence_fallback_used`
- `evidence_object_count`
- `evidence_thumbnail_count`
- `evidence_total_bytes`
- `local_verification_backend`

## Case Verification

The operator-friendly verification path is now:

- `POST /v1/cases/{case_id}/verify`

This endpoint reuses the stored original request and the stored structured issue for the case, so operators only need to provide the institution response plus optional actor and note fields. The response returns the updated case, including the persisted verification result and the original stored request context.

Verification in `localfree` mode now uses a real local policy layer instead of token-only matching:

- Unicode-normalized place comparison
- coordinate-aware same-place scoring when coordinates are present
- before/after image scene comparison for uploaded evidence
- issue-change scoring from local visual features
- institution and category specific resolution rules

## Model Provenance

Every stored case now keeps a structured `model_context` audit trail with:

- case-level provider, model, and provenance schema version
- prompt bundle version
- classifier bundle version
- threshold-set bundle version
- per-stage provenance for:
  - `intake`
  - `routing`
  - `priority`
  - `draft`
  - `verification`
  - `explanation`
  - `review`

Each stage provenance record captures the actual runtime producer for that decision, including the provider label, engine type, model version, prompt version, classifier version when applicable, threshold-set version, concrete threshold values, and decision notes.

## Review Queue

`GET /v1/review-queue` is now a real operator queue instead of a single unpaged list.

Supported query parameters:

- `page`
- `page_size`
- `review_queue`
- `priority_level`
- `status`
- `assignee_id`
- `assignment_state`
- `sort_by`

Supported queue ordering:

- `sla`
- `priority`
- `created_at`
- `assignee`

The default `sla` ordering uses a finite local SLA policy derived from priority and case age so urgent older cases rise first. Queue items now expose:

- `assignee_id`
- `assignment_state`
- `sla_deadline_at`
- `sla_breached`

## Specialized Human Review Routing

Human review routing is now specialized instead of collapsing all manual work into a single generic queue.

Primary review queues now include:

- `triage-review`
- `legal-review`
- `evidence-quality-review`
- `urgent-safety-review`
- `institution-review:<institution-slug>`

Each `human_review` block on the case can now also include:

- `secondary_queues`
- `candidate_groups`
- `institution_queue`

This allows the system to route cases by the actual review need:

- unclear public-service classification -> triage
- conflicting verification evidence -> legal
- weak or missing evidence -> evidence quality
- high-risk safety issues -> urgent safety
- institution-owned manual follow-up -> institution-specific queue

## SQLite Storage Operations

The SQLite repository now includes:

- schema versioning through `PRAGMA user_version`
- sequential migrations tracked in `schema_migrations`
- `storage_backups` and `storage_restores` operational tables
- queryable case-operation columns for category, institution, priority, reviewer, disposition, and model metadata
- WAL plus per-connection busy-timeout tuning
- `BEGIN IMMEDIATE` write transactions with bounded retry/backoff
- online backup and restore endpoints that work from `/docs`
- filtered case listing and summary analytics backed by SQL instead of JSON blob scans

Additional local storage settings:

```bash
export ASAN_SQLITE_MAX_WRITE_RETRIES=5
export ASAN_SQLITE_WRITE_RETRY_BACKOFF_MS=50
export ASAN_SQLITE_BACKUP_DIR=./.data/backups
export ASAN_SQLITE_BACKUP_PAGES_PER_STEP=128
```

`GET /health` now also exposes storage diagnostics such as:

- `sqlite_schema_version`
- `sqlite_migration_count`
- `sqlite_case_projection_ready`
- `sqlite_write_strategy`
- `sqlite_backup_count`
- `sqlite_restore_count`
- `sqlite_backup_dir`

## Provider Strategy

`localfree` mode gives you deterministic local behavior, no API dependency, and no private key requirement.

`gemini` mode uses:

- the official Google GenAI SDK
- the Gemini Developer API free tier
- structured parsing into Pydantic models
- image-aware evidence handling for local image paths and `data:` URLs
- heuristic fallback if the API call fails

`openai` mode remains available and uses:

- the OpenAI Python SDK
- the Responses API
- structured parsing into Pydantic models
- image-aware evidence handling for local image paths, image URLs, and `data:` URLs
- heuristic fallback if the API call fails

This means the system stays usable in development, but becomes genuinely live when an API key is configured.

## Suggested Next Build Steps

1. Replace the embedded Naive Bayes seed data with reviewed real-case exports from SQLite.
2. Add local eval datasets for intake, routing, priority, and verification.
3. Add institution and reviewer authentication.
4. Add a frontend for citizen submission and review operations.

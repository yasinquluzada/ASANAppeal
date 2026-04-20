from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Callable
from uuid import uuid4

from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

logger = logging.getLogger("asanappeal.observability")

_REQUEST_ID: ContextVar[str | None] = ContextVar("asan_request_id", default=None)
_TRACE_ID: ContextVar[str | None] = ContextVar("asan_trace_id", default=None)


def current_request_id() -> str | None:
    return _REQUEST_ID.get()


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


@dataclass
class ProviderCallStats:
    calls: int = 0
    errors: int = 0
    fallback_errors: int = 0
    latency_ms_total: float = 0.0
    latency_ms_max: float = 0.0


class ObservabilityService:
    def __init__(self) -> None:
        self.request_total = 0
        self.request_in_flight = 0
        self.request_rejections = 0
        self.request_status_counts: dict[str, int] = defaultdict(int)
        self.request_path_counts: dict[str, int] = defaultdict(int)
        self.request_latency_totals_ms: dict[str, float] = defaultdict(float)
        self.request_latency_max_ms: dict[str, float] = defaultdict(float)
        self.provider_calls: dict[str, ProviderCallStats] = defaultdict(ProviderCallStats)

    def diagnostics(self, repository=None) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_total_count": self.request_total,
            "request_in_flight": self.request_in_flight,
            "request_rejection_count": self.request_rejections,
            "request_status_counts": dict(sorted(self.request_status_counts.items())),
        }
        snapshot = self.snapshot(repository)
        payload.update(
            {
                "provider_error_total": snapshot["provider"]["errors_total"],
                "provider_fallback_error_total": snapshot["provider"]["fallback_errors_total"],
                "review_queue_depth": snapshot["queues"]["total_items"],
                "review_sla_breached_count": snapshot["queues"]["sla_breached_items"],
                "review_turnaround_avg_hours": snapshot["reviews"]["review_turnaround_avg_hours"],
            }
        )
        return payload

    def start_request(self, *, method: str, path: str, client_ip: str) -> tuple[str, str, float]:
        request_id = uuid4().hex[:16]
        trace_id = uuid4().hex
        _REQUEST_ID.set(request_id)
        _TRACE_ID.set(trace_id)
        self.request_total += 1
        self.request_in_flight += 1
        started_at = time.perf_counter()
        self._log(
            "request_start",
            method=method,
            path=path,
            client_ip=client_ip,
            request_id=request_id,
            trace_id=trace_id,
        )
        return request_id, trace_id, started_at

    def finish_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        started_at: float,
        client_ip: str,
    ) -> None:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        self.request_in_flight = max(0, self.request_in_flight - 1)
        self.request_status_counts[str(status_code)] += 1
        self.request_path_counts[path] += 1
        self.request_latency_totals_ms[path] += latency_ms
        self.request_latency_max_ms[path] = max(self.request_latency_max_ms[path], latency_ms)
        self._log(
            "request_finish",
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            client_ip=client_ip,
            request_id=current_request_id(),
            trace_id=current_trace_id(),
        )
        _REQUEST_ID.set(None)
        _TRACE_ID.set(None)

    def record_request_rejection(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        reason: str,
        category: str,
        client_ip: str,
    ) -> None:
        self.request_rejections += 1
        self._log(
            "request_rejected",
            method=method,
            path=path,
            status_code=status_code,
            reason=reason,
            category=category,
            client_ip=client_ip,
            request_id=current_request_id(),
            trace_id=current_trace_id(),
        )

    def record_provider_call(
        self,
        *,
        stage: str,
        provider_name: str,
        latency_ms: float,
        error: BaseException | None = None,
        fallback_error: bool = False,
    ) -> None:
        bucket = self.provider_calls[f"{provider_name}:{stage}"]
        bucket.calls += 1
        bucket.latency_ms_total += latency_ms
        bucket.latency_ms_max = max(bucket.latency_ms_max, latency_ms)
        if error is not None:
            bucket.errors += 1
        if fallback_error:
            bucket.fallback_errors += 1
        self._log(
            "provider_call",
            provider=provider_name,
            stage=stage,
            latency_ms=round(latency_ms, 2),
            error_type=type(error).__name__ if error is not None else None,
            fallback_error=fallback_error,
            request_id=current_request_id(),
            trace_id=current_trace_id(),
        )

    def record_provider_fallback_error(self, *, stage: str, provider_name: str, error: BaseException) -> None:
        bucket = self.provider_calls[f"{provider_name}:{stage}"]
        bucket.fallback_errors += 1
        bucket.errors += 1
        self._log(
            "provider_fallback_error",
            provider=provider_name,
            stage=stage,
            error_type=type(error).__name__,
            message=str(error),
            request_id=current_request_id(),
            trace_id=current_trace_id(),
        )

    def snapshot(self, repository=None) -> dict[str, object]:
        provider_items: dict[str, dict[str, object]] = {}
        errors_total = 0
        fallback_errors_total = 0
        for key, stats in sorted(self.provider_calls.items()):
            avg_latency = stats.latency_ms_total / stats.calls if stats.calls else 0.0
            provider_items[key] = {
                "calls": stats.calls,
                "errors": stats.errors,
                "fallback_errors": stats.fallback_errors,
                "avg_latency_ms": round(avg_latency, 2),
                "max_latency_ms": round(stats.latency_ms_max, 2),
            }
            errors_total += stats.errors
            fallback_errors_total += stats.fallback_errors

        request_paths: dict[str, dict[str, object]] = {}
        for path, count in sorted(self.request_path_counts.items()):
            avg_latency = (
                self.request_latency_totals_ms[path] / count
                if count
                else 0.0
            )
            request_paths[path] = {
                "count": count,
                "avg_latency_ms": round(avg_latency, 2),
                "max_latency_ms": round(self.request_latency_max_ms[path], 2),
            }

        queue_metrics = self._queue_metrics(repository)
        review_metrics = self._review_metrics(repository)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "request": {
                "total": self.request_total,
                "in_flight": self.request_in_flight,
                "rejections": self.request_rejections,
                "status_counts": dict(sorted(self.request_status_counts.items())),
                "paths": request_paths,
            },
            "provider": {
                "errors_total": errors_total,
                "fallback_errors_total": fallback_errors_total,
                "items": provider_items,
            },
            "queues": queue_metrics,
            "reviews": review_metrics,
        }

    def _queue_metrics(self, repository) -> dict[str, object]:
        if repository is None:
            return {
                "total_items": 0,
                "sla_breached_items": 0,
                "counts_by_queue": {},
            }
        queue_page = repository.query_review_queue(page=1, page_size=10000)
        items = queue_page["items"]
        counts_by_queue: dict[str, int] = defaultdict(int)
        sla_breached = 0
        for item in items:
            counts_by_queue[str(item["review_queue"])] += 1
            if item["sla_breached"]:
                sla_breached += 1
        return {
            "total_items": len(items),
            "sla_breached_items": sla_breached,
            "counts_by_queue": dict(sorted(counts_by_queue.items())),
        }

    def _review_metrics(self, repository) -> dict[str, object]:
        if repository is None:
            return {
                "reviewed_case_count": 0,
                "review_turnaround_avg_hours": 0.0,
                "resolution_turnaround_avg_hours": 0.0,
            }
        items = repository.list_cases(limit=10000)
        review_durations: list[float] = []
        resolution_durations: list[float] = []
        for item in items:
            case = repository.get_case(str(item["case_id"]))
            if case is None:
                continue
            created_at = case.created_at
            first_human_at = None
            for action in case.operations.workflow_history:
                if action.actor_id and action.actor_id != "system":
                    first_human_at = action.acted_at
                    break
            if first_human_at is None and case.operations.reviewer_id and case.operations.status_updated_at:
                first_human_at = case.operations.status_updated_at
            if first_human_at is not None:
                review_durations.append((first_human_at - created_at).total_seconds() / 3600.0)
            if case.operations.disposition_updated_at is not None:
                resolution_durations.append(
                    (case.operations.disposition_updated_at - created_at).total_seconds() / 3600.0
                )
        return {
            "reviewed_case_count": len(review_durations),
            "review_turnaround_avg_hours": round(
                sum(review_durations) / len(review_durations), 2
            )
            if review_durations
            else 0.0,
            "resolution_turnaround_avg_hours": round(
                sum(resolution_durations) / len(resolution_durations), 2
            )
            if resolution_durations
            else 0.0,
        }

    def _log(self, event: str, **payload: object) -> None:
        logger.info(
            json.dumps(
                {"event": event, **payload},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )


class InstrumentedProvider:
    def __init__(self, provider, observability: ObservabilityService) -> None:
        self._provider = provider
        self._observability = observability
        attach_provider_error_observer(provider, observability)

    def analyze_submission(self, submission):
        return self._call("intake", self._provider.analyze_submission, submission)

    def route_issue(self, submission, structured_issue):
        return self._call("routing", self._provider.route_issue, submission, structured_issue)

    def assess_priority(self, submission, structured_issue, routing):
        return self._call(
            "priority",
            self._provider.assess_priority,
            submission,
            structured_issue,
            routing,
        )

    def draft_appeal(self, submission, structured_issue, routing, priority):
        return self._call(
            "draft",
            self._provider.draft_appeal,
            submission,
            structured_issue,
            routing,
            priority,
        )

    def verify_resolution(self, original_submission, structured_issue, institution_response):
        return self._call(
            "verification",
            self._provider.verify_resolution,
            original_submission,
            structured_issue,
            institution_response,
        )

    def explain_case(self, structured_issue, routing, priority, human_review, verification=None):
        return self._call(
            "explanation",
            self._provider.explain_case,
            structured_issue,
            routing,
            priority,
            human_review,
            verification,
        )

    def clear_stage_provenance(self) -> None:
        self._provider.clear_stage_provenance()

    def get_stage_provenance(self):
        return self._provider.get_stage_provenance()

    def _call(self, stage: str, func: Callable[..., Any], *args: Any) -> Any:
        started = time.perf_counter()
        try:
            result = func(*args)
        except Exception as exc:
            self._observability.record_provider_call(
                stage=stage,
                provider_name=type(self._provider).__name__,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=exc,
            )
            raise
        self._observability.record_provider_call(
            stage=stage,
            provider_name=type(self._provider).__name__,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._provider, item)


def attach_provider_error_observer(provider, observability: ObservabilityService) -> None:
    if hasattr(provider, "_provider_error_observer"):
        setattr(
            provider,
            "_provider_error_observer",
            lambda *, stage, provider_name, error: observability.record_provider_fallback_error(
                stage=stage,
                provider_name=provider_name,
                error=error,
            ),
        )
    nested = getattr(provider, "_heuristic", None)
    if nested is not None and nested is not provider:
        attach_provider_error_observer(nested, observability)


class RequestControlMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        app = scope.get("app")
        runtime = getattr(getattr(app, "state", None), "runtime", None)
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        client_ip = (scope.get("client") or ("unknown", 0))[0]

        if runtime is None:
            await self.app(scope, receive, send)
            return

        observability = runtime.observability_service
        abuse = runtime.abuse_service
        request_id, trace_id, started_at = observability.start_request(
            method=method,
            path=path,
            client_ip=client_ip,
        )
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        scope["state"]["trace_id"] = trace_id

        decision = abuse.check_request_rate(client_key=client_ip, path=path)
        if not decision.allowed:
            observability.record_request_rejection(
                method=method,
                path=path,
                status_code=decision.status_code,
                reason=decision.reason or "Rejected",
                category=decision.category or "abuse_rejection",
                client_ip=client_ip,
            )
            response = JSONResponse({"detail": decision.reason}, status_code=decision.status_code)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            await response(scope, receive, send)
            observability.finish_request(
                method=method,
                path=path,
                status_code=decision.status_code,
                started_at=started_at,
                client_ip=client_ip,
            )
            return

        max_body_bytes = abuse.body_limit_for_path(path)
        content_length_header = None
        for raw_key, raw_value in scope.get("headers", []):
            if raw_key.lower() == b"content-length":
                content_length_header = raw_value.decode("latin-1")
                break
        if content_length_header and content_length_header.isdigit():
            if int(content_length_header) > max_body_bytes:
                observability.record_request_rejection(
                    method=method,
                    path=path,
                    status_code=413,
                    reason="Request body exceeded the configured payload limit.",
                    category="payload_too_large",
                    client_ip=client_ip,
                )
                response = JSONResponse(
                    {"detail": "Request body exceeded the configured payload limit."},
                    status_code=413,
                )
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Trace-ID"] = trace_id
                await response(scope, receive, send)
                observability.finish_request(
                    method=method,
                    path=path,
                    status_code=413,
                    started_at=started_at,
                    client_ip=client_ip,
                )
                return

        response_status = 500
        body_total = 0

        async def guarded_receive():
            nonlocal body_total
            message = await receive()
            if message["type"] == "http.request":
                body_total += len(message.get("body", b""))
                if body_total > max_body_bytes:
                    raise RequestBodyTooLarge()
            return message

        async def observed_send(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
                headers.append("X-Trace-ID", trace_id)
            await send(message)

        try:
            await self.app(scope, guarded_receive, observed_send)
        except RequestBodyTooLarge:
            observability.record_request_rejection(
                method=method,
                path=path,
                status_code=413,
                reason="Request body exceeded the configured payload limit.",
                category="payload_too_large",
                client_ip=client_ip,
            )
            response_status = 413
            response = JSONResponse(
                {"detail": "Request body exceeded the configured payload limit."},
                status_code=413,
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            await response(scope, receive, send)
        except Exception:
            observability.finish_request(
                method=method,
                path=path,
                status_code=500,
                started_at=started_at,
                client_ip=client_ip,
            )
            raise
        else:
            observability.finish_request(
                method=method,
                path=path,
                status_code=response_status,
                started_at=started_at,
                client_ip=client_ip,
            )


class RequestBodyTooLarge(RuntimeError):
    pass

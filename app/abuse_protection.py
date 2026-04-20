from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic
import re

from app.config import Settings
from app.models.api import ProcessCaseRequest
from app.models.domain import AuthenticatedUser

PROMOTION_KEYWORDS = {
    "buy now",
    "discount",
    "promo code",
    "telegram channel",
    "whatsapp group",
    "crypto",
    "casino",
    "forex",
    "investment opportunity",
}
ABUSIVE_KEYWORDS = {
    "kill",
    "murder",
    "bomb",
    "terrorist",
    "rape",
}


@dataclass(frozen=True)
class AbuseDecision:
    allowed: bool
    status_code: int
    reason: str | None = None
    category: str | None = None


class AbuseProtectionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rate_windows: dict[str, deque[float]] = defaultdict(deque)
        self._spam_windows: dict[str, deque[float]] = defaultdict(deque)
        self._blocked_counts: dict[str, int] = defaultdict(int)

    def diagnostics(self) -> dict[str, object]:
        return {
            "abuse_protection_enabled": True,
            "abuse_rate_limit_window_seconds": self.settings.request_rate_limit_window_seconds,
            "abuse_rate_limit_max_requests": self.settings.request_rate_limit_max_requests,
            "abuse_auth_rate_limit_max_requests": self.settings.auth_rate_limit_max_requests,
            "abuse_max_evidence_items": self.settings.abuse_max_evidence_items,
            "abuse_max_text_chars": self.settings.abuse_max_text_chars,
            "abuse_duplicate_threshold": self.settings.abuse_duplicate_threshold,
            "abuse_blocked_total": sum(self._blocked_counts.values()),
            "abuse_blocked_by_reason": dict(sorted(self._blocked_counts.items())),
        }

    def body_limit_for_path(self, path: str) -> int:
        if path.startswith("/v1/evidence/upload"):
            return self.settings.evidence_max_bytes
        if path.startswith("/docs") or path.startswith("/app-assets"):
            return max(self.settings.request_max_body_bytes, 1024 * 1024)
        return self.settings.request_max_body_bytes

    def check_request_rate(self, *, client_key: str, path: str) -> AbuseDecision:
        now = monotonic()
        window_seconds = float(self.settings.request_rate_limit_window_seconds)
        limit = (
            self.settings.auth_rate_limit_max_requests
            if path.startswith("/v1/auth/")
            else self.settings.request_rate_limit_max_requests
        )
        bucket = f"{client_key}:{'auth' if path.startswith('/v1/auth/') else 'general'}"
        queue = self._rate_windows[bucket]
        while queue and now - queue[0] > window_seconds:
            queue.popleft()
        if len(queue) >= limit:
            self._blocked_counts["rate_limit"] += 1
            return AbuseDecision(
                allowed=False,
                status_code=429,
                reason="Rate limit exceeded for this client.",
                category="rate_limit",
            )
        queue.append(now)
        return AbuseDecision(allowed=True, status_code=200)

    def validate_process_case(
        self,
        payload: ProcessCaseRequest,
        *,
        current_user: AuthenticatedUser,
    ) -> AbuseDecision:
        submission = payload.submission
        total_evidence = len(submission.evidence) + len(
            payload.institution_response.evidence if payload.institution_response else []
        )
        if total_evidence > self.settings.abuse_max_evidence_items:
            self._blocked_counts["too_many_evidence_items"] += 1
            return AbuseDecision(
                allowed=False,
                status_code=400,
                reason=f"Too many evidence items; the maximum is {self.settings.abuse_max_evidence_items}.",
                category="too_many_evidence_items",
            )

        text_fields = [
            submission.citizen_text,
            submission.location_hint or "",
            payload.institution_response.response_text if payload.institution_response else "",
            payload.institution_response.location_hint if payload.institution_response else "",
        ]
        for text in text_fields:
            if text and len(text) > self.settings.abuse_max_text_chars:
                self._blocked_counts["text_too_large"] += 1
                return AbuseDecision(
                    allowed=False,
                    status_code=400,
                    reason=f"Text fields must stay under {self.settings.abuse_max_text_chars} characters.",
                    category="text_too_large",
                )

        moderation = self._moderation_decision(text_fields)
        if not moderation.allowed:
            return moderation

        spam = self._spam_decision(submission.citizen_text, submission.location_hint, current_user.user_id)
        if not spam.allowed:
            return spam
        return AbuseDecision(allowed=True, status_code=200)

    def register_case_submission(
        self,
        payload: ProcessCaseRequest,
        *,
        current_user: AuthenticatedUser,
    ) -> None:
        fingerprint = self._spam_fingerprint(
            payload.submission.citizen_text,
            payload.submission.location_hint,
            current_user.user_id,
        )
        now = monotonic()
        queue = self._spam_windows[fingerprint]
        queue.append(now)
        while queue and now - queue[0] > float(self.settings.abuse_spam_window_seconds):
            queue.popleft()

    def _moderation_decision(self, text_fields: list[str]) -> AbuseDecision:
        normalized = " ".join(text_fields).lower()
        url_count = len(re.findall(r"https?://|www\.", normalized))
        repeated_characters = re.search(r"(.)\1{7,}", normalized)
        matched_promotion = next((keyword for keyword in PROMOTION_KEYWORDS if keyword in normalized), None)
        matched_abusive = next((keyword for keyword in ABUSIVE_KEYWORDS if keyword in normalized), None)
        if matched_abusive:
            self._blocked_counts["abusive_content"] += 1
            return AbuseDecision(
                allowed=False,
                status_code=400,
                reason="Submitted content contains prohibited abusive or violent language.",
                category="abusive_content",
            )
        if matched_promotion or url_count >= 3 or repeated_characters:
            self._blocked_counts["spam_content"] += 1
            return AbuseDecision(
                allowed=False,
                status_code=400,
                reason="Submitted content appears to be promotional or spam-like.",
                category="spam_content",
            )
        return AbuseDecision(allowed=True, status_code=200)

    def _spam_fingerprint(self, citizen_text: str, location_hint: str | None, actor_id: str) -> str:
        normalized_text = re.sub(r"[^a-z0-9]+", " ", (citizen_text or "").lower()).strip()
        normalized_location = re.sub(r"[^a-z0-9]+", " ", (location_hint or "").lower()).strip()
        normalized_text = " ".join(normalized_text.split()[:24])
        normalized_location = " ".join(normalized_location.split()[:12])
        return f"{actor_id}:{normalized_text}:{normalized_location}"

    def _spam_decision(
        self,
        citizen_text: str,
        location_hint: str | None,
        actor_id: str,
    ) -> AbuseDecision:
        now = monotonic()
        fingerprint = self._spam_fingerprint(citizen_text, location_hint, actor_id)
        queue = self._spam_windows[fingerprint]
        while queue and now - queue[0] > float(self.settings.abuse_spam_window_seconds):
            queue.popleft()
        if len(queue) >= self.settings.abuse_duplicate_threshold:
            self._blocked_counts["duplicate_spam"] += 1
            return AbuseDecision(
                allowed=False,
                status_code=429,
                reason="Repeated duplicate submissions were blocked as spam.",
                category="duplicate_spam",
            )
        return AbuseDecision(allowed=True, status_code=200)

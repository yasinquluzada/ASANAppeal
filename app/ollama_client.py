from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from app.models.api import SubmissionInput

logger = logging.getLogger(__name__)


IMAGE_SUFFIX_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class OllamaProbeResult:
    status: str
    base_url: str
    model_requested: str
    server_reachable: bool
    model_available: bool
    dependency_ok: bool
    available_model_count: int


def _load_submission_images(submission: SubmissionInput) -> list[str]:
    images: list[str] = []
    for item in submission.evidence[:2]:
        if item.kind.value != "image" or not item.uri:
            continue
        path = Path(item.uri).expanduser()
        if not path.exists() or path.suffix.lower() not in IMAGE_SUFFIX_TO_MIME:
            continue
        images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
    return images


class OllamaLocalClient:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._cached_probe: OllamaProbeResult | None = None
        self._cached_probe_at = 0.0
        self._probe_ttl_seconds = 2.0

    def _request(self, path: str, payload: dict | None = None, *, silent: bool = False) -> dict | None:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            if not silent:
                logger.warning("Ollama request failed, using heuristic fallback: %s", exc)
            return None

    def _normalize_model_name(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized.endswith(":latest"):
            return normalized[:-7]
        return normalized

    def probe(self, *, force_refresh: bool = False) -> OllamaProbeResult:
        now = time.monotonic()
        if (
            not force_refresh
            and self._cached_probe is not None
            and now - self._cached_probe_at < self._probe_ttl_seconds
        ):
            return self._cached_probe

        response = self._request("/api/tags", silent=True)
        if not response or not isinstance(response.get("models"), list):
            probe = OllamaProbeResult(
                status="unreachable",
                base_url=self.base_url,
                model_requested=self.model,
                server_reachable=False,
                model_available=False,
                dependency_ok=False,
                available_model_count=0,
            )
        else:
            available_names: set[str] = set()
            for item in response["models"]:
                if not isinstance(item, dict):
                    continue
                for key in ("name", "model"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        available_names.add(self._normalize_model_name(value))
            model_available = self._normalize_model_name(self.model) in available_names
            probe = OllamaProbeResult(
                status="ready" if model_available else "model_missing",
                base_url=self.base_url,
                model_requested=self.model,
                server_reachable=True,
                model_available=model_available,
                dependency_ok=model_available,
                available_model_count=len(available_names),
            )

        self._cached_probe = probe
        self._cached_probe_at = now
        return probe

    def is_available(self) -> bool:
        return self.probe().dependency_ok

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict | None:
        message: dict[str, object] = {
            "role": "user",
            "content": user_prompt,
        }
        if images:
            message["images"] = images

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                message,
            ],
        }
        response = self._request("/api/chat", payload)
        if not response:
            return None
        self._cached_probe = OllamaProbeResult(
            status="ready",
            base_url=self.base_url,
            model_requested=self.model,
            server_reachable=True,
            model_available=True,
            dependency_ok=True,
            available_model_count=max(self._cached_probe.available_model_count, 1)
            if self._cached_probe
            else 1,
        )
        self._cached_probe_at = time.monotonic()
        try:
            content = response["message"]["content"]
            return json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ollama returned non-JSON content, using heuristic fallback: %s", exc)
            return None

    def analyze_submission(self, submission: SubmissionInput, *, system_prompt: str, user_prompt: str) -> dict | None:
        return self.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=_load_submission_images(submission),
        )

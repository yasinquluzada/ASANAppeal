from __future__ import annotations

import base64
from typing import Any

from app.providers.structured_provider import (
    SchemaT,
    StructuredProviderBase,
    has_placeholder_secret,
)


class OpenAIProvider(StructuredProviderBase):
    """Live OpenAI-backed provider using structured responses with heuristic fallback."""

    provider_label = "OpenAI"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        reasoning_effort: str = "high",
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        if has_placeholder_secret(api_key):
            raise RuntimeError("OPENAI_API_KEY is missing or still a placeholder value.")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only if selected
            raise RuntimeError(
                "OpenAI provider requested but the `openai` package is not installed."
            ) from exc

        super().__init__()
        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model = model
        self._reasoning_effort = reasoning_effort

    def _structured_completion(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        sections: list[tuple[str, str]],
        evidence: list[Any],
    ) -> SchemaT:
        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": block}
            for block in self._iter_text_blocks(sections, evidence)
        ]
        for image_bytes, mime_type in self._iter_image_payloads(evidence):
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{mime_type};base64,"
                        f"{base64.b64encode(image_bytes).decode('utf-8')}"
                    ),
                }
            )

        request: dict[str, Any] = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "text_format": schema,
        }
        if self._reasoning_effort:
            request["reasoning"] = {"effort": self._reasoning_effort}

        response = self._client.responses.parse(**request)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("Structured output parsing returned no result.")
        return parsed

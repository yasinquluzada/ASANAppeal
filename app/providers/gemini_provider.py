from __future__ import annotations

from typing import Any

from app.providers.structured_provider import (
    SchemaT,
    StructuredProviderBase,
    has_placeholder_secret,
)


class GeminiProvider(StructuredProviderBase):
    """Live Gemini-backed provider using the free-tier-capable Gemini Developer API."""

    provider_label = "Gemini"

    def __init__(self, api_key: str | None, model: str) -> None:
        if has_placeholder_secret(api_key):
            raise RuntimeError("GEMINI_API_KEY is missing or still a placeholder value.")
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only if selected
            raise RuntimeError(
                "Gemini provider requested but the `google-genai` package is not installed."
            ) from exc

        super().__init__()
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _structured_completion(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        sections: list[tuple[str, str]],
        evidence: list[Any],
    ) -> SchemaT:
        from google.genai import types  # type: ignore

        contents: list[Any] = list(self._iter_text_blocks(sections, evidence))
        for image_bytes, mime_type in self._iter_image_payloads(evidence):
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned no text response.")
        return schema.model_validate_json(text)

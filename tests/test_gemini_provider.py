from __future__ import annotations

import sys
import types
from pathlib import Path

from app.models.api import SubmissionInput
from app.models.domain import StructuredIssue
from app.providers.gemini_provider import GeminiProvider


class _FakePart:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str):
        return {"inline_data": {"data": data, "mime_type": mime_type}}


class _FakeGenerateContentConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        payload = StructuredIssue(
            category="road_damage",
            issue_type="Road Damage",
            summary="Deep pothole on a busy road.",
            extracted_signals=["pothole", "busy road"],
            missing_information=[],
            confidence=0.88,
        )
        return _FakeResponse(payload.model_dump_json())


class _FakeGeminiClient:
    last_client = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = _FakeModels()
        _FakeGeminiClient.last_client = self


def test_gemini_provider_uses_structured_output_and_image_parts(
    tmp_path: Path, monkeypatch
) -> None:
    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=_FakeGeminiClient)
    fake_types_module = types.SimpleNamespace(
        Part=_FakePart,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    fake_google_genai = types.ModuleType("google.genai")
    fake_google_genai.types = fake_types_module

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_google_genai)

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    result = provider.analyze_submission(
        SubmissionInput(
            citizen_text="A dangerous pothole is on the main road.",
            location_hint="Baku city center",
            evidence=[{"kind": "image", "uri": str(image_path), "description": "road pothole"}],
        )
    )

    assert result.category == "road_damage"
    call = _FakeGeminiClient.last_client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"]["response_mime_type"] == "application/json"
    assert any(
        isinstance(item, dict) and item.get("inline_data", {}).get("mime_type") == "image/jpeg"
        for item in call["contents"]
    )


def test_gemini_provider_falls_back_to_heuristics_on_generation_error(monkeypatch) -> None:
    class _BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("boom")

    class _BrokenClient:
        def __init__(self, **kwargs):
            self.models = _BrokenModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=_BrokenClient)
    fake_types_module = types.SimpleNamespace(
        Part=_FakePart,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    fake_google_genai = types.ModuleType("google.genai")
    fake_google_genai.types = fake_types_module

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_google_genai)

    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    result = provider.analyze_submission(
        SubmissionInput(
            citizen_text="There is a pothole on the road.",
            location_hint="Baku",
        )
    )

    assert result.category == "road_damage"

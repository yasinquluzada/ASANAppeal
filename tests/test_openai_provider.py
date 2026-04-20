from __future__ import annotations

import sys
import types
from pathlib import Path

from app.models.api import SubmissionInput
from app.models.domain import StructuredIssue
from app.providers.openai_provider import OpenAIProvider


class _FakeParsedResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


class _FakeResponsesClient:
    def __init__(self) -> None:
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["text_format"]
        if schema is StructuredIssue:
            return _FakeParsedResponse(
                StructuredIssue(
                    category="road_damage",
                    issue_type="Road Damage",
                    summary="Deep pothole on a busy road.",
                    extracted_signals=["pothole", "busy road"],
                    missing_information=[],
                    confidence=0.91,
                )
            )
        raise AssertionError(f"Unexpected schema {schema}")


class _FakeOpenAIClient:
    last_client = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = _FakeResponsesClient()
        _FakeOpenAIClient.last_client = self


def test_openai_provider_uses_structured_parse_and_images(tmp_path: Path, monkeypatch) -> None:
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.4",
        reasoning_effort="high",
        timeout_seconds=45,
        max_retries=1,
    )
    result = provider.analyze_submission(
        SubmissionInput(
            citizen_text="A dangerous pothole is on the main road.",
            location_hint="Baku city center",
            evidence=[{"kind": "image", "uri": str(image_path), "description": "road pothole"}],
        )
    )

    assert result.category == "road_damage"
    call = _FakeOpenAIClient.last_client.responses.calls[0]
    assert call["model"] == "gpt-5.4"
    assert call["reasoning"] == {"effort": "high"}
    user_content = call["input"][1]["content"]
    assert any(
        block.get("type") == "input_image" and block.get("image_url", "").startswith("data:image/jpeg;base64,")
        for block in user_content
    )


def test_openai_provider_falls_back_to_heuristics_on_parse_error(monkeypatch) -> None:
    class _BrokenResponsesClient:
        def parse(self, **kwargs):
            raise RuntimeError("boom")

    class _BrokenOpenAIClient:
        def __init__(self, **kwargs):
            self.responses = _BrokenResponsesClient()

    fake_module = types.SimpleNamespace(OpenAI=_BrokenOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = OpenAIProvider(api_key="test-key", model="gpt-5.4")
    result = provider.analyze_submission(
        SubmissionInput(
            citizen_text="There is a pothole on the road.",
            location_hint="Baku",
        )
    )

    assert result.category == "road_damage"

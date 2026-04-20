from app.providers import create_provider
from app.providers.local_provider import LocalFreeProvider
from app.config import Settings
from app.ollama_client import OllamaProbeResult
from app.models.api import SubmissionInput
from app.models.domain import HumanReviewTask


def test_localfree_is_default_provider() -> None:
    settings = Settings(provider="localfree")
    provider = create_provider(settings)
    assert isinstance(provider, LocalFreeProvider)
    diagnostics = provider.diagnostics()
    assert diagnostics["local_reasoning_backend"] == "retrieval-template"
    assert diagnostics["local_reasoning_routing_examples"] >= 60
    assert diagnostics["local_image_reasoning_backend"] == "pillow-vision-heuristics"
    assert diagnostics["local_verification_backend"] == "geo-visual-policy"
    assert diagnostics["routing_model_training_examples"] >= 60
    assert diagnostics["priority_model_training_examples"] >= 30


def test_heuristic_alias_resolves_to_localfree() -> None:
    settings = Settings(provider="heuristic")
    provider = create_provider(settings)
    assert isinstance(provider, LocalFreeProvider)


def test_ollama_alias_resolves_to_localfree() -> None:
    settings = Settings(provider="ollama")
    provider = create_provider(settings)
    assert isinstance(provider, LocalFreeProvider)


def test_unknown_provider_preserves_local_settings() -> None:
    settings = Settings(
        provider="unknown-provider",
        local_ml_enabled=False,
        local_llm_backend="heuristic",
    )
    provider = create_provider(settings)

    assert isinstance(provider, LocalFreeProvider)
    assert provider.settings.local_ml_enabled is False
    assert provider.routing_model is None
    assert provider.priority_model is None


def test_gemini_fallback_preserves_local_settings(monkeypatch) -> None:
    from app import providers as provider_module

    class _BrokenGeminiProvider:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("gemini init failed")

    monkeypatch.setattr(provider_module, "GeminiProvider", _BrokenGeminiProvider)

    settings = Settings(
        provider="gemini",
        gemini_api_key="placeholder-key",
        local_ml_enabled=False,
        local_llm_backend="heuristic",
    )
    provider = create_provider(settings)

    assert isinstance(provider, LocalFreeProvider)
    assert provider.settings.local_ml_enabled is False
    assert provider.routing_model is None
    assert provider.priority_model is None


def test_openai_fallback_preserves_local_settings(monkeypatch) -> None:
    from app import providers as provider_module

    class _BrokenOpenAIProvider:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("openai init failed")

    monkeypatch.setattr(provider_module, "OpenAIProvider", _BrokenOpenAIProvider)

    settings = Settings(
        provider="openai",
        openai_api_key="placeholder-key",
        local_ml_enabled=False,
        local_llm_backend="heuristic",
    )
    provider = create_provider(settings)

    assert isinstance(provider, LocalFreeProvider)
    assert provider.settings.local_ml_enabled is False
    assert provider.routing_model is None
    assert provider.priority_model is None


def test_ollama_diagnostics_report_unreachable(monkeypatch) -> None:
    settings = Settings(provider="ollama", local_llm_backend="ollama")
    provider = create_provider(settings)

    monkeypatch.setattr(
        provider.ollama_client,
        "probe",
        lambda force_refresh=False: OllamaProbeResult(
            status="unreachable",
            base_url="http://127.0.0.1:11434",
            model_requested="gemma3:4b",
            server_reachable=False,
            model_available=False,
            dependency_ok=False,
            available_model_count=0,
        ),
    )

    diagnostics = provider.diagnostics(force_refresh=True)
    assert diagnostics["local_llm_status"] == "unreachable"
    assert diagnostics["local_llm_dependency_ok"] is False
    assert diagnostics["local_llm_server_reachable"] is False


def test_ollama_diagnostics_report_ready(monkeypatch) -> None:
    settings = Settings(provider="ollama", local_llm_backend="ollama")
    provider = create_provider(settings)

    monkeypatch.setattr(
        provider.ollama_client,
        "probe",
        lambda force_refresh=False: OllamaProbeResult(
            status="ready",
            base_url="http://127.0.0.1:11434",
            model_requested="gemma3:4b",
            server_reachable=True,
            model_available=True,
            dependency_ok=True,
            available_model_count=3,
        ),
    )

    diagnostics = provider.diagnostics(force_refresh=True)
    assert diagnostics["local_llm_status"] == "ready"
    assert diagnostics["local_llm_dependency_ok"] is True
    assert diagnostics["local_llm_model_available"] is True


def test_local_reasoning_upgrades_no_ollama_intake_to_specific_category() -> None:
    provider = LocalFreeProvider(Settings(provider="localfree", local_llm_backend="heuristic"))

    issue = provider.analyze_submission(
        SubmissionInput(
            citizen_text="Retaining wall collapse damaged the carriageway beside the library underpass.",
            location_hint="Library underpass",
            time_hint="2026-04-19 13:00",
        )
    )

    assert issue.category == "road_damage"
    assert issue.issue_type == "Road Damage"
    assert "underpass" in issue.summary.lower()
    assert any(signal in {"underpass", "carriageway"} for signal in issue.extracted_signals)


def test_local_reasoning_builds_richer_no_ollama_draft_and_explanation() -> None:
    provider = LocalFreeProvider(Settings(provider="localfree", local_llm_backend="heuristic"))
    submission = SubmissionInput(
        citizen_text="Retaining wall collapse damaged the carriageway beside the library underpass.",
        location_hint="Library underpass",
        time_hint="2026-04-19 13:00",
    )
    issue = provider.analyze_submission(submission)
    routing = provider.route_issue(submission, issue)
    priority = provider.assess_priority(submission, issue, routing)
    draft = provider.draft_appeal(submission, issue, routing, priority)
    explanation = provider.explain_case(
        issue,
        routing,
        priority,
        HumanReviewTask(needed=False, queue="triage-review", reasons=[], confidence=0.9),
    )

    assert "Operational impact:" in draft.body
    assert "Requested action:" in draft.body
    assert any("Comparable local patterns" in line for line in explanation.detailed_rationale)
    assert "local corpus-backed reasoning" in explanation.summary

from __future__ import annotations

import logging

from app.config import Settings
from app.providers.base import AIProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.local_provider import LocalFreeProvider
from app.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


def _local_fallback(settings: Settings, *, llm_backend: str | None = None) -> LocalFreeProvider:
    return LocalFreeProvider(settings, llm_backend=llm_backend)


def create_provider(settings: Settings) -> AIProvider:
    if settings.provider in {"localfree", "heuristic", "local"}:
        return _local_fallback(settings)
    if settings.provider == "ollama":
        return _local_fallback(settings, llm_backend="ollama")
    if settings.provider == "gemini":
        try:
            return GeminiProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
            )
        except RuntimeError as exc:
            logger.warning("Falling back to heuristic provider: %s", exc)
            return _local_fallback(settings)
    if settings.provider == "openai":
        try:
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                reasoning_effort=settings.openai_reasoning_effort,
                timeout_seconds=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )
        except RuntimeError as exc:
            logger.warning("Falling back to heuristic provider: %s", exc)
            return _local_fallback(settings)
    logger.warning("Unknown provider %r requested, falling back to local provider.", settings.provider)
    return _local_fallback(settings)

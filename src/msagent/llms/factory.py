"""Factory for creating supported LLM provider clients."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from msagent.configs.llm import LLMConfig
from msagent.core.settings import LLMSettings

_SUPPORTED_PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google_genai",
    "gemini": "google_genai",
}

_SETTINGS_API_KEY_ATTR: dict[str, str] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "google_genai": "google_api_key",
}

_DEFAULT_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

_PROVIDER_API_KEY_KWARG: dict[str, str] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "google_genai": "google_api_key",
}

_PROVIDER_BASE_URL_KWARG: dict[str, str] = {
    "openai": "openai_api_base",
    "anthropic": "anthropic_api_url",
}


class LLMFactory:
    """Factory for creating LangChain chat models from LLMConfig."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        default_llm: LLMConfig | None = None,
    ) -> None:
        self.settings = settings
        self.default_llm = default_llm

    def create(
        self,
        config: LLMConfig | None = None,
        *,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Create an LLM model from config."""
        cfg = config or self.default_llm
        if cfg is None:
            raise ValueError("No LLM config provided")

        provider = self._normalize_provider(cfg.provider.value)
        model_name = f"{provider}:{cfg.model}"

        kwargs: dict[str, Any] = dict(cfg.params or {})
        kwargs.update(
            {
                "temperature": cfg.temperature,
                "timeout": float(
                    cfg.request_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            }
        )
        if max_retries is not None:
            kwargs["max_retries"] = int(max_retries)

        if cfg.max_tokens > 0:
            kwargs["max_tokens"] = cfg.max_tokens

        normalized_base_url = self._normalize_base_url(provider, cfg.base_url)
        if normalized_base_url:
            kwargs["base_url"] = normalized_base_url
            provider_base_url_key = _PROVIDER_BASE_URL_KWARG.get(provider)
            if provider_base_url_key:
                kwargs[provider_base_url_key] = normalized_base_url

        if api_key := self._resolve_api_key(cfg, provider):
            kwargs["api_key"] = api_key
            provider_api_key = _PROVIDER_API_KEY_KWARG.get(provider)
            if provider_api_key:
                kwargs[provider_api_key] = api_key

        if provider == "openai":
            kwargs["use_responses_api"] = self._should_use_openai_responses_api(
                normalized_base_url
            )
            # Keep token usage metrics populated while streaming.
            kwargs["stream_usage"] = bool(cfg.streaming)

            resolved_trust_env = self._resolve_openai_trust_env(
                cfg.trust_env, normalized_base_url
            )
            if cfg.http2 or not resolved_trust_env:
                # Custom OpenAI-compatible gateways are often private endpoints.
                # Disable trust_env by default there to avoid unstable global proxy
                # interception causing TLS EOF errors.
                timeout = kwargs["timeout"]
                kwargs["http_client"] = httpx.Client(
                    timeout=timeout,
                    trust_env=resolved_trust_env,
                    http2=bool(cfg.http2),
                )
                kwargs["http_async_client"] = httpx.AsyncClient(
                    timeout=timeout,
                    trust_env=resolved_trust_env,
                    http2=bool(cfg.http2),
                )

        return init_chat_model(model_name, **kwargs)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        normalized = _SUPPORTED_PROVIDER_MAP.get(provider.strip().lower())
        if normalized is None:
            supported = ", ".join(sorted(_SUPPORTED_PROVIDER_MAP))
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported: {supported}"
            )
        return normalized

    def _resolve_api_key(self, cfg: LLMConfig, provider: str) -> str | None:
        if cfg.api_key_env:
            from_env = os.getenv(cfg.api_key_env, "").strip()
            if from_env:
                return from_env

        default_env = _DEFAULT_PROVIDER_API_KEY_ENV.get(provider)
        if default_env:
            from_env = os.getenv(default_env, "").strip()
            if from_env:
                return from_env

        if self.settings is None:
            return None

        attr = _SETTINGS_API_KEY_ATTR.get(provider)
        if not attr:
            return None

        value = getattr(self.settings, attr, None)
        if isinstance(value, SecretStr):
            raw = value.get_secret_value().strip()
        elif isinstance(value, str):
            raw = value.strip()
        else:
            return None

        if not raw:
            return None
        if raw.lower() == "dummy":
            return None
        return raw

    @staticmethod
    def _normalize_base_url(provider: str, base_url: str | None) -> str | None:
        """Normalize provider-specific base URLs for compatibility."""
        if base_url is None:
            return None

        normalized = base_url.strip()
        if not normalized:
            return None

        if provider != "openai":
            return normalized

        parsed = urlparse(normalized)
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/")

        # DeepSeek OpenAI-compatible endpoint requires /v1.
        if host.endswith("deepseek.com") and path in {"", "/"}:
            parsed = parsed._replace(path="/v1")
            return urlunparse(parsed)

        return normalized

    @staticmethod
    def _should_use_openai_responses_api(base_url: str | None) -> bool:
        """Only enable Responses API for OpenAI official endpoints."""
        if not base_url:
            return True

        host = (urlparse(base_url).hostname or "").lower()
        if not host:
            return True

        return host == "api.openai.com" or host.endswith(".openai.com")

    @classmethod
    def _resolve_openai_trust_env(
        cls,
        trust_env: bool | None,
        base_url: str | None,
    ) -> bool:
        if trust_env is not None:
            return bool(trust_env)
        return cls._should_use_openai_responses_api(base_url)

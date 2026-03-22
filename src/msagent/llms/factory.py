from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from botocore.config import Config
from pydantic import SecretStr

from msagent.configs import LLMConfig, LLMProvider
from msagent.core.logging import get_logger
from msagent.core.settings import LLMSettings
from msagent.utils.rate_limiter import TokenBucketLimiter

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _mask_key(key: str | None) -> str:
    """Mask API key for logging, showing only first 8 and last 4 characters."""
    if not key:
        return "<not_set>"
    if len(key) <= 16:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def _format_config_for_log(config: LLMConfig) -> dict:
    """Format LLMConfig for logging (with sensitive info masked)."""
    return {
        "provider": config.provider.value,
        "model": config.model,
        "alias": config.alias,
        "api_key_env": config.api_key_env,
        "base_url": config.base_url or "<default>",
        "max_tokens": config.max_tokens or "<default>",
        "temperature": config.temperature,
        "streaming": config.streaming,
        "context_window": config.context_window,
        "rate_config": "<configured>" if config.rate_config else "<none>",
        "extended_reasoning": "<enabled>" if config.extended_reasoning else "<none>",
    }


class LLMFactory:
    def __init__(self, llm_settings: LLMSettings):
        self.llm_settings = llm_settings
        self._proxy_dict = self._get_proxy_dict()
        self._proxy = self._proxy_dict.get("https") or self._proxy_dict.get("http")
        self._llm_cache: dict[int, BaseChatModel] = {}
        logger.debug(
            "LLMFactory initialized with proxy=%s",
            "<configured>" if self._proxy else "<none>",
        )

    def _get_proxy_dict(self):
        http_proxy = self.llm_settings.http_proxy.get_secret_value()
        https_proxy = self.llm_settings.https_proxy.get_secret_value()
        proxy_dict = {k: v for k, v in [("http", http_proxy), ("https", https_proxy)] if v}
        if proxy_dict:
            logger.debug("Proxy configured: schemes=%s", list(proxy_dict.keys()))
        return proxy_dict

    @staticmethod
    def _is_local_url(url: str) -> bool:
        return urlparse(url).hostname in _LOCAL_HOSTS

    def _get_http_clients(self, base_url: str | None = None):
        if not self._proxy_dict:
            logger.debug("HTTP clients: no proxy configured")
            return None, None
        
        if base_url and self._is_local_url(base_url):
            logger.debug("HTTP clients: local URL detected, skipping proxy for %s", base_url)
            return None, None

        # Use per-scheme mounts when http and https proxies differ
        if (
            len(self._proxy_dict) == 2
            and self._proxy_dict["http"] != self._proxy_dict["https"]
        ):
            logger.debug("HTTP clients: using per-scheme proxy mounts")
            sync_mounts = {
                f"{k}://": httpx.HTTPTransport(proxy=v)
                for k, v in self._proxy_dict.items()
            }
            async_mounts = {
                f"{k}://": httpx.AsyncHTTPTransport(proxy=v)
                for k, v in self._proxy_dict.items()
            }
            return httpx.Client(mounts=sync_mounts), httpx.AsyncClient(
                mounts=async_mounts
            )

        logger.debug("HTTP clients: using single proxy configuration")
        return httpx.Client(proxy=self._proxy), httpx.AsyncClient(proxy=self._proxy)

    def _get_ollama_kwargs(self, base_url: str):
        if not self._proxy_dict or self._is_local_url(base_url):
            return {}

        # Use per-scheme proxies when http and https differ
        if (
            len(self._proxy_dict) == 2
            and self._proxy_dict["http"] != self._proxy_dict["https"]
        ):
            return {
                "client_kwargs": {
                    "proxies": {f"{k}://": v for k, v in self._proxy_dict.items()}
                }
            }

        return {"client_kwargs": {"proxy": self._proxy}}

    def _get_bedrock_config(self):
        return Config(proxies=self._proxy_dict) if self._proxy_dict else None

    @staticmethod
    def _create_limiter(config: LLMConfig):
        if not config.rate_config:
            return None
        
        logger.debug(
            "Creating rate limiter: rps=%.2f, input_tps=%.2f, output_tps=%.2f",
            config.rate_config.requests_per_second,
            config.rate_config.input_tokens_per_second,
            config.rate_config.output_tokens_per_second,
        )
        return TokenBucketLimiter(
            requests_per_second=config.rate_config.requests_per_second,
            input_tokens_per_second=config.rate_config.input_tokens_per_second,
            output_tokens_per_second=config.rate_config.output_tokens_per_second,
            check_every_n_seconds=config.rate_config.check_every_n_seconds,
            max_bucket_size=config.rate_config.max_bucket_size,
        )

    def _resolve_api_key(
        self,
        config: LLMConfig,
        fallback: SecretStr,
        default_env: str | None = None,
    ) -> str:
        """Resolve API key from environment or fallback.
        
        Priority:
        1. Environment variable specified by config.api_key_env
        2. Environment variable specified by default_env
        3. Fallback from settings
        """
        if config.api_key_env:
            explicit = os.getenv(config.api_key_env, "")
            if explicit:
                logger.debug("API key resolved from env var: %s", config.api_key_env)
                return explicit
            else:
                logger.warning(
                    "Configured api_key_env '%s' is not set or empty",
                    config.api_key_env,
                )

        if default_env:
            implicit = os.getenv(default_env, "")
            if implicit:
                logger.debug("API key resolved from default env var: %s", default_env)
                return implicit
            else:
                logger.debug("Default env var '%s' is not set", default_env)

        fallback_value = fallback.get_secret_value()
        if fallback_value:
            logger.debug("API key resolved from settings fallback")
        else:
            logger.warning("API key is empty (no env var or fallback configured)")
        return fallback_value

    @staticmethod
    def _add_optional_kwarg(
        kwargs: dict[str, object], key: str, value: object | None
    ) -> None:
        if value not in {None, "", 0}:
            kwargs[key] = value

    @staticmethod
    def _get_config_hash(config: LLMConfig) -> int:
        return hash(
            (
                config.provider,
                config.model,
                config.api_key_env,
                config.base_url,
                config.temperature,
                config.max_tokens,
                config.streaming,
                str(config.extended_reasoning) if config.extended_reasoning else None,
            )
        )

    def _log_provider_config(self, provider: LLMProvider, kwargs: dict, config: LLMConfig) -> dict:
        """Log provider configuration with sensitive data masked."""
        log_kwargs = dict(kwargs)
        
        # Mask sensitive fields
        for key in ["api_key", "aws_access_key_id", "aws_secret_access_key", "aws_session_token"]:
            if key in log_kwargs:
                log_kwargs[key] = _mask_key(str(log_kwargs[key]))
        
        # Mask URL if it contains credentials
        if "base_url" in log_kwargs and log_kwargs["base_url"]:
            url = str(log_kwargs["base_url"])
            if "@" in url:
                parsed = urlparse(url)
                if parsed.username or parsed.password:
                    log_kwargs["base_url"] = f"{parsed.scheme}://***@{parsed.hostname}{parsed.path}"
        
        logger.info(
            "Creating %s LLM: model=%s, base_url=%s, api_key=%s, streaming=%s",
            provider.value,
            log_kwargs.get("model") or log_kwargs.get("model_name"),
            log_kwargs.get("base_url", "<default>"),
            log_kwargs.get("api_key", "<not_set>"),
            log_kwargs.get("streaming", "<default>"),
        )
        logger.debug("%s kwargs: %s", provider.value.upper(), log_kwargs)
        
        return log_kwargs

    def create(self, config: LLMConfig) -> BaseChatModel:
        config_hash = self._get_config_hash(config)
        if config_hash in self._llm_cache:
            logger.debug("LLM cache hit for hash=%d", config_hash)
            return self._llm_cache[config_hash]

        logger.info("Creating new LLM instance (hash=%d)", config_hash)
        logger.debug("LLM config: %s", _format_config_for_log(config))

        limiter = self._create_limiter(config)
        llm: BaseChatModel

        try:
            if config.provider == LLMProvider.OPENAI:
                from langchain_openai import ChatOpenAI

                http_client, http_async_client = self._get_http_clients(config.base_url)
                kwargs = {
                    "api_key": self._resolve_api_key(
                        config, self.llm_settings.openai_api_key, "OPENAI_API_KEY"
                    ),
                    "model": config.model,
                    "temperature": config.temperature,
                    "streaming": config.streaming,
                    "rate_limiter": limiter,
                    "http_client": http_client,
                    "http_async_client": http_async_client,
                }
                self._add_optional_kwarg(kwargs, "base_url", config.base_url)
                self._add_optional_kwarg(
                    kwargs, "max_completion_tokens", config.max_tokens
                )

                if config.extended_reasoning:
                    kwargs["reasoning"] = config.extended_reasoning
                    kwargs["output_version"] = "responses/v1"

                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatOpenAI(**kwargs)
                
            elif config.provider == LLMProvider.CUSTOM:
                from msagent.llms.wrappers.custom import ChatCustomHTTP

                url = config.base_url or self.llm_settings.custom_base_url
                if not url:
                    raise ValueError(
                        "CUSTOM provider requires base_url or CUSTOM_BASE_URL to be set"
                    )
                http_client, http_async_client = self._get_http_clients(url)
                kwargs = {
                    "url": url,
                    "api_key": self._resolve_api_key(
                        config, self.llm_settings.custom_api_key, "CUSTOM_API_KEY"
                    ),
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "streaming": config.streaming,
                    "disable_streaming": True,
                    "rate_limiter": limiter,
                    "http_client": http_client,
                    "http_async_client": http_async_client,
                }

                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatCustomHTTP(**kwargs)
                
            elif config.provider == LLMProvider.ANTHROPIC:
                from langchain_anthropic import ChatAnthropic

                kwargs = {
                    "api_key": self._resolve_api_key(
                        config, self.llm_settings.anthropic_api_key, "ANTHROPIC_API_KEY"
                    ),
                    "model_name": config.model,
                    "temperature": config.temperature,
                    "streaming": config.streaming,
                    "rate_limiter": limiter,
                    "timeout": None,
                    "stop": None,
                }
                self._add_optional_kwarg(kwargs, "base_url", config.base_url)
                self._add_optional_kwarg(kwargs, "max_tokens_to_sample", config.max_tokens)

                if config.extended_reasoning:
                    kwargs["thinking"] = config.extended_reasoning

                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatAnthropic(**kwargs)
                
            elif config.provider == LLMProvider.GOOGLE:
                from langchain_google_genai import ChatGoogleGenerativeAI

                kwargs = {
                    "api_key": self._resolve_api_key(
                        config, self.llm_settings.google_api_key, "GOOGLE_API_KEY"
                    ),
                    "model": config.model,
                    "temperature": config.temperature,
                    "disable_streaming": not config.streaming,
                    "rate_limiter": limiter,
                }
                self._add_optional_kwarg(kwargs, "max_tokens", config.max_tokens)

                if config.extended_reasoning:
                    kwargs.update(config.extended_reasoning)

                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatGoogleGenerativeAI(**kwargs)
                
            elif config.provider == LLMProvider.OLLAMA:
                from langchain_ollama import ChatOllama

                base_url = self.llm_settings.ollama_base_url
                kwargs = {
                    "base_url": base_url,
                    "model": config.model,
                    "temperature": config.temperature,
                    "disable_streaming": not config.streaming,
                    "rate_limiter": limiter,
                    **self._get_ollama_kwargs(base_url),
                }
                self._add_optional_kwarg(kwargs, "num_predict", config.max_tokens)
                
                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatOllama(**kwargs)
                
            elif config.provider == LLMProvider.LMSTUDIO:
                from langchain_openai import ChatOpenAI

                base_url = self.llm_settings.lmstudio_base_url
                http_client, http_async_client = self._get_http_clients(base_url)
                kwargs = {
                    "base_url": base_url,
                    "model": config.model,
                    "temperature": config.temperature,
                    "streaming": config.streaming,
                    "api_key": SecretStr("SOME_KEY"),
                    "rate_limiter": limiter,
                    "http_client": http_client,
                    "http_async_client": http_async_client,
                }
                self._add_optional_kwarg(
                    kwargs, "max_completion_tokens", config.max_tokens
                )
                
                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatOpenAI(**kwargs)
                
            elif config.provider == LLMProvider.BEDROCK:
                from langchain_aws import ChatBedrock

                kwargs = {
                    "aws_access_key_id": self.llm_settings.aws_access_key_id,
                    "aws_secret_access_key": self.llm_settings.aws_secret_access_key,
                    "aws_session_token": self.llm_settings.aws_session_token,
                    "model": config.model,
                    "temperature": config.temperature,
                    "streaming": config.streaming,
                    "rate_limiter": limiter,
                    "config": self._get_bedrock_config(),
                }
                self._add_optional_kwarg(kwargs, "max_tokens", config.max_tokens)

                if config.extended_reasoning:
                    kwargs["model_kwargs"] = {"thinking": config.extended_reasoning}

                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatBedrock(**kwargs)
                
            elif config.provider == LLMProvider.DEEPSEEK:
                from langchain_deepseek import ChatDeepSeek

                http_client, http_async_client = self._get_http_clients(config.base_url)
                kwargs = {
                    "api_key": self._resolve_api_key(
                        config, self.llm_settings.deepseek_api_key, "DEEPSEEK_API_KEY"
                    ),
                    "model": config.model,
                    "temperature": config.temperature,
                    "streaming": config.streaming,
                    "rate_limiter": limiter,
                    "http_client": http_client,
                    "http_async_client": http_async_client,
                }
                self._add_optional_kwarg(kwargs, "base_url", config.base_url)
                self._add_optional_kwarg(kwargs, "max_tokens", config.max_tokens)
                
                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatDeepSeek(**kwargs)
                
            elif config.provider == LLMProvider.ZHIPUAI:
                from msagent.llms.wrappers.zhipuai import ChatZhipuAI

                kwargs = {
                    "api_key": self._resolve_api_key(
                        config, self.llm_settings.zhipuai_api_key, "ZHIPUAI_API_KEY"
                    ),
                    "model": config.model,
                    "temperature": config.temperature,
                    "streaming": config.streaming,
                    "rate_limiter": limiter,
                }
                self._add_optional_kwarg(kwargs, "zhipuai_api_base", config.base_url)
                self._add_optional_kwarg(kwargs, "max_tokens", config.max_tokens)

                if config.extended_reasoning:
                    kwargs["thinking"] = config.extended_reasoning

                self._log_provider_config(config.provider, kwargs, config)
                llm = ChatZhipuAI(**kwargs)
                
            else:
                raise ValueError(f"Unknown LLM provider: {config.provider}")

        except Exception as e:
            logger.error(
                "Failed to create LLM for provider=%s, model=%s: %s (%s)",
                config.provider.value,
                config.model,
                type(e).__name__,
                str(e),
            )
            logger.debug(
                "Config that caused error: %s",
                _format_config_for_log(config),
            )
            raise

        self._llm_cache[config_hash] = llm
        logger.info("LLM instance created and cached successfully (hash=%d)", config_hash)
        return llm

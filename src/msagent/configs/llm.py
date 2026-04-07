"""LLM configuration classes."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from msagent.configs.base import VersionedConfig
from msagent.configs.utils import (
    _load_dir_items,
    _load_single_file,
    _validate_no_duplicates,
)
from msagent.core.constants import LLM_CONFIG_VERSION

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class RateConfig(BaseModel):
    requests_per_second: float = Field(
        description="The maximum number of requests per second"
    )
    input_tokens_per_second: float = Field(
        description="The maximum number of input tokens per second"
    )
    output_tokens_per_second: float = Field(
        description="The maximum number of output tokens per second"
    )
    check_every_n_seconds: float = Field(
        description="The interval in seconds to check the rate limit"
    )
    max_bucket_size: int = Field(
        description="The maximum number of requests that can be stored in the bucket"
    )


class LLMConfig(VersionedConfig):
    version: str = Field(
        default=LLM_CONFIG_VERSION, description="Config schema version"
    )
    provider: LLMProvider = Field(description="The provider of the LLM")
    model: str = Field(description="The model to use")
    alias: str = Field(default="", description="Display alias for the model")
    api_key_env: str | None = Field(
        default=None,
        description="Optional environment variable name that stores the API key",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Optional base URL for compatible providers or full request URL for custom"
        ),
    )
    max_tokens: int = Field(description="The maximum number of tokens to generate")
    temperature: float = Field(description="The temperature to use")
    streaming: bool = Field(default=True, description="Whether to stream the response")
    request_timeout_seconds: float = Field(
        default=120.0,
        description="LLM request timeout in seconds",
        gt=0,
    )
    trust_env: bool | None = Field(
        default=None,
        description=(
            "Whether HTTP client should trust proxy/SSL env vars (HTTP_PROXY, "
            "HTTPS_PROXY, SSL_CERT_FILE). Null means provider-specific default."
        ),
    )
    http2: bool = Field(
        default=False,
        description="Whether to enable HTTP/2 for model requests",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Extra keyword arguments forwarded to init_chat_model for "
            "provider-specific options"
        ),
    )
    rate_config: RateConfig | None = Field(
        default=None, description="The rate config to use"
    )
    context_window: int | None = Field(
        default=None, description="Context window size in tokens"
    )
    input_cost_per_mtok: float | None = Field(
        default=None, description="Input token cost per million tokens"
    )
    output_cost_per_mtok: float | None = Field(
        default=None, description="Output token cost per million tokens"
    )
    extended_reasoning: dict[str, Any] | None = Field(
        default=None,
        description="Extended reasoning/thinking configuration (provider-agnostic)",
    )

    @classmethod
    def get_latest_version(cls) -> str:
        return LLM_CONFIG_VERSION

    @model_validator(mode="after")
    def set_alias_default(self) -> LLMConfig:
        """Set alias to model name if not provided."""
        if not self.alias:
            self.alias = self.model
        return self

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider_alias(cls, value: Any) -> Any:
        """Support gemini alias by normalizing to google."""
        if isinstance(value, str) and value.strip().lower() == "gemini":
            return LLMProvider.GOOGLE.value
        return value


class BatchLLMConfig(BaseModel):
    llms: list[LLMConfig] = Field(description="The LLMs configurations")

    @property
    def llm_names(self) -> list[str]:
        return [llm.alias for llm in self.llms]

    def get_llm_config(self, llm_name: str) -> LLMConfig | None:
        return next((llm for llm in self.llms if llm.alias == llm_name), None)

    @classmethod
    async def from_yaml(
        cls,
        file_path: Path | None = None,
        dir_path: Path | None = None,
    ) -> BatchLLMConfig:
        raw_llms: list[dict[str, Any]] = []

        if file_path and file_path.exists():
            raw_llms.extend(await _load_single_file(file_path, "llms", LLMConfig))

        if dir_path:
            raw_llms.extend(await _load_dir_items(dir_path, config_class=LLMConfig))

        llms = cls._filter_supported_llms(raw_llms)

        _validate_no_duplicates(llms, key="alias", config_type="LLM")
        return cls.model_validate({"llms": llms})

    @staticmethod
    def _filter_supported_llms(llms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter unsupported providers to keep runtime backward compatible."""
        supported = {provider.value for provider in LLMProvider}
        accepted_aliases = supported | {"gemini"}
        filtered: list[dict[str, Any]] = []

        for item in llms:
            provider = str(item.get("provider", "")).strip().lower()
            alias = str(item.get("alias", item.get("model", "<unknown>")))
            if provider == "gemini":
                item = dict(item)
                item["provider"] = LLMProvider.GOOGLE.value
                provider = LLMProvider.GOOGLE.value

            if provider not in accepted_aliases:
                logger.debug(
                    "Ignoring unsupported LLM provider '%s' for alias '%s'. "
                    "Supported providers: %s",
                    provider,
                    alias,
                    ", ".join(sorted(supported | {"gemini"})),
                )
                continue

            filtered.append(item)

        return filtered

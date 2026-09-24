# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Typed configuration loading for the whl validation suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProtocolConfig:
    """One LLM protocol endpoint and its environment variable mapping."""

    provider: str
    base_url: str
    base_url_env: str
    provider_api_key_env: str


@dataclass(frozen=True)
class LlmConfig:
    """LLM settings shared by real-model validation cases."""

    model: str
    api_key_env: str
    protocols: dict[str, ProtocolConfig]

    def protocol(self, provider: str) -> ProtocolConfig:
        try:
            return self.protocols[provider]
        except KeyError as exc:
            raise ValueError(f"未配置 LLM 协议: {provider}") from exc


@dataclass(frozen=True)
class MsagentConfig:
    """CLI settings consumed by the subprocess runtime."""

    executable: str
    timeout_seconds: int


@dataclass(frozen=True)
class ArtifactConfig:
    """Retention behavior for per-test diagnostic artifacts."""

    root_dir: Path
    retention: str
    retain_workspace_on_failure: bool


@dataclass(frozen=True)
class ValidationConfig:
    """Validated top-level suite configuration."""

    llm: LlmConfig
    msagent: MsagentConfig
    artifacts: ArtifactConfig
    raw: dict[str, Any]


def _required_text(mapping: dict, key: str, location: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"{location}.{key} 不能为空")
    return value


def load_validation_config(path: Path) -> ValidationConfig:
    """Load YAML once and fail early on missing runtime-critical fields."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取测试配置 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 顶层配置必须是字典")

    llm_payload = payload.get("llm")
    if not isinstance(llm_payload, dict):
        raise ValueError(f"{path} 缺少 llm 配置")
    protocol_payloads = llm_payload.get("protocols")
    if not isinstance(protocol_payloads, dict) or not protocol_payloads:
        raise ValueError(f"{path} 缺少 llm.protocols 配置")

    protocols: dict[str, ProtocolConfig] = {}
    for provider, value in protocol_payloads.items():
        if not isinstance(provider, str) or not isinstance(value, dict):
            raise ValueError(f"{path} 中 llm.protocols 的结构无效")
        location = f"llm.protocols.{provider}"
        protocols[provider] = ProtocolConfig(
            provider=provider,
            base_url=_required_text(value, "base_url", location),
            base_url_env=_required_text(value, "base_url_env", location),
            provider_api_key_env=_required_text(value, "provider_api_key_env", location),
        )

    msagent_payload = payload.get("msagent") or {}
    if not isinstance(msagent_payload, dict):
        raise ValueError(f"{path} 中 msagent 必须是字典")

    artifact_payload = payload.get("artifacts") or {}
    if not isinstance(artifact_payload, dict):
        raise ValueError(f"{path} 中 artifacts 必须是字典")
    retention = str(artifact_payload.get("retention") or "failed").strip()
    if retention not in {"all", "failed", "none"}:
        raise ValueError("artifacts.retention 只能是 all、failed 或 none")
    configured_root = Path(str(artifact_payload.get("root_dir") or "artifacts"))
    root_dir = configured_root if configured_root.is_absolute() else path.parents[1] / configured_root

    return ValidationConfig(
        llm=LlmConfig(
            model=_required_text(llm_payload, "model", "llm"),
            api_key_env=_required_text(llm_payload, "api_key_env", "llm"),
            protocols=protocols,
        ),
        msagent=MsagentConfig(
            executable=str(msagent_payload.get("executable") or "msagent").strip(),
            timeout_seconds=int(msagent_payload.get("timeout_seconds") or 180),
        ),
        artifacts=ArtifactConfig(
            root_dir=root_dir.resolve(),
            retention=retention,
            retain_workspace_on_failure=bool(artifact_payload.get("retain_workspace_on_failure", True)),
        ),
        raw=payload,
    )

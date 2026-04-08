from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from msagent.core.constants import (
    AGENT_CONFIG_VERSION,
    APP_VERSION,
    CHECKPOINTER_CONFIG_VERSION,
    CONFIG_VERSION_TOKEN,
    LLM_CONFIG_VERSION,
    SANDBOX_CONFIG_VERSION,
)
from msagent.utils.version import get_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "resources" / "configs" / "default"


def _project_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _iter_default_config_versions() -> list[tuple[Path, str]]:
    versions: list[tuple[Path, str]] = []

    for path in sorted(DEFAULT_CONFIG_ROOT.rglob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "version" in data:
            versions.append((path, str(data["version"])))
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "version" in item:
                    versions.append((path, str(item["version"])))

    return versions


def _resolve_default_config_version(version: str, project_version: str) -> str:
    if version == CONFIG_VERSION_TOKEN:
        return project_version
    return version


def test_runtime_version_uses_project_version() -> None:
    project_version = _project_version()

    assert APP_VERSION == project_version
    assert get_version() == project_version


def test_config_version_constants_match_project_version() -> None:
    project_version = _project_version()

    assert AGENT_CONFIG_VERSION == project_version
    assert LLM_CONFIG_VERSION == project_version
    assert CHECKPOINTER_CONFIG_VERSION == project_version
    assert SANDBOX_CONFIG_VERSION == project_version


def test_default_resource_config_versions_match_project_version() -> None:
    project_version = _project_version()
    config_versions = _iter_default_config_versions()

    assert config_versions
    assert all(
        _resolve_default_config_version(version, project_version) == project_version
        for _, version in config_versions
    ), (
        "Found config versions that do not match project version: "
        + ", ".join(
            f"{path.relative_to(PROJECT_ROOT)}={version}"
            for path, version in config_versions
            if _resolve_default_config_version(version, project_version)
            != project_version
        )
    )

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from pathlib import Path
import json
from importlib.resources import files
import yaml

import pytest

from msagent.configs.registry import ConfigRegistry
from msagent.core.constants import CONFIG_APPROVAL_FILE_NAME, LLM_CONFIG_VERSION
from msagent.skills.factory import SkillFactory


def _load_default_msprof_server() -> dict:
    config_path = files("resources")
    for part in ("configs", "default", "config.mcp.json"):
        config_path = config_path.joinpath(part)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["mcpServers"]["msprof-mcp"]


def _load_default_profiler_config() -> dict:
    config_path = files("resources")
    for part in ("configs", "default", "agents", "Profiler.yml"):
        config_path = config_path.joinpath(part)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _load_default_modeling_config() -> dict:
    config_path = files("resources")
    for part in ("configs", "default", "agents", "Modeling.yml"):
        config_path = config_path.joinpath(part)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _load_default_minos_config() -> dict:
    config_path = files("resources")
    for part in ("configs", "default", "agents", "Minos.yml"):
        config_path = config_path.joinpath(part)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_default_msprof_server_uses_packaged_executable() -> None:
    default_msprof_server = _load_default_msprof_server()

    assert default_msprof_server["command"] == "msprof-mcp"
    assert default_msprof_server["args"] == []
    assert default_msprof_server["repair_timeout"] == 30
    assert "invoke_timeout" in default_msprof_server
    assert isinstance(default_msprof_server["invoke_timeout"], (int, float))
    assert default_msprof_server["invoke_timeout"] > 0


@pytest.mark.asyncio
async def test_config_registry_bootstraps_default_layout(tmp_path: Path) -> None:
    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    config_dir = tmp_path / ".msagent"
    assert config_dir.exists()
    assert (config_dir / "README.md").exists()
    assert (config_dir / "agents").is_dir()
    assert (config_dir / "llms").is_dir()
    assert (config_dir / "sandboxes").is_dir()
    assert (config_dir / "subagents").is_dir()
    assert (config_dir / "config.llms.yml").exists()
    assert (config_dir / "config.mcp.json").exists()
    assert (config_dir / CONFIG_APPROVAL_FILE_NAME.name).exists()

    mcp_config = json.loads((config_dir / "config.mcp.json").read_text())
    assert "msprof-mcp" in mcp_config["mcpServers"]
    msprof_server = mcp_config["mcpServers"]["msprof-mcp"]
    default_msprof_server = _load_default_msprof_server()
    assert msprof_server["args"] == default_msprof_server["args"]
    assert msprof_server["stateful"] == default_msprof_server["stateful"]

    approval_config = json.loads((config_dir / CONFIG_APPROVAL_FILE_NAME.name).read_text(encoding="utf-8"))
    assert "interrupt_on" in approval_config
    assert "execute" in approval_config["interrupt_on"]
    assert approval_config["interrupt_on"]["execute"]["allowed_decisions"] == [
        "approve",
        "reject",
    ]
    assert "decision_rules" in approval_config
    assert any(
        rule
        == {
            "name": "execute",
            "args": {"command": ".*"},
            "decision": "always_approve",
        }
        for rule in approval_config["decision_rules"]
    )

    profiler = yaml.safe_load((config_dir / "agents" / "Profiler.yml").read_text())
    modeling = yaml.safe_load((config_dir / "agents" / "Modeling.yml").read_text())
    minos = yaml.safe_load((config_dir / "agents" / "Minos.yml").read_text())
    default_profiler = _load_default_profiler_config()
    default_modeling = _load_default_modeling_config()
    default_minos = _load_default_minos_config()
    assert profiler["name"] == "Profiler"
    assert profiler["tools"]["patterns"] == default_profiler["tools"]["patterns"]
    assert profiler["skills"]["patterns"] == default_profiler["skills"]["patterns"]
    assert modeling["name"] == "Modeling"
    assert modeling["tools"]["patterns"] == default_modeling["tools"]["patterns"]
    assert modeling["skills"]["patterns"] == default_modeling["skills"]["patterns"]
    assert "default:msmodeling-env-installer" in modeling["skills"]["patterns"]
    assert modeling["default"] is False
    assert minos["name"] == "Minos"
    assert minos["skills"]["patterns"] == default_minos["skills"]["patterns"]


@pytest.mark.asyncio
async def test_config_registry_adds_missing_modeling_skill_patterns_to_existing_config_dir(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    agents_dir = config_dir / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "Modeling.yml").write_text(
        yaml.safe_dump(
            {
                "version": "__APP_VERSION__",
                "name": "Modeling",
                "description": "legacy local modeling config",
                "prompt": ["prompts/agents/Modeling.md"],
                "llm": "default",
                "checkpointer": "sqlite",
                "default": False,
                "tools": {"patterns": ["impl:deepagents:*"], "use_catalog": False},
                "skills": {"patterns": ["default:custom-local-skill"], "use_catalog": False},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    modeling = yaml.safe_load((agents_dir / "Modeling.yml").read_text(encoding="utf-8"))
    patterns = modeling["skills"]["patterns"]
    default_modeling = _load_default_modeling_config()
    for pattern in default_modeling["skills"]["patterns"]:
        assert pattern in patterns
    assert "default:custom-local-skill" in patterns


@pytest.mark.asyncio
async def test_default_skills_include_msmodeling_env_installer() -> None:
    skills = await SkillFactory().load_skills(SkillFactory.get_default_skills_dir())

    assert "msmodeling-env-installer" in skills["default"]


@pytest.mark.asyncio
async def test_config_registry_resolves_template_version_tokens_on_load(
    tmp_path: Path,
) -> None:
    registry = ConfigRegistry(tmp_path)

    llms = await registry.load_llms()

    assert llms.llms
    assert all(llm.version == LLM_CONFIG_VERSION for llm in llms.llms)


def test_default_agent_skill_bindings_are_split_between_profiler_and_minos() -> None:
    profiler = _load_default_profiler_config()
    minos = _load_default_minos_config()

    assert "default:document-ux-review" not in profiler["skills"]["patterns"]
    assert "default:gitcode-code-reviewer" not in profiler["skills"]["patterns"]
    assert minos["skills"]["patterns"] == [
        "default:document-ux-review",
        "default:gitcode-code-reviewer",
    ]


@pytest.mark.asyncio
async def test_config_registry_preserves_existing_mcp_server_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    config_dir.mkdir(parents=True)
    existing_mcp = {
        "mcpServers": {
            "msprof-mcp": {
                "command": "uvx",
                "args": [
                    "--isolated",
                    "--refresh",
                    "--from",
                    "git+https://gitcode.com/kali20gakki1/msprof_mcp.git",
                    "msprof-mcp",
                ],
                "transport": "stdio",
                "env": {},
                "include": [],
                "exclude": [],
                "enabled": False,
                "stateful": False,
            }
        }
    }
    (config_dir / "config.mcp.json").write_text(
        json.dumps(existing_mcp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    mcp_config = json.loads((config_dir / "config.mcp.json").read_text(encoding="utf-8"))
    msprof_server = mcp_config["mcpServers"]["msprof-mcp"]
    assert msprof_server["args"] == existing_mcp["mcpServers"]["msprof-mcp"]["args"]
    assert msprof_server["stateful"] is False
    assert msprof_server["enabled"] is False


@pytest.mark.asyncio
async def test_config_registry_adds_missing_default_mcp_server(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.mcp.json").write_text(
        json.dumps({"mcpServers": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    mcp_config = json.loads((config_dir / "config.mcp.json").read_text(encoding="utf-8"))
    msprof_server = mcp_config["mcpServers"]["msprof-mcp"]
    default_msprof_server = _load_default_msprof_server()
    assert msprof_server == default_msprof_server


@pytest.mark.asyncio
async def test_config_registry_copies_missing_approval_template_into_existing_config_dir(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.mcp.json").write_text(
        json.dumps({"mcpServers": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    approval_path = config_dir / CONFIG_APPROVAL_FILE_NAME.name
    assert approval_path.exists()
    approval_config = json.loads(approval_path.read_text(encoding="utf-8"))
    assert "interrupt_on" in approval_config
    assert "execute" in approval_config["interrupt_on"]
    assert "decision_rules" in approval_config

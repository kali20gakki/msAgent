#!/usr/bin/python3
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

from pathlib import Path
import json
from importlib.resources import files
import yaml

import pytest

from msagent.configs.registry import ConfigRegistry
from msagent.core.constants import LLM_CONFIG_VERSION
from msagent.core.paths import AppPaths
from msagent.skills.factory import SkillFactory
from msagent.tools.internal.memory import DEFAULT_MEMORY_FILE_CONTENT


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


def _create_registry(tmp_path: Path) -> tuple[ConfigRegistry, AppPaths]:
    app_paths = AppPaths.from_home(tmp_path / "global-home" / ".msagent")
    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    return ConfigRegistry(working_dir, app_paths), app_paths


def test_default_mcp_config_is_valid_json() -> None:
    config_path = files("resources")
    for part in ("configs", "default", "config.mcp.json"):
        config_path = config_path.joinpath(part)

    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert isinstance(config.get("mcpServers"), dict)


@pytest.mark.asyncio
async def test_config_registry_bootstraps_default_layout(tmp_path: Path) -> None:
    registry, app_paths = _create_registry(tmp_path)

    await registry.ensure_config_dir()

    assert app_paths.config_dir.is_dir()
    assert list(app_paths.config_dir.iterdir()) == []
    assert registry.project_paths.metadata_file.is_file()
    assert registry.project_paths.memory_file.is_file()
    assert not (registry.working_dir / ".msagent").exists()

    agents = await registry.load_agents()
    assert "Profiler" in agents.agent_names


@pytest.mark.asyncio
async def test_load_user_memory_ignores_default_template(tmp_path: Path) -> None:
    registry, _ = _create_registry(tmp_path)
    memory_file = registry.project_paths.memory_file
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(DEFAULT_MEMORY_FILE_CONTENT, encoding="utf-8")

    assert await registry.load_user_memory() == ""


@pytest.mark.asyncio
async def test_load_user_memory_wraps_user_content(tmp_path: Path) -> None:
    registry, _ = _create_registry(tmp_path)
    memory_file = registry.project_paths.memory_file
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text("- 用户喜欢中文回复\n", encoding="utf-8")

    assert await registry.load_user_memory() == "<user-memory>\n- 用户喜欢中文回复\n</user-memory>"


@pytest.mark.asyncio
async def test_config_registry_user_agent_replaces_packaged_agent(
    tmp_path: Path,
) -> None:
    registry, app_paths = _create_registry(tmp_path)
    config_dir = app_paths.config_dir
    agents_dir = config_dir / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "Modeling.yml").write_text(
        yaml.safe_dump(
            {
                "version": "__APP_VERSION__",
                "name": "Modeling",
                "description": "legacy local modeling config",
                "prompt": "custom prompt",
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

    modeling = await registry.get_agent("Modeling")

    assert modeling.description == "legacy local modeling config"
    assert modeling.prompt == "custom prompt"
    assert modeling.skills is not None
    assert modeling.skills.patterns == ["default:custom-local-skill"]


@pytest.mark.asyncio
async def test_generated_agent_override_contains_only_mutated_fields(tmp_path: Path) -> None:
    registry, app_paths = _create_registry(tmp_path)
    packaged = await registry.get_agent("Profiler")

    await registry.update_agent_llm("Profiler", "gpt-5-mini-thinking")

    override_file = app_paths.config_dir / "agents" / "Profiler.yml"
    override = yaml.safe_load(override_file.read_text(encoding="utf-8"))
    assert set(override) == {"version", "name", "llm"}
    assert override["name"] == "Profiler"
    assert override["llm"] == "gpt-5-mini-thinking"
    assert not app_paths.prompts_dir.exists() or list(app_paths.prompts_dir.rglob("*")) == []

    effective = await registry.get_agent("Profiler")
    assert effective.llm.alias == "gpt-5-mini-thinking"
    assert effective.prompt == packaged.prompt
    assert effective.tools == packaged.tools
    assert effective.skills == packaged.skills
    assert effective.compression == packaged.compression


@pytest.mark.asyncio
async def test_current_agent_is_scoped_to_project_state(tmp_path: Path) -> None:
    app_paths = AppPaths.from_home(tmp_path / "home")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = ConfigRegistry(first_dir, app_paths)
    second = ConfigRegistry(second_dir, app_paths)

    await first.set_current_agent("Minos")

    assert (await first.get_agent()).name == "Minos"
    assert (await second.get_agent()).name == "Profiler"
    assert not (app_paths.config_dir / "agents").exists()


@pytest.mark.asyncio
async def test_removed_workspace_agent_falls_back_to_packaged_default(tmp_path: Path) -> None:
    registry, _ = _create_registry(tmp_path)
    registry.project_paths.set_current_agent("RemovedAgent")

    assert (await registry.get_agent()).name == "Profiler"


@pytest.mark.asyncio
async def test_default_skills_include_msmodeling_env_installer() -> None:
    skills = await SkillFactory().load_skills(SkillFactory.get_default_skills_dir())

    assert "msmodeling-env-installer" in skills["modeling"]


@pytest.mark.asyncio
async def test_config_registry_resolves_template_version_tokens_on_load(
    tmp_path: Path,
) -> None:
    registry, _ = _create_registry(tmp_path)

    llms = await registry.load_llms()

    assert llms.llms
    assert all(llm.version == LLM_CONFIG_VERSION for llm in llms.llms)


def test_default_agent_skill_bindings_are_split_between_profiler_and_minos() -> None:
    profiler = _load_default_profiler_config()
    minos = _load_default_minos_config()

    assert "minos:document-ux-review" not in profiler["skills"]["patterns"]
    assert "minos:gitcode-code-reviewer" not in profiler["skills"]["patterns"]
    assert minos["skills"]["patterns"] == ["minos:*", "basic:*", "default:*"]


@pytest.mark.asyncio
async def test_config_registry_preserves_existing_mcp_server_config(
    tmp_path: Path,
) -> None:
    registry, app_paths = _create_registry(tmp_path)
    config_dir = app_paths.config_dir
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

    mcp_config = await registry.load_mcp()
    msprof_server = mcp_config.servers["msprof-mcp"]
    assert msprof_server.args == existing_mcp["mcpServers"]["msprof-mcp"]["args"]
    assert msprof_server.stateful is False
    assert msprof_server.enabled is False


@pytest.mark.asyncio
async def test_config_registry_empty_override_keeps_packaged_mcp_servers(
    tmp_path: Path,
) -> None:
    registry, app_paths = _create_registry(tmp_path)
    config_dir = app_paths.config_dir
    config_dir.mkdir(parents=True)
    (config_dir / "config.mcp.json").write_text(
        json.dumps({"mcpServers": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mcp_config = await registry.load_mcp()
    default_path = registry.default_config_dir / "config.mcp.json"
    default_names = set(json.loads(default_path.read_text(encoding="utf-8"))["mcpServers"])

    assert set(mcp_config.servers) == default_names

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

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from msagent.configs.agent import AgentConfig, BatchAgentConfig


def _agent(name: str, *, default: bool) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": name,
            "default": default,
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "max_tokens": 1024,
                "temperature": 0,
            },
        }
    )


def test_batch_agent_config_rejects_nondefault_agent_when_only_one_agent_exists() -> None:
    with pytest.raises(ValidationError, match="must be marked as default=true"):
        BatchAgentConfig(agents=[_agent("general", default=False)])


def test_batch_agent_config_rejects_multiple_defaults_when_more_than_one_agent_is_default() -> None:
    with pytest.raises(ValidationError, match="Multiple agents marked as default"):
        BatchAgentConfig(
            agents=[
                _agent("general", default=True),
                _agent("profiler", default=True),
            ]
        )


def test_batch_agent_config_returns_expected_agents_when_lookup_uses_name_or_default() -> None:
    config = BatchAgentConfig(
        agents=[
            _agent("general", default=True),
            _agent("profiler", default=False),
        ]
    )

    assert config.agent_names == ["general", "profiler"]
    assert config.get_agent_config(None) is config.agents[0]
    assert config.get_agent_config("profiler") is config.agents[1]
    assert config.get_agent_config("missing") is None


@pytest.mark.asyncio
async def test_update_agent_llm_updates_directory_file_when_agent_file_exists(
    tmp_path: Path,
) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    agent_file = agents_dir / "general.yml"
    agent_file.write_text("name: general\nllm: old-model\n", encoding="utf-8")
    aggregate_file = tmp_path / "agents.yml"
    aggregate_file.write_text("agents:\n  - name: general\n    llm: aggregate-model\n", encoding="utf-8")

    await BatchAgentConfig.update_agent_llm(aggregate_file, "general", "new-model", agents_dir)

    assert yaml.safe_load(agent_file.read_text(encoding="utf-8"))["llm"] == "new-model"
    assert yaml.safe_load(aggregate_file.read_text(encoding="utf-8"))["agents"][0]["llm"] == "aggregate-model"


@pytest.mark.asyncio
async def test_update_agent_llm_updates_aggregate_file_when_directory_agent_is_missing(
    tmp_path: Path,
) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    aggregate_file = tmp_path / "agents.yml"
    aggregate_file.write_text(
        "agents:\n  - name: general\n    llm: old-model\n  - name: profiler\n    llm: profiler-model\n",
        encoding="utf-8",
    )

    await BatchAgentConfig.update_agent_llm(aggregate_file, "general", "new-model", agents_dir)

    agents = yaml.safe_load(aggregate_file.read_text(encoding="utf-8"))["agents"]
    assert agents == [
        {"name": "general", "llm": "new-model"},
        {"name": "profiler", "llm": "profiler-model"},
    ]


@pytest.mark.asyncio
async def test_update_default_agent_marks_single_target_when_directory_and_aggregate_exist(
    tmp_path: Path,
) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for name, default in (("general", True), ("profiler", False)):
        (agents_dir / f"{name}.yml").write_text(
            yaml.safe_dump({"name": name, "default": default}),
            encoding="utf-8",
        )
    aggregate_file = tmp_path / "agents.yml"
    aggregate_file.write_text(
        yaml.safe_dump(
            {
                "agents": [
                    {"name": "general", "default": True},
                    {"name": "profiler", "default": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    await BatchAgentConfig.update_default_agent(aggregate_file, "profiler", agents_dir)

    assert yaml.safe_load((agents_dir / "general.yml").read_text(encoding="utf-8"))["default"] is False
    assert yaml.safe_load((agents_dir / "profiler.yml").read_text(encoding="utf-8"))["default"] is True
    aggregate_agents = yaml.safe_load(aggregate_file.read_text(encoding="utf-8"))["agents"]
    assert [agent["default"] for agent in aggregate_agents] == [False, True]


@pytest.mark.asyncio
async def test_add_agent_skill_pattern_persists_once_when_directory_agent_matches(
    tmp_path: Path,
) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    agent_file = agents_dir / "general.yml"
    agent_file.write_text(
        yaml.safe_dump({"name": "general", "skills": {"patterns": ["existing"]}}),
        encoding="utf-8",
    )

    first_changed = await BatchAgentConfig.add_agent_skill_pattern(
        tmp_path / "missing.yml", "general", "new-skill", agents_dir
    )
    second_changed = await BatchAgentConfig.add_agent_skill_pattern(
        tmp_path / "missing.yml", "general", "new-skill", agents_dir
    )

    assert first_changed is True
    assert second_changed is False
    assert yaml.safe_load(agent_file.read_text(encoding="utf-8"))["skills"]["patterns"] == [
        "existing",
        "new-skill",
    ]


@pytest.mark.asyncio
async def test_add_agent_skill_pattern_returns_false_when_agent_is_absent(
    tmp_path: Path,
) -> None:
    aggregate_file = tmp_path / "agents.yml"
    original = "agents:\n  - name: general\n"
    aggregate_file.write_text(original, encoding="utf-8")

    changed = await BatchAgentConfig.add_agent_skill_pattern(aggregate_file, "missing", "new-skill")

    assert changed is False
    assert aggregate_file.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_from_yaml_reports_available_llms_when_agent_references_unknown_llm(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "agents.yml"
    config_file.write_text(
        "agents:\n  - name: general\n    default: true\n    llm: missing-model\n",
        encoding="utf-8",
    )

    class _LLMs:
        llm_names = ["known-model"]

        @staticmethod
        def get_llm_config(_name: str) -> None:
            return None

    with pytest.raises(
        ValueError,
        match=r"LLM 'missing-model' not found\. Available: \['known-model'\]",
    ):
        await BatchAgentConfig.from_yaml(file_path=config_file, batch_llm_config=_LLMs())


@pytest.mark.asyncio
async def test_from_yaml_reports_agent_context_when_subagent_reference_is_unknown(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "agents.yml"
    config_file.write_text(
        """
agents:
  - name: general
    default: true
    llm:
      provider: openai
      model: gpt-4o-mini
      max_tokens: 1024
      temperature: 0
    subagents: [missing-subagent]
""".strip(),
        encoding="utf-8",
    )

    class _Subagents:
        subagent_names = ["known-subagent"]

        @staticmethod
        def get_subagent_config(_name: str) -> None:
            return None

    with pytest.raises(ValueError, match="For agent 'general': subagent 'missing-subagent' not found"):
        await BatchAgentConfig.from_yaml(file_path=config_file, batch_subagent_config=_Subagents())

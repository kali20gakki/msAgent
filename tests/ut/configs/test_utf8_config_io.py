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

import json
from pathlib import Path

import aiofiles
import pytest

from msagent.configs.agent import BatchAgentConfig
from msagent.configs.approval import ToolApprovalConfig
from msagent.configs.mcp import MCPConfig

PROMPT_TEXT = "Follow the prompt with \u201csmart quotes\u201d and \u4e2d\u6587\u3002"
CHINESE_TEXT = "\u4e2d\u6587"


def test_default_approval_interrupts_cover_filesystem_boundary_tools() -> None:
    payload = ToolApprovalConfig().to_interrupt_on_payload()

    assert payload is not None
    assert set(payload) >= {"execute", "ls", "read_file", "write_file", "edit_file", "glob", "grep"}


@pytest.mark.asyncio
async def test_batch_agent_config_reads_yaml_and_prompts_with_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "agents"
    prompt_dir = tmp_path / "prompts" / "agents"
    agents_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)

    (prompt_dir / "general.md").write_text(PROMPT_TEXT, encoding="utf-8")
    (agents_dir / "general.yml").write_text(
        """
name: general
default: true
prompt:
  - prompts/agents/general.md
llm:
  provider: openai
  model: gpt-4o-mini
  alias: default
  max_tokens: 0
  temperature: 0
""".strip(),
        encoding="utf-8",
    )

    path_type = type(tmp_path)
    original_read_text = path_type.read_text

    def strict_read_text(self, *args, **kwargs):
        if "encoding" not in kwargs:
            raise AssertionError(f"encoding is required for {self}")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "read_text", strict_read_text)

    config = await BatchAgentConfig.from_yaml(dir_path=agents_dir)
    assert config.agents[0].prompt == PROMPT_TEXT


def test_tool_approval_config_utf8_round_trip_and_temp_cleanup(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"
    config = ToolApprovalConfig()
    config.prepend_decision_rule(
        tool_name="execute",
        tool_args={"command": f"echo {CHINESE_TEXT}"},
        decision="always_reject",
    )

    config.save_to_json_file(config_path)
    reloaded = ToolApprovalConfig.from_json_file(config_path)

    assert reloaded.resolve_decision("execute", {"command": f"echo {CHINESE_TEXT}"}) == "always_reject"
    assert CHINESE_TEXT in config_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".config.approval.json.*.tmp"))


@pytest.mark.parametrize("content", ["{broken", '"not-an-object"', "invalid-utf8"])
def test_tool_approval_config_corrupt_file_falls_back_to_empty(tmp_path: Path, content: str) -> None:
    config_path = tmp_path / "config.approval.json"
    if content == "invalid-utf8":
        config_path.write_bytes(b"\xff")
    else:
        config_path.write_text(content, encoding="utf-8")

    config = ToolApprovalConfig.from_json_file(config_path)

    assert config.decision_rules == []


def test_tool_approval_config_merges_each_update_with_latest_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"

    first = ToolApprovalConfig.prepend_rule_to_json_file(
        config_path,
        tool_name="execute",
        tool_args={"command": "python first.py"},
        decision="always_approve",
    )
    second = ToolApprovalConfig.prepend_rule_to_json_file(
        config_path,
        tool_name="execute",
        tool_args={"command": "python second.py"},
        decision="always_reject",
    )

    assert len(first.decision_rules) == 1
    assert len(second.decision_rules) == 2
    reloaded = ToolApprovalConfig.from_json_file(config_path)
    assert reloaded.resolve_decision("execute", {"command": "python first.py"}) == "always_approve"
    assert reloaded.resolve_decision("execute", {"command": "python second.py"}) == "always_reject"


def test_tool_approval_config_persists_project_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"

    saved = ToolApprovalConfig.update_mode_in_json_file(config_path, "auto")
    reloaded = ToolApprovalConfig.from_json_file(config_path)

    assert saved.execute_approval_mode == "auto"
    assert reloaded.execute_approval_mode == "auto"


def test_family_rule_matches_opaque_commands_with_different_payloads(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"

    ToolApprovalConfig.prepend_family_rule_to_json_file(
        config_path,
        family="python3 -c",
        decision="always_approve",
    )
    reloaded = ToolApprovalConfig.from_json_file(config_path)

    assert reloaded.resolve_family_decision("python3 -c") == "always_approve"
    assert reloaded.resolve_family_decision("bash -c") == "ask"


def test_script_rule_matches_arguments_but_not_another_script(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"
    script = tmp_path / "scripts" / "report.py"
    script.parent.mkdir()
    script.write_text("print('ok')", encoding="utf-8")

    ToolApprovalConfig.prepend_script_rule_to_json_file(
        config_path,
        interpreter="python3",
        path=script,
        decision="always_approve",
    )
    reloaded = ToolApprovalConfig.from_json_file(config_path)

    assert reloaded.resolve_script_decision(interpreter="python3", path=str(script.resolve())) == "always_approve"
    assert (
        reloaded.resolve_script_decision(interpreter="python3", path=str(script.with_name("other.py").resolve()))
        == "ask"
    )


def test_external_directory_rule_matches_only_path_boundary(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"
    external = tmp_path / "external"
    sibling = tmp_path / "external-other"

    ToolApprovalConfig.prepend_external_directory_rule_to_json_file(config_path, external)
    reloaded = ToolApprovalConfig.from_json_file(config_path)

    assert reloaded.allows_external_path(external / "nested" / "file.txt")
    assert not reloaded.allows_external_path(sibling / "file.txt")


def test_project_rule_has_stable_id_and_can_be_removed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.approval.json"
    ToolApprovalConfig.prepend_family_rule_to_json_file(
        config_path,
        family="python3 -c",
        decision="always_approve",
    )
    first = ToolApprovalConfig.from_json_file(config_path)
    second = ToolApprovalConfig.from_json_file(config_path)
    rule_id = first.family_rules[0].id

    assert rule_id
    assert second.family_rules[0].id == rule_id
    assert ToolApprovalConfig.remove_rule_from_json_file(config_path, rule_id)
    assert ToolApprovalConfig.from_json_file(config_path).family_rules == []


@pytest.mark.asyncio
async def test_mcp_config_reads_utf8_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "msprof-mcp": {
                        "command": "uvx",
                        "args": ["server"],
                        "env": {"LABEL": CHINESE_TEXT},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    original_aiofiles_open = aiofiles.open

    def strict_aiofiles_open(*args, **kwargs):
        if "encoding" not in kwargs:
            raise AssertionError("encoding is required for aiofiles.open")
        return original_aiofiles_open(*args, **kwargs)

    monkeypatch.setattr(aiofiles, "open", strict_aiofiles_open)

    config = await MCPConfig.from_json(config_path)
    assert config.servers["msprof-mcp"].env["LABEL"] == CHINESE_TEXT

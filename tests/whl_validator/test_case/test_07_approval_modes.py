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

"""通过 Mock session 验证 execute 审批模式、黑白名单和规则持久化。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.handlers.interrupts import InterruptHandler
from msagent.configs import ToolApprovalConfig


pytestmark = [pytest.mark.local]


def _build_session(tmp_path: Path, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path, state_dir=tmp_path),
        prompt=SimpleNamespace(mode_change_callback=None),
        approval_session_rules=[],
        execute_approval_mode=mode,
    )


def _execute_payload(command: str) -> dict[str, object]:
    return {
        "action_requests": [{"name": "execute", "args": {"command": command}}],
        "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
    }


def _rules(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "config.approval.json"
    return list(json.loads(path.read_text(encoding="utf-8")).get("decision_rules") or []) if path.exists() else []


async def _arun_with_prompt(
    handler: InterruptHandler,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    selected: str = "always_approve",
) -> tuple[dict[str, object] | None, bool, list[str]]:
    prompted: list[str] = []

    async def _prompt(**kwargs):
        prompted.append(str(kwargs["tool_args"]["command"]))
        return selected

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)
    result, interacted = await handler._get_hitl_decisions(_execute_payload(command))
    return result, interacted, prompted


def test_safe_mode_approval_policy_and_project_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path, "safe")
    handler = InterruptHandler(session)

    whitelist_result, whitelist_interacted = asyncio.run(handler._get_hitl_decisions(_execute_payload("pwd && ls -la")))
    ordinary_command = "chmod 777 /tmp/msagent-approval-demo && ls -l /tmp/msagent-approval-demo"
    ordinary_result, ordinary_interacted, prompted = asyncio.run(
        _arun_with_prompt(handler, monkeypatch, ordinary_command)
    )

    assert whitelist_result == {"decisions": [{"type": "approve"}]}
    assert whitelist_interacted is False
    assert ordinary_result == {"decisions": [{"type": "approve"}]}
    assert ordinary_interacted is True
    assert prompted == [ordinary_command]
    assert session.approval_session_rules == []
    assert _rules(tmp_path) == [
        {
            "name": "execute",
            "args": {"command": rf"^{re.escape(ordinary_command)}$"},
            "decision": "always_approve",
        }
    ]

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")
    assert config.resolve_decision("execute", {"command": ordinary_command}) == "always_approve"
    assert config.resolve_decision("execute", {"command": ordinary_command + " --changed"}) == "ask"


def test_convenience_mode_policy_and_session_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path, "convenience")
    handler = InterruptHandler(session)

    whitelist_result, whitelist_interacted = asyncio.run(
        handler._get_hitl_decisions(_execute_payload("git status --short"))
    )
    ordinary_command = "python approval_demo.py"
    ordinary_result, ordinary_interacted = asyncio.run(handler._get_hitl_decisions(_execute_payload(ordinary_command)))
    blacklist_command = "rm -rf /tmp/msagent-approval-demo"
    blacklist_result, blacklist_interacted, prompted = asyncio.run(
        _arun_with_prompt(handler, monkeypatch, blacklist_command)
    )

    assert whitelist_result == {"decisions": [{"type": "approve"}]}
    assert whitelist_interacted is False
    assert ordinary_result == {"decisions": [{"type": "approve"}]}
    assert ordinary_interacted is False
    assert blacklist_result == {"decisions": [{"type": "approve"}]}
    assert blacklist_interacted is True
    assert prompted == [blacklist_command]
    assert _rules(tmp_path) == []
    assert len(session.approval_session_rules) == 1
    assert session.approval_session_rules[0].decision == "always_approve"
    assert session.approval_session_rules[0].matches_call("execute", {"command": blacklist_command})


@pytest.mark.parametrize("mode", ["safe", "convenience"])
def test_blacklist_always_approve_stays_session_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    session = _build_session(tmp_path, mode)
    handler = InterruptHandler(session)
    command = "echo cleanup && rm -rf /tmp/msagent-approval-demo"

    result, interacted, _prompted = asyncio.run(_arun_with_prompt(handler, monkeypatch, command))

    assert result == {"decisions": [{"type": "approve"}]}
    assert interacted is True
    assert _rules(tmp_path) == []
    assert len(session.approval_session_rules) == 1
    assert session.approval_session_rules[0].matches_call("execute", {"command": command})

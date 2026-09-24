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

#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.handlers import interrupts as interrupts_module
from msagent.cli.handlers.interrupts import InterruptHandler, SWITCH_TO_CONVENIENCE
from msagent.configs import ToolApprovalConfig


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path, state_dir=tmp_path),
        prompt=SimpleNamespace(mode_change_callback=None),
        approval_session_rules=[],
        execute_approval_mode=None,
    )


def _execute_payload(command: str) -> dict[str, object]:
    return {
        "action_requests": [{"name": "execute", "args": {"command": command}}],
        "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "command", "prompt_choice", "expected", "interacted"),
    [
        ("convenience", 'python -c "print(1)"', "reject", {"decisions": [{"type": "reject"}]}, True),
        ("safe", 'python -c "print(1)"', "approve", {"decisions": [{"type": "approve"}]}, True),
        ("safe", "ls -la", "reject", {"decisions": [{"type": "approve"}]}, False),
        ("convenience", "rm -rf /tmp/demo", "reject", {"decisions": [{"type": "reject"}]}, True),
    ],
)
async def test_execute_mode_default_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    command: str,
    prompt_choice: str,
    expected: dict[str, object],
    interacted: bool,
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = mode
    handler = InterruptHandler(session)
    prompt_calls = 0

    async def _prompt(**_kwargs):
        nonlocal prompt_calls
        prompt_calls += 1
        return prompt_choice

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload(command))

    assert result == expected
    assert user_interacted is interacted
    assert prompt_calls == int(interacted)


@pytest.mark.parametrize(
    "command",
    [
        "rm file.txt",
        "rm -rf /tmp/demo",
        "del file.txt",
        "Remove-Item file.txt",
        "Remove-Item -Recurse .\\build",
        "rmdir build",
        "echo ok && rm file.txt",
    ],
)
def test_delete_commands_are_blacklisted(tmp_path: Path, command: str) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "convenience"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": command})

    assert profile["category"] == "blacklist"
    assert profile["default_decision"] == "ask"
    assert profile["persistence_scope"] == "session"


def test_compound_whitelist_commands_auto_approve(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": "pwd && ls -la"})

    assert profile["category"] == "whitelist"
    assert profile["default_decision"] == "approve"
    assert profile["persistence_scope"] == "session"


def test_rm_substring_does_not_match_delete_blacklist(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "convenience"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={"command": "python script.py --message confirm"},
    )

    assert profile["category"] == "ordinary"


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "rm -rf /tmp/demo"',
        'env /bin/bash -c "rm -rf /tmp/demo"',
        'sh -c "rm -rf /tmp/demo"',
        'python -c "import os; os.system(\'rm -rf /tmp/demo\')"',
        '/usr/bin/python3 -I -c "print(1)"',
        "printf '%s\\n' /tmp/demo | xargs rm -rf",
        "echo $(rm -rf /tmp/demo)",
        "echo `rm -rf /tmp/demo`",
    ],
)
def test_wrapped_commands_require_approval_in_convenience_mode(tmp_path: Path, command: str) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "convenience"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": command})

    assert profile == {
        "category": "blacklist",
        "default_decision": "ask",
        "persistence_scope": "session",
    }


def test_whitelist_command_in_compound_execute_does_not_auto_approve(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={
            "command": "chmod 777 /home/ylb/msagent/code/333/123.txt && ls -l /home/ylb/msagent/code/333/123.txt"
        },
    )

    assert profile["category"] == "ordinary"
    assert profile["default_decision"] == "ask"
    assert profile["persistence_scope"] == "project"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "expected_result", "expected_rule"),
    [
        ("always_approve", {"decisions": [{"type": "approve"}]}, "always_approve"),
        (
            "always_reject",
            {"decisions": [{"type": "reject", "message": "Rejected by local approval policy."}]},
            "always_reject",
        ),
    ],
)
async def test_safe_mode_always_rules_persist_to_project_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    expected_result: dict[str, object],
    expected_rule: str,
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    handler = InterruptHandler(session)

    async def _prompt(**_kwargs):
        return choice

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("python script.py"))

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")

    assert result == expected_result
    assert user_interacted is True
    assert session.approval_session_rules == []
    assert config.resolve_decision("execute", {"command": "python script.py"}) == expected_rule


def test_stale_session_snapshots_merge_project_rules(tmp_path: Path) -> None:
    first_handler = InterruptHandler(_build_session(tmp_path))
    second_handler = InterruptHandler(_build_session(tmp_path))
    first_snapshot = first_handler._load_project_approval_config()
    second_snapshot = second_handler._load_project_approval_config()

    first_handler._persist_decision(
        approval_config=first_snapshot,
        tool_name="execute",
        tool_args={"command": "python first.py"},
        decision="always_approve",
        scope="project",
    )
    second_handler._persist_decision(
        approval_config=second_snapshot,
        tool_name="execute",
        tool_args={"command": "python second.py"},
        decision="always_reject",
        scope="project",
    )

    reloaded = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")
    assert reloaded.resolve_decision("execute", {"command": "python first.py"}) == "always_approve"
    assert reloaded.resolve_decision("execute", {"command": "python second.py"}) == "always_reject"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["safe", "convenience"])
async def test_blacklist_always_approve_is_session_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = mode
    handler = InterruptHandler(session)

    async def _prompt(**_kwargs):
        return "always_approve"

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("rm -rf /tmp/demo"))

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")

    assert result == {"decisions": [{"type": "approve"}]}
    assert user_interacted is True
    assert session.approval_session_rules
    assert config.resolve_decision("execute", {"command": "rm -rf /tmp/demo"}) == "ask"


@pytest.mark.asyncio
async def test_project_rules_do_not_auto_approve_blacklist_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    handler = InterruptHandler(session)
    project_config = ToolApprovalConfig(decision_rules=[])
    project_config.prepend_decision_rule(
        tool_name="execute",
        tool_args={"command": "rm -rf /tmp/demo"},
        decision="always_approve",
    )
    project_config.save_to_json_file(tmp_path / "config.approval.json")

    async def _prompt(**_kwargs):
        return "reject"

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("rm -rf /tmp/demo"))

    assert result == {"decisions": [{"type": "reject"}]}
    assert user_interacted is True


@pytest.mark.asyncio
async def test_safe_mode_prompt_includes_switch_to_convenience_and_project_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    handler = InterruptHandler(session)
    captured: dict[str, object] = {}

    async def _prompt_from_options(*, question: str, options: list[str]) -> str | None:
        captured["question"] = question
        captured["options"] = options
        return "approve"

    monkeypatch.setattr(handler, "_prompt_from_options", _prompt_from_options)

    await handler._prompt_hitl_decision(
        tool_name="execute",
        tool_args={"command": "python script.py"},
        description="Run script",
        options=handler._approval_options(
            allowed=["approve", "reject"],
            command_profile=handler._classify_command(tool_name="execute", tool_args={"command": "python script.py"}),
        ),
        command_profile=handler._classify_command(tool_name="execute", tool_args={"command": "python script.py"}),
    )

    assert SWITCH_TO_CONVENIENCE in captured["options"]
    question = str(captured["question"])
    assert "exactly the same command text" in question


def test_project_approval_path_requires_state_dir(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.context.state_dir = None
    handler = InterruptHandler(session)

    with pytest.raises(RuntimeError, match="Project state directory is required"):
        handler._project_approval_path()

    assert not (tmp_path / ".msagent").exists()


@pytest.mark.asyncio
async def test_switch_to_convenience_updates_mode_and_approves_current_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    handler = InterruptHandler(session)

    async def _prompt(**_kwargs):
        return SWITCH_TO_CONVENIENCE

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("python script.py"))

    assert result == {"decisions": [{"type": "approve"}]}
    assert user_interacted is True
    assert session.execute_approval_mode == "convenience"


@pytest.mark.asyncio
async def test_handle_records_audit_only_when_user_prompted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, object]] = []
    session = _build_session(tmp_path)
    session.execute_approval_mode = "safe"
    session.audit_writer = SimpleNamespace(
        enabled=True,
        emit_user_response=lambda **kwargs: recorded.append(kwargs),
    )
    handler = InterruptHandler(session)

    async def _prompt(**_kwargs):
        return "approve"

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    interrupt = SimpleNamespace(id="int-manual", value=_execute_payload("python script.py"))
    result = await handler.handle([interrupt])

    assert result == {"decisions": [{"type": "approve"}]}
    assert len(recorded) == 1
    assert recorded[0]["response"] == "approve"


@pytest.mark.asyncio
async def test_hitl_prompts_for_execute_approval_mode_once_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    handler = InterruptHandler(session)
    prompt_calls: list[str] = []

    async def _prompt_mode() -> str | None:
        prompt_calls.append("mode")
        return "convenience"

    monkeypatch.setattr(handler, "_prompt_execute_approval_mode", _prompt_mode)

    first, first_interacted = await handler._get_hitl_decisions(_execute_payload("echo hello"))
    second, second_interacted = await handler._get_hitl_decisions(_execute_payload("echo again"))

    assert first == {"decisions": [{"type": "approve"}]}
    assert second == {"decisions": [{"type": "approve"}]}
    assert first_interacted is True
    assert second_interacted is False
    assert session.execute_approval_mode == "convenience"
    assert prompt_calls == ["mode"]


@pytest.mark.asyncio
async def test_interrupt_handler_returns_none_for_unknown_payload_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    monkeypatch.setattr(interrupts_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(interrupts_module.console, "print", lambda *_args, **_kwargs: None)

    fake_interrupt = SimpleNamespace(id="int-1", value="just a string")
    result, _user_interacted = await handler._get_choice(fake_interrupt)

    assert result is None

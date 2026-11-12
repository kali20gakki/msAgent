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
from msagent.cli.handlers.interrupts import ALWAYS_ALLOW_PROJECT_SCRIPT, InterruptHandler, SWITCH_TO_AUTO
from msagent.configs import ToolApprovalConfig


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path, state_dir=tmp_path),
        prompt=SimpleNamespace(mode_change_callback=None),
        execute_approval_mode=None,
    )


def _execute_payload(command: str, **extra_args: object) -> dict[str, object]:
    return {
        "action_requests": [{"name": "execute", "args": {"command": command, **extra_args}}],
        "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
    }


def _tool_payload(name: str, args: dict[str, object]) -> dict[str, object]:
    return {
        "action_requests": [{"name": name, "args": args}],
        "review_configs": [{"action_name": name, "allowed_decisions": ["approve", "reject"]}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "command", "prompt_choice", "expected", "interacted"),
    [
        ("auto", 'python -c "print(1)"', "reject", {"decisions": [{"type": "approve"}]}, False),
        ("manual", 'python -c "print(1)"', "approve", {"decisions": [{"type": "approve"}]}, True),
        ("manual", "ls -la", "reject", {"decisions": [{"type": "approve"}]}, False),
        ("auto", "rm -rf /tmp/demo", "reject", {"decisions": [{"type": "reject"}]}, True),
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
def test_delete_commands_are_dangerous(tmp_path: Path, command: str) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": command})

    assert profile["category"] == "dangerous"
    assert profile["default_decision"] == "ask"
    assert profile["persistence_scope"] == "none"


def test_compound_whitelist_commands_auto_approve(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": "pwd && ls -la"})

    assert profile["category"] == "readonly"
    assert profile["default_decision"] == "approve"
    assert profile["persistence_scope"] == "none"


def test_readonly_pipeline_and_version_probe_auto_approve(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={
            "command": "ls -la /home/fanglanyue/.local/share/perfetto/prebuilts/ 2>&1 | head -30; "
            "ldd --version 2>&1 | head"
        },
    )

    assert profile == {
        "category": "readonly",
        "default_decision": "approve",
        "persistence_scope": "none",
    }


def test_readonly_directory_summary_and_sorted_listing_auto_approve(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={
            "command": "cd /home/fanglanyue/data_space/profile_count_0 && du -sh . && "
            "ls -la g340663150/ASCEND_PROFILER_OUTPUT/ | sort -k5 -n -r | head -20 && "
            "echo '--- advisor html ---' && ls -la /home/fanglanyue/workspace/msagent/mstt_advisor_*.html"
        },
    )

    assert profile == {
        "category": "readonly",
        "default_decision": "approve",
        "persistence_scope": "none",
    }


def test_output_redirection_prevents_readonly_classification(tmp_path: Path) -> None:
    handler = InterruptHandler(_build_session(tmp_path))

    profile = handler._classify_command(tool_name="execute", tool_args={"command": "ls -la > listing.txt"})

    assert profile["category"] == "ordinary"


def test_rm_substring_does_not_match_delete_blacklist(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={"command": "python script.py --message confirm"},
    )

    assert profile["category"] == "ordinary"


@pytest.mark.parametrize(
    "command_template",
    [
        "python3 tools/report.py --input trace.json",
        "time python3 tools/report.py --input trace.json 2>&1 | tail -80",
        "cd {project} && time python3 tools/report.py --input trace.json 2>&1 | tail -80",
        "cd {project} && python3 tools/report.py | sort | tail -10",
    ],
)
def test_readonly_shell_chain_recognizes_project_script(tmp_path: Path, command_template: str) -> None:
    script = tmp_path / "tools" / "report.py"
    script.parent.mkdir()
    script.write_text("print('ok')", encoding="utf-8")
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={"command": command_template.format(project=tmp_path)},
    )

    assert profile["category"] == "ordinary"
    assert profile["script_interpreter"] == "python3"
    assert profile["script_path"] == str(script.resolve())
    assert ALWAYS_ALLOW_PROJECT_SCRIPT in handler._approval_options(
        allowed=["approve", "reject"],
        command_profile=profile,
        approval_config=ToolApprovalConfig(),
    )


@pytest.mark.parametrize(
    "command_template",
    [
        "python3 tools/report.py > output.txt",
        "python3 tools/report.py && curl https://example.invalid",
        "python3 tools/report.py && python3 tools/report.py",
        "time env python3 tools/report.py",
        "cd {project} && python3 tools/report.py && mkdir generated",
    ],
)
def test_non_readonly_shell_chain_does_not_recognize_project_script(tmp_path: Path, command_template: str) -> None:
    script = tmp_path / "tools" / "report.py"
    script.parent.mkdir()
    script.write_text("print('ok')", encoding="utf-8")
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={"command": command_template.format(project=tmp_path)},
    )

    assert "script_interpreter" not in profile
    assert "script_path" not in profile


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "rm -rf /tmp/demo"',
        'env /bin/bash -c "rm -rf /tmp/demo"',
        'sh -c "rm -rf /tmp/demo"',
        'python -c "import os; os.system(\'rm -rf /tmp/demo\')"',
        '/usr/bin/python3 -I -c "print(1)"',
        "python3 - <<'EOF'\nprint(1)\nEOF",
        "printf '%s\\n' /tmp/demo | xargs rm -rf",
        "echo $(rm -rf /tmp/demo)",
        "echo `rm -rf /tmp/demo`",
    ],
)
def test_wrapped_commands_auto_approve_in_auto_mode(tmp_path: Path, command: str) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": command})

    assert profile["category"] == "opaque"
    assert profile["default_decision"] == "approve"
    assert profile["persistence_scope"] == "project"
    assert profile["family"]


def test_python_stdin_and_dash_c_share_inline_code_family(tmp_path: Path) -> None:
    handler = InterruptHandler(_build_session(tmp_path))

    dash_c = handler._classify_command(tool_name="execute", tool_args={"command": 'python3 -c "print(1)"'})
    stdin = handler._classify_command(tool_name="execute", tool_args={"command": "python3 - <<'EOF'\nprint(2)\nEOF"})

    assert dash_c["family"] == "python3 inline code"
    assert stdin["family"] == "python3 inline code"


def test_catastrophic_delete_is_forbidden(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    profile = handler._classify_command(tool_name="execute", tool_args={"command": "rm -rf /"})

    assert profile["category"] == "forbidden"
    assert profile["default_decision"] == "always_reject"
    assert profile["persistence_scope"] == "none"


def test_dangerous_compound_segment_overrides_opaque_family(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    profile = handler._classify_command(
        tool_name="execute",
        tool_args={"command": 'python3 -c "print(1)" && rm /tmp/demo'},
    )

    assert profile["category"] == "dangerous"


@pytest.mark.asyncio
async def test_forbidden_command_is_rejected_without_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    async def _unexpected_prompt(**_kwargs):
        raise AssertionError("forbidden commands must be rejected without prompting")

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _unexpected_prompt)
    result, interacted = await handler._get_hitl_decisions(_execute_payload("rm -rf /"))

    assert result == {"decisions": [{"type": "reject", "message": "Rejected by local approval policy."}]}
    assert interacted is False


def test_whitelist_command_in_compound_execute_does_not_auto_approve(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
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
async def test_manual_mode_always_rules_persist_to_project_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    expected_result: dict[str, object],
    expected_rule: str,
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    async def _prompt(**_kwargs):
        return choice

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("python script.py"))

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")

    assert result == expected_result
    assert user_interacted is True
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


def test_dangerous_commands_cannot_be_remembered(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)
    profile = handler._classify_command(tool_name="execute", tool_args={"command": "rm /tmp/demo"})

    assert handler._approval_options(allowed=["approve", "reject"], command_profile=profile) == [
        "approve",
        "reject",
    ]


def test_opaque_commands_offer_only_once_project_family_or_reject(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)
    profile = handler._classify_command(tool_name="execute", tool_args={"command": 'python3 -c "print(1)"'})

    assert handler._approval_options(allowed=["approve", "reject"], command_profile=profile) == [
        "approve",
        "always_approve",
        "reject",
    ]


def test_manual_inline_commands_offer_switch_to_auto(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)
    profile = handler._classify_command(tool_name="execute", tool_args={"command": 'python3 -c "print(1)"'})

    assert handler._approval_options(allowed=["approve", "reject"], command_profile=profile) == [
        "approve",
        "always_approve",
        "reject",
        SWITCH_TO_AUTO,
    ]


@pytest.mark.asyncio
async def test_project_rules_do_not_auto_approve_dangerous_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
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
async def test_opaque_family_allow_persists_for_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    async def _always_allow(**_kwargs):
        return "always_approve"

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _always_allow)
    first, first_interacted = await handler._get_hitl_decisions(_execute_payload('python3 -c "print(1)"'))

    restarted_session = _build_session(tmp_path)
    restarted_session.execute_approval_mode = "manual"
    restarted_handler = InterruptHandler(restarted_session)

    async def _unexpected_prompt(**_kwargs):
        raise AssertionError("family rule should auto-approve a different python3 -c payload")

    monkeypatch.setattr(restarted_handler, "_prompt_hitl_decision", _unexpected_prompt)
    second, second_interacted = await restarted_handler._get_hitl_decisions(_execute_payload('python3 -c "print(2)"'))

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")
    assert first == {"decisions": [{"type": "approve"}]}
    assert first_interacted is True
    assert second == {"decisions": [{"type": "approve"}]}
    assert second_interacted is False
    assert config.resolve_family_decision("python3 inline code") == "always_approve"


@pytest.mark.asyncio
async def test_manual_project_script_rule_ignores_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "report.py"
    script.write_text("print('ok')", encoding="utf-8")
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)
    captured: dict[str, object] = {}

    async def _allow_script(**kwargs):
        captured["options"] = kwargs["options"]
        return ALWAYS_ALLOW_PROJECT_SCRIPT

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _allow_script)
    first, first_interacted = await handler._get_hitl_decisions(_execute_payload("python3 report.py --first"))

    restarted_session = _build_session(tmp_path)
    restarted_session.execute_approval_mode = "manual"
    restarted_handler = InterruptHandler(restarted_session)

    async def _unexpected_prompt(**_kwargs):
        raise AssertionError("project script rule should ignore invocation arguments")

    monkeypatch.setattr(restarted_handler, "_prompt_hitl_decision", _unexpected_prompt)
    second, second_interacted = await restarted_handler._get_hitl_decisions(
        _execute_payload("python3 report.py --second")
    )

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")
    assert captured["options"] == [
        "approve",
        ALWAYS_ALLOW_PROJECT_SCRIPT,
        "reject",
        "always_reject",
        SWITCH_TO_AUTO,
    ]
    assert first == {"decisions": [{"type": "approve"}]}
    assert first_interacted is True
    assert second == {"decisions": [{"type": "approve"}]}
    assert second_interacted is False
    assert config.resolve_script_decision(interpreter="python3", path=str(script.resolve())) == "always_approve"


@pytest.mark.asyncio
async def test_external_script_needs_directory_grant_before_script_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    (project / ".git").mkdir()
    external.mkdir()
    script = external / "report.py"
    script.write_text("print('ok')", encoding="utf-8")
    session = _build_session(tmp_path)
    session.context.working_dir = project
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)
    captured: dict[str, object] = {}

    async def _allow_directory(*, directory: Path):
        assert directory == external.resolve()
        return "always_approve"

    async def _allow_script(**kwargs):
        captured["options"] = kwargs["options"]
        return ALWAYS_ALLOW_PROJECT_SCRIPT

    monkeypatch.setattr(handler, "_prompt_external_directory", _allow_directory)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _allow_script)
    result, interacted = await handler._get_hitl_decisions(_execute_payload(f"python3 {script}"))

    assert result == {"decisions": [{"type": "approve"}]}
    assert interacted is True
    assert ALWAYS_ALLOW_PROJECT_SCRIPT in captured["options"]


@pytest.mark.asyncio
async def test_external_workdir_can_be_allowed_for_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    nested = external / "nested"
    project.mkdir()
    (project / ".git").mkdir()
    nested.mkdir(parents=True)
    session = _build_session(tmp_path)
    session.context.working_dir = project
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)

    async def _allow_external(**kwargs):
        assert kwargs["directory"] == external.resolve()
        return "always_approve"

    monkeypatch.setattr(handler, "_prompt_external_directory", _allow_external)
    first, first_interacted = await handler._get_hitl_decisions(_execute_payload("echo hello", cwd=str(external)))

    restarted_session = _build_session(tmp_path)
    restarted_session.context.working_dir = project
    restarted_session.execute_approval_mode = "auto"
    restarted_handler = InterruptHandler(restarted_session)

    async def _unexpected_external_prompt(**_kwargs):
        raise AssertionError("saved external-directory rule should cover descendants")

    monkeypatch.setattr(restarted_handler, "_prompt_external_directory", _unexpected_external_prompt)
    second, second_interacted = await restarted_handler._get_hitl_decisions(
        _execute_payload("echo again", cwd=str(nested))
    )

    config = ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json")
    assert first == {"decisions": [{"type": "approve"}]}
    assert first_interacted is True
    assert second == {"decisions": [{"type": "approve"}]}
    assert second_interacted is False
    assert config.allows_external_path(nested)


@pytest.mark.asyncio
async def test_external_file_path_prompts_but_project_file_auto_approves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    (project / ".git").mkdir()
    external.mkdir()
    internal_file = project / "inside.txt"
    external_file = external / "outside.txt"
    internal_file.write_text("inside", encoding="utf-8")
    external_file.write_text("outside", encoding="utf-8")
    session = _build_session(tmp_path)
    session.context.working_dir = project
    handler = InterruptHandler(session)
    prompted: list[Path] = []

    async def _reject_external(*, directory: Path):
        prompted.append(directory)
        return "reject"

    monkeypatch.setattr(handler, "_prompt_external_directory", _reject_external)
    internal, internal_interacted = await handler._get_hitl_decisions(
        _tool_payload("read_file", {"file_path": str(internal_file)})
    )
    outside, outside_interacted = await handler._get_hitl_decisions(
        _tool_payload("read_file", {"file_path": str(external_file)})
    )

    assert internal == {"decisions": [{"type": "approve"}]}
    assert internal_interacted is False
    assert outside == {"decisions": [{"type": "reject", "message": "External directory access was rejected."}]}
    assert outside_interacted is True
    assert prompted == [external.resolve()]


@pytest.mark.asyncio
async def test_escaping_relative_command_argument_triggers_external_directory_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    (project / ".git").mkdir()
    external.mkdir()
    session = _build_session(tmp_path)
    session.context.working_dir = project
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)
    prompted: list[Path] = []

    async def _reject_external(*, directory: Path):
        prompted.append(directory)
        return "reject"

    monkeypatch.setattr(handler, "_prompt_external_directory", _reject_external)
    result, interacted = await handler._get_hitl_decisions(_execute_payload("ls ../external"))

    assert result == {"decisions": [{"type": "reject", "message": "External directory access was rejected."}]}
    assert interacted is True
    assert prompted == [external.resolve()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "tool_args", "expected_path"),
    [
        (r'type "C:\outside files\file.txt"', {}, r"C:\outside files\file.txt"),
        (r"type \\server\share\file.txt", {}, r"\\server\share\file.txt"),
        (r"type \outside\file.txt", {}, r"\outside\file.txt"),
        (r"cd C:\outside", {}, r"C:\outside"),
        ("dir", {"cwd": r"C:\outside"}, r"C:\outside"),
    ],
)
async def test_windows_paths_trigger_external_directory_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, tool_args: dict[str, str], expected_path: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    session = _build_session(project)
    session.execute_approval_mode = "auto"
    handler = InterruptHandler(session)
    monkeypatch.setattr(interrupts_module, "sys", SimpleNamespace(platform="win32"))

    def resolve_path(value: str, *, base: Path, directory: bool) -> Path:
        assert value == expected_path
        return outside

    async def reject_external(*, directory: Path) -> str:
        assert directory == outside
        return "reject"

    monkeypatch.setattr(handler, "_resolve_path", resolve_path)
    monkeypatch.setattr(handler, "_prompt_external_directory", reject_external)

    result, interacted = await handler._get_hitl_decisions(_execute_payload(command, **tool_args))

    assert result == {"decisions": [{"type": "reject", "message": "External directory access was rejected."}]}
    assert interacted


@pytest.mark.asyncio
async def test_manual_mode_approval_menu_includes_switch_and_project_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)
    captured: dict[str, object] = {}

    async def _run_menu(menu):
        captured["title"] = menu.title
        captured["details"] = menu.details
        captured["metadata"] = menu.metadata
        captured["notes"] = menu.notes
        captured["options"] = [option.value for option in menu.options]
        captured["labels"] = [option.label for option in menu.options]
        return interrupts_module.ApprovalChoice("approve")

    monkeypatch.setattr(interrupts_module.ApprovalMenu, "run_async", _run_menu)

    result = await handler._prompt_hitl_decision(
        tool_name="execute",
        tool_args={"command": "python script.py"},
        description="Run script",
        options=handler._approval_options(
            allowed=["approve", "reject"],
            command_profile=handler._classify_command(tool_name="execute", tool_args={"command": "python script.py"}),
        ),
        command_profile=handler._classify_command(tool_name="execute", tool_args={"command": "python script.py"}),
    )

    assert result == interrupts_module.ApprovalChoice("approve")
    assert captured["title"] == "execute Requires Approval"
    assert SWITCH_TO_AUTO in captured["options"]
    assert "Always allow this exact command for this project" in captured["labels"]
    assert "Run script\npython script.py" in str(captured["details"])
    assert captured["metadata"] == ["Mode: manual"]
    assert "Note:" in str(captured["notes"])
    assert "exactly the same command text" in str(captured["notes"])


@pytest.mark.asyncio
async def test_hitl_includes_rejection_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    async def _reject_with_feedback(**_kwargs):
        return interrupts_module.ApprovalChoice("reject", "Use a safer target.")

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _reject_with_feedback)
    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("python script.py"))

    assert result == {"decisions": [{"type": "reject", "message": "Use a safer target."}]}
    assert user_interacted is True


@pytest.mark.asyncio
async def test_external_directory_prompt_uses_approval_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    captured: dict[str, object] = {}

    async def _run_menu(menu):
        captured["title"] = menu.title
        captured["details"] = menu.details
        captured["metadata"] = menu.metadata
        captured["notes"] = menu.notes
        captured["labels"] = [option.label for option in menu.options]
        return interrupts_module.ApprovalChoice("always_approve")

    monkeypatch.setattr(interrupts_module.ApprovalMenu, "run_async", _run_menu)
    selected = await handler._prompt_external_directory(directory=tmp_path / "external")

    assert selected == "always_approve"
    assert captured["title"] == "External Directory Requires Approval"
    assert str(tmp_path / "external") in str(captured["metadata"])
    assert "Note:" in str(captured["notes"])
    assert "Always allow this directory for this project" in captured["labels"]


@pytest.mark.asyncio
async def test_mode_prompt_uses_approval_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    captured: dict[str, object] = {}

    async def _run_menu(menu):
        captured["title"] = menu.title
        captured["options"] = [option.value for option in menu.options]
        return interrupts_module.ApprovalChoice("Auto Mode")

    monkeypatch.setattr(interrupts_module.ApprovalMenu, "run_async", _run_menu)
    selected = await handler._prompt_execute_approval_mode()

    assert selected == "auto"
    assert captured["title"] == "Choose Approval Mode"
    assert captured["options"] == ["Manual Mode", "Auto Mode"]


@pytest.mark.asyncio
async def test_legacy_interrupt_uses_approval_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))

    async def _run_menu(menu):
        assert menu.title == "Approval Required"
        assert menu.details == "Continue?"
        return interrupts_module.ApprovalChoice("yes")

    monkeypatch.setattr(interrupts_module.ApprovalMenu, "run_async", _run_menu)
    selected = await handler._get_legacy_choice({"question": "Continue?", "options": ["yes", "no"]})

    assert selected == "yes"


def test_project_approval_path_requires_state_dir(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.context.state_dir = None
    handler = InterruptHandler(session)

    with pytest.raises(RuntimeError, match="Project state directory is required"):
        handler._project_approval_path()

    assert not (tmp_path / ".msagent").exists()


@pytest.mark.asyncio
async def test_switch_to_auto_updates_mode_and_approves_current_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
    handler = InterruptHandler(session)

    async def _prompt(**_kwargs):
        return SWITCH_TO_AUTO

    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result, user_interacted = await handler._get_hitl_decisions(_execute_payload("python script.py"))

    assert result == {"decisions": [{"type": "approve"}]}
    assert user_interacted is True
    assert session.execute_approval_mode == "auto"


@pytest.mark.asyncio
async def test_handle_records_audit_only_when_user_prompted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, object]] = []
    session = _build_session(tmp_path)
    session.execute_approval_mode = "manual"
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
async def test_hitl_prompts_for_execute_approval_mode_once_and_persists_for_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    handler = InterruptHandler(session)
    prompt_calls: list[str] = []

    async def _prompt_mode() -> str | None:
        prompt_calls.append("mode")
        return "auto"

    monkeypatch.setattr(handler, "_prompt_execute_approval_mode", _prompt_mode)

    first, first_interacted = await handler._get_hitl_decisions(_execute_payload("echo hello"))
    second, second_interacted = await handler._get_hitl_decisions(_execute_payload("echo again"))

    assert first == {"decisions": [{"type": "approve"}]}
    assert second == {"decisions": [{"type": "approve"}]}
    assert first_interacted is True
    assert second_interacted is False
    assert session.execute_approval_mode == "auto"
    assert prompt_calls == ["mode"]

    restarted_session = _build_session(tmp_path)
    restarted_handler = InterruptHandler(restarted_session)

    async def _unexpected_mode_prompt() -> str | None:
        raise AssertionError("saved project mode should be restored")

    monkeypatch.setattr(restarted_handler, "_prompt_execute_approval_mode", _unexpected_mode_prompt)
    restarted, restarted_interacted = await restarted_handler._get_hitl_decisions(_execute_payload("echo restored"))

    assert restarted == {"decisions": [{"type": "approve"}]}
    assert restarted_interacted is False
    assert restarted_session.execute_approval_mode == "auto"
    assert ToolApprovalConfig.from_json_file(tmp_path / "config.approval.json").execute_approval_mode == "auto"


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

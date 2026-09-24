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

from types import SimpleNamespace
from pathlib import Path

from msagent.cli.dispatchers.commands import CommandDispatcher


def _build_session(working_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(
            working_dir=working_dir,
            bash_mode=False,
            thread_id="thread-1",
            current_input_tokens=None,
            current_output_tokens=None,
        ),
        prompt=SimpleNamespace(hotkeys={}),
        renderer=SimpleNamespace(
            render_help=lambda *_args, **_kwargs: None,
            render_hotkeys=lambda *_args, **_kwargs: None,
        ),
        update_context=lambda **_kwargs: None,
        clear_tool_output=lambda: None,
        running=True,
    )


def test_command_dispatcher_removes_resume_and_replay_commands() -> None:
    session = _build_session(Path.cwd())
    dispatcher = CommandDispatcher(session)

    assert "/resume" not in dispatcher.commands
    assert "/replay" not in dispatcher.commands
    assert "/approve" not in dispatcher.commands
    assert "/memory" not in dispatcher.commands
    assert "/remember" in dispatcher.commands
    assert "/showmemory" in dispatcher.commands
    assert "/shommemory" not in dispatcher.commands
    assert "/compress" not in dispatcher.commands
    assert "/todo" not in dispatcher.commands
    assert "/graph" not in dispatcher.commands
    assert "/offload" in dispatcher.commands
    assert "/threads" in dispatcher.commands
    assert "/add-skill" in dispatcher.commands


def test_command_dispatcher_offload_delegates_to_compression_handler(monkeypatch) -> None:
    session = _build_session(Path.cwd())
    dispatcher = CommandDispatcher(session)
    calls: list[str] = []

    async def fake_handle() -> None:
        calls.append("offload")

    monkeypatch.setattr(dispatcher.compression_handler, "handle", fake_handle)

    import asyncio

    asyncio.run(dispatcher.cmd_offload([]))

    assert calls == ["offload"]


def test_command_dispatcher_threads_delegates_to_threads_handler(monkeypatch) -> None:
    session = _build_session(Path.cwd())
    session.message_dispatcher = SimpleNamespace()
    dispatcher = CommandDispatcher(session)
    calls: list[str] = []

    async def fake_handle() -> None:
        calls.append("threads")

    monkeypatch.setattr(dispatcher.threads_handler, "handle", fake_handle)

    import asyncio

    asyncio.run(dispatcher.cmd_threads([]))

    assert calls == ["threads"]


def test_command_dispatcher_remember_writes_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("msagent.cli.handlers.memory.console.print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("msagent.cli.handlers.memory.console.print", lambda *_args, **_kwargs: None)

    dispatcher = CommandDispatcher(_build_session(tmp_path))

    import asyncio

    asyncio.run(dispatcher.dispatch("/remember 用户喜欢中文回复"))

    assert "用户喜欢中文回复" in (tmp_path / ".msagent" / "memory.md").read_text(encoding="utf-8")


def test_command_dispatcher_showmemory_is_registered_without_typo_alias(tmp_path: Path) -> None:
    dispatcher = CommandDispatcher(_build_session(tmp_path))

    assert dispatcher.commands["/showmemory"] == dispatcher.cmd_showmemory
    assert "/shommemory" not in dispatcher.commands


def test_command_dispatcher_registers_permissions_command(tmp_path: Path) -> None:
    dispatcher = CommandDispatcher(_build_session(tmp_path))

    assert dispatcher.commands["/permissions"] == dispatcher.cmd_permissions


def test_command_dispatcher_permissions_mode_switches_session(tmp_path: Path, monkeypatch) -> None:
    session = _build_session(tmp_path)
    session.execute_approval_mode = None
    dispatcher = CommandDispatcher(session)
    monkeypatch.setattr("msagent.cli.handlers.permissions.console.print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("msagent.cli.handlers.permissions.console.print", lambda *_args, **_kwargs: None)

    import asyncio

    asyncio.run(dispatcher.dispatch("/permissions mode safe"))

    assert session.execute_approval_mode == "safe"


def test_command_dispatcher_permissions_clear_project_resets_rules(tmp_path: Path, monkeypatch) -> None:
    session = _build_session(tmp_path)
    session.context.state_dir = tmp_path
    dispatcher = CommandDispatcher(session)
    monkeypatch.setattr("msagent.cli.handlers.permissions.console.print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("msagent.cli.handlers.permissions.console.print", lambda *_args, **_kwargs: None)

    approval_file = tmp_path / "config.approval.json"
    approval_file.write_text(
        '{\n  "decision_rules": [\n    {"name": "execute", "args": {"command": "^echo$"}, "decision": "always_approve"}\n  ]\n}\n',
        encoding="utf-8",
    )

    import asyncio

    asyncio.run(dispatcher.dispatch("/permissions clear-project"))

    assert '"decision_rules": []' in approval_file.read_text(encoding="utf-8")

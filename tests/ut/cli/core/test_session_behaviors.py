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

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest

from msagent.cli.core.context import Context
from msagent.cli.core.session import Session
from msagent.cli.core.tool_output import ToolOutputEntry
from msagent.configs import ApprovalMode
import msagent.cli.core.session as session_module


def _context(**overrides: object) -> Context:
    values: dict[str, object] = {
        "agent": "Profiler",
        "model": "default",
        "thread_id": "thread-1",
        "working_dir": Path.cwd(),
        "approval_mode": ApprovalMode.ACTIVE,
        "recursion_limit": 80,
    }
    values.update(overrides)
    return Context.model_validate(values)


class _GraphContext:
    def __init__(self, graph: object = None) -> None:
        self.graph = graph if graph is not None else object()

    async def __aenter__(self) -> object:
        return self.graph

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _Status:
    def __init__(self) -> None:
        self.events: list[object] = []

    def __enter__(self) -> _Status:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def stop(self) -> None:
        self.events.append("stop")

    def start(self) -> None:
        self.events.append("start")

    def update(self, message: str) -> None:
        self.events.append(("update", message))


class _Recorder:
    def __init__(self) -> None:
        self.start_calls: list[tuple[object, bool]] = []
        self.finish_codes: list[int] = []

    def start(self, *, context: object, stream_output: bool) -> None:
        self.start_calls.append((context, stream_output))

    def finish(self, *, context: object, exit_code: int) -> None:
        del context
        self.finish_codes.append(exit_code)

    def record_error(self, error: BaseException) -> None:
        raise AssertionError(f"unexpected error: {error}")


def _bare_session(*, bash_mode: bool = False) -> Session:
    session = Session.__new__(Session)
    session.context = _context(bash_mode=bash_mode)
    session.renderer = Mock()
    session.command_dispatcher = SimpleNamespace(dispatch=AsyncMock())
    session.message_dispatcher = SimpleNamespace(dispatch=AsyncMock())
    session.bash_dispatcher = SimpleNamespace(dispatch=AsyncMock())
    session.prompt = SimpleNamespace(
        get_input=AsyncMock(side_effect=EOFError),
        refresh_style=Mock(),
        handle_external_sigint=Mock(return_value=False),
    )
    session.graph = None
    session.graph_context = None
    session.running = False
    session.needs_reload = False
    session.current_stream_task = None
    session._sigint_registered = False
    session._previous_sigint = None
    session._sigint_handler = None
    session.tool_outputs = []
    session.latest_tool_output = None
    session.run_recorder = None
    session.audit_writer = SimpleNamespace(rebind=Mock(), enabled=False)
    return session


@pytest.mark.asyncio
async def test_create_prompt_with_fallback_returns_stub_when_prompt_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module,
        "InteractivePrompt",
        Mock(side_effect=RuntimeError("no terminal")),
    )
    monkeypatch.setattr(
        session_module,
        "CommandDispatcher",
        lambda _session: SimpleNamespace(commands={}),
    )
    monkeypatch.setattr(session_module, "MessageDispatcher", Mock())
    monkeypatch.setattr(session_module, "BashDispatcher", Mock())

    session = Session(_context())

    assert session.prompt.hotkeys == {}
    assert session.prompt.handle_external_sigint() is False
    with pytest.raises(RuntimeError, match="Interactive prompt is unavailable"):
        await session.prompt.get_input()


@pytest.mark.asyncio
async def test_start_initializes_graph_and_welcome_when_first_interactive_start_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _bare_session()
    graph = object()
    status = _Status()
    recorder = _Recorder()
    session.run_recorder = recorder
    session._main_loop = AsyncMock()  # type: ignore[method-assign]
    session._check_updates_background = AsyncMock()  # type: ignore[method-assign]
    session._register_sigint_handler = Mock()  # type: ignore[method-assign]
    session._restore_sigint = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr(session_module.initializer, "get_graph", lambda **_kwargs: _GraphContext(graph))
    monkeypatch.setattr(session_module.console.console, "status", lambda _message: status)
    monkeypatch.setattr(session_module.console, "print", Mock())

    await session.start(show_welcome=True)

    assert session.graph is graph
    session.renderer.show_welcome.assert_called_once_with(session.context)
    session._check_updates_background.assert_awaited_once()
    session._main_loop.assert_awaited_once()
    assert recorder.start_calls == [(session.context, True)]
    assert recorder.finish_codes == [0]
    assert status.events[0] == "stop"
    assert status.events[-2:] == [
        "start",
        (
            "update",
            f"[{session_module.theme.spinner_color}]Cleaning...[/{session_module.theme.spinner_color}]",
        ),
    ]
    session._restore_sigint.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bash_mode", "content", "is_slash", "expected_dispatcher"),
    [
        (True, "pwd", False, "bash_dispatcher"),
        (False, "/help", True, "command_dispatcher"),
        (False, "hello", False, "message_dispatcher"),
    ],
)
async def test_main_loop_routes_input_when_mode_or_command_type_changes(
    bash_mode: bool,
    content: str,
    is_slash: bool,
    expected_dispatcher: str,
) -> None:
    session = _bare_session(bash_mode=bash_mode)
    session.prompt.get_input = AsyncMock(side_effect=[(content, is_slash), EOFError])

    await session._main_loop()

    expected = getattr(session, expected_dispatcher)
    expected.dispatch.assert_awaited_once_with(content)
    for name in {"bash_dispatcher", "command_dispatcher", "message_dispatcher"} - {expected_dispatcher}:
        getattr(session, name).dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_loop_continues_after_reporting_when_input_processing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _bare_session()
    session.prompt.get_input = AsyncMock(side_effect=[RuntimeError("bad input"), EOFError])
    print_error = Mock()
    monkeypatch.setattr(session_module.console, "print_error", print_error)
    monkeypatch.setattr(session_module.console, "print", Mock())

    await session._main_loop()

    print_error.assert_called_once_with("Error processing input: bad input")
    assert session.prompt.get_input.await_count == 2


@pytest.mark.asyncio
async def test_send_returns_success_and_records_interrupt_when_dispatch_is_cancelled_by_keyboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _bare_session()
    recorder = _Recorder()
    session.run_recorder = recorder
    session.message_dispatcher.dispatch = AsyncMock(side_effect=KeyboardInterrupt)
    session._register_sigint_handler = Mock()
    session._restore_sigint = Mock()
    monkeypatch.setattr(session_module.initializer, "get_graph", lambda **_kwargs: _GraphContext())

    result = await session.send("stop")

    assert result == 0
    assert recorder.finish_codes == [130]
    session._restore_sigint.assert_called_once()


def test_update_context_stops_session_and_rebinds_audit_when_agent_changes() -> None:
    session = _bare_session()
    session.running = True

    session.update_context(agent="Accuracy", thread_id="thread-2", unknown="ignored")

    assert session.context.agent == "Accuracy"
    assert session.context.thread_id == "thread-2"
    assert not hasattr(session.context, "unknown")
    assert session.needs_reload is True
    assert session.running is False
    session.audit_writer.rebind.assert_called_once_with(thread_id="thread-2", agent_name="Accuracy")


def test_remember_tool_output_replaces_existing_entry_when_tool_call_id_matches() -> None:
    session = _bare_session()
    original = ToolOutputEntry("call-1", "execute", "old", "old full")
    replacement = ToolOutputEntry("call-1", "execute", "new", "new full")
    second = ToolOutputEntry("call-2", "read_file", "preview", "full")

    session.remember_tool_output(original)
    session.remember_tool_output(second)
    session.remember_tool_output(replacement)

    assert [entry.tool_call_id for entry in session.tool_outputs] == [
        "call-1",
        "call-2",
    ]
    assert replacement.sequence == 1
    assert second.sequence == 2
    assert session.tool_outputs[0] is replacement
    assert session.latest_tool_output is replacement

    session.clear_tool_output()
    assert session.tool_outputs == []
    assert session.latest_tool_output is None


def test_mode_callbacks_update_context_and_prompt_when_keyboard_toggle_runs() -> None:
    session = _bare_session()

    session._handle_approval_mode_change()
    session._handle_bash_mode_toggle()

    assert session.context.approval_mode == ApprovalMode.AGGRESSIVE
    assert session.context.bash_mode is True
    assert session.prompt.refresh_style.call_count == 2


@pytest.mark.asyncio
async def test_sigint_handler_cancels_active_stream_when_stream_task_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _bare_session()
    installed: list[object] = []
    previous = Mock()
    monkeypatch.setattr(session_module.signal, "getsignal", lambda _signal: previous)
    monkeypatch.setattr(
        session_module.signal,
        "signal",
        lambda _signal, handler: installed.append(handler),
    )
    task = asyncio.create_task(asyncio.sleep(60))
    session.current_stream_task = task

    session._register_sigint_handler()
    handler = installed[-1]
    assert callable(handler)
    handler(signal.SIGINT, None)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    previous.assert_not_called()
    session.prompt.handle_external_sigint.assert_not_called()


def test_restore_sigint_restores_previous_handler_when_session_owns_current_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _bare_session()
    current = Mock()
    previous = Mock()
    restored: list[object] = []
    session._sigint_registered = True
    session._sigint_handler = current
    session._previous_sigint = previous
    monkeypatch.setattr(session_module.signal, "getsignal", lambda _signal: current)
    monkeypatch.setattr(
        session_module.signal,
        "signal",
        lambda _signal, handler: restored.append(handler),
    )

    session._restore_sigint()

    assert restored == [previous]
    assert session._sigint_registered is False
    assert session._previous_sigint is None
    assert session._sigint_handler is None


@pytest.mark.asyncio
async def test_check_updates_background_prints_upgrade_when_new_version_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _bare_session()
    warning = Mock()
    monkeypatch.setattr(session_module, "check_for_updates", lambda: ("27.0.0", "uv tool upgrade"))
    monkeypatch.setattr(session_module.console, "print_warning", warning)
    monkeypatch.setattr(session_module.console, "print", Mock())

    await session._check_updates_background()

    warning.assert_called_once()
    assert "27.0.0" in warning.call_args.args[0]
    assert "uv tool upgrade" in warning.call_args.args[0]

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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest

import msagent.cli.bootstrap.chat as chat_module


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "timer": False,
        "agent": "Profiler",
        "model": "default",
        "working_dir": str(tmp_path),
        "approval_mode": "active",
        "execute_approval_mode": None,
        "stream": True,
        "trace_jsonl": None,
        "message": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_handle_chat_command_sends_once_when_message_is_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = object()
    create = AsyncMock(return_value=context)
    session = SimpleNamespace(send=AsyncMock(return_value=7))
    session_factory = Mock(return_value=session)
    enable_timer = Mock()
    monkeypatch.setattr(chat_module.Context, "create", create)
    monkeypatch.setattr(chat_module, "Session", session_factory)
    monkeypatch.setattr(chat_module, "enable_timer", enable_timer)
    trace_path = tmp_path / "trace.jsonl"

    result = await chat_module.handle_chat_command(
        _args(
            tmp_path,
            timer=True,
            message="analyze",
            stream=False,
            trace_jsonl=str(trace_path),
        )
    )

    assert result == 7
    enable_timer.assert_called_once()
    create.assert_awaited_once_with(
        agent="Profiler",
        model="default",
        working_dir=tmp_path,
        approval_mode="active",
        execute_approval_mode=None,
        stream_output=False,
        trace_jsonl=trace_path,
    )
    session_factory.assert_called_once_with(context)
    session.send.assert_awaited_once_with("analyze")


@pytest.mark.asyncio
async def test_handle_chat_command_recreates_session_when_active_agent_needs_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = object()
    first = SimpleNamespace(start=AsyncMock(), needs_reload=True)
    second = SimpleNamespace(start=AsyncMock(), needs_reload=False)
    session_factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(chat_module.Context, "create", AsyncMock(return_value=context))
    monkeypatch.setattr(chat_module, "Session", session_factory)

    result = await chat_module.handle_chat_command(_args(tmp_path))

    assert result == 0
    assert session_factory.call_args_list == [((context,),), ((context,),)]
    first.start.assert_awaited_once_with(show_welcome=True)
    second.start.assert_awaited_once_with(show_welcome=False)


@pytest.mark.asyncio
async def test_handle_chat_command_returns_success_when_user_interrupts_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_module.Context, "create", AsyncMock(side_effect=KeyboardInterrupt))
    print_error = Mock()
    monkeypatch.setattr(chat_module.console, "print_error", print_error)

    assert await chat_module.handle_chat_command(_args(tmp_path)) == 0
    print_error.assert_not_called()


@pytest.mark.asyncio
async def test_handle_chat_command_reports_failure_when_context_creation_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_module.Context,
        "create",
        AsyncMock(side_effect=RuntimeError("invalid config")),
    )
    print_error = Mock()
    monkeypatch.setattr(chat_module.console, "print_error", print_error)
    monkeypatch.setattr(chat_module.console, "print", Mock())

    assert await chat_module.handle_chat_command(_args(tmp_path)) == 1
    print_error.assert_called_once_with("Error starting chat session: invalid config")

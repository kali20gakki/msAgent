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
import signal
from typing import Any

import pytest

from msagent.cli.core.context import Context
from msagent.cli.core.session import Session
from msagent.cli.ui.prompt import InteractivePrompt
from msagent.configs import ApprovalMode


def _build_context() -> Context:
    return Context(
        agent="msagent",
        model="default",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.ACTIVE,
        recursion_limit=80,
    )


def _patch_prompt_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_setup_session(self) -> None:
        self.prompt_session = None

    monkeypatch.setattr(InteractivePrompt, "_setup_session", fake_setup_session)


class _FakeGraphContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeRunRecorder:
    def __init__(self) -> None:
        self.finish_codes: list[int] = []
        self.errors: list[BaseException] = []
        self.started = False

    def start(self, **_kwargs: Any) -> None:
        self.started = True

    def finish(self, *, exit_code: int, **_kwargs: Any) -> None:
        self.finish_codes.append(exit_code)

    def record_error(self, error: BaseException) -> None:
        self.errors.append(error)


@pytest.mark.asyncio
async def test_check_updates_background_keeps_sigint_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_setup(monkeypatch)
    session = Session(_build_context())
    session._sigint_registered = True

    monkeypatch.setattr(
        "msagent.cli.core.session.check_for_updates",
        lambda: None,
    )

    await session._check_updates_background()

    assert session._sigint_registered is True


@pytest.mark.asyncio
async def test_send_finishes_recorder_once_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_prompt_setup(monkeypatch)
    session = Session(_build_context())
    recorder = _FakeRunRecorder()
    restore_calls: list[None] = []

    async def fake_dispatch(_message: str) -> None:
        return None

    monkeypatch.setattr("msagent.cli.core.session.initializer.get_graph", lambda **_kwargs: _FakeGraphContext())
    session.run_recorder = recorder
    session.message_dispatcher.dispatch = fake_dispatch
    session._register_sigint_handler = lambda: None
    session._restore_sigint = lambda: restore_calls.append(None)

    exit_code = await session.send("hello")

    assert exit_code == 0
    assert recorder.started is True
    assert recorder.finish_codes == [0]
    assert restore_calls == [None]


@pytest.mark.asyncio
async def test_send_finishes_recorder_once_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_prompt_setup(monkeypatch)
    session = Session(_build_context())
    recorder = _FakeRunRecorder()
    restore_calls: list[None] = []
    error = RuntimeError("boom")

    async def fake_dispatch(_message: str) -> None:
        raise error

    monkeypatch.setattr("msagent.cli.core.session.initializer.get_graph", lambda **_kwargs: _FakeGraphContext())
    monkeypatch.setattr("msagent.cli.core.session.console.print_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("msagent.cli.core.session.console.print", lambda *_args, **_kwargs: None)
    session.run_recorder = recorder
    session.message_dispatcher.dispatch = fake_dispatch
    session._register_sigint_handler = lambda: None
    session._restore_sigint = lambda: restore_calls.append(None)

    exit_code = await session.send("hello")

    assert exit_code == 1
    assert recorder.finish_codes == [1]
    assert recorder.errors == [error]
    assert restore_calls == [None]


def test_sigint_handler_delegates_to_prompt_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_setup(monkeypatch)
    session = Session(_build_context())
    session._previous_sigint = signal.default_int_handler
    session.current_stream_task = None
    calls: list[str] = []

    session.prompt.handle_external_sigint = lambda: calls.append("prompt") or True

    session._sigint_registered = True
    try:
        handler = None
        session._register_sigint_handler()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
    finally:
        if handler is not None:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        session._sigint_registered = False

    assert calls == ["prompt"]

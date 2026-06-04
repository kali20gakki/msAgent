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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.handlers import bash as bash_module
from msagent.cli.handlers.bash import BashDispatcher


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
    )


@pytest.mark.asyncio
async def test_bash_dispatcher_skips_empty_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = BashDispatcher(_build_session(tmp_path))
    executed: list[tuple[list[str], str]] = []

    async def fake_execute(cmd: list[str], *, cwd: str = "") -> tuple[int, str, str]:
        executed.append((cmd, cwd))
        return (0, "", "")

    monkeypatch.setattr(bash_module, "execute_bash_command", fake_execute)

    await handler.dispatch("   ")

    assert executed == []


@pytest.mark.asyncio
async def test_bash_dispatcher_executes_command_with_working_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = BashDispatcher(_build_session(tmp_path))
    executed: list[tuple[list[str], str]] = []

    async def fake_execute(cmd: list[str], *, cwd: str = "") -> tuple[int, str, str]:
        executed.append((cmd, cwd))
        return (0, "hello world", "")

    monkeypatch.setattr(bash_module, "execute_bash_command", fake_execute)

    printed: list[str] = []
    monkeypatch.setattr(
        bash_module.console.console, "print", lambda *args, **kwargs: printed.append(str(args[0])) if args else None
    )
    monkeypatch.setattr(bash_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(bash_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.dispatch("echo hello")

    assert len(executed) == 1
    assert executed[0][0] == ["bash", "-c", "echo hello"]
    assert executed[0][1] == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_dispatcher_prints_stderr_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = BashDispatcher(_build_session(tmp_path))

    async def fake_execute(cmd: list[str], *, cwd: str = "") -> tuple[int, str, str]:
        return (1, "out", "error message")

    errors: list[str] = []
    monkeypatch.setattr(bash_module, "execute_bash_command", fake_execute)
    monkeypatch.setattr(bash_module.console, "print_error", errors.append)
    monkeypatch.setattr(bash_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.dispatch("ls /nonexistent")

    assert "error message" in errors
    assert "Command exited with code 1" in errors


@pytest.mark.asyncio
async def test_bash_dispatcher_prints_stdout_only_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = BashDispatcher(_build_session(tmp_path))

    async def fake_execute(cmd: list[str], *, cwd: str = "") -> tuple[int, str, str]:
        return (0, "output line 1\noutput line 2", "")

    stdout_printed: list[str] = []
    monkeypatch.setattr(bash_module, "execute_bash_command", fake_execute)
    monkeypatch.setattr(
        bash_module.console.console,
        "print",
        lambda *args, **kwargs: stdout_printed.append(str(args[0])) if args else None,
    )
    monkeypatch.setattr(bash_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(bash_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.dispatch("pwd")

    assert len(stdout_printed) >= 1


@pytest.mark.asyncio
async def test_bash_dispatcher_handles_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = BashDispatcher(_build_session(tmp_path))

    async def fake_execute(cmd: list[str], *, cwd: str = "") -> tuple[int, str, str]:
        raise ValueError("bad syntax")

    errors: list[str] = []
    monkeypatch.setattr(bash_module, "execute_bash_command", fake_execute)
    monkeypatch.setattr(bash_module.console, "print_error", errors.append)
    monkeypatch.setattr(bash_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.dispatch("bad command")

    assert any("Invalid command syntax" in e for e in errors)


@pytest.mark.asyncio
async def test_bash_dispatcher_handles_generic_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = BashDispatcher(_build_session(tmp_path))

    async def fake_execute(cmd: list[str], *, cwd: str = "") -> tuple[int, str, str]:
        raise RuntimeError("unexpected failure")

    errors: list[str] = []
    monkeypatch.setattr(bash_module, "execute_bash_command", fake_execute)
    monkeypatch.setattr(bash_module.console, "print_error", errors.append)
    monkeypatch.setattr(bash_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.dispatch("crash")

    assert any("Error executing command" in e for e in errors)

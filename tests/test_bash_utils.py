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

import asyncio
import sys

import pytest

from msagent.utils import bash as bash_module


class _DummyStream:
    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self._chunks = list(chunks or [])

    async def read(self, _size: int = -1) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _DummyProcess:
    def __init__(
        self,
        *,
        wait_delay: float = 0.0,
        returncode: int | None = 0,
        stdout_chunks: list[bytes] | None = None,
        stderr_chunks: list[bytes] | None = None,
    ) -> None:
        self.wait_delay = wait_delay
        self.returncode = returncode
        self.pid = 12345
        self.stdout = _DummyStream(stdout_chunks)
        self.stderr = _DummyStream(stderr_chunks)
        self.kill_called = False

    async def wait(self) -> int:
        if self.wait_delay:
            await asyncio.sleep(self.wait_delay)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_bash_helper_functions_cover_stream_cancel_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = bytearray()
    await bash_module._pump_stream(_DummyStream([b"a", b"b"]), buffer)
    assert buffer == b"ab"
    await bash_module._pump_stream(None, bytearray())

    task = asyncio.create_task(asyncio.sleep(10))
    await bash_module._cancel_task(task)
    assert task.cancelled() is True
    await bash_module._cancel_task(None)

    await bash_module._finish_stream_tasks([None])

    cancelled: list[asyncio.Task[None]] = []

    async def fake_cancel_task(task):
        cancelled.append(task)

    async def fake_wait_for(_awaitable, timeout):
        raise RuntimeError("flush failed")

    monkeypatch.setattr(bash_module, "_cancel_task", fake_cancel_task)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    pending = asyncio.create_task(asyncio.sleep(0))
    await bash_module._finish_stream_tasks([pending])
    assert cancelled == [pending]


@pytest.mark.asyncio
async def test_terminate_process_tree_covers_windows_and_posix_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _DummyProcess(returncode=None)

    original_platform = bash_module.sys.platform
    monkeypatch.setattr(bash_module.sys, "platform", "win32")

    class _FakeKiller:
        async def wait(self) -> int:
            return 0

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeKiller()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    await bash_module._terminate_process_tree(process)

    process = _DummyProcess(returncode=None)

    async def raising_create_subprocess_exec(*args, **kwargs):
        raise RuntimeError("taskkill unavailable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", raising_create_subprocess_exec)
    await bash_module._terminate_process_tree(process)
    assert process.kill_called is True

    process = _DummyProcess(returncode=None)
    monkeypatch.setattr(bash_module.sys, "platform", "linux")
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(bash_module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(bash_module.os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    await bash_module._terminate_process_tree(process)
    assert calls == [(process.pid, 9)]

    process = _DummyProcess(returncode=None)

    def raise_permission(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(bash_module.os, "killpg", raise_permission, raising=False)
    await bash_module._terminate_process_tree(process)
    assert process.kill_called is True

    monkeypatch.setattr(bash_module.sys, "platform", original_platform)


@pytest.mark.asyncio
async def test_execute_bash_command_returns_error_for_spawn_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    status, stdout, stderr = await bash_module.execute_bash_command(["bash", "-c", "echo test"])

    assert status == -1
    assert stdout == ""
    assert stderr == "spawn failed"


@pytest.mark.asyncio
async def test_execute_bash_command_collects_incremental_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _DummyProcess(
        stdout_chunks=[b"hello ", b"world"],
        stderr_chunks=[b"warn"],
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    status, stdout, stderr = await bash_module.execute_bash_command(
        ["bash", "-c", "echo test"],
        timeout=1,
    )

    assert status == 0
    assert stdout == "hello world"
    assert stderr == "warn"


@pytest.mark.asyncio
async def test_execute_bash_command_uses_isolated_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _DummyProcess()
    captured_kwargs: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await bash_module.execute_bash_command(["bash", "-c", "true"], timeout=1)

    if sys.platform == "win32":
        assert "creationflags" in captured_kwargs
    else:
        assert captured_kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_execute_bash_command_timeout_terminates_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _DummyProcess(wait_delay=10.0, returncode=None)
    terminated: list[int] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    async def fake_terminate_process_tree(proc):
        terminated.append(proc.pid)
        proc.returncode = -9

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        bash_module,
        "_terminate_process_tree",
        fake_terminate_process_tree,
    )

    status, stdout, stderr = await bash_module.execute_bash_command(
        ["bash", "-c", "sleep 60 | tail -50"],
        timeout=0.01,
    )

    assert status == -1
    assert stdout == ""
    assert stderr == "Command timed out"
    assert terminated == [process.pid]


@pytest.mark.asyncio
async def test_execute_bash_command_timeout_preserves_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _DummyProcess(
        wait_delay=10.0,
        returncode=None,
        stdout_chunks=[b"compile line 1\n", b"compile line 2\n"],
        stderr_chunks=[b"warning: x\n"],
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    async def fake_terminate_process_tree(proc):
        proc.returncode = -9

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        bash_module,
        "_terminate_process_tree",
        fake_terminate_process_tree,
    )

    status, stdout, stderr = await bash_module.execute_bash_command(
        ["bash", "-c", "make -j4"],
        timeout=0.01,
    )

    assert status == -1
    assert "compile line 1" in stdout
    assert "compile line 2" in stdout
    assert stderr.startswith("Command timed out")
    assert "warning: x" in stderr

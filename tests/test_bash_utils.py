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

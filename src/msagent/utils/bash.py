import asyncio
import contextlib
import os
import signal
import subprocess
import sys


def _subprocess_spawn_kwargs() -> dict[str, object]:
    """Create platform-specific subprocess kwargs for isolating child processes."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _pump_stream(
    stream: asyncio.StreamReader | None,
    buffer: bytearray,
) -> None:
    """Read a subprocess stream incrementally so partial output is preserved."""
    if stream is None:
        return

    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        buffer.extend(chunk)


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel a background task without leaking cancellation warnings."""
    if task is None:
        return

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _finish_stream_tasks(tasks: list[asyncio.Task[None] | None]) -> None:
    """Wait briefly for stream pump tasks to flush buffered output after exit/kill."""
    active_tasks = [task for task in tasks if task is not None]
    if not active_tasks:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*active_tasks), timeout=1)
    except Exception:
        for task in active_tasks:
            await _cancel_task(task)


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    """Terminate the subprocess and its descendants after a timeout."""
    if process.returncode is not None:
        return

    if sys.platform == "win32":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
        except Exception:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()

    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), timeout=5)


async def execute_bash_command(
    command: list[str], cwd: str | None = None, timeout: int | None = None
) -> tuple[int, str, str]:
    process = None
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            **_subprocess_spawn_kwargs(),
        )

        stdout_task = asyncio.create_task(_pump_stream(process.stdout, stdout_buffer))
        stderr_task = asyncio.create_task(_pump_stream(process.stderr, stderr_buffer))

        await asyncio.wait_for(process.wait(), timeout=timeout)
        await _finish_stream_tasks([stdout_task, stderr_task])

        return (
            process.returncode or 0,
            stdout_buffer.decode("utf-8", errors="replace"),
            stderr_buffer.decode("utf-8", errors="replace"),
        )

    except TimeoutError:
        if process is not None:
            await _terminate_process_tree(process)
        await _finish_stream_tasks([stdout_task, stderr_task])

        stdout = stdout_buffer.decode("utf-8", errors="replace")
        stderr = stderr_buffer.decode("utf-8", errors="replace").strip()
        timeout_stderr = "Command timed out"
        if stderr:
            timeout_stderr = f"{timeout_stderr}\n\n{stderr}"
        return -1, stdout, timeout_stderr
    except Exception as e:
        if process is not None and process.returncode is None:
            await _terminate_process_tree(process)
        await _finish_stream_tasks([stdout_task, stderr_task])
        return -1, "", str(e)

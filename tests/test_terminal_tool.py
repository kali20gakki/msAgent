from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException

from msagent.tools.impl import terminal as terminal_module


def _runtime(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
        tool_call_id="call-run-command",
    )


@pytest.mark.asyncio
async def test_run_command_uses_default_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_execute(command: list[str], cwd: str | None = None, timeout: int | None = None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return 0, "ok", ""

    monkeypatch.setattr(terminal_module, "execute_bash_command", fake_execute)

    result = await terminal_module.run_command.coroutine(
        command="echo ok",
        runtime=_runtime(tmp_path),
    )

    assert result == "ok"
    assert captured["command"] == ["bash", "-c", "echo ok"]
    assert captured["cwd"] == str(tmp_path)
    assert captured["timeout"] == terminal_module.DEFAULT_COMMAND_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_run_command_allows_custom_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_execute(command: list[str], cwd: str | None = None, timeout: int | None = None):
        captured["timeout"] = timeout
        return 0, "", ""

    monkeypatch.setattr(terminal_module, "execute_bash_command", fake_execute)

    result = await terminal_module.run_command.coroutine(
        command="sleep 1",
        timeout_seconds=7200,
        runtime=_runtime(tmp_path),
    )

    assert result == "Command completed successfully"
    assert captured["timeout"] == 7200


@pytest.mark.asyncio
async def test_run_command_reports_timeout_with_requested_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(command: list[str], cwd: str | None = None, timeout: int | None = None):
        return -1, "", "Command timed out"

    monkeypatch.setattr(terminal_module, "execute_bash_command", fake_execute)

    with pytest.raises(ToolException, match="Command timed out after 42s"):
        await terminal_module.run_command.coroutine(
            command="long-job",
            timeout_seconds=42,
            runtime=_runtime(tmp_path),
        )


@pytest.mark.asyncio
async def test_run_command_includes_partial_output_when_timeout_occurs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(command: list[str], cwd: str | None = None, timeout: int | None = None):
        return -1, "build step ok", "Command timed out\n\nwarning: unused symbol"

    monkeypatch.setattr(terminal_module, "execute_bash_command", fake_execute)

    with pytest.raises(ToolException) as exc_info:
        await terminal_module.run_command.coroutine(
            command="make -j4",
            timeout_seconds=42,
            runtime=_runtime(tmp_path),
        )

    message = str(exc_info.value)
    assert "Command timed out after 42s" in message
    assert "Partial output before timeout" in message
    assert "[stdout]" in message
    assert "build step ok" in message
    assert "[stderr]" in message
    assert "warning: unused symbol" in message


def test_tail_output_lines_keeps_recent_lines_only() -> None:
    output = "\n".join(f"line {i}" for i in range(1, 61))

    tailed, omitted = terminal_module._tail_output_lines(output, max_lines=5)

    assert tailed == "\n".join(f"line {i}" for i in range(56, 61))
    assert omitted == 55


def test_build_timeout_error_message_truncates_to_last_lines() -> None:
    stdout = "\n".join(f"stdout {i}" for i in range(1, 56))
    stderr = "Command timed out\n\n" + "\n".join(f"stderr {i}" for i in range(1, 4))

    message = terminal_module._build_timeout_error_message(42, stdout, stderr)
    message_lines = message.splitlines()

    assert "Command timed out after 42s" in message
    assert "[stdout] last 50 lines (omitted 5 earlier lines)" in message
    assert "stdout 1" not in message_lines
    assert "stdout 6" in message_lines
    assert "stdout 55" in message_lines
    assert "[stderr]" in message
    assert "stderr 1" in message_lines


def test_run_command_schema_exposes_timeout_guidance() -> None:
    timeout_schema = terminal_module.run_command.args["timeout_seconds"]

    assert timeout_schema["default"] == terminal_module.DEFAULT_COMMAND_TIMEOUT_SECONDS
    assert "long-running tasks" in timeout_schema["description"]
    assert timeout_schema["minimum"] == 1


def test_render_command_args_shows_custom_timeout_only() -> None:
    rendered_default = terminal_module._render_command_args({"command": "ls"}, {})
    rendered_custom = terminal_module._render_command_args(
        {"command": "pytest", "timeout_seconds": 3600},
        {},
    )

    assert "timeout_seconds" not in rendered_default
    assert "timeout_seconds=3600" in rendered_custom

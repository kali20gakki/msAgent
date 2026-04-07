from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _msagent_command() -> list[str]:
    script_dir = Path(sys.executable).parent
    candidates = [
        script_dir / "msagent",
        script_dir / "msagent.exe",
        script_dir / "msagent.cmd",
        script_dir / "msagent.bat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return [sys.executable, str(PROJECT_ROOT / "run.py")]


def _run_msagent(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged_env.update(env)

    return subprocess.run(
        [*_msagent_command(), *args],
        cwd=str(cwd),
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )


def _normalize_terminal_output(content: str) -> str:
    return ANSI_ESCAPE_RE.sub("", content).replace("\r", "")


def test_entrypoint_version_and_config_show(tmp_path: Path) -> None:
    version = _run_msagent("--version", cwd=PROJECT_ROOT)
    assert version.returncode == 0
    assert "msAgent" in version.stdout

    config = _run_msagent("config", "--show", "-w", str(tmp_path), cwd=PROJECT_ROOT)
    assert config.returncode == 0
    assert "Current Configuration" in config.stdout
    assert "LLM Provider" in config.stdout


def test_entrypoint_one_shot_tool_call_and_todo_render(tmp_path: Path) -> None:
    env = {"MSAGENT_FAKE_BACKEND": "1"}

    tool = _run_msagent(
        "-w",
        str(tmp_path),
        "please run one tool call",
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert tool.returncode == 0
    tool_stdout = _normalize_terminal_output(tool.stdout)
    assert "Use tool run_command" in tool_stdout
    assert "fake-tool-output" in tool_stdout

    todo = _run_msagent(
        "-w",
        str(tmp_path),
        "please update todo list",
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert todo.returncode == 0
    todo_stdout = _normalize_terminal_output(todo.stdout)
    assert "TODOs" in todo_stdout
    assert "Review profile bottleneck" in todo_stdout

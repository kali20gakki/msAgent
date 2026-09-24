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

"""Run the installed msagent CLI and collect validation evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CA_BUNDLE_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def remove_invalid_ca_bundle_env(env: dict[str, str]) -> None:
    """移除失效的 CA 文件变量，避免 Conda 将已删除环境中的路径传给 httpx。"""
    for variable in _CA_BUNDLE_ENV_VARS:
        configured_path = env.get(variable, "").strip()
        if variable in env and (not configured_path or not Path(configured_path).expanduser().is_file()):
            env.pop(variable, None)


@dataclass(frozen=True)
class RunResult:
    """Artifacts produced by one non-streaming msagent invocation."""

    returncode: int
    stdout: str
    stderr: str
    traces: list[dict]
    app_log: str
    command: tuple[str, ...]
    duration_seconds: float
    trace_path: Path
    app_log_path: Path
    stdout_path: Path
    stderr_path: Path


def _read_traces(trace_path: Path) -> tuple[list[dict], list[str]]:
    """Read JSONL events while preserving useful diagnostics for bad lines."""
    traces: list[dict] = []
    diagnostics: list[str] = []

    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [f"trace file was not created: {trace_path}"]
    except OSError as exc:
        return [], [f"failed to read trace file {trace_path}: {exc}"]

    if not lines:
        return [], [f"trace file is empty: {trace_path}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"invalid JSON in trace file {trace_path} at line {line_number}: {exc}")
            continue

        if not isinstance(event, dict):
            diagnostics.append(f"trace event at line {line_number} is {type(event).__name__}, expected object")
            continue
        traces.append(event)

    if not traces and not diagnostics:
        diagnostics.append(f"trace file contains no events: {trace_path}")
    return traces, diagnostics


def _read_latest_app_log(log_path: Path, previous_size: int) -> tuple[str, list[str]]:
    """Read only content appended by this run when an older log already exists."""
    try:
        content = log_path.read_bytes()
    except FileNotFoundError:
        return "", [f"app log was not created: {log_path}"]
    except OSError as exc:
        return "", [f"failed to read app log {log_path}: {exc}"]

    # A shared MSAGENT_HOME may already have historical logs. Slice at the
    # previous byte offset so assertions inspect this invocation only. If the
    # file was rotated or truncated during execution, read the new file fully.
    if previous_size <= len(content):
        content = content[previous_size:]
    return content.decode("utf-8", errors="replace"), []


def _append_diagnostics(stderr: str, diagnostics: list[str]) -> str:
    """Attach collection failures without discarding subprocess stderr."""
    if not diagnostics:
        return stderr

    suffix = "\n".join(f"[agent_runner] {message}" for message in diagnostics)
    if not stderr:
        return suffix
    return f"{stderr.rstrip()}\n{suffix}"


def run_msagent(
    prompt: str,
    workspace_dir: str,
    extra_env: dict | None = None,
    agent_name: str | None = None,
    *,
    executable: str = "msagent",
    artifact_dir: str | Path | None = None,
    timeout_seconds: float | None = None,
    execute_approval_mode: str | None = None,
) -> RunResult:
    """Run msagent and collect stdout, stderr, JSONL traces, and the latest log.

    The command is executed without a shell. Passing each argument separately
    avoids quoting bugs and ensures that ``prompt`` is treated as one positional
    argument even when it contains spaces or shell metacharacters.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    workspace = Path(workspace_dir).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace directory does not exist: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace path is not a directory: {workspace}")

    if extra_env is not None and not isinstance(extra_env, dict):
        raise TypeError("extra_env must be a dict or None")

    if agent_name is not None:
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string or None")
        agent_name = agent_name.strip()

    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("executable must be a non-empty string")
    executable = executable.strip()
    env = os.environ.copy()
    if extra_env:
        # subprocess requires string keys and values. Reject invalid data here
        # so configuration errors are reported before msagent is started.
        invalid_items = [
            key for key, value in extra_env.items() if not isinstance(key, str) or not isinstance(value, str)
        ]
        if invalid_items:
            raise TypeError("extra_env keys and values must all be strings")
        env.update(extra_env)

    # DEBUG is mandatory for this validation runner and intentionally overrides
    # a conflicting value supplied by the parent process or extra_env.
    env["MSAGENT_LOG_LEVEL"] = "DEBUG"
    # 同时控制子进程写出编码；subprocess 的 encoding 只控制父进程解码。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    remove_invalid_ca_bundle_env(env)

    resolved_executable = shutil.which(executable, path=env.get("PATH"))
    if resolved_executable is None and sys.platform == "win32" and not executable.lower().endswith(".exe"):
        resolved_executable = shutil.which(f"{executable}.exe", path=env.get("PATH"))
    if resolved_executable is None:
        raise RuntimeError("msagent executable was not found; install the whl and activate its environment")

    configured_home = env.get("MSAGENT_HOME", "").strip()
    msagent_home = (
        Path(configured_home).expanduser().resolve() if configured_home else (Path.home() / ".msagent").resolve()
    )
    app_log_path = msagent_home / "logs" / "app.log"
    try:
        previous_log_size = app_log_path.stat().st_size
    except OSError:
        previous_log_size = 0

    # The caller may provide a durable per-invocation directory. Direct callers
    # still get a unique directory below MSAGENT_HOME.
    if artifact_dir is None:
        trace_dir = msagent_home / "validation-artifacts" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(tempfile.mkdtemp(prefix="msagent-run-", dir=trace_dir))
    else:
        artifact_path = Path(artifact_dir).expanduser().resolve()
    try:
        artifact_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not artifact_path.is_dir() or any(artifact_path.iterdir()):
            raise RuntimeError(f"artifact directory already exists and is not empty: {artifact_path}")
    except OSError as exc:
        raise RuntimeError(f"failed to create artifact directory {artifact_path}: {exc}") from exc

    trace_path = artifact_path / "trace.jsonl"
    app_log_artifact_path = artifact_path / "app.log"
    stdout_path = artifact_path / "stdout.txt"
    stderr_path = artifact_path / "stderr.txt"

    command = [
        resolved_executable,
        "-v",
        "--no-stream",
        "--trace-jsonl",
        str(trace_path),
        "-w",
        str(workspace),
    ]
    if agent_name is not None:
        command.extend(["--agent", agent_name])
    if execute_approval_mode is not None:
        command.extend(["--execute-approval-mode", execute_approval_mode])
    command.append(prompt)

    started_at = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("msagent executable was not found; install the whl and activate its environment") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"msagent command timed out after {timeout_seconds} seconds; "
            "check whether the process is waiting for interactive approval input"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to start msagent: {exc}") from exc
    duration_seconds = time.monotonic() - started_at

    traces, trace_diagnostics = _read_traces(trace_path)
    app_log, log_diagnostics = _read_latest_app_log(app_log_path, previous_log_size)
    stderr = _append_diagnostics(
        completed.stderr,
        [*trace_diagnostics, *log_diagnostics],
    )

    try:
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        app_log_artifact_path.write_text(app_log, encoding="utf-8")
        (artifact_path / "command.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "duration_seconds": duration_seconds,
                    "workspace": str(workspace),
                    "msagent_home": str(msagent_home),
                    "agent": agent_name,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        stderr = _append_diagnostics(
            stderr,
            [f"failed to persist invocation artifacts in {artifact_path}: {exc}"],
        )

    return RunResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=stderr,
        traces=traces,
        app_log=app_log,
        command=tuple(command),
        duration_seconds=duration_seconds,
        trace_path=trace_path,
        app_log_path=app_log_artifact_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

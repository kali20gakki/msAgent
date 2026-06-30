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

import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _msagent_command() -> list[str]:
    source_command = [sys.executable, str(PROJECT_ROOT / "run.py")]
    script_dir = Path(sys.executable).parent
    candidates = [
        script_dir / "msagent",
        script_dir / "msagent.exe",
        script_dir / "msagent.cmd",
        script_dir / "msagent.bat",
    ]
    if os.environ.get("MSAGENT_E2E_USE_INSTALLED", "").strip().lower() not in {"1", "true", "yes"}:
        return source_command
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return source_command


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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def test_entrypoint_trace_jsonl_records_tools_tokens_and_time(tmp_path: Path) -> None:
    trace_path = tmp_path / "msagent.events.jsonl"
    result = _run_msagent(
        "--no-stream",
        "--trace-jsonl",
        str(trace_path),
        "-w",
        str(tmp_path),
        "please run one tool call",
        cwd=PROJECT_ROOT,
        env={"MSAGENT_FAKE_BACKEND": "1"},
    )

    assert result.returncode == 0
    events = _read_jsonl(trace_path)
    assert [event["type"] for event in events][:2] == ["session_started", "token_usage"]

    tool_call = next(event for event in events if event["type"] == "tool_call")
    assert tool_call["tool"] == "run_command"
    assert tool_call["input"] == {"command": "echo fake-tool-output"}

    tool_result = next(event for event in events if event["type"] == "tool_result")
    assert tool_result["tool"] == "run_command"
    assert tool_result["item_id"] == "call-tool-1"
    assert tool_result["duration_ms"] == 500

    finished = events[-1]
    assert finished["type"] == "session_finished"
    assert finished["duration_ms"] >= 0
    assert finished["token_usage"] == {
        "available": True,
        "source": "msagent-cli-jsonl",
        "input_tokens": 105,
        "output_tokens": 14,
        "total_tokens": 119,
    }

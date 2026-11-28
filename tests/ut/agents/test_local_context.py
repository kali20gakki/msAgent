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

import subprocess
from pathlib import Path

from msagent.agents.local_context import (
    _run_git,
    build_local_environment_context,
    ensure_local_context_prompt,
)


class _FakeProcess:
    def __init__(self, *, timeout_error: bool) -> None:
        self.returncode = 0
        self.killed = False
        self._timeout_error = timeout_error
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicate_calls.append(timeout)
        if self._timeout_error:
            raise subprocess.TimeoutExpired("git", timeout if timeout is not None else 0)
        return "true\n", ""

    def __enter__(self) -> "_FakeProcess":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_run_git_does_not_inherit_stdin(monkeypatch) -> None:
    captured: dict[str, object] = {}
    process = _FakeProcess(timeout_error=False)

    def fake_popen(*args: object, **kwargs: object) -> _FakeProcess:
        captured.update(kwargs)
        return process

    monkeypatch.setattr("msagent.agents.local_context.subprocess.Popen", fake_popen)

    assert _run_git(Path("."), "rev-parse", "--is-inside-work-tree") == "true"
    assert captured["stdin"] is subprocess.DEVNULL


def test_run_git_timeout_is_bounded_and_returns_none(monkeypatch) -> None:
    process = _FakeProcess(timeout_error=True)
    monkeypatch.setattr(
        "msagent.agents.local_context.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    assert _run_git(Path("."), "status", "--short") is None
    assert process.killed is True
    assert all(timeout is not None for timeout in process.communicate_calls)


def test_ensure_local_context_prompt_appends_placeholder_once() -> None:
    prompt = "You are a coding assistant."
    augmented = ensure_local_context_prompt(prompt)

    assert "{local_environment_context}" in augmented
    assert augmented.count("{local_environment_context}") == 1

    augmented_again = ensure_local_context_prompt(augmented)
    assert augmented_again == augmented


def test_build_local_environment_context_includes_project_signals(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("# lock", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "src").mkdir()

    context = build_local_environment_context(tmp_path)

    assert "## Local Runtime Snapshot" in context
    assert "## Project Signals" in context
    assert "Python" in context
    assert "uv" in context
    assert "## Top-Level Workspace Layout" in context

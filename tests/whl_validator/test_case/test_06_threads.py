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

"""Validate that /threads can discover a prior persisted conversation."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from validator_core.assertions import (
    assert_app_log_has_no_fatal_exception,
    assert_session_succeeded,
)
from validator_core.llm_capture_server import LlmCaptureServer
from validator_core.msagent_runtime import MsagentRuntime


DUMMY_API_KEY = "threads-validation-dummy-key"
THREAD_PROMPT = "THREADS_VALIDATION_PREVIOUS_CONVERSATION"
_THREADS_PROBE = """
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from msagent.cli.handlers.threads import ThreadsHandler

session = SimpleNamespace(
    context=SimpleNamespace(
        agent=sys.argv[1],
        working_dir=Path(sys.argv[2]),
        thread_id=sys.argv[3],
    )
)
entries = asyncio.run(ThreadsHandler(session)._load_thread_entries())
print(json.dumps([{"thread_id": entry.thread_id, "preview": entry.preview} for entry in entries]))
"""


def _disable_default_mcp_server(msagent_home) -> None:
    """Keep this persistence-focused case independent of msprof-mcp binaries."""
    config_path = msagent_home / "config" / "config.mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"mcpServers": {"msprof-mcp": {"enabled": False}}}) + "\n",
        encoding="utf-8",
    )


def _load_previous_threads(
    runtime: MsagentRuntime,
    *,
    agent: str,
    current_thread_id: str,
) -> list[dict[str, str]]:
    """Load /threads entries from a fresh process sharing the runtime state."""
    env = os.environ.copy()
    env.update(runtime.extra_env)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            _THREADS_PROBE,
            agent,
            str(runtime.workspace_dir),
            current_thread_id,
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert probe.returncode == 0, f"/threads 探针失败。stdout:\n{probe.stdout}\nstderr:\n{probe.stderr}"
    entries = json.loads(probe.stdout)
    assert isinstance(entries, list), entries
    assert all(isinstance(entry, dict) for entry in entries), entries
    return entries


def test_threads_discovers_a_previous_mock_llm_session(
    msagent_runtime_factory,
    llm_capture_server: LlmCaptureServer,
) -> None:
    """A fresh session must list the conversation created by the prior run."""
    runtime = msagent_runtime_factory(
        "openai",
        base_url=llm_capture_server.base_url,
        api_key=DUMMY_API_KEY,
    )
    _disable_default_mcp_server(runtime.msagent_home)

    first_result = runtime.run(THREAD_PROMPT, agent_name="Accuracy")

    assert_session_succeeded(first_result)
    assert_app_log_has_no_fatal_exception(first_result)
    assert not llm_capture_server.errors, llm_capture_server.errors
    assert llm_capture_server.requests, "Mock LLM 未收到创建历史会话的请求"

    started_events = [event for event in first_result.traces if event.get("type") == "session_started"]
    assert len(started_events) == 1, first_result.traces
    previous_thread_id = started_events[0].get("thread_id")
    assert isinstance(previous_thread_id, str) and previous_thread_id, started_events[0]

    entries = _load_previous_threads(
        runtime,
        agent="Accuracy",
        current_thread_id="threads-browser-validation-session",
    )
    matching_entries = [entry for entry in entries if entry.get("thread_id") == previous_thread_id]

    assert len(matching_entries) == 1, [entry.get("thread_id") for entry in entries]
    assert THREAD_PROMPT in matching_entries[0].get("preview", ""), matching_entries[0]

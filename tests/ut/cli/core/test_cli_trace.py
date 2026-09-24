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

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage

from msagent.cli.core.trace import CliRunRecorder


def test_elapsed_ms_is_zero_before_start(tmp_path: Path) -> None:
    recorder = CliRunRecorder(tmp_path / "events.jsonl")

    assert recorder.elapsed_ms == 0


def test_usage_dedupe_without_message_id_does_not_use_python_hash(monkeypatch, tmp_path: Path) -> None:
    original_hash = builtins.hash

    def fail_on_tuple_hash(value: Any) -> int:
        if isinstance(value, tuple) and len(value) == 3:
            raise AssertionError("unstable Python tuple hash should not be used for usage dedupe")
        return original_hash(value)

    monkeypatch.setattr(builtins, "hash", fail_on_tuple_hash)
    recorder = CliRunRecorder(tmp_path / "events.jsonl")
    context = SimpleNamespace(
        agent="agent",
        model="model",
        model_display="model",
        thread_id="thread",
        working_dir=tmp_path,
        approval_mode="active",
        current_input_tokens=None,
        current_output_tokens=None,
    )
    message = AIMessage(
        content="same message",
        usage_metadata={
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        },
    )

    recorder.start(context=context, stream_output=False)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "hash", fail_on_tuple_hash)
        recorder.record_usage_from_message(message)
        recorder.record_usage_from_message(message)
    recorder.finish(context=context, exit_code=0)

    events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [event["type"] for event in events].count("token_usage") == 1

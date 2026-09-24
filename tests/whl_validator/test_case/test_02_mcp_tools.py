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

"""通过真实 msprof-mcp 和合成 trace_view.json 验证 MCP 核心链路。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validator_core.agent_runner import RunResult
from validator_core.assertions import (
    assert_app_log_has_no_fatal_exception,
    assert_mcp_tool_invoked,
    assert_session_succeeded,
)
from validator_core.llm_capture_server import LlmCaptureServer, ToolCallScript
from validator_core.llm_payload_parser import extract_tool_names
from validator_core.trace_parser import get_tool_result_json, is_mcp_tool_name


MCP_SERVER_NAME = "msprof-mcp"
DUMMY_API_KEY = "mcp-validation-dummy-key"
MOCK_COMPLETION = "MCP_TOOL_VALIDATION_COMPLETED"

EXPECTED_OVERLAP = {
    "Computing": {"duration_ms": 4.0, "percentage": "66.67%"},
    "Communication": {"duration_ms": 1.0, "percentage": "16.67%"},
    "Communication(Not Overlapped)": {
        "duration_ms": 0.5,
        "percentage": "8.33%",
    },
    "Free": {"duration_ms": 0.5, "percentage": "8.33%"},
}


def _synthetic_trace_events() -> dict:
    """Build deterministic Chrome Trace Event data for both MCP tools.

    Chrome trace ``ts`` and ``dur`` values are microseconds. The target process
    uses ``tid == pid`` for its main thread so Perfetto marks it as the process
    main thread. Worker-thread, similar-name, and second-process events are
    deliberate distractors used by the filtering assertions.
    """
    return {
        "traceEvents": [
            {
                "name": "process_name",
                "cat": "__metadata",
                "ph": "M",
                "pid": 100,
                "args": {"name": "Overlap Analysis"},
            },
            {
                "name": "thread_name",
                "cat": "__metadata",
                "ph": "M",
                "pid": 100,
                "tid": 100,
                "args": {"name": "MainThread"},
            },
            {
                "name": "thread_name",
                "cat": "__metadata",
                "ph": "M",
                "pid": 100,
                "tid": 200,
                "args": {"name": "WorkerThread"},
            },
            {
                "name": "Computing",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 0,
                "dur": 4000,
            },
            {
                "name": "Communication",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 4000,
                "dur": 1000,
            },
            {
                "name": "Communication(Not Overlapped)",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 5000,
                "dur": 500,
            },
            {
                "name": "Free",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 5500,
                "dur": 500,
            },
            {
                "name": "MatMulValidation",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 10000,
                "dur": 250,
            },
            {
                "name": "MatMulValidation",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 20000,
                "dur": 750,
            },
            {
                "name": "MatMulValidation",
                "ph": "X",
                "pid": 100,
                "tid": 200,
                "ts": 30000,
                "dur": 2000,
            },
            {
                "name": "MatMulValidationExtra",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": 40000,
                "dur": 3000,
            },
            {
                "name": "process_name",
                "cat": "__metadata",
                "ph": "M",
                "pid": 101,
                "args": {"name": "Distractor Process"},
            },
            {
                "name": "thread_name",
                "cat": "__metadata",
                "ph": "M",
                "pid": 101,
                "tid": 101,
                "args": {"name": "MainThread"},
            },
            {
                "name": "Computing",
                "ph": "X",
                "pid": 101,
                "tid": 101,
                "ts": 0,
                "dur": 9000,
            },
            {
                "name": "Communication",
                "ph": "X",
                "pid": 101,
                "tid": 101,
                "ts": 9000,
                "dur": 9000,
            },
            {
                "name": "MatMulValidation",
                "ph": "X",
                "pid": 101,
                "tid": 101,
                "ts": 50000,
                "dur": 4000,
            },
        ]
    }


@pytest.fixture
def synthetic_trace_view(case_artifact_dir: Path) -> Path:
    """Write the trace input beside this case's retained diagnostic artifacts."""
    input_dir = case_artifact_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    trace_path = input_dir / "synthetic_trace_view.json"
    trace_path.write_text(
        json.dumps(_synthetic_trace_events(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return trace_path.resolve()


def _create_mock_runtime(msagent_runtime_factory, mock_server: LlmCaptureServer):
    """Create a Profiler runtime that uses a dummy key and local Mock LLM."""
    return msagent_runtime_factory(
        "openai",
        base_url=mock_server.base_url,
        api_key=DUMMY_API_KEY,
        # 发布门禁必须验证 whl 自带的 trace_processor_shell，不能被调用
        # pytest 时偶然存在的本机 override 掩盖打包或 glibc 兼容问题。
        runtime_env={"MSPROF_MCP_TRACE_PROCESSOR_SHELL": ""},
    )


def _assert_mcp_run_is_healthy(
    result: RunResult,
    mock_server: LlmCaptureServer,
) -> None:
    """Apply common process, trace, log, and Mock LLM protocol assertions."""
    assert not mock_server.errors, "\n".join(mock_server.errors)
    assert len(mock_server.requests) >= 2, "脚本化 LLM 应至少收到工具调用前和工具结果后的两次请求"
    assert MOCK_COMPLETION in result.stdout, result.stdout
    assert_session_succeeded(result)
    assert_app_log_has_no_fatal_exception(result)


def _assert_only_target_tool_was_called(
    result: RunResult,
    tool_name: str,
) -> None:
    """Prevent an MCP test from silently falling back to another tool."""
    calls = [event for event in result.traces if event.get("type") == "tool_call"]
    assert len(calls) == 1, (
        f"用例只应调用 {MCP_SERVER_NAME}/{tool_name}，实际调用为：{json.dumps(calls, ensure_ascii=False, indent=2)}"
    )
    assert is_mcp_tool_name(calls[0].get("tool"), MCP_SERVER_NAME, tool_name), calls[0]


@pytest.mark.mcp
def test_msprof_mcp_server_exposes_required_trace_view_tools(
    msagent_runtime_factory,
    llm_capture_server: LlmCaptureServer,
) -> None:
    """Profiler 初始化后应从真实 msprof-mcp 发现两个核心 trace 工具。"""
    runtime = _create_mock_runtime(msagent_runtime_factory, llm_capture_server)
    result = runtime.run(
        "请只回复 SYS_PROMPT_CAPTURE_OK，不要调用任何工具。",
        agent_name="Profiler",
    )

    # 如果 MCP 启动或 list_tools 失败，通常不会进入 LLM 请求；先展示 trace
    # 和日志，让失败直接指向 Server 初始化问题。
    errors = [event for event in result.traces if event.get("type") == "error"]
    assert not errors, (
        json.dumps(errors, ensure_ascii=False, indent=2) + f"\nstderr:\n{result.stderr}\napp.log:\n{result.app_log}"
    )
    payload = llm_capture_server.latest_request.get("body")
    assert isinstance(payload, dict), llm_capture_server.latest_request
    tool_names = extract_tool_names(payload)
    assert len(tool_names) == len(set(tool_names)), f"LLM payload 中存在重复工具名: {tool_names!r}"

    for required_tool in ("analyze_overlap", "find_slices"):
        matching = [name for name in tool_names if is_mcp_tool_name(name, MCP_SERVER_NAME, required_tool)]
        assert len(matching) == 1, (
            f"{MCP_SERVER_NAME} 应唯一暴露 {required_tool!r}，实际匹配为 {matching!r}；全部工具为 {tool_names!r}"
        )

    assert_session_succeeded(result)
    assert_app_log_has_no_fatal_exception(result)


@pytest.mark.mcp
def test_msprof_mcp_analyze_overlap_returns_expected_breakdown(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    synthetic_trace_view: Path,
) -> None:
    """analyze_overlap 应只统计目标进程并返回精确占比。"""
    arguments = {"trace_path": str(synthetic_trace_view)}
    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            server_name=MCP_SERVER_NAME,
            tool_name="analyze_overlap",
            arguments=arguments,
            final_content=MOCK_COMPLETION,
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server)
    prompt = (
        "请调用 msprof-mcp 的 analyze_overlap 工具分析以下 trace_view.json：\n"
        f"{synthetic_trace_view}\n"
        "只使用 analyze_overlap，不要改用 shell、SQL 或文件读取工具。"
    )

    result = runtime.run(prompt, agent_name="Profiler")

    call, tool_result = assert_mcp_tool_invoked(
        result,
        MCP_SERVER_NAME,
        "analyze_overlap",
        input_matches=lambda value: value == arguments,
    )
    assert call.get("input") == arguments
    _assert_only_target_tool_was_called(result, "analyze_overlap")

    output = tool_result.get("output")
    assert isinstance(output, dict), tool_result
    assert output.get("is_error") is False, output
    assert output.get("content_truncated") is False, "MCP 结果在 trace 中被截断，无法作为完整发布断言依据"
    payload = get_tool_result_json(tool_result)
    if "success" in payload:
        # analyze_overlap normally returns its analysis object directly. On a
        # trace-processor startup/query failure, msprof-mcp forwards a standard
        # error envelope instead; surface that envelope before field checks.
        assert payload.get("success") is True, payload
    assert payload.get("process") == "Overlap Analysis", payload
    assert float(payload.get("total_duration_ms")) == pytest.approx(6.0)

    breakdown = payload.get("breakdown")
    assert isinstance(breakdown, list), payload
    by_name = {
        item.get("name"): item for item in breakdown if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    assert set(by_name) == set(EXPECTED_OVERLAP), by_name
    for name, expected in EXPECTED_OVERLAP.items():
        assert float(by_name[name].get("duration_ms")) == pytest.approx(expected["duration_ms"])
        assert by_name[name].get("percentage") == expected["percentage"]

    percentage_sum = sum(float(str(item["percentage"]).removesuffix("%")) for item in breakdown)
    assert percentage_sum == pytest.approx(100.0, abs=0.02)
    _assert_mcp_run_is_healthy(result, mock_server)


@pytest.mark.mcp
def test_msprof_mcp_find_slices_filters_target_process_and_main_thread(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    synthetic_trace_view: Path,
) -> None:
    """find_slices 应排除近似名称、工作线程和其他进程的数据。"""
    arguments = {
        "trace_path": str(synthetic_trace_view),
        "pattern": "MatMulValidation",
        "process_name": "Overlap Analysis",
        "match_mode": "exact",
        "limit": 20,
        "main_thread_only": True,
    }
    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            server_name=MCP_SERVER_NAME,
            tool_name="find_slices",
            arguments=arguments,
            final_content=MOCK_COMPLETION,
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server)
    prompt = (
        "请调用 msprof-mcp 的 find_slices，在以下文件中精确查找 "
        "Overlap Analysis 进程主线程上的 MatMulValidation，limit=20：\n"
        f"{synthetic_trace_view}\n"
        "只使用 find_slices。"
    )

    result = runtime.run(prompt, agent_name="Profiler")

    call, tool_result = assert_mcp_tool_invoked(
        result,
        MCP_SERVER_NAME,
        "find_slices",
        input_matches=lambda value: value == arguments,
    )
    assert call.get("input") == arguments
    _assert_only_target_tool_was_called(result, "find_slices")

    output = tool_result.get("output")
    assert isinstance(output, dict), tool_result
    assert output.get("is_error") is False, output
    assert output.get("content_truncated") is False, "MCP 结果在 trace 中被截断，无法作为完整发布断言依据"
    envelope = get_tool_result_json(tool_result)
    assert envelope.get("success") is True, envelope
    assert envelope.get("error") is None, envelope
    assert envelope.get("processName") == "Overlap Analysis", envelope
    assert Path(str(envelope.get("tracePath"))).resolve() == synthetic_trace_view

    payload = envelope.get("result")
    assert isinstance(payload, dict), envelope
    assert payload.get("matchMode") == "exact", payload
    assert payload.get("timeRangeMs") is None, payload
    filters = payload.get("filters")
    assert isinstance(filters, dict), payload
    assert filters == {
        "processName": "Overlap Analysis",
        "mainThreadOnly": True,
        "limit": 20,
        "pattern": "MatMulValidation",
    }

    aggregates = payload.get("aggregates")
    assert isinstance(aggregates, list) and len(aggregates) == 1, payload
    aggregate = aggregates[0]
    assert aggregate.get("name") == "MatMulValidation", aggregate
    assert aggregate.get("count") == 2, aggregate
    assert float(aggregate.get("minMs")) == pytest.approx(0.25)
    assert float(aggregate.get("avgMs")) == pytest.approx(0.5)
    assert float(aggregate.get("maxMs")) == pytest.approx(0.75)
    assert aggregate.get("linkable") is True, aggregate

    examples = payload.get("examples")
    assert isinstance(examples, list) and len(examples) == 2, payload
    actual_durations_by_ts = {}
    for example in examples:
        assert example.get("process_name") == "Overlap Analysis", example
        assert example.get("pid") == 100, example
        assert example.get("tid") == 100, example
        assert bool(example.get("is_main_thread")) is True, example
        actual_durations_by_ts[float(example.get("tsMs"))] = float(example.get("durMs"))
    assert actual_durations_by_ts == pytest.approx({10.0: 0.25, 20.0: 0.75})

    _assert_mcp_run_is_healthy(result, mock_server)

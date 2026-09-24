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

"""通过本地 OpenAI 兼容服务捕获并验证最终 System Prompt payload。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from validator_core.llm_payload_parser import extract_system_prompt, extract_tool_names


CAPTURE_PROMPT = "请只回复 SYS_PROMPT_CAPTURE_OK，不要调用任何工具。"
DUMMY_API_KEY = "system-prompt-capture-dummy-key"


@dataclass(frozen=True)
class AgentPromptCase:
    """描述一个 Agent 的 Prompt、Skill 和工具边界。"""

    agent_name: str
    expected_prompt_keywords: tuple[str, ...]
    forbidden_prompt_keywords: tuple[str, ...]
    expected_skill_names: tuple[str, ...]
    expected_tool_prefixes: tuple[str, ...] = ()
    forbidden_tool_prefixes: tuple[str, ...] = ()


AGENT_PROMPT_CASES = [
    pytest.param(
        AgentPromptCase(
            agent_name="Profiler",
            expected_prompt_keywords=(
                "你是 Profiler",
                "Ascend NPU Profiling 性能分析助手",
                "真实 Profiling 数据",
                "数据驱动",
                "工具优先",
            ),
            forbidden_prompt_keywords=(
                "你是 Accuracy",
                "accuracy/nan-overflow-detection",
            ),
            expected_skill_names=(
                "profiler/op-mfu-calculator",
                "profiler/ascend-profiler-data-validation",
            ),
            expected_tool_prefixes=("msprof-mcp",),
        ),
        id="Profiler",
    ),
    pytest.param(
        AgentPromptCase(
            agent_name="Accuracy",
            expected_prompt_keywords=(
                "你是 Accuracy",
                "Ascend NPU 精度分析助手",
                "真实 dump 数据",
                "数据驱动",
                "证据闭环",
            ),
            forbidden_prompt_keywords=(
                "你是 Profiler",
                "profiler/op-mfu-calculator",
            ),
            expected_skill_names=(
                "accuracy/nan-overflow-detection",
                "accuracy/deterministic-calculation-analysis",
            ),
            forbidden_tool_prefixes=("msprof-mcp",),
        ),
        id="Accuracy",
    ),
]


def _run_and_capture(
    llm_runtime_factory,
    llm_capture_server,
    agent_name: str,
):
    """运行指定 Agent，并返回运行时、结果和捕获到的请求 payload。"""
    runtime = llm_runtime_factory(
        "openai",
        base_url=llm_capture_server.base_url,
        api_key=DUMMY_API_KEY,
    )
    result = runtime.run(CAPTURE_PROMPT, agent_name=agent_name)

    # Report startup/graph errors before trying to inspect HTTP capture state.
    # This preserves the real root cause when, for example, an Agent-required
    # MCP executable is missing and no LLM request can be sent.
    error_events = [event for event in result.traces if event.get("type") == "error"]
    assert not error_events, (
        "msagent 在发送 LLM 请求前失败：\n"
        + json.dumps(error_events, ensure_ascii=False, indent=2)
        + f"\nstderr:\n{result.stderr}\napp.log:\n{result.app_log}"
    )
    captured = llm_capture_server.latest_request

    assert captured.get("path", "").endswith("/chat/completions"), captured
    payload = captured.get("body")
    assert isinstance(payload, dict), f"捕获到的请求 body 无效: {captured!r}"
    assert "SYS_PROMPT_CAPTURE_OK" in result.stdout, result.stdout

    session_started = [event for event in result.traces if event.get("type") == "session_started"]
    assert session_started, f"缺少 session_started 事件: {result.traces!r}"
    assert session_started[-1].get("agent") == agent_name, session_started[-1]
    return runtime, result, payload


@pytest.mark.parametrize("agent_case", AGENT_PROMPT_CASES)
def test_agent_system_prompt_contains_expected_identity_skills_and_tools(
    llm_runtime_factory,
    llm_capture_server,
    agent_case: AgentPromptCase,
) -> None:
    """至少验证 Profiler 和 Accuracy 的身份、Skills 及工具隔离。"""
    _, result, payload = _run_and_capture(
        llm_runtime_factory,
        llm_capture_server,
        agent_case.agent_name,
    )
    system_prompt = extract_system_prompt(payload)
    tool_names = extract_tool_names(payload)

    for keyword in agent_case.expected_prompt_keywords:
        assert keyword in system_prompt, f"{agent_case.agent_name} System Prompt 缺少关键字 {keyword!r}"
    for keyword in agent_case.forbidden_prompt_keywords:
        assert keyword not in system_prompt, f"{agent_case.agent_name} System Prompt 错误包含 {keyword!r}"
    for skill_name in agent_case.expected_skill_names:
        assert skill_name in system_prompt, f"{agent_case.agent_name} System Prompt 缺少 Skill {skill_name!r}"

    # get_skill 是所有带 Skill 的 Agent 都必须具备的目录工具。
    assert "get_skill" in tool_names, f"请求 tools 中缺少 get_skill: {tool_names!r}"
    for prefix in agent_case.expected_tool_prefixes:
        assert any(name.startswith(prefix) for name in tool_names), (
            f"{agent_case.agent_name} 请求 tools 缺少前缀 {prefix!r}: {tool_names!r}"
        )
    for prefix in agent_case.forbidden_tool_prefixes:
        assert not any(name.startswith(prefix) for name in tool_names), (
            f"{agent_case.agent_name} 请求 tools 不应包含前缀 {prefix!r}: {tool_names!r}"
        )


def test_system_prompt_contains_runtime_environment(
    llm_runtime_factory,
    llm_capture_server,
) -> None:
    """断言环境占位符已替换为当前隔离工作区和真实运行环境。"""
    runtime, _, payload = _run_and_capture(
        llm_runtime_factory,
        llm_capture_server,
        "Profiler",
    )
    system_prompt = extract_system_prompt(payload)

    assert "<local-context>" in system_prompt
    assert "## Local Runtime Snapshot" in system_prompt
    assert "Working directory:" in system_prompt
    assert str(runtime.workspace_dir.resolve()) in system_prompt
    assert "- OS:" in system_prompt
    assert "- Python:" in system_prompt

    unresolved_placeholders = (
        "{working_dir}",
        "{platform}",
        "{os_version}",
        "{current_date_time_zoned}",
        "{local_environment_context}",
    )
    for placeholder in unresolved_placeholders:
        assert placeholder not in system_prompt, f"System Prompt 仍包含未替换的环境变量: {placeholder}"

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

"""通过结构化轨迹验证 Skill 的显式选择与自然语言触发。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from validator_core.agent_runner import RunResult
from validator_core.assertions import assert_session_succeeded
from validator_core.trace_parser import get_tool_calls, get_tool_results


@dataclass(frozen=True)
class SkillRoutingCase:
    """描述一个与具体业务领域无关的 Skill 路由期望。"""

    prompt: str
    skill_name: str
    skill_category: str
    skill_content_keywords: tuple[str, ...] = ()


EXPLICIT_SKILL_CASES = [
    pytest.param(
        SkillRoutingCase(
            prompt=(
                "/op-mfu-calculator 计算Ascend 910B3上Matmul算子的MFU，"
                "input_shapes=[24480,1152;3456,1152;3456]，"
                "task_duration=858300纳秒"
            ),
            skill_name="op-mfu-calculator",
            skill_category="profiler",
            skill_content_keywords=("MFU",),
        ),
        id="op-mfu-calculator",
    ),
]

KEYWORD_SKILL_CASES = [
    pytest.param(
        SkillRoutingCase(
            prompt=(
                "计算Ascend 910B3上Matmul算子的MFU，input_shapes=[24480,1152;3456,1152;3456]，task_duration=858300纳秒"
            ),
            skill_name="op-mfu-calculator",
            skill_category="profiler",
            skill_content_keywords=("MFU",),
        ),
        id="op-mfu-calculator",
    ),
]


def _assert_skill_invoked(
    result: RunResult,
    case: SkillRoutingCase,
) -> None:
    """断言目标 Skill 被成功读取，且调用和返回可以按 item_id 关联。"""
    calls = get_tool_calls(result.traces, "get_skill")
    target_calls = [
        event
        for event in calls
        if event.get("input", {}).get("name") == case.skill_name
        and (
            event.get("input", {}).get("category") is None
            or event.get("input", {}).get("category") == case.skill_category
        )
    ]
    assert target_calls, (
        f"没有发现 {case.skill_category}/{case.skill_name} 的 get_skill 调用，"
        f"实际调用为：{json.dumps(calls, ensure_ascii=False, indent=2)}"
    )

    call = target_calls[-1]
    assert call.get("item_id"), f"get_skill 调用缺少 item_id: {call!r}"

    results = get_tool_results(result.traces, "get_skill")
    matching_results = [event for event in results if event.get("item_id") == call["item_id"]]
    assert matching_results, f"没有找到对应的 get_skill 结果: {results!r}"

    output = matching_results[-1].get("output")
    assert isinstance(output, dict), f"get_skill 返回结构无效: {matching_results[-1]!r}"
    assert output.get("is_error") is False, f"get_skill 执行失败: {output!r}"

    content = str(output.get("content") or "")
    assert content.strip(), "get_skill 返回了空的 SKILL.md 内容"
    assert case.skill_name in content, f"get_skill 返回内容不属于目标 Skill: {content!r}"
    missing_keywords = [
        keyword for keyword in case.skill_content_keywords if keyword.casefold() not in content.casefold()
    ]
    assert not missing_keywords, f"SKILL.md 缺少预期关键字 {missing_keywords!r}: {content!r}"


def _run_case(llm_runtime_factory, case: SkillRoutingCase) -> RunResult:
    """使用共享的 OpenAI 兼容运行时执行一个 Skill 路由用例。"""
    runtime = llm_runtime_factory("openai")
    return runtime.run(case.prompt)


@pytest.mark.llm
@pytest.mark.skill
@pytest.mark.parametrize("skill_case", EXPLICIT_SKILL_CASES)
def test_explicit_skill_is_routed(
    llm_runtime_factory,
    skill_case: SkillRoutingCase,
) -> None:
    """显式给出 Skill 快捷方式时，应读取并执行对应 Skill。"""
    result = _run_case(llm_runtime_factory, skill_case)

    _assert_skill_invoked(result, skill_case)
    assert_session_succeeded(result)


@pytest.mark.llm
@pytest.mark.skill
@pytest.mark.parametrize("skill_case", KEYWORD_SKILL_CASES)
def test_keyword_triggers_expected_skill(
    llm_runtime_factory,
    skill_case: SkillRoutingCase,
) -> None:
    """自然语言不出现 Skill 名称时，应根据语义路由到对应 Skill。"""
    result = _run_case(llm_runtime_factory, skill_case)

    _assert_skill_invoked(result, skill_case)
    assert_session_succeeded(result)

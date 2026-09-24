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

"""通过结构化运行轨迹验证真实 LLM 的连通性和会话状态。"""

from __future__ import annotations

import pytest

from validator_core.assertions import (
    assert_app_log_has_no_fatal_exception,
    assert_session_succeeded,
)


LLM_PROVIDERS = [
    pytest.param("openai", id="openai"),
    pytest.param("anthropic", id="anthropic"),
]


def _run_ping(llm_provider: str, llm_runtime_factory):
    runtime = llm_runtime_factory(llm_provider)
    return runtime.run("Ping. Please reply 'Pong' only.")


@pytest.mark.llm
@pytest.mark.parametrize("llm_provider", LLM_PROVIDERS)
def test_llm_response_trace_is_successful(llm_provider: str, llm_runtime_factory) -> None:
    result = _run_ping(llm_provider, llm_runtime_factory)

    # JSONL 中必须出现至少一条非空 assistant_message，证明模型确实返回内容。
    assistant_messages = [event for event in result.traces if event.get("type") == "assistant_message"]
    assert assistant_messages, f"未捕获 assistant_message，traces={result.traces!r}"
    assert any(str(event.get("content") or "").strip() for event in assistant_messages), (
        f"assistant_message 内容均为空: {assistant_messages!r}"
    )

    # 简单确定性问题的最终控制台输出应包含 Pong，忽略模型的大小写差异。
    assert "pong" in result.stdout.lower(), f"最终输出不包含 Pong: {result.stdout!r}"


@pytest.mark.llm
@pytest.mark.parametrize("llm_provider", LLM_PROVIDERS)
def test_llm_session_has_no_runtime_errors(llm_provider: str, llm_runtime_factory) -> None:
    result = _run_ping(llm_provider, llm_runtime_factory)

    # 同时校验真实进程返回码、结构化会话收口和应用日志。
    assert_session_succeeded(result)
    assert_app_log_has_no_fatal_exception(result)

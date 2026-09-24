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

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.agents.context import AgentContext
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.configs import ApprovalMode, LLMProvider


@pytest.mark.asyncio
async def test_context_create_keeps_alias_and_exposes_resolved_model(
    monkeypatch,
) -> None:
    llm_config = SimpleNamespace(
        alias="default",
        model="deepseek-chat",
        provider=LLMProvider.OPENAI,
        context_window=128000,
    )
    agent_config = SimpleNamespace(
        name="Profiler",
        description="Ascend NPU profiling analysis agent with msprof-mcp-first workflow",
        llm=llm_config,
        tools=None,
        recursion_limit=80,
    )

    async def fake_load_agent_config(agent, working_dir):
        return agent_config

    async def fake_get_current_model(agent, working_dir):
        return None

    monkeypatch.setattr(initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(initializer, "get_current_model", fake_get_current_model)

    context = await Context.create(
        agent=None,
        model=None,
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        working_dir=Path.cwd(),
    )

    assert context.model == "default"
    assert context.agent_description == agent_config.description
    assert context.model_display == "deepseek-chat (openai)"


@pytest.mark.asyncio
async def test_context_create_uses_workspace_model_preference(monkeypatch) -> None:
    default_llm = SimpleNamespace(
        alias="default",
        model="default-model",
        provider=LLMProvider.OPENAI,
        context_window=64000,
    )
    preferred_llm = SimpleNamespace(
        alias="fast",
        model="fast-model",
        provider=LLMProvider.OPENAI,
        context_window=128000,
    )
    agent_config = SimpleNamespace(
        name="Profiler",
        description="Profiler",
        llm=default_llm,
        tools=None,
        recursion_limit=80,
    )

    async def fake_load_agent_config(_agent, _working_dir):
        return agent_config

    async def fake_get_current_model(_agent, _working_dir):
        return "fast"

    async def fake_load_llm_config(model, _working_dir):
        assert model == "fast"
        return preferred_llm

    monkeypatch.setattr(initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(initializer, "get_current_model", fake_get_current_model)
    monkeypatch.setattr(initializer, "load_llm_config", fake_load_llm_config)

    context = await Context.create(
        agent=None,
        model=None,
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        working_dir=Path.cwd(),
    )

    assert context.agent == "Profiler"
    assert context.model == "fast"
    assert context.model_display == "fast-model (openai)"


@pytest.mark.asyncio
async def test_context_create_keeps_execute_approval_mode(monkeypatch) -> None:
    llm_config = SimpleNamespace(
        alias="default",
        model="deepseek-chat",
        provider=LLMProvider.OPENAI,
        context_window=128000,
    )
    agent_config = SimpleNamespace(
        name="Profiler",
        description="Profiler",
        llm=llm_config,
        tools=None,
        recursion_limit=80,
    )

    async def fake_load_agent_config(_agent, _working_dir):
        return agent_config

    async def fake_get_current_model(_agent, _working_dir):
        return None

    monkeypatch.setattr(initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(initializer, "get_current_model", fake_get_current_model)

    context = await Context.create(
        agent=None,
        model=None,
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        execute_approval_mode="safe",
        working_dir=Path.cwd(),
    )

    assert context.execute_approval_mode == "safe"


def test_agent_context_defaults_to_cwd() -> None:
    context = AgentContext()

    assert context.approval_mode == ApprovalMode.ACTIVE
    assert context.working_dir == Path.cwd().resolve()

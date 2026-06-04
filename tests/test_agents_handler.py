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

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.core.context import Context
from msagent.cli.handlers import agents as agents_module
from msagent.cli.handlers.agents import AgentHandler
from msagent.cli.bootstrap.initializer import initializer
from msagent.configs import AgentConfig, ApprovalMode


def _build_context() -> Context:
    return Context(
        agent="Profiler",
        agent_description="Ascend NPU profiling analysis agent",
        model="default",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        recursion_limit=80,
    )


@pytest.mark.asyncio
async def test_agents_handler_keeps_single_agent_list_without_warning(
    monkeypatch,
) -> None:
    agent = AgentConfig.model_construct(
        name="Profiler",
        description="Ascend NPU profiling analysis agent",
        llm=SimpleNamespace(alias="default"),
        default=True,
    )

    async def fake_load_agents_config(_working_dir):
        return SimpleNamespace(agents=[agent])

    selections: list[tuple[list[AgentConfig], str]] = []
    monkeypatch.setattr(initializer, "load_agents_config", fake_load_agents_config)

    async def fake_get_agent_selection(agents, current_agent_name):
        selections.append((agents, current_agent_name))
        return "Profiler"

    session = SimpleNamespace(context=_build_context(), update_context=lambda **kwargs: None)
    handler = AgentHandler(session)
    monkeypatch.setattr(handler, "_get_agent_selection", fake_get_agent_selection)

    await handler.handle()

    assert len(selections) == 1
    agents, current_agent_name = selections[0]
    assert [agent.name for agent in agents] == ["Profiler"]
    assert current_agent_name == "Profiler"


def test_format_agent_list_shows_all_state_markers() -> None:
    agents = [
        AgentConfig.model_construct(
            name="Profiler",
            description="Ascend NPU profiling analysis agent",
            llm=SimpleNamespace(alias="default"),
            default=True,
        ),
        AgentConfig.model_construct(
            name="Minos",
            description="Documentation onboarding and GitCode PR review agent",
            llm=SimpleNamespace(alias="default"),
            default=False,
        ),
    ]

    formatted = AgentHandler._format_agent_list(agents, selected_index=0, current_agent_name="Profiler")
    text = "".join(fragment[1] for fragment in formatted)

    assert "Profiler [current]" in text
    assert "Ascend NPU profiling analysis agent" in text
    assert "Minos" in text
    assert "Documentation onboarding and GitCode PR review agent" in text


@pytest.mark.asyncio
async def test_agents_handler_reports_no_agents_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(agents_module.console, "print_error", errors.append)
    monkeypatch.setattr(agents_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_load_agents_config(_working_dir):
        return SimpleNamespace(agents=[])

    monkeypatch.setattr(agents_module.initializer, "load_agents_config", fake_load_agents_config)

    session = SimpleNamespace(context=_build_context(), update_context=lambda **kwargs: None)
    handler = AgentHandler(session)
    await handler.handle()

    assert "No agents configured" in errors


@pytest.mark.asyncio
async def test_agents_handler_skips_update_when_same_agent_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = AgentConfig.model_construct(
        name="Profiler",
        description="Ascend NPU profiling analysis agent",
        llm=SimpleNamespace(alias="default"),
        default=True,
    )

    context_updates: dict[str, object] = {}

    async def fake_load_agents_config(_working_dir):
        return SimpleNamespace(agents=[agent])

    monkeypatch.setattr(agents_module.initializer, "load_agents_config", fake_load_agents_config)

    async def fake_get_agent_selection(_agents, _current):
        return "Profiler"

    session = SimpleNamespace(context=_build_context(), update_context=lambda **kwargs: context_updates.update(kwargs))
    handler = AgentHandler(session)
    monkeypatch.setattr(handler, "_get_agent_selection", fake_get_agent_selection)

    await handler.handle()

    assert context_updates == {}


@pytest.mark.asyncio
async def test_agents_handler_updates_context_when_different_agent_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hermes = AgentConfig.model_construct(
        name="Profiler",
        description="NPU profiling agent",
        llm=SimpleNamespace(alias="default"),
        default=True,
    )
    minos = AgentConfig.model_construct(
        name="Minos",
        description="Documentation agent",
        llm=SimpleNamespace(alias="fast"),
        default=False,
    )

    context_updates: dict[str, object] = {}

    async def fake_load_agents_config(_working_dir):
        return SimpleNamespace(agents=[hermes, minos])

    async def fake_load_agent_config(agent_name, _working_dir):
        if agent_name == "Minos":
            return minos
        return hermes

    async def fake_update_default_agent(_agent_name, _working_dir):
        pass

    async def fake_get_agent_selection(_agents, _current):
        return "Minos"

    monkeypatch.setattr(agents_module.initializer, "load_agents_config", fake_load_agents_config)
    monkeypatch.setattr(agents_module.initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(agents_module.initializer, "update_default_agent", fake_update_default_agent)

    session = SimpleNamespace(context=_build_context(), update_context=lambda **kwargs: context_updates.update(kwargs))
    handler = AgentHandler(session)
    monkeypatch.setattr(handler, "_get_agent_selection", fake_get_agent_selection)

    await handler.handle()

    assert context_updates.get("agent") == "Minos"
    assert context_updates.get("model") == "fast"


@pytest.mark.asyncio
async def test_agents_handler_handles_exception_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(agents_module.console, "print_error", errors.append)
    monkeypatch.setattr(agents_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_load_agents_config(_working_dir):
        raise RuntimeError("config load error")

    monkeypatch.setattr(agents_module.initializer, "load_agents_config", fake_load_agents_config)

    session = SimpleNamespace(context=_build_context(), update_context=lambda **kwargs: None)
    handler = AgentHandler(session)
    await handler.handle()

    assert any("Error switching agents" in e for e in errors)


def test_format_agent_list_hides_description_when_missing() -> None:
    agent = AgentConfig.model_construct(
        name="Profiler",
        description=None,
        llm=SimpleNamespace(alias="default"),
        default=True,
    )

    formatted = AgentHandler._format_agent_list([agent], selected_index=0, current_agent_name="Profiler")
    text = "".join(fragment[1] for fragment in formatted)

    assert "Profiler [current]" in text


def test_format_agent_list_does_not_show_current_tag_for_non_current_agent() -> None:
    agents = [
        AgentConfig.model_construct(
            name="Profiler",
            description="Agent A",
            llm=SimpleNamespace(alias="default"),
            default=True,
        ),
        AgentConfig.model_construct(
            name="Minos",
            description="Agent B",
            llm=SimpleNamespace(alias="fast"),
            default=False,
        ),
    ]

    formatted = AgentHandler._format_agent_list(agents, selected_index=1, current_agent_name="Profiler")
    text = "".join(fragment[1] for fragment in formatted)

    assert "[current]" not in text.replace("Profiler [current]", "")


@pytest.mark.asyncio
async def test_agent_handler_get_agent_selection_returns_empty_for_no_agents() -> None:
    session = SimpleNamespace(context=_build_context())
    handler = AgentHandler(session)
    result = await handler._get_agent_selection([], "Profiler")
    assert result == ""

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.core.context import Context
from msagent.cli.handlers.agents import AgentHandler
from msagent.cli.bootstrap.initializer import initializer
from msagent.configs import AgentConfig, ApprovalMode


def _build_context() -> Context:
    return Context(
        agent="Hermes",
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
        name="Hermes",
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
        return "Hermes"

    session = SimpleNamespace(
        context=_build_context(), update_context=lambda **kwargs: None
    )
    handler = AgentHandler(session)
    monkeypatch.setattr(handler, "_get_agent_selection", fake_get_agent_selection)

    await handler.handle()

    assert len(selections) == 1
    agents, current_agent_name = selections[0]
    assert [agent.name for agent in agents] == ["Hermes"]
    assert current_agent_name == "Hermes"


def test_format_agent_list_shows_all_state_markers() -> None:
    agents = [
        AgentConfig.model_construct(
            name="Hermes",
            description="Ascend NPU profiling analysis agent",
            llm=SimpleNamespace(alias="default"),
            default=True,
        ),
        AgentConfig.model_construct(
            name="Minos",
            description="Documentation onboarding and UX review agent",
            llm=SimpleNamespace(alias="default"),
            default=False,
        ),
    ]

    formatted = AgentHandler._format_agent_list(
        agents, selected_index=0, current_agent_name="Hermes"
    )
    text = "".join(fragment[1] for fragment in formatted)

    assert "Hermes [current]" in text
    assert "Ascend NPU profiling analysis agent" in text
    assert "Minos" in text
    assert "Documentation onboarding and UX review agent" in text

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
        name="Hermes",
        description="Ascend NPU profiling analysis agent with msprof-mcp-first workflow",
        llm=llm_config,
        tools=None,
        recursion_limit=80,
    )

    async def fake_load_agent_config(agent, working_dir):
        return agent_config

    monkeypatch.setattr(initializer, "load_agent_config", fake_load_agent_config)

    context = await Context.create(
        agent=None,
        model=None,
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        working_dir=Path.cwd(),
    )

    assert context.model == "default"
    assert context.agent_description == agent_config.description
    assert context.model_display == "deepseek-chat (openai)"


def test_agent_context_defaults_support_web_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MSAGENT_WEB_WORKING_DIR", str(tmp_path))

    context = AgentContext()

    assert context.approval_mode == ApprovalMode.ACTIVE
    assert context.working_dir == tmp_path.resolve()

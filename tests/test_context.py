from pathlib import Path
from types import SimpleNamespace

import pytest

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
        name="msagent",
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
        resume=False,
        working_dir=Path.cwd(),
    )

    assert context.model == "default"
    assert context.model_display == "deepseek-chat (openai)"

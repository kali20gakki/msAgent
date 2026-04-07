from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from msagent.agents.context import AgentContext
from msagent.cli.bootstrap.initializer import Initializer
from msagent.configs import ToolApprovalConfig
from msagent.skills.factory import Skill


class _DummyAsyncContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _DummyMcpClient:
    async def tools(self):
        return []

    async def close(self):
        return None


def test_initializer_resolves_cached_mcp_server_names_from_filtered_tools() -> None:
    init = Initializer()
    mcp_config = SimpleNamespace(
        servers={
            "msprof-mcp": SimpleNamespace(enabled=True),
            "other-mcp": SimpleNamespace(enabled=True),
        }
    )
    tools = [
        SimpleNamespace(name="msprof-mcp_ping"),
        SimpleNamespace(name="get_skill"),
    ]

    resolved = init._resolve_cached_mcp_server_names(
        tools=tools,
        mcp_config=mcp_config,
        mcp_module_map={"msprof-mcp_ping": "mcp:msprof-mcp"},
    )

    assert resolved == ["msprof-mcp"]


@pytest.mark.asyncio
async def test_initializer_passes_agent_context_schema_to_agent_factory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init = Initializer()

    agent_config = SimpleNamespace(
        checkpointer=None,
        tools=None,
        llm=SimpleNamespace(),
        name="msagent",
        prompt="prompt",
    )
    mcp_config = SimpleNamespace(servers={})
    registry = SimpleNamespace(
        get_agent=AsyncMock(return_value=agent_config),
        load_mcp=AsyncMock(return_value=mcp_config),
        load_approval=lambda: ToolApprovalConfig.model_validate(
            {"interrupt_on": {"execute": {"allowed_decisions": ["approve", "reject"]}}}
        ),
    )
    monkeypatch.setattr(init, "get_registry", lambda _wd: registry)

    monkeypatch.setattr(
        init, "_create_checkpointer", lambda *_args, **_kwargs: _DummyAsyncContext()
    )
    monkeypatch.setattr(
        init.mcp_factory, "create", AsyncMock(return_value=_DummyMcpClient())
    )

    fake_skill = Skill(
        name="demo",
        description="demo skill",
        category="default",
        path=tmp_path / "skills" / "demo" / "SKILL.md",
    )
    monkeypatch.setattr(
        init.skill_factory,
        "load_skills",
        AsyncMock(return_value={"default": {"demo": fake_skill}}),
    )

    fake_graph = SimpleNamespace(_llm_tools=[], _tools_in_catalog=[])
    create_mock = AsyncMock(return_value=fake_graph)
    monkeypatch.setattr(init.agent_factory, "create", create_mock)

    _graph, cleanup = await init.create_graph(
        agent=None,
        model=None,
        working_dir=tmp_path,
    )

    assert create_mock.await_args.kwargs["context_schema"] is AgentContext
    assert create_mock.await_args.kwargs["skills_dir"] is None
    assert create_mock.await_args.kwargs["interrupt_on"] == {
        "execute": {"allowed_decisions": ["approve", "reject"]}
    }
    await cleanup()


def test_initializer_filters_skills_by_patterns_with_negative_rules(
    tmp_path: Path,
) -> None:
    keep = Skill(
        name="mindstudio_profiler_data_check",
        description="keep",
        category="default",
        path=tmp_path / "skills" / "mindstudio_profiler_data_check" / "SKILL.md",
    )
    drop = Skill(
        name="op-mfu-calculator",
        description="drop",
        category="default",
        path=tmp_path / "skills" / "op-mfu-calculator" / "SKILL.md",
    )
    other_category = Skill(
        name="my-skill",
        description="other",
        category="analysis",
        path=tmp_path / "skills" / "analysis" / "my-skill" / "SKILL.md",
    )

    filtered = Initializer._filter_skills_by_patterns(
        [keep, drop, other_category],
        patterns=[
            "default:*",
            "!default:op-mfu-calculator",
        ],
    )

    assert [skill.name for skill in filtered] == ["mindstudio_profiler_data_check"]

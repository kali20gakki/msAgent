import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException
from pydantic import BaseModel

from msagent.skills.factory import Skill
from msagent.skills.factory import SkillFactory
from msagent.cli.bootstrap.initializer import initializer
from msagent.tools.catalog.skills import fetch_skills, get_skill
from msagent.tools.catalog.tools import (
    fetch_tools,
    get_tool,
    run_tool,
)
from msagent.tools.web_search import web_search


class RuntimeArgs(BaseModel):
    query: str
    runtime: object | None = None


def _make_runtime(*, tools=None, skills=None):
    return SimpleNamespace(
        context=SimpleNamespace(
            tool_catalog=list(tools or []),
            skill_catalog=list(skills or []),
        )
    )


def _make_runtime_with_dict_context(*, tools=None, skills=None):
    return SimpleNamespace(
        context={
            "tool_catalog": list(tools or []),
            "skill_catalog": list(skills or []),
        }
    )


@pytest.mark.asyncio
async def test_fetch_tools_filters_by_name_and_description() -> None:
    tools = [
        SimpleNamespace(
            name="read_file",
            description="Read a file from disk",
            tool_call_schema={"type": "object", "properties": {}},
            args_schema=None,
        ),
        SimpleNamespace(
            name="web_search",
            description="Search the web and return results with source URLs",
            tool_call_schema={"type": "object", "properties": {}},
            args_schema=None,
        ),
    ]

    result = await fetch_tools.coroutine(
        runtime=_make_runtime(tools=tools),
        pattern="web|disk",
    )

    assert result.splitlines() == ["read_file", "web_search"]


@pytest.mark.asyncio
async def test_fetch_tools_supports_dict_context_shape() -> None:
    tools = [
        SimpleNamespace(
            name="read_file",
            description="Read a file from disk",
            tool_call_schema={"type": "object", "properties": {}},
            args_schema=None,
        ),
    ]
    result = await fetch_tools.coroutine(
        runtime=_make_runtime_with_dict_context(tools=tools),
        pattern="read_file",
    )

    assert result.splitlines() == ["read_file"]


@pytest.mark.asyncio
async def test_fetch_tools_falls_back_to_builtin_catalog_when_runtime_is_empty() -> None:
    runtime = SimpleNamespace(context=SimpleNamespace(tool_catalog=[]))
    result = await fetch_tools.coroutine(runtime=runtime, pattern="fetch_tools")

    assert result.splitlines() == ["fetch_tools"]


@pytest.mark.asyncio
async def test_fetch_tools_rejects_invalid_regex() -> None:
    with pytest.raises(ToolException, match="Invalid regex pattern"):
        await fetch_tools.coroutine(runtime=_make_runtime(tools=[]), pattern="(")


@pytest.mark.asyncio
async def test_get_tool_returns_json_schema_for_tool() -> None:
    tool = SimpleNamespace(
        name="search",
        description="Search indexed traces",
        tool_call_schema=RuntimeArgs,
        args_schema=RuntimeArgs,
    )

    result = await get_tool.coroutine(
        tool_name="search",
        runtime=_make_runtime(tools=[tool]),
    )
    payload = json.loads(result)

    assert payload["name"] == "search"
    assert payload["description"] == "Search indexed traces"
    assert "query" in payload["parameters"]["properties"]


@pytest.mark.asyncio
async def test_get_tool_returns_builtin_web_search_metadata() -> None:
    result = await get_tool.coroutine(
        tool_name="web_search",
        runtime=_make_runtime(tools=[web_search]),
    )
    payload = json.loads(result)

    assert payload["name"] == "web_search"
    assert payload["description"] == "Search the web and return compact results with source URLs."
    assert "query" in payload["parameters"]["properties"]
    assert "allowed_domains" in payload["parameters"]["properties"]


@pytest.mark.asyncio
async def test_get_tool_not_found_returns_error_payload() -> None:
    result = await get_tool.coroutine(
        tool_name="missing_tool",
        runtime=_make_runtime(tools=[]),
    )
    payload = json.loads(result)

    assert "error" in payload
    assert payload["error"] == "Tool 'missing_tool' not found"
    assert "fetch_tools" in payload["available_tools"]


@pytest.mark.asyncio
async def test_run_tool_injects_runtime_when_tool_accepts_it() -> None:
    invoke = AsyncMock(return_value="ok")
    tool = SimpleNamespace(
        name="search",
        description="Search indexed traces",
        tool_call_schema=RuntimeArgs,
        args_schema=RuntimeArgs,
        ainvoke=invoke,
    )
    runtime = _make_runtime(tools=[tool])

    result = await run_tool.coroutine(
        tool_name="search",
        tool_args={"query": "slow rank"},
        runtime=runtime,
    )

    assert result == "ok"
    invoke.assert_awaited_once()
    call_args = invoke.await_args.args[0]
    assert call_args["query"] == "slow rank"
    assert call_args["runtime"] is runtime


@pytest.mark.asyncio
async def test_run_tool_not_found_returns_error_payload() -> None:
    result = await run_tool.coroutine(
        tool_name="missing_tool",
        tool_args={},
        runtime=_make_runtime(tools=[]),
    )

    assert result["error"] == "Tool 'missing_tool' not found"
    assert "fetch_tools" in result["available_tools"]


@pytest.mark.asyncio
async def test_fetch_skills_returns_display_name_and_filters(tmp_path: Path) -> None:
    skills = [
        Skill(
            name="cluster-fast-slow-rank-detector",
            description="Detect slow ranks in distributed runs",
            category="analysis",
            path=tmp_path / "analysis" / "cluster-fast-slow-rank-detector" / "SKILL.md",
        ),
        Skill(
            name="op-mfu-calculator",
            description="Compute operator MFU",
            category="default",
            path=tmp_path / "op-mfu-calculator" / "SKILL.md",
        ),
    ]

    result = await fetch_skills.coroutine(
        runtime=_make_runtime(skills=skills),
        pattern="analysis|mfu",
    )
    payload = json.loads(result)

    assert payload == [
        {
            "display_name": "analysis/cluster-fast-slow-rank-detector",
            "category": "analysis",
            "name": "cluster-fast-slow-rank-detector",
            "description": "Detect slow ranks in distributed runs",
        },
        {
            "display_name": "op-mfu-calculator",
            "category": "default",
            "name": "op-mfu-calculator",
            "description": "Compute operator MFU",
        },
    ]


@pytest.mark.asyncio
async def test_fetch_skills_supports_dict_context_shape(tmp_path: Path) -> None:
    skills = [
        Skill(
            name="op-mfu-calculator",
            description="Compute operator MFU",
            category="default",
            path=tmp_path / "op-mfu-calculator" / "SKILL.md",
        ),
    ]

    result = await fetch_skills.coroutine(
        runtime=_make_runtime_with_dict_context(skills=skills),
        pattern="mfu",
    )
    payload = json.loads(result)

    assert payload[0]["name"] == "op-mfu-calculator"


@pytest.mark.asyncio
async def test_fetch_skills_falls_back_to_disk_when_runtime_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: fallback test\n---\ncontent",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        SkillFactory,
        "get_default_skills_dir",
        staticmethod(lambda: tmp_path / "missing-default"),
    )
    monkeypatch.setattr(initializer, "cached_agent_skills", [])

    runtime = SimpleNamespace(context=SimpleNamespace(skill_catalog=[]))
    result = await fetch_skills.coroutine(runtime=runtime, pattern="demo-skill")
    payload = json.loads(result)

    assert payload[0]["name"] == "demo-skill"


@pytest.mark.asyncio
async def test_fetch_skills_falls_back_to_initializer_cache_when_runtime_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill = Skill(
        name="cached-skill",
        description="from initializer cache",
        category="default",
        path=tmp_path / "skills" / "cached-skill" / "SKILL.md",
    )
    monkeypatch.setattr(initializer, "cached_agent_skills", [skill])
    monkeypatch.setattr(
        SkillFactory,
        "load_skills",
        AsyncMock(side_effect=AssertionError("disk fallback should not be used")),
    )

    result = await fetch_skills.coroutine(runtime=None, pattern="cached-skill")
    payload = json.loads(result)

    assert payload == [
        {
            "display_name": "cached-skill",
            "category": "default",
            "name": "cached-skill",
            "description": "from initializer cache",
        }
    ]


@pytest.mark.asyncio
async def test_get_skill_requires_category_when_names_are_duplicated(
    tmp_path: Path,
) -> None:
    alpha_dir = tmp_path / "analysis" / "shared-skill"
    beta_dir = tmp_path / "debug" / "shared-skill"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)
    (alpha_dir / "SKILL.md").write_text(
        "---\nname: shared-skill\ndescription: analysis skill\n---\nalpha",
        encoding="utf-8",
    )
    (beta_dir / "SKILL.md").write_text(
        "---\nname: shared-skill\ndescription: debug skill\n---\nbeta",
        encoding="utf-8",
    )

    skills = [
        Skill(
            name="shared-skill",
            description="analysis skill",
            category="analysis",
            path=alpha_dir / "SKILL.md",
        ),
        Skill(
            name="shared-skill",
            description="debug skill",
            category="debug",
            path=beta_dir / "SKILL.md",
        ),
    ]

    with pytest.raises(ToolException, match="Specify category: analysis, debug"):
        await get_skill.coroutine(
            name="shared-skill",
            runtime=_make_runtime(skills=skills),
        )


@pytest.mark.asyncio
async def test_get_skill_reads_selected_skill_content(tmp_path: Path) -> None:
    skill_dir = tmp_path / "analysis" / "rank-detector"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\nname: rank-detector\ndescription: detect slow ranks\n---\nrun steps",
        encoding="utf-8",
    )
    skill = Skill(
        name="rank-detector",
        description="detect slow ranks",
        category="analysis",
        path=skill_path,
    )

    result = await get_skill.coroutine(
        name="rank-detector",
        category="analysis",
        runtime=_make_runtime(skills=[skill]),
    )

    assert "run steps" in result


@pytest.mark.asyncio
async def test_get_skill_falls_back_to_initializer_cache_when_runtime_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_dir = tmp_path / "default" / "cached-skill"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\nname: cached-skill\ndescription: cache skill\n---\nfrom cache",
        encoding="utf-8",
    )

    skill = Skill(
        name="cached-skill",
        description="cache skill",
        category="default",
        path=skill_path,
    )
    monkeypatch.setattr(initializer, "cached_agent_skills", [skill])
    monkeypatch.setattr(
        SkillFactory,
        "load_skills",
        AsyncMock(side_effect=AssertionError("disk fallback should not be used")),
    )

    result = await get_skill.coroutine(
        name="cached-skill",
        category="default",
        runtime=None,
    )

    assert "from cache" in result

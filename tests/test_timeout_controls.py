from __future__ import annotations

import asyncio
import inspect

import pytest
from langchain_core.tools import ToolException, tool

from msagent.configs import LLMConfig, ToolsConfig
from msagent.tools.catalog.skills import fetch_skills
from msagent.tools.factory import ToolFactory


def test_default_timeout_settings_are_applied() -> None:
    llm = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        alias="default",
        max_tokens=0,
        temperature=0.0,
        streaming=True,
    )
    tools = ToolsConfig()

    assert llm.request_timeout_seconds == 120
    assert tools.execution_timeout_seconds == 300


@tool("slow_tool")
async def _slow_tool(*, seconds: float = 0.2) -> str:
    """Sleep for a while and return done."""
    await asyncio.sleep(seconds)
    return "done"


@pytest.mark.asyncio
async def test_tool_factory_wraps_timeout_with_user_readable_error() -> None:
    wrapped = ToolFactory().wrap_tool_with_timeout(
        _slow_tool,
        timeout_seconds=0.05,
        source="unit-test",
    )

    with pytest.raises(ToolException, match="timed out after 0s|timed out after 1s|timed out"):
        await wrapped.ainvoke({"seconds": 0.2})


def test_timeout_wrapper_preserves_runtime_parameter_for_injected_context() -> None:
    wrapped = ToolFactory().wrap_tool_with_timeout(
        fetch_skills,
        timeout_seconds=1.0,
        source="unit-test",
    )

    assert wrapped.coroutine is not None
    assert "runtime" in inspect.signature(wrapped.coroutine).parameters

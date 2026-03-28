from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest
from langchain_core.tools import ToolException

from msagent.agents.context import AgentContext
from msagent.configs import ApprovalMode
from msagent.middlewares.approval import ALLOW, ApprovalMiddleware
from msagent.utils.render import TOOL_TIMING_RESPONSE_METADATA_KEY


@pytest.mark.asyncio
async def test_approval_middleware_attaches_tool_timing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ApprovalMiddleware()
    monkeypatch.setattr(middleware, "_handle_approval", lambda request: ALLOW)

    request = SimpleNamespace(
        tool_call={"name": "run_command", "id": "call-1"},
        runtime=SimpleNamespace(
            context=AgentContext(
                approval_mode=ApprovalMode.ACTIVE,
                working_dir=tmp_path,
            )
        ),
        tool=None,
    )

    async def handler(_request) -> str:
        await asyncio.sleep(0.01)
        return "ok"

    result = await middleware.awrap_tool_call(request, handler)

    timing = result.response_metadata[TOOL_TIMING_RESPONSE_METADATA_KEY]
    assert timing["duration_seconds"] >= 0.0
    assert timing["finished_at"] >= timing["started_at"]


@pytest.mark.asyncio
async def test_approval_middleware_preserves_tool_exception_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ApprovalMiddleware()
    monkeypatch.setattr(middleware, "_handle_approval", lambda request: ALLOW)

    request = SimpleNamespace(
        tool_call={"name": "run_command", "id": "call-1"},
        runtime=SimpleNamespace(
            context=AgentContext(
                approval_mode=ApprovalMode.ACTIVE,
                working_dir=tmp_path,
            )
        ),
        tool=None,
    )

    async def handler(_request) -> str:
        raise ToolException("Command timed out after 30s\n\nPartial output before timeout:\n[stdout]\nstep 1")

    result = await middleware.awrap_tool_call(request, handler)

    assert getattr(result, "is_error", False) is True
    assert result.content.startswith("Command timed out after 30s")
    assert "Failed to execute tool:" not in result.content

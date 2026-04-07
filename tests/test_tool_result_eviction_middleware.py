from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from msagent.middlewares.tool_result_eviction import ToolResultEvictionMiddleware


class _FakeBackend:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    def write(self, path: str, content: str):
        self.writes.append((path, content))
        return SimpleNamespace(error=None, files_update={path: {"content": content}})

    async def awrite(self, path: str, content: str):
        self.writes.append((path, content))
        return SimpleNamespace(error=None, files_update={path: {"content": content}})


def _make_request(tool_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={"name": tool_name},
        runtime=SimpleNamespace(),
    )


def _non_string_tool_message() -> ToolMessage:
    # Bypass validation to emulate providers/toolchains that surface structured content.
    return ToolMessage.model_construct(
        content=[{"file": "op_statistic.csv", "total_rows": 123}],
        tool_call_id="call-structured",
        name="run_tool",
    )


def test_evicts_large_tool_result_for_non_excluded_tool() -> None:
    backend = _FakeBackend()
    middleware = ToolResultEvictionMiddleware(
        backend=backend,
        tool_token_limit_before_evict=1,
    )
    message = ToolMessage(
        content="x" * 50,
        tool_call_id="call-1",
        name="run_tool",
    )

    result = middleware.wrap_tool_call(
        _make_request("run_tool"),
        lambda _request: message,
    )

    assert isinstance(result, Command)
    update = result.update or {}
    assert backend.writes
    assert backend.writes[0][0].startswith("/large_tool_results/")
    evicted_message = update["messages"][0]
    assert isinstance(evicted_message, ToolMessage)
    assert "Tool result too large" in str(evicted_message.content)


def test_skips_eviction_for_excluded_tool() -> None:
    backend = _FakeBackend()
    middleware = ToolResultEvictionMiddleware(
        backend=backend,
        tool_token_limit_before_evict=1,
    )
    message = ToolMessage(
        content="x" * 50,
        tool_call_id="call-1",
        name="read_file",
    )

    result = middleware.wrap_tool_call(
        _make_request("read_file"),
        lambda _request: message,
    )

    assert isinstance(result, ToolMessage)
    assert not backend.writes


@pytest.mark.asyncio
async def test_async_evicts_large_tool_result() -> None:
    backend = _FakeBackend()
    middleware = ToolResultEvictionMiddleware(
        backend=backend,
        tool_token_limit_before_evict=1,
    )
    message = ToolMessage(
        content="x" * 50,
        tool_call_id="call-async",
        name="run_tool",
    )

    async def _handler(_request):
        return message

    result = await middleware.awrap_tool_call(_make_request("run_tool"), _handler)

    assert isinstance(result, Command)
    assert backend.writes


def test_normalizes_non_string_tool_message_content() -> None:
    backend = _FakeBackend()
    middleware = ToolResultEvictionMiddleware(
        backend=backend,
        tool_token_limit_before_evict=10_000,
    )
    message = _non_string_tool_message()

    result = middleware.wrap_tool_call(
        _make_request("run_tool"),
        lambda _request: message,
    )

    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, str)
    assert "op_statistic.csv" in result.content


def test_normalizes_non_string_content_inside_command_updates() -> None:
    command_like = SimpleNamespace(update={"messages": [_non_string_tool_message()]})

    normalized = ToolResultEvictionMiddleware._normalize_tool_result(command_like)

    assert normalized is command_like
    message = command_like.update["messages"][0]
    assert isinstance(message.content, str)
    assert "total_rows" in message.content

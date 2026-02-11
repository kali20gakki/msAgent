"""Tests for llm module."""

from __future__ import annotations

import json
from typing import Any

import pytest

from msagent.config import LLMConfig
from msagent.llm import (
    AnthropicClient,
    GeminiClient,
    Message,
    OpenAIClient,
    create_llm_client,
)


class FakeHTTPResponse:
    """Fake HTTP response supporting both normal and streaming modes."""

    def __init__(self, payload: dict[str, Any] | None = None, lines: list[str] | None = None) -> None:
        self.payload = payload or {}
        self.lines = lines or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def __aenter__(self) -> "FakeHTTPResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeAsyncHTTPClient:
    """Fake async client capturing requests."""

    def __init__(self, post_response: FakeHTTPResponse, stream_response: FakeHTTPResponse | None = None) -> None:
        self.post_response = post_response
        self.stream_response = stream_response or FakeHTTPResponse()
        self.last_post: tuple[str, dict[str, Any]] | None = None
        self.last_stream: tuple[str, str, dict[str, Any]] | None = None

    async def post(self, url: str, json: dict[str, Any]) -> FakeHTTPResponse:
        self.last_post = (url, json)
        return self.post_response

    def stream(self, method: str, url: str, json: dict[str, Any]) -> FakeHTTPResponse:
        self.last_stream = (method, url, json)
        return self.stream_response


@pytest.mark.asyncio
async def test_openai_chat_parses_content_and_payload() -> None:
    config = LLMConfig(provider="openai", api_key="key", model="gpt-test")
    client = OpenAIClient(config)
    await client.client.aclose()

    fake_client = FakeAsyncHTTPClient(
        post_response=FakeHTTPResponse(
            {"choices": [{"message": {"content": "hello from openai"}}]}
        )
    )
    client.client = fake_client

    tools = [{"type": "function", "function": {"name": "calc"}}]
    result = await client.chat([Message("user", "hi")], tools=tools)

    assert result == "hello from openai"
    assert fake_client.last_post is not None
    assert fake_client.last_post[0] == "/chat/completions"
    assert fake_client.last_post[1]["tools"] == tools
    assert fake_client.last_post[1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openai_chat_stream_yields_incremental_chunks() -> None:
    config = LLMConfig(provider="openai", api_key="key", model="gpt-test")
    client = OpenAIClient(config)
    await client.client.aclose()

    lines = [
        'data: {"choices":[{"delta":{"content":"hello "}}]}',
        "data: not-json",
        'data: {"choices":[{"delta":{"content":"world"}}]}',
        "data: [DONE]",
    ]
    fake_client = FakeAsyncHTTPClient(
        post_response=FakeHTTPResponse(),
        stream_response=FakeHTTPResponse(lines=lines),
    )
    client.client = fake_client

    chunks = [chunk async for chunk in client.chat_stream([Message("user", "hi")])]
    assert "".join(chunks) == "hello world"


@pytest.mark.asyncio
async def test_anthropic_convert_messages_splits_system_and_conversation() -> None:
    config = LLMConfig(provider="anthropic", api_key="key", model="claude-test")
    client = AnthropicClient(config)

    system, messages = client._convert_messages(
        [Message("system", "rules"), Message("user", "hello"), Message("assistant", "hi")]
    )
    await client.client.aclose()

    assert system == "rules"
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_maps_tool_use_block() -> None:
    config = LLMConfig(provider="anthropic", api_key="key", model="claude-test")
    client = AnthropicClient(config)
    await client.client.aclose()

    fake_client = FakeAsyncHTTPClient(
        post_response=FakeHTTPResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "calculator",
                        "input": {"a": 1, "b": 2},
                    }
                ]
            }
        )
    )
    client.client = fake_client

    result = await client.chat_with_tools([Message("user", "sum")], tools=[{"name": "calculator"}])

    assert result["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_gemini_convert_tools_and_map_function_call() -> None:
    config = LLMConfig(provider="gemini", api_key="key", model="gemini-test")
    client = GeminiClient(config)
    await client.client.aclose()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search docs",
                "parameters": {"type": "object"},
            },
        }
    ]
    converted = client._convert_tools(tools)
    assert converted == [
        {
            "function_declarations": [
                {
                    "name": "search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object"},
                }
            ]
        }
    ]

    fake_client = FakeAsyncHTTPClient(
        post_response=FakeHTTPResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search_docs",
                                        "args": {"query": "pytest"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    client.client = fake_client

    result = await client.chat_with_tools([Message("user", "find pytest docs")], tools=tools)

    assert result["tool_calls"][0]["function"]["name"] == "search_docs"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"query": "pytest"}


@pytest.mark.asyncio
async def test_message_to_dict_and_factory() -> None:
    message = Message("user", "hello")
    assert message.to_dict() == {"role": "user", "content": "hello"}

    created_clients = [
        create_llm_client(LLMConfig(provider="openai", api_key="k")),
        create_llm_client(LLMConfig(provider="anthropic", api_key="k")),
        create_llm_client(LLMConfig(provider="gemini", api_key="k")),
        create_llm_client(LLMConfig(provider="custom", api_key="k")),
    ]

    assert isinstance(created_clients[0], OpenAIClient)
    assert isinstance(created_clients[1], AnthropicClient)
    assert isinstance(created_clients[2], GeminiClient)
    assert isinstance(created_clients[3], OpenAIClient)

    for client in created_clients:
        await client.client.aclose()

    invalid = LLMConfig(provider="openai", api_key="k")
    invalid.provider = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported provider"):
        create_llm_client(invalid)

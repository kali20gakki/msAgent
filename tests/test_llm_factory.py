from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from msagent.configs import LLMConfig, LLMProvider
from msagent.core.settings import LLMSettings
from msagent.llms.factory import LLMFactory
from msagent.llms.wrappers.custom import ChatCustomHTTP


def test_custom_http_wrapper_posts_plain_payload_and_parses_tool_calls() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "custom-model",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ping",
                                        "arguments": '{"text":"pong"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = ChatCustomHTTP(
        url="https://custom.example.com/api/chat",
        api_key=SecretStr("secret-key"),
        model="custom-model",
        temperature=0.3,
        max_tokens=256,
        http_client=client,
    )

    tool_schema = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Ping tool",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }

    response = llm.bind_tools([tool_schema], tool_choice="ping").invoke("hello")

    assert captured["url"] == "https://custom.example.com/api/chat"
    assert captured["headers"]["authorization"] == "Bearer secret-key"
    assert captured["body"] == {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.3,
        "max_tokens": 256,
        "tools": [tool_schema],
        "tool_choice": {
            "type": "function",
            "function": {"name": "ping"},
        },
    }
    assert response.tool_calls == [
        {"name": "ping", "args": {"text": "pong"}, "id": "call_1", "type": "tool_call"}
    ]
    assert response.response_metadata["usage"]["prompt_tokens"] == 11
    assert response.response_metadata["finish_reason"] == "tool_calls"


def test_factory_returns_custom_http_wrapper() -> None:
    factory = LLMFactory(
        LLMSettings(
            custom_api_key=SecretStr("settings-key"),
            custom_base_url="https://custom.example.com/request",
        )
    )
    config = LLMConfig(
        provider=LLMProvider.CUSTOM,
        model="custom-model",
        max_tokens=128,
        temperature=0.0,
        streaming=True,
    )

    llm = factory.create(config)

    assert isinstance(llm, ChatCustomHTTP)
    assert llm.url == "https://custom.example.com/request"
    assert llm.api_key.get_secret_value() == "settings-key"
    assert llm.disable_streaming is True
    assert llm.streaming is True


def test_factory_custom_requires_url() -> None:
    factory = LLMFactory(LLMSettings(custom_api_key=SecretStr("settings-key")))
    config = LLMConfig(
        provider=LLMProvider.CUSTOM,
        model="custom-model",
        max_tokens=128,
        temperature=0.0,
        streaming=True,
    )

    with pytest.raises(
        ValueError,
        match="CUSTOM provider requires base_url or CUSTOM_BASE_URL to be set",
    ):
        factory.create(config)

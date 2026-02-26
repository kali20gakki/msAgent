"""Tests for llm module (deepagents)."""

from __future__ import annotations

from typing import Any

import pytest

import msagent.llm as llm_module
from msagent.config import LLMConfig
from msagent.llm import DeepAgentsClient, Message, create_llm_client


class FakeAgent:
    def __init__(self, result: Any, events: list[dict[str, Any]] | None = None) -> None:
        self.result = result
        self.events = events or []

    async def ainvoke(self, _payload: dict[str, Any]) -> Any:
        return self.result

    async def astream_events(self, _payload: dict[str, Any], version: str = "v2"):
        assert version == "v2"
        for event in self.events:
            yield event


def _patch_model_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DeepAgentsClient, "_build_model", lambda self, _cfg: object())


def test_message_to_dict() -> None:
    assert Message("user", "hello").to_dict() == {"role": "user", "content": "hello"}


def test_create_llm_client_returns_deepagents_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)

    clients = [
        create_llm_client(LLMConfig(provider="openai", api_key="k", model="m")),
        create_llm_client(LLMConfig(provider="anthropic", api_key="k", model="m")),
        create_llm_client(LLMConfig(provider="gemini", api_key="k", model="m")),
        create_llm_client(LLMConfig(provider="custom", api_key="k", model="m")),
    ]

    assert all(isinstance(c, DeepAgentsClient) for c in clients)


def test_create_llm_client_unsupported_provider_raises() -> None:
    invalid = LLMConfig(provider="openai", api_key="k", model="m")
    invalid.provider = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported provider"):
        DeepAgentsClient(invalid)


@pytest.mark.asyncio
async def test_chat_extracts_final_assistant_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    fake_agent = FakeAgent(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello from deepagents"},
            ]
        }
    )
    monkeypatch.setattr(llm_module, "create_deep_agent", lambda **_kwargs: fake_agent)

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    text = await client.chat([Message("system", "rules"), Message("user", "hi")], tools=[])

    assert text == "hello from deepagents"


@pytest.mark.asyncio
async def test_chat_stream_reads_chunk_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    fake_agent = FakeAgent(
        result={"messages": [{"role": "assistant", "content": "fallback"}]},
        events=[
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": [{"text": "one "}]}}},
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": [{"text": "two"}]}}},
        ],
    )
    monkeypatch.setattr(llm_module, "create_deep_agent", lambda **_kwargs: fake_agent)

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    chunks = [c async for c in client.chat_stream([Message("user", "hi")], tools=[])]

    assert "".join(chunks) == "one two"


@pytest.mark.asyncio
async def test_chat_stream_fallbacks_to_chat_when_no_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    fake_agent = FakeAgent(
        result={"messages": [{"role": "assistant", "content": "fallback-text"}]},
        events=[],
    )
    monkeypatch.setattr(llm_module, "create_deep_agent", lambda **_kwargs: fake_agent)

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    chunks = [c async for c in client.chat_stream([Message("user", "hi")], tools=[])]

    assert "".join(chunks) == "fallback-text"


@pytest.mark.asyncio
async def test_chat_passes_skills_and_memory_to_create_deep_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model_build(monkeypatch)
    captured: dict[str, Any] = {}

    fake_agent = FakeAgent(
        {
            "messages": [
                {"role": "assistant", "content": "ok"},
            ]
        }
    )

    def _fake_create_deep_agent(**kwargs: Any):
        captured.update(kwargs)
        return fake_agent

    monkeypatch.setattr(llm_module, "create_deep_agent", _fake_create_deep_agent)

    client = DeepAgentsClient(
        LLMConfig(provider="openai", api_key="k", model="m"),
        skills=["/skills/user/", "/skills/project/"],
        memory=["/memory/AGENTS.md"],
    )
    text = await client.chat([Message("user", "hi")], tools=[])

    assert text == "ok"
    assert captured["skills"] == ["/skills/user/", "/skills/project/"]
    assert captured["memory"] == ["/memory/AGENTS.md"]

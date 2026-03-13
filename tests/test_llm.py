"""Tests for llm module (deepagents)."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest

import msagent.llm as llm_module
from deepagents.backends import LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from msagent.config import LLMConfig
from msagent.llm import DeepAgentsClient, Message, create_llm_client


class FakeAgent:
    def __init__(self, result: Any, events: list[dict[str, Any]] | None = None) -> None:
        self.result = result
        self.events = events or []
        self.ainvoke_calls = 0

    async def ainvoke(self, _payload: dict[str, Any], **_kwargs: Any) -> Any:
        self.ainvoke_calls += 1
        return self.result

    async def astream_events(
        self,
        _payload: dict[str, Any],
        version: str = "v2",
        **_kwargs: Any,
    ):
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
async def test_chat_stream_events_filters_nested_tool_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    fake_agent = FakeAgent(
        result={"messages": [{"role": "assistant", "content": "should-not-fallback"}]},
        events=[
            {
                "event": "on_tool_start",
                "name": "msprof-mcp__msprof_analyze_advisor",
                "data": {"input": {"mode": "all"}},
            },
            {
                "event": "on_tool_start",
                "name": "msprof_analyze_advisor",
                "data": {"input": {"mode": "all"}},
            },
            {
                "event": "on_tool_end",
                "name": "msprof_analyze_advisor",
                "data": {"output": "nested-output"},
            },
            {
                "event": "on_tool_end",
                "name": "msprof-mcp__msprof_analyze_advisor",
                "data": {"output": "tool-output"},
            },
            {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": [{"role": "assistant", "content": "final-from-event"}]
                    }
                },
            },
        ],
    )
    monkeypatch.setattr(llm_module, "create_deep_agent", lambda **_kwargs: fake_agent)

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    events = [
        event
        async for event in client.chat_stream_events(
            [Message("user", "hi")],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "msprof-mcp__msprof_analyze_advisor"},
                }
            ],
        )
    ]

    tool_starts = [event for event in events if event.get("type") == "tool_start"]
    tool_ends = [event for event in events if event.get("type") == "tool_end"]
    text_events = [event for event in events if event.get("type") == "text"]

    assert tool_starts == [
        {
            "type": "tool_start",
            "name": "msprof-mcp__msprof_analyze_advisor",
            "input": {"mode": "all"},
        }
    ]
    assert tool_ends == [
        {
            "type": "tool_end",
            "name": "msprof-mcp__msprof_analyze_advisor",
            "output": "tool-output",
        }
    ]
    assert text_events == [{"type": "text", "content": "final-from-event"}]
    assert fake_agent.ainvoke_calls == 0


@pytest.mark.asyncio
async def test_chat_stream_events_does_not_fallback_after_tool_only_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model_build(monkeypatch)
    fake_agent = FakeAgent(
        result={"messages": [{"role": "assistant", "content": "should-not-run"}]},
        events=[
            {
                "event": "on_tool_start",
                "name": "msprof-mcp__msprof_analyze_advisor",
                "data": {"input": {"mode": "all"}},
            },
            {
                "event": "on_tool_end",
                "name": "msprof-mcp__msprof_analyze_advisor",
                "data": {"output": "tool-output"},
            },
        ],
    )
    monkeypatch.setattr(llm_module, "create_deep_agent", lambda **_kwargs: fake_agent)

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    events = [
        event
        async for event in client.chat_stream_events(
            [Message("user", "hi")],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "msprof-mcp__msprof_analyze_advisor"},
                }
            ],
        )
    ]

    assert [event["type"] for event in events] == ["tool_start", "tool_end"]
    assert fake_agent.ainvoke_calls == 0


@pytest.mark.asyncio
async def test_build_structured_tool_uses_injected_tool_invoker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model_build(monkeypatch)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _tool_invoker(name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, arguments))
        return "tool-ok"

    client = DeepAgentsClient(
        LLMConfig(provider="openai", api_key="k", model="m"),
        tool_invoker=_tool_invoker,
    )
    tool = client._build_structured_tool(
        {
            "type": "function",
            "function": {
                "name": "builtin__execute_python",
                "description": "run python",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    )

    result = await tool.ainvoke({"code": "print(1)"})

    assert result == "tool-ok"
    assert calls == [("builtin__execute_python", {"code": "print(1)"})]


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
    assert isinstance(captured["backend"], FilesystemBackend)
    assert captured["backend"].virtual_mode is True


def test_client_uses_workspace_root_for_filesystem_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    workspace_root = Path("/tmp/msagent-workspace")
    client = DeepAgentsClient(
        LLMConfig(provider="openai", api_key="k", model="m"),
        workspace_root=workspace_root,
    )

    assert isinstance(client._backend, FilesystemBackend)
    assert client._backend.cwd == workspace_root.resolve()
    assert client._backend.virtual_mode is True


def test_client_can_use_local_shell_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    workspace_root = Path("/tmp/msagent-workspace")

    client = DeepAgentsClient(
        LLMConfig(provider="openai", api_key="k", model="m"),
        workspace_root=workspace_root,
        backend_mode="local_shell",
    )

    assert isinstance(client._backend, LocalShellBackend)
    assert client._backend.cwd == workspace_root.resolve()
    assert client._backend.virtual_mode is True


def test_count_tokens_uses_model_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def get_num_tokens_from_messages(self, messages, tools=None) -> int:
            assert len(messages) == 2
            assert tools is None
            return 123

    monkeypatch.setattr(DeepAgentsClient, "_build_model", lambda self, _cfg: FakeModel())

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    count = client.count_tokens(
        [Message("system", "rules"), Message("user", "hi")],
        tools=[{"type": "function", "function": {"name": "sum"}}],
    )

    assert count > 123


def test_count_tokens_falls_back_to_estimate_when_model_counter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def get_num_tokens_from_messages(self, messages, tools=None) -> int:
            raise RuntimeError("no tokenizer")

    monkeypatch.setattr(DeepAgentsClient, "_build_model", lambda self, _cfg: FakeModel())

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    count = client.count_tokens([Message("user", "hello world")], tools=None)

    assert isinstance(count, int)
    assert count > 0


def test_local_shell_backend_uses_minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_build(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("HOME", "/Users/tester")

    client = DeepAgentsClient(
        LLMConfig(provider="openai", api_key="k", model="m"),
        backend_mode="local_shell",
    )

    assert isinstance(client._backend, LocalShellBackend)
    assert client._backend._env["PATH"] == "/usr/local/bin:/usr/bin"
    assert client._backend._env["HOME"] == "/Users/tester"
    assert client._backend._env["PYTHONIOENCODING"] == "utf-8"
    assert "OPENAI_API_KEY" not in client._backend._env


@pytest.mark.asyncio
async def test_chat_stream_events_emits_builtin_execute_tool_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model_build(monkeypatch)
    fake_agent = FakeAgent(
        result={"messages": [{"role": "assistant", "content": "done"}]},
        events=[
            {
                "event": "on_tool_start",
                "name": "execute",
                "data": {"input": {"command": "pwd"}},
            },
            {
                "event": "on_tool_end",
                "name": "execute",
                "data": {"output": "ok"},
            },
        ],
    )
    monkeypatch.setattr(llm_module, "create_deep_agent", lambda **_kwargs: fake_agent)

    client = DeepAgentsClient(LLMConfig(provider="openai", api_key="k", model="m"))
    events = [event async for event in client.chat_stream_events([Message("user", "run")])]

    assert events == [
        {"type": "tool_start", "name": "execute", "input": {"command": "pwd"}},
        {"type": "tool_end", "name": "execute", "output": "ok"},
    ]


def test_build_model_uses_resolved_max_tokens_for_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_chat_openai(**kwargs: Any):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "ChatOpenAI", _fake_chat_openai)

    DeepAgentsClient(
        LLMConfig(
            provider="openai",
            api_key="k",
            model="deepseek-chat",
            max_tokens=0,
        )
    )

    assert "max_completion_tokens" not in captured


def test_build_model_normalizes_responses_endpoint_and_enables_responses_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_chat_openai(**kwargs: Any):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "ChatOpenAI", _fake_chat_openai)

    DeepAgentsClient(
        LLMConfig(
            provider="openai",
            api_key="k",
            model="gpt-5.2",
            base_url="https://gmn.chuangzuoli.com/v1/responses",
        )
    )

    assert captured["base_url"] == "https://gmn.chuangzuoli.com/v1"
    assert captured["use_responses_api"] is True


def test_build_model_normalizes_chat_completions_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_chat_openai(**kwargs: Any):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "ChatOpenAI", _fake_chat_openai)

    DeepAgentsClient(
        LLMConfig(
            provider="custom",
            api_key="k",
            model="deepseek-chat",
            base_url="https://example.com/v1/chat/completions",
        )
    )

    assert captured["base_url"] == "https://example.com/v1"
    assert "use_responses_api" not in captured

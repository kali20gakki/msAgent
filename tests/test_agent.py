"""Tests for agent module."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import msagent.agent as agent_module
from msagent.agent import Agent
from msagent.config import AppConfig, MCPConfig


class FakeMCPManager:
    """Test double for global MCP manager."""

    def __init__(self) -> None:
        self.added_servers: list[str] = []
        self.connected_servers: list[str] = []
        self.tools: list[dict[str, Any]] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.disconnect_called = False
        self.call_result = "tool-result"

    async def add_server(self, config: MCPConfig) -> bool:
        self.added_servers.append(config.name)
        return True

    def get_connected_servers(self) -> list[str]:
        return self.connected_servers

    def get_all_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.tool_calls.append((tool_name, arguments))
        return self.call_result

    async def disconnect_all(self) -> None:
        self.disconnect_called = True


class FakeLLMClient:
    """Test double for LLM client."""

    def __init__(self) -> None:
        self.chat_response = "assistant-response"
        self.stream_chunks = ["chunk-1", "chunk-2"]

    async def chat(self, messages: list[Any], tools: list[dict] | None = None) -> str:
        return self.chat_response

    async def chat_stream(self, messages: list[Any], tools: list[dict] | None = None):
        for chunk in self.stream_chunks:
            yield chunk
            await asyncio.sleep(0)


def make_config(api_key: str = "test-key", mcp_servers: list[MCPConfig] | None = None) -> AppConfig:
    config = AppConfig()
    config.llm.api_key = api_key
    config.mcp_servers = mcp_servers or []
    return config


@pytest.mark.asyncio
async def test_initialize_fails_when_llm_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    config = make_config(api_key="")
    agent = Agent(config)

    initialized = await agent.initialize()

    assert initialized is False
    assert "LLM not configured" in agent.error_message


@pytest.mark.asyncio
async def test_initialize_loads_only_enabled_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_llm = FakeLLMClient()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)
    monkeypatch.setattr(agent_module, "create_llm_client", lambda _cfg: fake_llm)

    config = make_config(
        api_key="configured",
        mcp_servers=[
            MCPConfig(name="enabled", command="python", enabled=True),
            MCPConfig(name="disabled", command="python", enabled=False),
        ],
    )
    agent = Agent(config)

    initialized = await agent.initialize()

    assert initialized is True
    assert agent.is_initialized is True
    assert agent.llm_client is fake_llm
    assert fake_mcp.added_servers == ["enabled"]


def test_get_system_prompt_includes_connected_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.connected_servers = ["alpha", "beta"]
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    prompt = agent.get_system_prompt()

    assert "alpha, beta" in prompt
    assert "You are msagent" in prompt


@pytest.mark.asyncio
async def test_chat_returns_error_if_not_initialized() -> None:
    agent = Agent(make_config())
    result = await agent.chat("hello")
    assert "Agent not initialized" in result


@pytest.mark.asyncio
async def test_chat_without_tools_uses_plain_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = []
    fake_llm = FakeLLMClient()
    fake_llm.chat_response = "plain-response"
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    result = await agent.chat("hi")

    assert result == "plain-response"
    assert [message.role for message in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].content == "plain-response"


@pytest.mark.asyncio
async def test_chat_with_tools_passes_tools_through_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = [{"type": "function", "function": {"name": "calc__sum"}}]
    fake_llm = FakeLLMClient()
    fake_llm.chat_response = "The answer is 42"
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    result = await agent.chat("what is 40 + 2?")

    assert result == "The answer is 42"
    assert fake_mcp.tool_calls == []
    assert [message.role for message in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].content == "The answer is 42"


@pytest.mark.asyncio
async def test_chat_stream_with_tools_yields_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = [{"type": "function", "function": {"name": "calc__sum"}}]
    fake_llm = FakeLLMClient()
    fake_llm.stream_chunks = ["one ", "two"]
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    chunks = [chunk async for chunk in agent.chat_stream("sum 1 and 9")]

    assert "".join(chunks[:2]) == "one two"
    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].content == "one two"


def test_clear_history_and_get_history_copy() -> None:
    agent = Agent(make_config())
    agent.messages = [agent_module.Message("user", "hello")]

    history = agent.get_history()
    history.append(agent_module.Message("assistant", "world"))

    assert len(agent.messages) == 1
    agent.clear_history()
    assert agent.messages == []


@pytest.mark.asyncio
async def test_shutdown_disconnects_mcp_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True

    await agent.shutdown()

    assert fake_mcp.disconnect_called is True
    assert agent.is_initialized is False

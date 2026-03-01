"""Tests for mcp_client module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import msagent.mcp_client as mcp_module
from msagent.config import MCPConfig
from msagent.mcp_client import MCPClient, MCPManager


@pytest.mark.asyncio
async def test_mcp_client_connect_passes_empty_args_to_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAdapter:
        def __init__(self, *, connections: dict[str, Any], **_: Any) -> None:
            captured["connections"] = connections

    monkeypatch.setattr(mcp_module, "MultiServerMCPClient", FakeAdapter)
    monkeypatch.setattr(MCPClient, "_fetch_tools", AsyncMock())

    client = MCPClient(MCPConfig(name="srv", command="msprof-mcp"))
    connected = await client.connect()

    assert connected is True
    assert captured["connections"]["srv"]["args"] == []


@pytest.mark.asyncio
async def test_mcp_client_call_tool_returns_combined_text_and_images() -> None:
    client = MCPClient(MCPConfig(name="test", command="python"))
    client._connected = True
    fake_tool = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=[
                {"type": "text", "text": "hello"},
                {"type": "image", "mime_type": "image/png"},
            ]
        )
    )
    client._tool_map["tool"] = fake_tool

    result = await client.call_tool("tool", {"x": 1})

    assert result == "hello\n[Image: image/png]"
    fake_tool.ainvoke.assert_awaited_once_with({"x": 1})


@pytest.mark.asyncio
async def test_mcp_client_call_tool_requires_connection() -> None:
    client = MCPClient(MCPConfig(name="test", command="python"))
    client._connected = False
    result = await client.call_tool("tool", {})
    assert result == "Error: Not connected to MCP server"


@pytest.mark.asyncio
async def test_mcp_client_fetch_tools_populates_openai_tool_schema() -> None:
    fake_tool = SimpleNamespace(
        name="sum",
        description="Add numbers",
        args_schema={"type": "object"},
    )
    client = MCPClient(MCPConfig(name="test", command="python"))
    client._adapter_client = SimpleNamespace(get_tools=AsyncMock(return_value=[fake_tool]))

    await client._fetch_tools()

    tools = client.get_tools()
    assert tools[0]["function"]["name"] == "sum"
    assert tools[0]["function"]["description"] == "Add numbers"
    assert tools[0]["function"]["parameters"] == {"type": "object"}


def test_mcp_manager_get_all_tools_prefixes_and_preserves_original() -> None:
    manager = MCPManager()
    original_tool = {
        "type": "function",
        "function": {"name": "echo", "description": "Echo", "parameters": {}},
    }
    fake_client = SimpleNamespace(
        name="srv",
        is_connected=True,
        get_tools=lambda: [original_tool],
    )
    manager.clients["srv"] = fake_client

    tools = manager.get_all_tools()

    assert tools[0]["function"]["name"] == "srv__echo"
    assert original_tool["function"]["name"] == "echo"


@pytest.mark.asyncio
async def test_mcp_manager_call_tool_validates_server_and_format() -> None:
    manager = MCPManager()
    assert await manager.call_tool("invalidname", {}) == "Error: Invalid tool name format: invalidname"
    assert await manager.call_tool("missing__tool", {}) == "Error: MCP server 'missing' not found"

    disconnected = SimpleNamespace(is_connected=False)
    manager.clients["srv"] = disconnected
    assert await manager.call_tool("srv__tool", {}) == "Error: MCP server 'srv' is not connected"


@pytest.mark.asyncio
async def test_mcp_manager_call_tool_routes_to_correct_client() -> None:
    manager = MCPManager()
    fake_client = SimpleNamespace(is_connected=True, call_tool=AsyncMock(return_value="done"))
    manager.clients["srv"] = fake_client

    result = await manager.call_tool("srv__echo", {"msg": "hi"})

    assert result == "done"
    fake_client.call_tool.assert_awaited_once_with("echo", {"msg": "hi"})


@pytest.mark.asyncio
async def test_mcp_manager_add_and_remove_server(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[Any] = []

    class FakeClient:
        def __init__(self, config: MCPConfig) -> None:
            self.config = config
            self.is_connected = True
            self.disconnected = False
            created_clients.append(self)

        async def connect(self) -> bool:
            return True

        async def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(mcp_module, "MCPClient", FakeClient)
    manager = MCPManager()

    added = await manager.add_server(MCPConfig(name="srv", command="python"))
    removed = await manager.remove_server("srv")

    assert added is True
    assert removed is True
    assert created_clients[0].disconnected is True
    assert "srv" not in manager.clients


@pytest.mark.asyncio
async def test_mcp_manager_disconnect_all_clears_clients() -> None:
    manager = MCPManager()
    client_a = SimpleNamespace(disconnect=AsyncMock())
    client_b = SimpleNamespace(disconnect=AsyncMock())
    manager.clients = {"a": client_a, "b": client_b}

    await manager.disconnect_all()

    client_a.disconnect.assert_awaited_once()
    client_b.disconnect.assert_awaited_once()
    assert manager.clients == {}

from __future__ import annotations

from types import SimpleNamespace

import pytest

from msagent.configs import MCPConfig, MCPServerConfig, MCPTransport
from msagent.mcp.client import MCPClient


class _StubToolFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, str]] = []

    def wrap_tool_with_timeout(self, tool, timeout_seconds: float, *, source: str):
        self.calls.append((tool.name, timeout_seconds, source))
        return tool


@pytest.mark.asyncio
async def test_mcp_client_filters_include_exclude_and_enabled_servers(monkeypatch) -> None:
    captured_connections: dict[str, dict] = {}

    class FakeMultiServerMCPClient:
        def __init__(self, connections, *, tool_name_prefix=False):
            captured_connections.update(connections)
            assert tool_name_prefix is True

        async def get_tools(self):
            return [
                SimpleNamespace(name="alpha_ping", description="ping", ainvoke=lambda *_args, **_kwargs: None),
                SimpleNamespace(name="alpha_secret", description="secret", ainvoke=lambda *_args, **_kwargs: None),
                SimpleNamespace(name="beta_info", description="info", ainvoke=lambda *_args, **_kwargs: None),
            ]

    monkeypatch.setattr("msagent.mcp.client.MultiServerMCPClient", FakeMultiServerMCPClient)

    config = MCPConfig(
        servers={
            "alpha": MCPServerConfig(
                command="alpha-server",
                transport=MCPTransport.STDIO,
                include=["ping"],
                exclude=["secret"],
                invoke_timeout=15,
                enabled=True,
            ),
            "beta": MCPServerConfig(
                command="beta-server",
                transport=MCPTransport.STDIO,
                enabled=False,
            ),
        }
    )
    tool_factory = _StubToolFactory()
    client = MCPClient(config, default_invoke_timeout=300, tool_factory=tool_factory)

    tools = await client.tools()

    assert sorted(captured_connections.keys()) == ["alpha"]
    assert [tool.name for tool in tools] == ["alpha_ping"]
    assert tool_factory.calls == [("alpha_ping", 15.0, "mcp:alpha")]
    assert client.module_map == {"alpha_ping": "mcp:alpha"}


@pytest.mark.asyncio
async def test_mcp_client_uses_default_timeout_when_server_timeout_missing(monkeypatch) -> None:
    class FakeMultiServerMCPClient:
        def __init__(self, connections, *, tool_name_prefix=False):
            self.connections = connections

        async def get_tools(self):
            return [SimpleNamespace(name="alpha_ping", description="ping", ainvoke=lambda *_args, **_kwargs: None)]

    monkeypatch.setattr("msagent.mcp.client.MultiServerMCPClient", FakeMultiServerMCPClient)

    config = MCPConfig(
        servers={
            "alpha": MCPServerConfig(
                command="alpha-server",
                transport=MCPTransport.STDIO,
                enabled=True,
            ),
        }
    )
    tool_factory = _StubToolFactory()
    client = MCPClient(config, default_invoke_timeout=123, tool_factory=tool_factory)

    await client.tools()

    assert tool_factory.calls == [("alpha_ping", 123.0, "mcp:alpha")]


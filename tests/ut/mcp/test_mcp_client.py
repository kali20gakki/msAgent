#!/usr/bin/python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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


@pytest.mark.asyncio
async def test_mcp_client_skips_unavailable_server_instead_of_failing(monkeypatch) -> None:
    """One unreachable server must not abort the whole session."""

    class FakeMultiServerMCPClient:
        def __init__(self, connections, *, tool_name_prefix=False):
            assert tool_name_prefix is True
            self.connections = connections

        async def get_tools(self):
            if len(self.connections) > 1:
                # Mirrors the adapter's anyio task-group failure text.
                raise RuntimeError("unhandled errors in a TaskGroup (1 sub-exception)")
            if "beta" in self.connections:
                raise RuntimeError("beta-server: not found")
            return [
                SimpleNamespace(name="alpha_ping", description="ping", ainvoke=lambda *_args, **_kwargs: None),
            ]

    monkeypatch.setattr("msagent.mcp.client.MultiServerMCPClient", FakeMultiServerMCPClient)

    config = MCPConfig(
        servers={
            "alpha": MCPServerConfig(command="alpha-server", transport=MCPTransport.STDIO, enabled=True),
            "beta": MCPServerConfig(command="beta-server", transport=MCPTransport.STDIO, enabled=True),
        }
    )
    client = MCPClient(config, tool_factory=_StubToolFactory())

    tools = await client.tools()

    assert [tool.name for tool in tools] == ["alpha_ping"]
    assert client.module_map == {"alpha_ping": "mcp:alpha"}
    assert list(client.unavailable_servers) == ["beta"]
    assert "not found" in client.unavailable_servers["beta"]


@pytest.mark.asyncio
async def test_mcp_client_survives_every_server_failing(monkeypatch) -> None:
    class FakeMultiServerMCPClient:
        def __init__(self, connections, *, tool_name_prefix=False):
            self.connections = connections

        async def get_tools(self):
            raise RuntimeError("unhandled errors in a TaskGroup (1 sub-exception)")

    monkeypatch.setattr("msagent.mcp.client.MultiServerMCPClient", FakeMultiServerMCPClient)

    config = MCPConfig(
        servers={
            "ascend-doc-mcp": MCPServerConfig(command="msagent-ascend-doc-mcp", enabled=True),
        }
    )
    client = MCPClient(config, tool_factory=_StubToolFactory())

    assert await client.tools() == []
    assert sorted(client.unavailable_servers) == ["ascend-doc-mcp"]


@pytest.mark.asyncio
async def test_mcp_client_reports_no_unavailable_server_on_success(monkeypatch) -> None:
    class FakeMultiServerMCPClient:
        def __init__(self, connections, *, tool_name_prefix=False):
            self.connections = connections

        async def get_tools(self):
            return [
                SimpleNamespace(name="alpha_ping", description="ping", ainvoke=lambda *_args, **_kwargs: None),
            ]

    monkeypatch.setattr("msagent.mcp.client.MultiServerMCPClient", FakeMultiServerMCPClient)

    config = MCPConfig(
        servers={"alpha": MCPServerConfig(command="alpha-server", enabled=True)},
    )
    client = MCPClient(config, tool_factory=_StubToolFactory())

    await client.tools()

    assert client.unavailable_servers == {}

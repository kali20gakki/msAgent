"""MCP client that loads tools from enabled servers in MCPConfig."""

from __future__ import annotations

import importlib
from datetime import timedelta
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from msagent.configs import MCPConfig, MCPServerConfig, MCPTransport
from msagent.tools.factory import ToolFactory

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

try:
    _mcp_client_module: ModuleType | None = importlib.import_module(
        "langchain_mcp_adapters.client"
    )
except ImportError:  # pragma: no cover - fallback for partial local environments
    _mcp_client_module = None

MultiServerMCPClient: Any = (
    getattr(_mcp_client_module, "MultiServerMCPClient", None)
    if _mcp_client_module is not None
    else None
)


class MCPClient:
    """Client for managing MCP tool loading from in-memory configuration."""

    def __init__(
        self,
        config: MCPConfig,
        *,
        default_invoke_timeout: float | None = None,
        tool_factory: ToolFactory | None = None,
    ) -> None:
        self.config = config
        self.default_invoke_timeout = default_invoke_timeout
        self.tool_factory = tool_factory or ToolFactory()
        self._tools: list[BaseTool] = []
        self._module_map: dict[str, str] = {}

    async def tools(self) -> list[BaseTool]:
        """Load and return tools from all enabled MCP servers."""
        if MultiServerMCPClient is None:
            raise RuntimeError(
                "langchain-mcp-adapters is required but not installed."
            )
        connections = self._build_connections()
        if not connections:
            self._tools = []
            self._module_map = {}
            return []

        client = MultiServerMCPClient(
            connections=cast(Any, connections),
            tool_name_prefix=True,
        )
        loaded_tools = await client.get_tools()

        filtered_tools: list[BaseTool] = []
        module_map: dict[str, str] = {}
        for tool in loaded_tools:
            tool_name = getattr(tool, "name", "")
            server_name, raw_name = self._parse_tool_name(tool_name)
            if server_name is None:
                # Unexpected naming format from adapter; keep tool enabled.
                filtered_tools.append(tool)
                module_map[tool_name] = "mcp:unknown"
                continue

            server = self.config.servers.get(server_name)
            if server is None or not self._is_tool_enabled(server, tool_name, raw_name):
                continue

            timeout = (
                float(server.invoke_timeout)
                if server.invoke_timeout is not None
                else self.default_invoke_timeout
            )
            wrapped = self.tool_factory.wrap_tool_with_timeout(
                tool,
                timeout_seconds=float(timeout),
                source=f"mcp:{server_name}",
            ) if timeout else tool

            filtered_tools.append(wrapped)
            module_map[getattr(wrapped, "name", tool_name)] = f"mcp:{server_name}"

        self._tools = filtered_tools
        self._module_map = module_map
        return filtered_tools

    @property
    def module_map(self) -> dict[str, str]:
        return dict(self._module_map)

    async def close(self) -> None:
        # MultiServerMCPClient uses per-call sessions, no long-lived close required.
        return None

    def _build_connections(self) -> dict[str, dict[str, Any]]:
        connections: dict[str, dict[str, Any]] = {}
        for server_name, server in self.config.servers.items():
            if not server.enabled:
                continue

            connection = self._build_connection(server)
            if connection is None:
                continue
            connections[server_name] = connection
        return connections

    @staticmethod
    def _build_connection(server: MCPServerConfig) -> dict[str, Any] | None:
        if server.transport == MCPTransport.STDIO:
            if not server.command:
                return None
            connection: dict[str, Any] = {
                "transport": "stdio",
                "command": server.command,
                "args": list(server.args),
            }
            if server.env:
                connection["env"] = dict(server.env)
            return connection

        if server.transport == MCPTransport.SSE:
            if not server.url:
                return None
            connection = {"transport": "sse", "url": server.url}
            if server.headers:
                connection["headers"] = dict(server.headers)
            if server.timeout is not None:
                connection["timeout"] = float(server.timeout)
            if server.sse_read_timeout is not None:
                connection["sse_read_timeout"] = float(server.sse_read_timeout)
            return connection

        if server.transport == MCPTransport.HTTP:
            if not server.url:
                return None
            connection = {"transport": "streamable_http", "url": server.url}
            if server.headers:
                connection["headers"] = dict(server.headers)
            if server.timeout is not None:
                connection["timeout"] = timedelta(seconds=float(server.timeout))
            if server.sse_read_timeout is not None:
                connection["sse_read_timeout"] = timedelta(
                    seconds=float(server.sse_read_timeout)
                )
            return connection

        if server.transport == MCPTransport.WEBSOCKET:
            if not server.url:
                return None
            return {"transport": "websocket", "url": server.url}

        return None

    def _parse_tool_name(self, tool_name: str) -> tuple[str | None, str]:
        server_names = sorted(self.config.servers.keys(), key=len, reverse=True)
        for server_name in server_names:
            for separator in ("__", "_"):
                prefix = f"{server_name}{separator}"
                if tool_name.startswith(prefix):
                    return server_name, tool_name[len(prefix) :]
        return None, tool_name

    @staticmethod
    def _is_tool_enabled(
        server: MCPServerConfig,
        full_name: str,
        raw_name: str,
    ) -> bool:
        if not server.enabled:
            return False
        include = {name.strip() for name in server.include if name.strip()}
        exclude = {name.strip() for name in server.exclude if name.strip()}
        if include and full_name not in include and raw_name not in include:
            return False
        if full_name in exclude or raw_name in exclude:
            return False
        return True

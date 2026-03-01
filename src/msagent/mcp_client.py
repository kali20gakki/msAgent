"""MCP client for msagent."""

import asyncio
import os
import time
import copy
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import MCPConfig


class MCPClient:
    """MCP client for connecting to MCP servers via langchain-mcp-adapters."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self._adapter_client: MultiServerMCPClient | None = None
        self._tools: list[dict[str, Any]] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._connected = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            connection: dict[str, Any] = {
                "transport": "stdio",
                "command": self.config.command,
                "args": list(self.config.args),
            }
            if self.config.env:
                connection["env"] = {**self.config.env}

            self._adapter_client = MultiServerMCPClient(
                connections={self.config.name: connection},
            )
            await self._fetch_tools()
            self._connected = True
            return True

        except Exception:
            self._adapter_client = None
            self._tool_map.clear()
            self._tools = []
            self._connected = False
            return False

    async def _fetch_tools(self) -> None:
        """Fetch available tools and cache OpenAI-compatible schema."""
        if not self._adapter_client:
            return

        tools = await self._adapter_client.get_tools(server_name=self.name)
        self._tool_map = {tool.name: tool for tool in tools}
        self._tools = [self._to_openai_tool_schema(tool) for tool in tools]

    def _to_openai_tool_schema(self, tool: BaseTool) -> dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        parameters: dict[str, Any]

        if isinstance(args_schema, dict):
            parameters = args_schema
        elif hasattr(args_schema, "model_json_schema"):
            parameters = args_schema.model_json_schema()
        else:
            parameters = {"type": "object", "properties": {}}

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from this MCP server."""
        return copy.deepcopy(self._tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        if not self._connected:
            return "Error: Not connected to MCP server"

        try:
            tool = self._tool_map.get(tool_name)
            if tool is None:
                await self._fetch_tools()
                tool = self._tool_map.get(tool_name)

            if tool is None:
                return f"Error calling tool {tool_name}: tool not found"

            timeout_s = float(os.getenv("MSAGENT_TOOL_TIMEOUT", "300"))
            start = time.monotonic()
            result = await asyncio.wait_for(
                tool.ainvoke(arguments),
                timeout=timeout_s,
            )
            elapsed = time.monotonic() - start
            if elapsed >= 1.0:
                print(f"[mcp] tool {self.name}__{tool_name} completed in {elapsed:.2f}s")

            content = self._stringify_tool_result(result)
            return content if content else "Tool executed successfully"

        except asyncio.TimeoutError:
            return f"Error calling tool {tool_name}: timed out after {timeout_s:.0f}s"
        except Exception as e:
            return f"Error calling tool {tool_name}: {e}"

    def _stringify_tool_result(self, result: Any) -> str:
        if isinstance(result, tuple):
            result = result[0]

        parts = self._collect_content_parts(result)
        return "\n".join(parts).strip()

    def _collect_content_parts(self, payload: Any) -> list[str]:
        if payload is None:
            return []

        if isinstance(payload, str):
            return [payload]

        if isinstance(payload, list):
            items: list[str] = []
            for part in payload:
                items.extend(self._collect_content_parts(part))
            return items

        if isinstance(payload, dict):
            item_type = payload.get("type")
            if item_type == "text":
                text = payload.get("text")
                return [text] if isinstance(text, str) else []
            if item_type == "image":
                mime_type = payload.get("mime_type") or payload.get("mimeType") or "image"
                return [f"[Image: {mime_type}]"]
            if item_type == "file":
                mime_type = payload.get("mime_type") or payload.get("mimeType") or "file"
                url = payload.get("url")
                if isinstance(url, str) and url:
                    return [f"[File: {mime_type}] {url}"]
                return [f"[File: {mime_type}]"]
            return [str(payload)]

        content = getattr(payload, "content", None)
        if content is not None:
            return self._collect_content_parts(content)

        return [str(payload)]

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._connected:
            self._connected = False
            self._adapter_client = None
            self._tool_map.clear()
            self._tools = []


class MCPManager:
    """Manages multiple MCP clients."""

    def __init__(self):
        self.clients: dict[str, MCPClient] = {}

    async def add_server(self, config: MCPConfig) -> bool:
        """Add and connect to an MCP server."""
        client = MCPClient(config)
        if await client.connect():
            self.clients[config.name] = client
            return True
        return False
    
    async def remove_server(self, name: str) -> bool:
        """Remove and disconnect from an MCP server."""
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]
            return True
        return False

    def get_all_tools(self) -> list[dict]:
        """Get all tools from all connected MCP servers."""
        all_tools = []
        for client in self.clients.values():
            if client.is_connected:
                tools = client.get_tools()
                # Add server prefix to tool names to avoid conflicts
                for tool in tools:
                    tool_copy = copy.deepcopy(tool)
                    original_name = tool_copy["function"]["name"]
                    tool_copy["function"]["name"] = f"{client.name}__{original_name}"
                    all_tools.append(tool_copy)
        return all_tools

    async def call_tool(self, full_tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool by its full name (server__tool_name)."""
        if "__" not in full_tool_name:
            return f"Error: Invalid tool name format: {full_tool_name}"

        server_name, tool_name = full_tool_name.split("__", 1)

        if server_name not in self.clients:
            return f"Error: MCP server '{server_name}' not found"

        client = self.clients[server_name]
        if not client.is_connected:
            return f"Error: MCP server '{server_name}' is not connected"

        return await client.call_tool(tool_name, arguments)

    def get_connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return [name for name, client in self.clients.items() if client.is_connected]

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()


# Global MCP manager instance
mcp_manager = MCPManager()

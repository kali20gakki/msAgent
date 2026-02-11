"""MCP client for msagent."""

import asyncio
import os
import time
import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import MCPConfig




class MCPClient:
    """MCP client for connecting to MCP servers via stdio."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._tools: list[dict] = []
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
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env={**self.config.env} if self.config.env else None,
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await self.session.initialize()
            self._connected = True
            
            # Fetch available tools
            await self._fetch_tools()
            
            return True
            
        except Exception as e:
            return False
    
    async def _fetch_tools(self) -> None:
        """Fetch available tools from the MCP server."""
        if not self.session:
            return
        
        try:
            response = await self.session.list_tools()
            self._tools = []
            for tool in response.tools:
                self._tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    }
                })
        except Exception as e:
            pass
    
    def get_tools(self) -> list[dict]:
        """Get available tools from this MCP server."""
        return self._tools
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        if not self.session or not self._connected:
            return "Error: Not connected to MCP server"
        
        try:
            timeout_s = float(os.getenv("MSAGENT_TOOL_TIMEOUT", "300"))
            start = time.monotonic()
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments=arguments),
                timeout=timeout_s,
            )
            elapsed = time.monotonic() - start
            if elapsed >= 1.0:
                print(f"[mcp] tool {self.name}__{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            contents = []
            for content in result.content:
                if content.type == "text":
                    contents.append(content.text)
                elif content.type == "image":
                    contents.append(f"[Image: {content.mimeType}]")
            
            return "\n".join(contents) if contents else "Tool executed successfully"
            
        except asyncio.TimeoutError:
            return f"Error calling tool {tool_name}: timed out after {timeout_s:.0f}s"
        except Exception as e:
            return f"Error calling tool {tool_name}: {e}"
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._connected:
            await self.exit_stack.aclose()
            self._connected = False
            self.session = None


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
                    tool_copy = json.loads(json.dumps(tool))  # Deep copy
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

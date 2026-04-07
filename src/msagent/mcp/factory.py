"""Factory for creating MCP clients from parsed MCPConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from msagent.configs import MCPConfig
from msagent.mcp.client import MCPClient
from msagent.tools.factory import ToolFactory


class MCPFactory:
    """Factory for creating MCP clients."""

    def __init__(self, tool_factory: ToolFactory | None = None) -> None:
        self.tool_factory = tool_factory or ToolFactory()

    async def create(
        self,
        config: MCPConfig,
        cache_dir: Path | None = None,
        oauth_dir: Path | None = None,
        sandbox_bindings: list[Any] | None = None,
        *,
        default_invoke_timeout: float | None = None,
    ) -> MCPClient:
        del cache_dir, oauth_dir, sandbox_bindings
        return MCPClient(
            config=config,
            default_invoke_timeout=default_invoke_timeout,
            tool_factory=self.tool_factory,
        )


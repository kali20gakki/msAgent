"""Helpers for keeping MCP stdio servers quiet in the CLI."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langchain_mcp_adapters import sessions as adapter_sessions

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import TextIO

    from mcp.client.stdio import StdioServerParameters


def silence_mcp_stdio_stderr() -> None:
    """Prevent MCP stdio server stderr logs from polluting the terminal UI."""
    if getattr(adapter_sessions, "_msagent_stdio_stderr_silenced", False):
        return

    original_stdio_client = adapter_sessions.stdio_client

    @asynccontextmanager
    async def quiet_stdio_client(
        server: StdioServerParameters,
        errlog: TextIO = sys.stderr,
    ) -> AsyncIterator[tuple[object, object]]:
        if errlog is not sys.stderr:
            async with original_stdio_client(server, errlog=errlog) as streams:
                yield streams
            return

        with open(os.devnull, "w", encoding="utf-8") as silent_errlog:
            async with original_stdio_client(server, errlog=silent_errlog) as streams:
                yield streams

    adapter_sessions.stdio_client = quiet_stdio_client
    adapter_sessions._msagent_stdio_stderr_silenced = True


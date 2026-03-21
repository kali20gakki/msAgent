from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager

import pytest
from langchain_mcp_adapters import sessions as adapter_sessions

from msagent.mcp.stdio import silence_mcp_stdio_stderr


@pytest.mark.asyncio
async def test_silence_mcp_stdio_stderr_redirects_default_stderr(monkeypatch) -> None:
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_stdio_client(server, errlog=sys.stderr):
        captured["server"] = server
        captured["errlog"] = errlog
        yield ("read", "write")

    monkeypatch.setattr(adapter_sessions, "stdio_client", fake_stdio_client)
    monkeypatch.delattr(
        adapter_sessions,
        "_msagent_stdio_stderr_silenced",
        raising=False,
    )

    silence_mcp_stdio_stderr()

    async with adapter_sessions.stdio_client("server") as streams:
        assert streams == ("read", "write")

    assert captured["server"] == "server"
    assert captured["errlog"] is not sys.stderr
    assert getattr(captured["errlog"], "name", None) == "/dev/null"


@pytest.mark.asyncio
async def test_silence_mcp_stdio_stderr_preserves_explicit_errlog(monkeypatch) -> None:
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_stdio_client(server, errlog=sys.stderr):
        captured["server"] = server
        captured["errlog"] = errlog
        yield ("read", "write")

    monkeypatch.setattr(adapter_sessions, "stdio_client", fake_stdio_client)
    monkeypatch.delattr(
        adapter_sessions,
        "_msagent_stdio_stderr_silenced",
        raising=False,
    )

    silence_mcp_stdio_stderr()

    custom_errlog = io.StringIO()
    async with adapter_sessions.stdio_client("server", errlog=custom_errlog) as streams:
        assert streams == ("read", "write")

    assert captured["server"] == "server"
    assert captured["errlog"] is custom_errlog


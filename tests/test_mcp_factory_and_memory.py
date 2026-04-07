from __future__ import annotations

from pathlib import Path

import pytest

from msagent.configs import MCPConfig
from msagent.mcp.factory import MCPFactory
from msagent.tools.factory import ToolFactory
from msagent.tools.internal.memory import read_memory_file


def test_mcp_factory_uses_default_tool_factory_when_none_provided() -> None:
    factory = MCPFactory()

    assert isinstance(factory.tool_factory, ToolFactory)


@pytest.mark.asyncio
async def test_mcp_factory_create_builds_client_with_timeout_and_tool_factory(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMCPClient:
        def __init__(self, config, *, default_invoke_timeout, tool_factory):
            captured["config"] = config
            captured["default_invoke_timeout"] = default_invoke_timeout
            captured["tool_factory"] = tool_factory

    monkeypatch.setattr("msagent.mcp.factory.MCPClient", FakeMCPClient)

    tool_factory = ToolFactory()
    config = MCPConfig()
    factory = MCPFactory(tool_factory=tool_factory)

    client = await factory.create(
        config=config,
        cache_dir=Path("cache"),
        oauth_dir=Path("oauth"),
        sandbox_bindings=[{"name": "ignored"}],
        default_invoke_timeout=42.0,
    )

    assert isinstance(client, FakeMCPClient)
    assert captured["config"] is config
    assert captured["default_invoke_timeout"] == 42.0
    assert captured["tool_factory"] is tool_factory


def test_read_memory_file_returns_empty_when_memory_does_not_exist(
    tmp_path: Path,
) -> None:
    assert read_memory_file(tmp_path) == ""


def test_read_memory_file_reads_utf8_content(tmp_path: Path) -> None:
    memory_file = tmp_path / ".msagent" / "memory.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text("remember this", encoding="utf-8")

    assert read_memory_file(tmp_path) == "remember this"


def test_read_memory_file_returns_empty_when_read_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory_file = tmp_path / ".msagent" / "memory.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text("will fail", encoding="utf-8")

    def broken_read_text(self, encoding="utf-8"):
        del self, encoding
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", broken_read_text)

    assert read_memory_file(tmp_path) == ""

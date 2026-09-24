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

from pathlib import Path

import pytest

from msagent.configs import MCPConfig
from msagent.mcp.factory import MCPFactory
from msagent.tools.factory import ToolFactory
from msagent.tools.internal.memory import (
    DEFAULT_MEMORY_FILE_CONTENT,
    append_memory_entry,
    ensure_memory_file,
    is_default_memory_content,
    read_memory_file,
)


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


def test_ensure_memory_file_creates_default_template(tmp_path: Path) -> None:
    memory_file = ensure_memory_file(tmp_path)

    assert memory_file == tmp_path / ".msagent" / "memory.md"
    assert memory_file.read_text(encoding="utf-8") == DEFAULT_MEMORY_FILE_CONTENT


def test_default_memory_content_detects_template_only() -> None:
    assert is_default_memory_content(DEFAULT_MEMORY_FILE_CONTENT)
    assert not is_default_memory_content(DEFAULT_MEMORY_FILE_CONTENT + "- 用户喜欢中文回复\n")


def test_ensure_memory_file_preserves_existing_memory(tmp_path: Path) -> None:
    memory_file = tmp_path / ".msagent" / "memory.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text("existing memory", encoding="utf-8")

    assert ensure_memory_file(tmp_path) == memory_file
    assert memory_file.read_text(encoding="utf-8") == "existing memory"


def test_append_memory_entry_replaces_default_empty_bullet(tmp_path: Path) -> None:
    memory_file = append_memory_entry("用户喜欢中文回复", tmp_path)

    assert memory_file.read_text(encoding="utf-8") == (
        "# msagent memory\n\nStore durable user preferences and project facts here.\n\n- 用户喜欢中文回复\n"
    )


def test_append_memory_entry_preserves_existing_entries(tmp_path: Path) -> None:
    memory_file = tmp_path / ".msagent" / "memory.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text("# memory\n\n- first\n", encoding="utf-8")

    append_memory_entry("second", tmp_path)

    assert memory_file.read_text(encoding="utf-8") == "# memory\n\n- first\n- second\n"


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

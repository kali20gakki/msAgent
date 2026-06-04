#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.handlers import mcp as mcp_module
from msagent.cli.handlers.mcp import MCPHandler
from msagent.configs import MCPServerConfig


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
        needs_reload=False,
        running=True,
    )


@pytest.mark.asyncio
async def test_mcp_handler_reports_no_servers_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(mcp_module.console, "print_error", errors.append)
    monkeypatch.setattr(mcp_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_load_mcp_config(_working_dir):
        return SimpleNamespace(servers={})

    monkeypatch.setattr(mcp_module.initializer, "load_mcp_config", fake_load_mcp_config)

    handler = MCPHandler(_build_session(tmp_path))
    await handler.handle()

    assert "No MCP servers configured" in errors


@pytest.mark.asyncio
async def test_mcp_handler_saves_config_and_marks_reload_on_modification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    server_a = MCPServerConfig(command="node", args=["server-a.js"], enabled=True)
    server_b = MCPServerConfig(command="node", args=["server-b.js"], enabled=False)
    mcp_config = SimpleNamespace(servers={"server-a": server_a, "server-b": server_b})

    saved_configs: list[object] = []

    async def fake_load_mcp_config(_working_dir):
        return mcp_config

    async def fake_save_mcp_config(config, _working_dir):
        saved_configs.append(config)

    monkeypatch.setattr(mcp_module.initializer, "load_mcp_config", fake_load_mcp_config)
    monkeypatch.setattr(mcp_module.initializer, "save_mcp_config", fake_save_mcp_config)
    monkeypatch.setattr(mcp_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(mcp_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_get_mcp_selection(_servers):
        server_b.enabled = True
        return True

    handler = MCPHandler(session)
    monkeypatch.setattr(handler, "_get_mcp_selection", fake_get_mcp_selection)

    await handler.handle()

    assert session.needs_reload is True
    assert session.running is False
    assert len(saved_configs) == 1


@pytest.mark.asyncio
async def test_mcp_handler_skips_reload_when_no_modification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path)
    server_a = MCPServerConfig(command="node", args=["server-a.js"], enabled=True)
    mcp_config = SimpleNamespace(servers={"server-a": server_a})

    async def fake_load_mcp_config(_working_dir):
        return mcp_config

    monkeypatch.setattr(mcp_module.initializer, "load_mcp_config", fake_load_mcp_config)
    monkeypatch.setattr(mcp_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(mcp_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_get_mcp_selection(_servers):
        return False

    handler = MCPHandler(session)
    monkeypatch.setattr(handler, "_get_mcp_selection", fake_get_mcp_selection)

    await handler.handle()

    assert session.needs_reload is False


@pytest.mark.asyncio
async def test_mcp_handler_handles_load_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(mcp_module.console, "print_error", errors.append)
    monkeypatch.setattr(mcp_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_load_mcp_config(_working_dir):
        raise RuntimeError("config file corrupt")

    monkeypatch.setattr(mcp_module.initializer, "load_mcp_config", fake_load_mcp_config)

    handler = MCPHandler(_build_session(tmp_path))
    await handler.handle()

    assert any("Error managing MCP servers" in e for e in errors)


def test_format_server_list_shows_enabled_checkbox() -> None:
    server_a = MCPServerConfig(command="node", args=["server-a.js"], enabled=True)
    server_b = MCPServerConfig(command="node", args=["server-b.js"], enabled=False)
    servers = {"server-a": server_a, "server-b": server_b}

    formatted = MCPHandler._format_server_list(servers, ["server-a", "server-b"], selected_index=0)
    text = "".join(fragment[1] for fragment in formatted)

    assert "[x] server-a" in text
    assert "[ ] server-b" in text


def test_format_server_list_highlights_selected_server() -> None:
    server_a = MCPServerConfig(command="node", args=["server-a.js"], enabled=True)
    server_b = MCPServerConfig(command="node", args=["server-b.js"], enabled=True)
    servers = {"server-a": server_a, "server-b": server_b}

    formatted = MCPHandler._format_server_list(servers, ["server-a", "server-b"], selected_index=1)

    selected_line_style = None
    for style, text in formatted:
        if "server-b" in text and "[x]" in text:
            selected_line_style = style
            break

    assert selected_line_style is not None


@pytest.mark.asyncio
async def test_mcp_handler_get_mcp_selection_returns_false_for_empty_servers(
    tmp_path: Path,
) -> None:
    handler = MCPHandler(_build_session(tmp_path))
    result = await handler._get_mcp_selection({})
    assert result is False

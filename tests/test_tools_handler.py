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

from msagent.cli.handlers import tools as tools_module
from msagent.cli.handlers.tools import ToolsHandler


def _build_tool(*, name: str, description: str = "No description") -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description)


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
    )


@pytest.mark.asyncio
async def test_tools_handler_reports_no_tools_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(tools_module.console, "print_error", errors.append)
    monkeypatch.setattr(tools_module.console, "print", lambda *_args, **_kwargs: None)

    handler = ToolsHandler(_build_session(tmp_path))
    await handler.handle([])

    assert "No tools available" in errors


def test_format_tool_list_shows_tool_names() -> None:
    tools = [
        _build_tool(name="read_file", description="Read a file from disk"),
        _build_tool(name="execute", description="Run a shell command"),
    ]

    formatted = ToolsHandler._format_tool_list(
        tools, selected_index=0, expanded_indices=set(), scroll_offset=0, window_size=10
    )
    text = "".join(fragment[1] for fragment in formatted)

    assert "read_file" in text
    assert "execute" in text


def test_format_tool_list_highlights_selected_tool() -> None:
    tools = [
        _build_tool(name="read_file", description="Read a file"),
        _build_tool(name="execute", description="Run a command"),
    ]

    formatted = ToolsHandler._format_tool_list(
        tools, selected_index=1, expanded_indices=set(), scroll_offset=0, window_size=10
    )

    selected_style = None
    for style, text_content in formatted:
        if "execute" in text_content:
            selected_style = style
            break

    assert selected_style is not None
    assert selected_style != ""


def test_format_tool_list_expands_description_on_enter() -> None:
    tools = [
        _build_tool(name="execute", description="Run a shell command.\nSupports timeout and process isolation."),
    ]

    formatted = ToolsHandler._format_tool_list(
        tools, selected_index=0, expanded_indices={0}, scroll_offset=0, window_size=10
    )
    text = "".join(fragment[1] for fragment in formatted)

    assert "Run a shell command" in text
    assert "Supports timeout" in text


def test_format_tool_list_respects_scroll_offset_and_window_size() -> None:
    tools = [_build_tool(name=f"tool-{i}", description=f"Tool number {i}") for i in range(20)]

    formatted = ToolsHandler._format_tool_list(
        tools, selected_index=5, expanded_indices=set(), scroll_offset=5, window_size=5
    )
    text = "".join(fragment[1] for fragment in formatted)

    assert "tool-5" in text
    assert "tool-9" in text
    assert "tool-0" not in text
    assert "tool-19" not in text


def test_format_tool_list_wraps_long_description_lines() -> None:
    long_desc = "This is a very long description that should be wrapped " * 10
    tools = [_build_tool(name="big_tool", description=long_desc)]

    formatted = ToolsHandler._format_tool_list(
        tools, selected_index=0, expanded_indices={0}, scroll_offset=0, window_size=10
    )
    text = "".join(fragment[1] for fragment in formatted)

    assert "big_tool" in text


def test_format_tool_list_handles_unknown_tool_name() -> None:
    tool = SimpleNamespace(description="Some description")
    tools = [tool]

    formatted = ToolsHandler._format_tool_list(
        tools, selected_index=0, expanded_indices=set(), scroll_offset=0, window_size=10
    )
    text = "".join(fragment[1] for fragment in formatted)

    assert "Unknown" in text


@pytest.mark.asyncio
async def test_tools_handler_handles_exception_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(tools_module.console, "print_error", errors.append)
    monkeypatch.setattr(tools_module.console, "print", lambda *_args, **_kwargs: None)

    handler = ToolsHandler(_build_session(tmp_path))

    async def fake_get_tool_selection(_tools):
        raise RuntimeError("UI failure")

    monkeypatch.setattr(handler, "_get_tool_selection", fake_get_tool_selection)

    await handler.handle([_build_tool(name="test")])

    assert any("Error displaying tools" in e for e in errors)

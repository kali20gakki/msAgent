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

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from msagent.utils import render as render_module


def test_render_templates_recurses_and_preserves_unrenderable_strings() -> None:
    data = {
        "title": "Hello {name}",
        "items": ["{name}", {"missing": "{unknown}"}],
        "count": 3,
    }

    rendered = render_module.render_templates(data, {"name": "Alice"})

    assert rendered == {
        "title": "Hello Alice",
        "items": ["Alice", {"missing": "{unknown}"}],
        "count": 3,
    }


def test_format_tool_response_handles_strings_collections_dicts_and_none() -> None:
    tool_message = ToolMessage(
        content="full text",
        tool_call_id="call-1",
        name="tool",
        short_content="short text",
    )
    assert render_module.format_tool_response(tool_message) == ("full text", "short text")

    assert render_module.format_tool_response(' {"value": 1} ') == ('{\n  "value": "1"\n}', None)
    assert render_module.format_tool_response("[1, null, 2]") == ("1\n2", None)
    assert render_module.format_tool_response("{not json") == ("{not json", None)
    assert render_module.format_tool_response(None) == ("", None)
    assert render_module.format_tool_response(42) == ("42", None)


def test_truncate_text_handles_zero_and_overflow_lengths() -> None:
    assert render_module.truncate_text("", 0) == ""
    assert render_module.truncate_text("abcdef", 0) == "... (truncated, original length: 6)"
    assert render_module.truncate_text("abc", 5) == "abc"
    assert render_module.truncate_text("abcdef", 3) == "abc... (truncated, original length: 6)"


def test_create_tool_message_extracts_metadata_and_short_content() -> None:
    result = SimpleNamespace(
        is_error=False,
        return_direct=True,
        additional_kwargs={"source": "base"},
        response_metadata={"latency": 1},
    )

    message = render_module.create_tool_message(
        result=result,
        tool_name="search",
        tool_call_id="call-1",
        additional_kwargs={"extra": "value"},
        response_metadata={render_module.TOOL_TIMING_RESPONSE_METADATA_KEY: 5},
    )

    assert message.name == "search"
    assert message.tool_call_id == "call-1"
    assert message.return_direct is True
    assert "namespace(" in message.short_content
    assert message.additional_kwargs == {"source": "base", "extra": "value"}
    assert message.response_metadata == {"latency": 1, render_module.TOOL_TIMING_RESPONSE_METADATA_KEY: 5}


def test_create_tool_message_uses_ai_message_text_and_error_content() -> None:
    message = render_module.create_tool_message(
        result=AIMessage(content="assistant result"),
        tool_name="tool",
        tool_call_id="call-2",
        is_error=True,
    )

    assert message.content == "assistant result"
    assert message.short_content == "assistant result"
    assert message.is_error is True


def test_generate_diff_and_line_number_helpers_adjust_hunks() -> None:
    diff_lines = render_module.generate_diff(
        old_content="beta\ngamma",
        new_content="beta\nGAMMA",
        full_content="alpha\nbeta\ngamma\ndelta",
    )

    assert diff_lines[0:2] == ["--- ", "+++ "]
    assert "@@ -2,2 +2,2 @@" in diff_lines[2]
    assert render_module._find_content_line_number(["a", "b", "c"], ["b", "c"]) == 2
    assert render_module._find_content_line_number(["a"], ["missing"]) == 0
    assert render_module._adjust_diff_line_numbers(["@@ -1,2 +1,2 @@"], 3) == ["@@ -3,2 +3,2 @@"]


def test_wrap_diff_line_and_format_diff_rich_cover_wrapping_and_markers(monkeypatch) -> None:
    wrapped = render_module._wrap_diff_line(
        code="[bold]" + ("x" * 40),
        marker="+",
        color="green",
        line_num=7,
        width=3,
        term_width=18,
    )

    assert len(wrapped) > 1
    assert wrapped[0].startswith("[green]  7 +  ")
    assert "\\[bold]" in wrapped[0]

    monkeypatch.setattr(render_module.shutil, "get_terminal_size", lambda: os.terminal_size((40, 20)))
    rich_diff = render_module.format_diff_rich(
        [
            "--- ",
            "+++ ",
            "@@ -3,2 +3,2 @@",
            " context",
            "-old",
            "+new",
            "...",
        ]
    )

    assert "No changes detected" not in rich_diff
    assert "  3    context" in rich_diff
    assert "  4 -  old" in rich_diff
    assert "  4 +  new" in rich_diff
    assert "..." in rich_diff


def test_format_diff_rich_handles_empty_diff() -> None:
    assert render_module.format_diff_rich([]) == "[muted]No changes detected[/muted]"

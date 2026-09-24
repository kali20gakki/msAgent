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

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from msagent.cli.theme import theme
from msagent.cli.ui import renderer as renderer_module
from msagent.cli.ui.renderer import Renderer


class _CaptureConsole:
    def __init__(self) -> None:
        self.console = Console(record=True, width=120, theme=theme.rich_theme)

    def print(self, *args: object, **kwargs: object) -> None:
        self.console.print(*args, **kwargs)


def test_extract_thinking_and_text_from_blocks_separates_provider_blocks_when_formats_are_mixed() -> None:
    blocks = [
        "plain text",
        {"type": "text", "text": "structured text"},
        {"type": "thinking", "thinking": "anthropic thought"},
        {
            "type": "reasoning",
            "summary": [{"text": "summary one"}, "ignored", {"text": "summary two"}],
        },
        {"type": "reasoning_content", "reasoning_content": "reasoning detail"},
        {"type": "image", "url": "ignored"},
    ]

    texts, thinking = Renderer._extract_thinking_and_text_from_blocks(blocks)

    assert texts == ["plain text\n", "structured text\n"]
    assert thinking == [
        "anthropic thought",
        "summary one\nsummary two",
        "reasoning detail",
    ]


@pytest.mark.parametrize(
    ("content", "expected_content", "expected_thinking"),
    [
        ("<think>reasoning</think>answer", "answer", "reasoning"),
        ("prefix <think>literal</think>", "prefix <think>literal</think>", None),
        ("<think>unfinished", "<think>unfinished", None),
        ("<think>one</think><think>two</think>done", "done", "one\n\ntwo"),
    ],
)
def test_extract_thinking_tags_returns_expected_parts_when_tag_position_or_completeness_varies(
    content: str,
    expected_content: str,
    expected_thinking: str | None,
) -> None:
    assert Renderer._extract_thinking_tags(content) == (
        expected_content,
        expected_thinking,
    )


def test_extract_thinking_from_metadata_returns_text_when_thinking_payload_is_mapping() -> None:
    message = AIMessage(content="answer", additional_kwargs={"thinking": {"text": "metadata thought"}})

    assert Renderer._extract_thinking_from_metadata(message) == "metadata thought"


def test_extract_thinking_from_metadata_returns_none_when_thinking_payload_is_not_mapping() -> None:
    message = AIMessage(content="answer", additional_kwargs={"thinking": "metadata thought"})

    assert Renderer._extract_thinking_from_metadata(message) is None


def test_render_assistant_message_shows_thinking_text_and_tools_when_content_contains_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)
    message = AIMessage(
        content="<think>check inputs</think>Run the command.",
        tool_calls=[{"name": "execute", "args": {"command": "pwd"}, "id": "call-1"}],
    )

    Renderer.render_assistant_message(message)
    output = capture.console.export_text()

    assert "check inputs" in output
    assert "Run the command." in output
    assert "Use tool execute" in output
    assert "command: pwd" in output


def test_render_assistant_message_hides_tool_calls_when_show_tool_calls_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)
    message = AIMessage(
        content="Visible answer",
        tool_calls=[{"name": "execute", "args": {"command": "pwd"}, "id": "call-1"}],
    )

    Renderer.render_assistant_message(message, show_tool_calls=False)
    output = capture.console.export_text()

    assert "Visible answer" in output
    assert "Use tool" not in output


def test_render_assistant_message_shows_error_style_content_when_message_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)
    message = AIMessage(content="request failed")
    message.is_error = True  # type: ignore[attr-defined]

    Renderer.render_assistant_message(message, indent_level=1)

    assert capture.console.export_text().strip() == "request failed"


def test_build_tool_message_display_returns_none_when_preview_and_content_are_blank() -> None:
    message = ToolMessage(content="   ", tool_call_id="call-1")

    assert Renderer._build_tool_message_display(message) is None


def test_build_tool_message_display_marks_error_when_tool_status_is_error() -> None:
    message = ToolMessage(content="permission denied", tool_call_id="call-1", status="error")

    display = Renderer._build_tool_message_display(message)

    assert display is not None
    assert display.is_error is True
    assert display.display_content == "permission denied"


def test_render_user_message_prefers_short_content_when_message_was_offloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)
    monkeypatch.setattr(renderer_module.settings.cli, "prompt_style", "> ")
    message = HumanMessage(content="full sensitive content")
    message.short_content = "short preview\nsecond line"  # type: ignore[attr-defined]

    Renderer().render_user_message(message)
    output = capture.console.export_text()

    assert "short preview" in output
    assert "second line" in output
    assert "full sensitive content" not in output


def test_render_help_uses_docstring_and_fallback_when_command_descriptions_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)

    def documented() -> None:
        """Open the selected thread."""

    undocumented = SimpleNamespace(__doc__=None)

    Renderer.render_help({"/open": documented, "/unknown": undocumented})
    output = capture.console.export_text()

    assert "Help" in output
    assert "/open" in output and "Open the selected thread." in output
    assert "/unknown" in output and "No description available" in output


def test_render_hotkeys_lists_shortcuts_when_multiple_bindings_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)

    Renderer.render_hotkeys({"Ctrl+C": "Interrupt", "Ctrl+O": "Browse tool output"})
    output = capture.console.export_text()

    assert "Keyboard Shortcuts" in output
    assert "Ctrl+C" in output and "Interrupt" in output
    assert "Ctrl+O" in output and "Browse tool output" in output

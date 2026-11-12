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

from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.output import DummyOutput

from msagent.cli.ui import approval
from msagent.cli.ui.approval import (
    ApprovalChoice,
    ApprovalMenu,
    format_approval_details,
)
from msagent.configs import ApprovalMode


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        approval_mode=ApprovalMode.ACTIVE,
        bash_mode=False,
        working_dir=Path.cwd(),
        model="test-model",
        model_display="test-model",
        current_input_tokens=None,
        current_output_tokens=None,
        context_window=None,
    )


def test_options_follow_deepagents_style_order_and_labels() -> None:
    menu = ApprovalMenu(
        context=_context(),
        title="execute Requires Approval",
        details="rm -rf /tmp/demo",
        options=["approve", "reject", "always_approve", "always_reject"],
    )

    assert [option.value for option in menu.options] == [
        "approve",
        "always_approve",
        "reject",
        "always_reject",
    ]
    assert [option.hotkey for option in menu.options] == ["y", "a", "n", "r"]


def test_option_label_can_describe_project_family_scope() -> None:
    menu = ApprovalMenu(
        context=_context(),
        title="execute Requires Approval",
        details='python3 -c "print(1)"',
        options=["approve", "always_approve", "reject"],
        option_labels={"always_approve": "Always allow python3 -c for this project"},
    )

    assert [option.label for option in menu.options] == [
        "Approve",
        "Always allow python3 -c for this project",
        "Reject",
    ]


def test_menu_renders_title_selection_and_help() -> None:
    menu = ApprovalMenu(
        context=_context(),
        title="execute Requires Approval",
        details="mkdir tmp_rm",
        options=["approve", "reject"],
    )

    rendered = "".join(text for _style, text in menu._format_menu())
    help_text = "".join(text for _style, text in menu._format_help())

    assert ">>> execute Requires Approval <<<" in rendered
    assert "1. Approve (y)" in rendered
    assert "2. Reject (n)" in rendered
    assert "navigate" in help_text
    assert "Tab reject with feedback" in help_text
    assert "Esc reject" in help_text


def test_menu_renders_metadata_and_notes_with_distinct_styles() -> None:
    menu = ApprovalMenu(
        context=_context(),
        title="execute Requires Approval",
        details="python3 -c \"print(1)\"",
        metadata=["Mode: manual", "Policy: opaque"],
        notes=["Note: an allow rule applies to Python inline code in this project."],
        options=["approve", "reject"],
    )

    fragments = menu._format_menu()

    assert ("class:approval.details", 'python3 -c "print(1)"') in fragments
    assert ("class:approval.metadata", "Mode: manual") in fragments
    assert any(style == "class:approval.note" and text.startswith("Note:") for style, text in fragments)


def test_format_approval_details_uses_command_preview() -> None:
    details = format_approval_details(
        tool_name="execute",
        tool_args={"command": "mkdir tmp_rm"},
        description="Create a directory",
    )

    assert details == "Create a directory\nmkdir tmp_rm"


def test_application_can_be_constructed() -> None:
    menu = ApprovalMenu(
        context=_context(),
        title="execute Requires Approval",
        details="echo ok",
        options=["approve", "reject"],
    )

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        app = menu.create_application()

    assert isinstance(app.layout.container, HSplit)
    assert menu._choice(1) == ApprovalChoice("reject")
    assert menu.menu_window.wrap_lines()


@pytest.mark.asyncio
async def test_approval_cancels_when_console_buffer_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    menu = ApprovalMenu(context=_context(), title="Approval", details="echo ok", options=["approve", "reject"])
    messages: list[str] = []
    errors: list[str] = []

    def fail():
        raise approval.NoConsoleScreenBufferError("missing console")

    monkeypatch.setattr(menu, "create_application", fail)
    monkeypatch.setattr(approval.console, "print_warning", messages.append)
    monkeypatch.setattr(approval.logger, "exception", errors.append)

    assert await menu.run_async() is None
    assert messages == ["Approval cancelled because the terminal has no console screen buffer."]
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_approval_propagates_unexpected_initialization_error(monkeypatch: pytest.MonkeyPatch) -> None:
    menu = ApprovalMenu(context=_context(), title="Approval", details="echo ok", options=["approve", "reject"])

    def fail():
        raise ValueError("invalid layout")

    monkeypatch.setattr(approval, "Application", lambda **kwargs: fail())

    with pytest.raises(ValueError, match="invalid layout"):
        await menu.run_async()

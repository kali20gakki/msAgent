from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.layout.containers import Window

from msagent.cli.core.tool_output import ToolOutputEntry
from msagent.cli.dispatchers.commands import CommandDispatcher
from msagent.cli.handlers.tool_outputs import ToolOutputHandler
from msagent.configs import ApprovalMode


def _build_context() -> SimpleNamespace:
    return SimpleNamespace(
        agent="msagent",
        model="default",
        model_display=None,
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        recursion_limit=80,
        bash_mode=False,
        current_input_tokens=None,
        current_output_tokens=None,
        context_window=128000,
    )


@pytest.mark.asyncio
async def test_tool_output_handler_warns_when_no_expandable_output(monkeypatch) -> None:
    printed = []
    monkeypatch.setattr(
        "msagent.cli.handlers.tool_outputs.console.print_warning", printed.append
    )
    monkeypatch.setattr(
        "msagent.cli.handlers.tool_outputs.console.print", lambda *args, **kwargs: None
    )

    session = SimpleNamespace(context=_build_context(), latest_tool_output=None)
    handler = ToolOutputHandler(session)

    await handler.handle()

    assert printed == ["No expandable tool output available yet"]


@pytest.mark.asyncio
async def test_tool_output_handler_uses_expandable_entries_and_opens_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    run_calls: list[str] = []

    class FakeApp:
        async def run_async(self) -> None:
            run_calls.append("run")

    def fake_create_selector_application(**kwargs: Any) -> FakeApp:
        captured_kwargs.update(kwargs)
        return FakeApp()

    monkeypatch.setattr(
        "msagent.cli.handlers.tool_outputs.create_selector_application",
        fake_create_selector_application,
    )

    session = SimpleNamespace(
        context=_build_context(),
        tool_outputs=[
            ToolOutputEntry(
                tool_call_id="call-1",
                tool_name="read_file",
                preview_content="short",
                full_content="short",
            ),
            ToolOutputEntry(
                tool_call_id="call-2",
                tool_name="run_command",
                preview_content="preview-2",
                full_content="full-2\nline-2",
            ),
        ],
        latest_tool_output=None,
    )
    handler = ToolOutputHandler(session)

    await handler.handle()

    assert run_calls == ["run"]
    assert captured_kwargs["context"] is session.context
    assert captured_kwargs["full_screen"] is True
    assert captured_kwargs["mouse_support"] is True
    assert isinstance(captured_kwargs["content_window"], Window)
    assert captured_kwargs["header_windows"]


@pytest.mark.asyncio
async def test_tool_output_handler_falls_back_to_latest_expandable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    run_calls: list[str] = []

    class FakeApp:
        async def run_async(self) -> None:
            run_calls.append("run")

    def fake_create_selector_application(**kwargs: Any) -> FakeApp:
        captured_kwargs.update(kwargs)
        return FakeApp()

    monkeypatch.setattr(
        "msagent.cli.handlers.tool_outputs.create_selector_application",
        fake_create_selector_application,
    )

    latest = ToolOutputEntry(
        tool_call_id="call-latest",
        tool_name="run_command",
        preview_content="preview-latest",
        full_content="full-latest",
    )
    session = SimpleNamespace(
        context=_build_context(),
        tool_outputs=[],
        latest_tool_output=latest,
    )
    handler = ToolOutputHandler(session)

    await handler.handle()

    assert run_calls == ["run"]
    assert captured_kwargs["context"] is session.context


@pytest.mark.asyncio
async def test_tool_output_handler_ignores_non_expandable_latest_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed = []
    monkeypatch.setattr(
        "msagent.cli.handlers.tool_outputs.console.print_warning", printed.append
    )
    monkeypatch.setattr(
        "msagent.cli.handlers.tool_outputs.console.print", lambda *args, **kwargs: None
    )

    latest = ToolOutputEntry(
        tool_call_id="call-1",
        tool_name="read_file",
        preview_content="same",
        full_content="same",
    )
    session = SimpleNamespace(
        context=_build_context(),
        tool_outputs=[],
        latest_tool_output=latest,
    )
    handler = ToolOutputHandler(session)

    await handler.handle()

    assert printed == ["No expandable tool output available yet"]


def test_command_dispatcher_registers_tool_output_command() -> None:
    session = SimpleNamespace(
        context=_build_context(),
        renderer=SimpleNamespace(render_help=lambda *_args, **_kwargs: None),
        prompt=SimpleNamespace(hotkeys={}),
        running=True,
        update_context=lambda **_kwargs: None,
        clear_tool_output=lambda: None,
    )
    dispatcher = CommandDispatcher(session)

    assert "/tool-output" in dispatcher.commands


def test_session_like_tool_output_list_keeps_multiple_entries() -> None:
    tool_outputs: list[ToolOutputEntry] = []
    latest_tool_output = None

    def remember(entry: ToolOutputEntry) -> None:
        nonlocal latest_tool_output
        for index, existing in enumerate(tool_outputs):
            if existing.tool_call_id and existing.tool_call_id == entry.tool_call_id:
                entry.sequence = existing.sequence
                tool_outputs[index] = entry
                latest_tool_output = entry
                return
        entry.sequence = len(tool_outputs) + 1
        tool_outputs.append(entry)
        latest_tool_output = entry

    remember(
        ToolOutputEntry(
            tool_call_id="call-1",
            tool_name="run_command",
            preview_content="preview-1",
            full_content="full-1",
        )
    )
    remember(
        ToolOutputEntry(
            tool_call_id="call-2",
            tool_name="read_file",
            preview_content="preview-2",
            full_content="full-2",
        )
    )

    assert [entry.tool_call_id for entry in tool_outputs] == ["call-1", "call-2"]
    assert latest_tool_output is tool_outputs[-1]
    assert [entry.sequence for entry in tool_outputs] == [1, 2]

"""Tests for TODO UI functionality.

This module tests the todo panel rendering, tool call hiding,
and related UI features.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from msagent.cli.dispatchers.messages import MessageDispatcher
from msagent.cli.theme import theme
from msagent.cli.ui import renderer as renderer_module
from msagent.cli.ui.renderer import Renderer
from msagent.tools.internal.todo import (
    TODO_PANEL_MARKER,
    _coerce_sequential_todos,
    _build_todos_table,
    format_todos,
    parse_todos_for_panel,
    render_todos_panel,
)


class _CaptureConsole:
    def __init__(self) -> None:
        self.console = Console(record=True, width=120, theme=theme.rich_theme)

    def print(self, *args, **kwargs) -> None:
        self.console.print(*args, **kwargs)


class TestTodoCoerce:
    """Test _coerce_sequential_todos function."""

    def test_empty_todos(self):
        """Test with empty list."""
        result = _coerce_sequential_todos([])
        assert result == []

    def test_none_todos(self):
        """Test with None."""
        result = _coerce_sequential_todos(None)
        assert result == []

    def test_sequential_progression(self):
        """Test that todos after non-completed are forced to pending."""
        todos = [
            {"content": "Task 1", "status": "completed"},
            {"content": "Task 2", "status": "pending"},
            {"content": "Task 3", "status": "completed"},  # Should become pending
            {"content": "Task 4", "status": "in_progress"},  # Should become pending
        ]
        result = _coerce_sequential_todos(todos)
        assert result[0]["status"] == "completed"
        assert result[1]["status"] == "pending"
        assert result[2]["status"] == "pending"  # Forced to pending
        assert result[3]["status"] == "pending"  # Forced to pending

    def test_hyphen_to_underscore_conversion(self):
        """Test that in-progress is converted to in_progress."""
        todos = [{"content": "Task", "status": "in-progress"}]
        result = _coerce_sequential_todos(todos)
        assert result[0]["status"] == "in_progress"


class TestBuildTodosTable:
    """Test _build_todos_table function."""

    def test_empty_todos(self):
        """Test with empty list."""
        table = _build_todos_table([])
        assert isinstance(table, Table)

    def test_todos_rendered(self):
        """Test that todos are rendered in table."""
        todos = [
            {"content": "Task 1", "status": "completed"},
            {"content": "Task 2", "status": "in_progress"},
            {"content": "Task 3", "status": "pending"},
        ]
        table = _build_todos_table(todos)
        assert isinstance(table, Table)

    def test_max_items_limit(self):
        """Test that max_items limits the output."""
        todos = [{"content": f"Task {i}", "status": "pending"} for i in range(10)]
        _build_todos_table(todos, max_items=5, max_completed=0)
        # Table should have limited rows

    def test_completed_indicator_hidden(self):
        """Test that show_completed_indicator=False hides the indicator."""
        todos = [
            {"content": "Task 1", "status": "completed"},
            {"content": "Task 2", "status": "completed"},
            {"content": "Task 3", "status": "completed"},
        ]
        _build_todos_table(
            todos, max_items=10, max_completed=1, show_completed_indicator=False
        )
        # Should not show "+X more completed" row


class TestRenderTodosPanel:
    """Test render_todos_panel function."""

    def test_empty_todos_panel(self):
        """Test panel rendering with empty todos."""
        panel = render_todos_panel([])
        assert isinstance(panel, Panel)

    def test_panel_with_todos(self):
        """Test panel rendering with todos."""
        todos = [
            {"content": "Completed task", "status": "completed"},
            {"content": "In progress task", "status": "in_progress"},
            {"content": "Pending task", "status": "pending"},
        ]
        panel = render_todos_panel(todos)
        assert isinstance(panel, Panel)
        assert panel.title == "TODOs"

    def test_full_list_shown(self):
        """Test that all todos are shown (no hidden indicator)."""
        todos = [
            {"content": f"Task {i}", "status": "completed"}
            for i in range(10)
        ]
        panel = render_todos_panel(
            todos, max_items=50, max_completed=50, show_completed_indicator=False
        )
        assert isinstance(panel, Panel)

    def test_panel_uses_single_visual_border(self):
        """Test that inner todo table is borderless (no nested frame)."""
        panel = render_todos_panel([{"content": "Task", "status": "pending"}])
        table = panel.renderable
        assert isinstance(table, Table)
        assert table.show_edge is False
        assert table.box is None


class TestFormatTodos:
    """Test format_todos function."""

    def test_empty_todos(self):
        """Test with empty list."""
        result = format_todos([])
        assert result == "[muted]No todos[/muted]"

    def test_all_statuses(self):
        """Test that all statuses are formatted correctly."""
        todos = [
            {"content": "Completed", "status": "completed"},
            {"content": "In Progress", "status": "in_progress"},
            {"content": "Pending", "status": "pending"},
        ]
        result = format_todos(todos, max_items=50, max_completed=50, show_completed_indicator=False)
        assert "Completed" in result
        assert "In Progress" in result
        assert "Pending" in result

    def test_strike_for_completed(self):
        """Test that completed items have strike markup."""
        todos = [{"content": "Done", "status": "completed"}]
        result = format_todos(todos, max_items=50, max_completed=50, show_completed_indicator=False)
        assert "[strike]" in result


class TestWriteTodosPayload:
    """Test write_todos payload parsing compatibility."""

    def test_format_todos_includes_marker(self):
        """Test that format_todos output can be combined with marker."""
        todos = [{"content": "Task", "status": "pending"}]

        formatted = format_todos(
            todos, max_items=50, max_completed=50, show_completed_indicator=False
        )

        # Legacy panel format still uses marker + formatted body
        marked_content = f"{TODO_PANEL_MARKER}{formatted}"

        assert TODO_PANEL_MARKER in marked_content
        assert "Task" in marked_content

    def test_parse_deepagents_write_todos_payload(self):
        """Test parsing deepagents default write_todos response text."""
        payload = (
            "Updated todo list to "
            "[{'content': 'Task 1', 'status': 'completed'}, "
            "{'content': 'Task 2', 'status': 'in_progress'}]"
        )
        parsed = parse_todos_for_panel(payload)
        assert parsed is not None
        assert len(parsed) == 2
        assert parsed[0]["content"] == "Task 1"
        assert parsed[0]["status"] == "completed"
        assert parsed[1]["status"] == "in_progress"

    def test_parse_legacy_marker_payload(self):
        """Test parsing legacy marker-based payload."""
        formatted = format_todos(
            [{"content": "Task", "status": "pending"}],
            max_items=50,
            max_completed=50,
            show_completed_indicator=False,
        )
        parsed = parse_todos_for_panel(f"{TODO_PANEL_MARKER}{formatted}")
        assert parsed is not None
        assert parsed[0]["content"] == "Task"


class TestRendererSkipToolCall:
    """Test Renderer._should_skip_tool_call."""

    def test_write_todos_is_skipped(self):
        """Test that write_todos is marked for skipping."""
        assert Renderer._should_skip_tool_call({"name": "write_todos"})

    def test_other_tools_not_skipped(self):
        """Test that other tools are not skipped."""
        assert not Renderer._should_skip_tool_call({"name": "read_file"})
        assert not Renderer._should_skip_tool_call({"name": "run_command"})
        assert not Renderer._should_skip_tool_call({"name": "write_file"})


class TestMessageDispatcherHiddenTools:
    """Test MessageDispatcher._HIDDEN_ACTIVITY_TOOLS."""

    def test_write_todos_in_hidden_set(self):
        """Test that write_todos is in hidden tools set."""
        assert "write_todos" in MessageDispatcher._HIDDEN_ACTIVITY_TOOLS

    def test_remember_tool_headers_skips_hidden(self):
        """Test that _remember_tool_headers skips hidden tools."""
        dispatcher = MessageDispatcher.__new__(MessageDispatcher)
        dispatcher._pending_tool_headers = {}
        dispatcher._HIDDEN_ACTIVITY_TOOLS = {"write_todos"}

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "write_todos", "args": {"todos": []}, "id": "call-1"},
                {"name": "read_file", "args": {"path": "/test"}, "id": "call-2"},
            ],
        )

        dispatcher._remember_tool_headers(ai_msg, indent_level=0)

        assert "call-1" not in dispatcher._pending_tool_headers
        assert "call-2" in dispatcher._pending_tool_headers

    def test_render_pending_tool_header_skips_hidden(self):
        """Test that _render_pending_tool_header skips hidden tools."""
        from unittest.mock import MagicMock

        dispatcher = MessageDispatcher.__new__(MessageDispatcher)
        dispatcher._HIDDEN_ACTIVITY_TOOLS = {"write_todos"}
        dispatcher.session = MagicMock()
        dispatcher.session.renderer = MagicMock()

        # Store a hidden tool header
        dispatcher._pending_tool_headers = {
            "call-1": ({"name": "write_todos", "id": "call-1"}, 0, 0.0)
        }

        tool_msg = ToolMessage(
            name="write_todos",
            content="Done",
            tool_call_id="call-1",
        )

        dispatcher._render_pending_tool_header(tool_msg, indent_level=0)

        # Should not call render_tool_call for hidden tool
        dispatcher.session.renderer.render_tool_call.assert_not_called()


class TestTodoPanelIntegration:
    """Integration tests for todo panel rendering."""

    def test_full_flow_todo_panel_rendering(self):
        """Test the full flow of todo panel rendering."""
        todos = [
            {"content": "First task", "status": "completed"},
            {"content": "Second task", "status": "in_progress"},
            {"content": "Third task", "status": "pending"},
        ]

        # Step 1: Format todos
        formatted = format_todos(
            todos, max_items=50, max_completed=50, show_completed_indicator=False
        )
        assert TODO_PANEL_MARKER not in formatted  # format_todos doesn't add marker

        # Step 2: Render panel
        panel = render_todos_panel(
            todos, max_items=50, max_completed=50, show_completed_indicator=False
        )
        assert isinstance(panel, Panel)
        assert panel.title == "TODOs"

        # Step 3: Check tool call skipping
        assert Renderer._should_skip_tool_call({"name": "write_todos"})

    def test_todos_with_special_characters(self):
        """Test todos with special characters in content."""
        todos = [
            {"content": "Task with [brackets] and {braces}", "status": "completed"},
            {"content": "Task with <html> tags", "status": "pending"},
            {"content": "Task with \"quotes\" and 'apostrophes'", "status": "in_progress"},
        ]
        result = format_todos(todos, max_items=50, max_completed=50, show_completed_indicator=False)
        # Should not raise and should contain the task content
        assert "brackets" in result

    def test_render_tool_message_keeps_todo_panel_in_subagent_flow(self, monkeypatch):
        capture = _CaptureConsole()
        monkeypatch.setattr(renderer_module, "console", capture)

        todos = [
            {"content": "Read skill", "status": "completed"},
            {"content": "Analyze advisor output", "status": "in_progress"},
        ]
        formatted = format_todos(
            todos, max_items=50, max_completed=50, show_completed_indicator=False
        )
        message = ToolMessage(
            name="write_todos",
            content='[{"content":"Read skill","status":"completed"}]',
            tool_call_id="call-todo-1",
        )
        setattr(message, "short_content", f"{TODO_PANEL_MARKER}{formatted}")

        Renderer.render_tool_message(message, indent_level=1)

        output = capture.console.export_text()
        assert "[Subagent] Update TODOs" in output
        assert "TODOs" in output
        assert "Analyze advisor output" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

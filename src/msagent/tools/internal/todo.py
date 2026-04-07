"""Todo rendering and payload parsing helpers used by the TUI."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Literal, cast

from rich.panel import Panel
from rich.table import Table

from msagent.agents.state import Todo

TODO_PANEL_MARKER = "<!-- TODOS_PANEL -->"

_STATUS_ORDER = {"completed": 0, "in_progress": 1, "pending": 2}
_STATUS_ICON = {"completed": "✓", "in_progress": "◔", "pending": "○"}
TodoStatus = Literal["pending", "in_progress", "completed"]


def _normalize_status(value: str | None) -> TodoStatus:
    normalized = (value or "pending").strip().replace("-", "_").lower()
    if normalized not in _STATUS_ORDER:
        return "pending"
    return cast(TodoStatus, normalized)


def _coerce_sequential_todos(todos: list[dict[str, Any]] | None) -> list[Todo]:
    """Ensure todo statuses follow sequential progression."""
    if not todos:
        return []

    coerced: list[Todo] = []
    first_unfinished_seen = False
    for item in todos:
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        status = _normalize_status(cast(str | None, item.get("status")))
        if first_unfinished_seen and status != "pending":
            status = "pending"
        if status != "completed":
            first_unfinished_seen = True

        coerced.append({"content": content, "status": status})
    return coerced


def _build_todos_table(
    todos: list[Todo],
    *,
    max_items: int = 10,
    max_completed: int = 3,
    show_completed_indicator: bool = True,
) -> Table:
    # Keep the table borderless so the surrounding panel is the only frame.
    table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        expand=True,
        padding=(0, 0),
    )
    table.add_column("Status", width=1, no_wrap=True)
    table.add_column("Task", min_width=20, overflow="fold")

    if not todos:
        table.add_row("", "[muted]No todos[/muted]")
        return table

    completed = [todo for todo in todos if todo["status"] == "completed"]
    active = [todo for todo in todos if todo["status"] != "completed"]
    visible_completed = completed[:max_completed]

    visible: list[Todo] = [*visible_completed, *active]
    visible = visible[:max_items]

    for todo in visible:
        icon = _STATUS_ICON[todo["status"]]
        content = todo["content"]
        if todo["status"] == "completed":
            content = f"[strike]{content}[/strike]"
        table.add_row(icon, content)

    hidden_completed = len(completed) - len(visible_completed)
    if show_completed_indicator and hidden_completed > 0:
        table.add_row("", f"[muted]+{hidden_completed} more completed[/muted]")

    return table


def render_todos_panel(
    todos: list[dict[str, Any]] | None,
    *,
    max_items: int = 10,
    max_completed: int = 3,
    show_completed_indicator: bool = True,
) -> Panel:
    coerced = _coerce_sequential_todos(todos)
    table = _build_todos_table(
        coerced,
        max_items=max_items,
        max_completed=max_completed,
        show_completed_indicator=show_completed_indicator,
    )
    return Panel(table, title="TODOs", border_style="muted")


def format_todos(
    todos: list[dict[str, Any]] | None,
    *,
    max_items: int = 10,
    max_completed: int = 3,
    show_completed_indicator: bool = True,
) -> str:
    coerced = _coerce_sequential_todos(todos)
    if not coerced:
        return "[muted]No todos[/muted]"

    completed = [todo for todo in coerced if todo["status"] == "completed"]
    active = [todo for todo in coerced if todo["status"] != "completed"]
    visible_completed = completed[:max_completed]
    visible = [*visible_completed, *active][:max_items]

    lines: list[str] = []
    for todo in visible:
        icon = _STATUS_ICON[todo["status"]]
        content = todo["content"]
        if todo["status"] == "completed":
            content = f"[strike]{content}[/strike]"
        lines.append(f"{icon} {content}")

    hidden_completed = len(completed) - len(visible_completed)
    if show_completed_indicator and hidden_completed > 0:
        lines.append(f"[muted]+{hidden_completed} more completed[/muted]")

    return "\n".join(lines)


def parse_todos_for_panel(content: str) -> list[Todo] | None:
    """Parse todos from either legacy panel payloads or deepagents tool output."""
    payload = content.strip()
    if not payload:
        return None

    if payload.startswith(TODO_PANEL_MARKER):
        payload = payload[len(TODO_PANEL_MARKER) :]
        todos = _parse_todos_from_formatted_lines(payload)
        return todos if todos else None

    prefix = "Updated todo list to "
    if not payload.startswith(prefix):
        return None

    raw_todos = payload[len(prefix) :].strip()
    parsed = _parse_todos_payload(raw_todos)
    if not isinstance(parsed, list):
        return None
    return _coerce_sequential_todos(cast(list[dict[str, Any]], parsed))


def _parse_todos_payload(raw_todos: str) -> Any:
    """Parse the serialized todo payload from write_todos tool output."""
    try:
        return json.loads(raw_todos)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_todos)
        except (SyntaxError, ValueError):
            return None


def _parse_todos_from_formatted_lines(content: str) -> list[Todo]:
    """Parse todos from rendered panel lines."""
    todos: list[Todo] = []
    for raw_line in content.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("+"):
            continue

        # Strip Rich markup before matching status icon + task content.
        clean_line = re.sub(r"\[/?[^\]]*?\]", "", line).strip()
        if not clean_line:
            continue

        status_text = _parse_status_from_icon(clean_line)
        if status_text is None:
            continue

        status, todo_content = status_text
        if todo_content:
            todos.append({"content": todo_content, "status": status})
    return todos


def _parse_status_from_icon(line: str) -> tuple[TodoStatus, str] | None:
    """Parse todo status from a rendered icon-prefixed line."""
    for status, icon in _STATUS_ICON.items():
        if line.startswith(icon):
            return cast(TodoStatus, status), line[len(icon) :].strip()
    return None

"""TODO management tools for task planning and progress tracking.

This module provides tools for creating and managing structured task lists
that enable agents to plan complex workflows and track progress through
multi-step operations.
"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from msagent.agents.context import AgentContext
from msagent.agents.state import AgentState, Todo
from msagent.cli.theme import theme


# Icon mapping for todo status (参考 langchain-code 的设计)
_TODO_ICON = {
    "pending": "○",
    "in_progress": "◔",
    "completed": "✓",
}


def _coerce_sequential_todos(todos: list[Todo] | None) -> list[dict]:
    """Ensure visual progression is strictly sequential.
    
    Once a todo is not completed, subsequent todos should not appear
    as in_progress or completed.
    """
    todos = list(todos or [])
    blocked = False
    out: list[dict] = []
    for item in todos:
        status = (item.get("status") or "pending").lower().replace("-", "_")
        if blocked and status in {"in_progress", "completed"}:
            status = "pending"
        if status != "completed":
            blocked = True
        out.append({**item, "status": status})
    return out


def _build_todos_table(
    todos: list[Todo],
    max_items: int = 10,
    max_completed: int = 2,
    show_completed_indicator: bool = True,
) -> Table:
    """Build a Rich Table with todos content.
    
    参考 langchain-code 的设计:
    - 待办: ○ (空心圆), 进行中: ◔ (半圆), 完成: ✓ (对勾)
    - 已完成的任务使用删除线(strike)划掉
    """
    todos = _coerce_sequential_todos(todos)

    # Filter and limit todos
    completed = [t for t in todos if t.get("status") == "completed"]
    active = [t for t in todos if t.get("status") in ("in_progress", "pending")]

    # Sort active by status priority (in_progress first)
    priority = {"in_progress": 0, "pending": 1}
    active_sorted = sorted(
        active,
        key=lambda todo: priority.get(todo.get("status", "pending"), 2),
    )

    # Build table
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=3, no_wrap=True)
    table.add_column()

    items_shown = 0

    # Show hidden completed indicator if needed
    if show_completed_indicator and len(completed) > max_completed:
        hidden_count = len(completed) - max_completed
        table.add_row("", Text(f"+{hidden_count} more completed", style=f"{theme.muted_text}"))

    # Show completed items (limited) - 使用删除线划掉
    completed_to_show = completed[-max_completed:]
    for todo in completed_to_show:
        if items_shown >= max_items:
            break
        content = (todo.get("content") or "").strip() or "(empty)"
        icon = _TODO_ICON["completed"]
        # 已完成: 绿色 + 删除线(strike)
        text = Text(content)
        text.stylize("strike")
        text.stylize(f"{theme.success_color}")
        row_text = Text.assemble(
            Text(f"{icon} ", style=f"{theme.success_color}"),
            text,
        )
        table.add_row(f"{items_shown + 1}.", row_text)
        items_shown += 1

    # Show active items
    active_shown = 0
    for todo in active_sorted:
        if items_shown >= max_items:
            remaining = len(active_sorted) - active_shown
            if remaining > 0:
                table.add_row("", Text(f"+{remaining} more", style=f"{theme.muted_text}"))
            break
        status = todo.get("status", "pending")
        content = (todo.get("content") or "").strip() or "(empty)"
        icon = _TODO_ICON.get(status, "○")
        
        if status == "in_progress":
            # 进行中: 信息色
            color = theme.info_color
        else:
            # 待办: 灰色
            color = theme.muted_text
        
        text = Text(content)
        text.stylize(f"{color}")
        row_text = Text.assemble(
            Text(f"{icon} ", style=f"{color}"),
            text,
        )
        table.add_row(f"{items_shown + 1}.", row_text)
        items_shown += 1
        active_shown += 1

    return table


def render_todos_panel(
    todos: list[Todo],
    max_items: int = 10,
    max_completed: int = 2,
    show_completed_indicator: bool = True,
) -> Panel:
    """Render todos as a Rich Panel with rounded box border.
    
    参考 langchain-code 的设计，使用 box.ROUNDED 风格的边框包围整个 todo 列表。
    """
    if not todos:
        return Panel(
            Text("No todos", style=f"{theme.muted_text}"),
            title="TODOs",
            border_style=f"{theme.muted_text}",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    table = _build_todos_table(
        todos,
        max_items=max_items,
        max_completed=max_completed,
        show_completed_indicator=show_completed_indicator,
    )

    return Panel(
        table,
        title="TODOs",
        border_style=f"{theme.muted_text}",
        box=box.ROUNDED,
        padding=(1, 1),
        expand=True,
    )


def format_todos(
    todos: list[Todo],
    max_items: int = 10,
    max_completed: int = 2,
    show_completed_indicator: bool = True,
) -> str:
    """Render todos as Rich markup string.
    
    用于需要字符串输出的场景（如 ToolMessage 的 short_content）。
    """
    if not todos:
        return "[muted]No todos[/muted]"

    # Use the table builder but convert to markup string
    table = _build_todos_table(
        todos,
        max_items=max_items,
        max_completed=max_completed,
        show_completed_indicator=show_completed_indicator,
    )
    
    # Build lines manually for string output
    lines: list[str] = []
    
    # Filter and limit todos for string output
    todos_list = _coerce_sequential_todos(todos)
    completed = [t for t in todos_list if t.get("status") == "completed"]
    active = [t for t in todos_list if t.get("status") in ("in_progress", "pending")]
    
    priority = {"in_progress": 0, "pending": 1}
    active_sorted = sorted(
        active,
        key=lambda todo: priority.get(todo.get("status", "pending"), 2),
    )
    
    items_shown = 0
    
    if show_completed_indicator and len(completed) > max_completed:
        hidden_count = len(completed) - max_completed
        lines.append(f"[{theme.muted_text}]+{hidden_count} more completed[/]")
    
    completed_to_show = completed[-max_completed:]
    for todo in completed_to_show:
        if items_shown >= max_items:
            break
        content = escape(todo.get("content", "").strip())
        icon = _TODO_ICON["completed"]
        lines.append(f"[{theme.success_color}]{icon} [strike]{content}[/strike][/]")
        items_shown += 1
    
    active_shown = 0
    for todo in active_sorted:
        if items_shown >= max_items:
            remaining = len(active_sorted) - active_shown
            if remaining > 0:
                lines.append(f"[{theme.muted_text}]+{remaining} more[/]")
            break
        status = todo.get("status", "pending")
        content = escape(todo.get("content", "").strip())
        icon = _TODO_ICON.get(status, "○")
        
        if status == "in_progress":
            color = theme.info_color
        else:
            color = theme.muted_text
        
        lines.append(f"[{color}]{icon} {content}[/]")
        items_shown += 1
        active_shown += 1

    return "\n".join(lines)


@tool()
def write_todos(
    todos: list[Todo],
    runtime: ToolRuntime[AgentContext, AgentState],
) -> Command:
    """Create and manage structured task lists for tracking progress through complex workflows.

    ## When to Use
    - Multi-step or non-trivial tasks requiring coordination
    - When user provides multiple tasks or explicitly requests todo list
    - Avoid for single, trivial actions unless directed otherwise

    ## Best Practices
    - Only one in_progress task at a time
    - Mark completed immediately when task is fully done
    - Always send the full updated list when making changes
    - Prune irrelevant items to keep list focused

    ## Progress Updates
    - Call write_todos again to change task status or edit content
    - Reflect real-time progress; don't batch completions
    - If blocked, keep in_progress and add new task describing blocker

    Args:
        todos: List of Todo items with content and status

    """
    # Format with high limits to show all todos, no hidden indicator
    formatted_todos = format_todos(
        todos,
        max_items=50,
        max_completed=50,
        show_completed_indicator=False,
    )

    # Add marker for renderer to display as panel
    marked_content = f"{TODO_PANEL_MARKER}{formatted_todos}"

    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    name=write_todos.name,
                    content=f"Updated todo list to {todos}",
                    tool_call_id=runtime.tool_call_id,
                    short_content=marked_content,
                )
            ],
        }
    )


write_todos.metadata = {"approval_config": {"always_approve": True}}


@tool()
def read_todos(
    runtime: ToolRuntime[AgentContext, AgentState],
) -> str:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.
    """
    todos = runtime.state.get("todos")
    if not todos:
        return "No todos currently in the list."

    return format_todos(todos, max_items=50)


read_todos.metadata = {"approval_config": {"always_approve": True}}


# Marker for detecting todo panel in renderer
TODO_PANEL_MARKER = "[TODO_PANEL]"

TODO_TOOLS = [write_todos, read_todos]

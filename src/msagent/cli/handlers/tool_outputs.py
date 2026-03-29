"""Interactive viewer for expandable tool outputs."""

from __future__ import annotations

import shutil
import textwrap
from collections.abc import Callable

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from msagent.cli.core.tool_output import ToolOutputEntry
from msagent.cli.theme import console, theme
from msagent.cli.ui.shared import (
    create_instruction,
    create_selector_application,
)

_VIEWER_BODY_PADDING = 4
_VIEWER_RESERVED_LINES = 7
_PAGE_STEP_LINES = 12


class ToolOutputHandler:
    """Show expandable tool results in an interactive viewer."""

    def __init__(self, session) -> None:
        self.session = session

    async def handle(self) -> None:
        """Open the tool-output viewer when expandable results exist."""
        entries = [
            entry
            for entry in getattr(self.session, "tool_outputs", [])
            if entry.can_expand
        ]
        if not entries:
            latest = getattr(self.session, "latest_tool_output", None)
            if latest is not None and latest.can_expand:
                entries = [latest]
            else:
                console.print_warning("No expandable tool output available yet")
                console.print("")
                return

        context = self.session.context
        state = {
            "index": len(entries) - 1,
            "scroll_offset": 0,
        }

        def current_entry() -> ToolOutputEntry:
            return entries[state["index"]]

        def body_lines() -> list[str]:
            width = max(20, shutil.get_terminal_size((120, 40)).columns - _VIEWER_BODY_PADDING)
            body = (
                current_entry().full_content
                if current_entry().expanded
                else current_entry().preview_content
            )
            if not body:
                body = "(empty output)"

            wrapped: list[str] = []
            for source_line in body.splitlines() or [body]:
                chunks = textwrap.wrap(
                    source_line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
                if chunks:
                    wrapped.extend(chunks)
                else:
                    wrapped.append("")
            return wrapped or ["(empty output)"]

        def page_size() -> int:
            height = shutil.get_terminal_size((120, 40)).lines
            return max(8, height - _VIEWER_RESERVED_LINES)

        def clamp_scroll() -> None:
            total_lines = len(body_lines())
            state["scroll_offset"] = max(
                0,
                min(state["scroll_offset"], max(0, total_lines - page_size())),
            )

        def toggle() -> None:
            entry = current_entry()
            entry.expanded = not entry.expanded
            state["scroll_offset"] = 0

        def move(delta: int) -> None:
            next_index = max(0, min(state["index"] + delta, len(entries) - 1))
            if next_index != state["index"]:
                state["index"] = next_index
                state["scroll_offset"] = 0

        def scroll(delta: int) -> None:
            state["scroll_offset"] += delta
            clamp_scroll()

        def on_message_click(mouse_event: MouseEvent) -> None:
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                toggle()

        def build_text() -> FormattedText:
            entry = current_entry()
            visible_lines = body_lines()
            total_lines = len(visible_lines)
            visible_count = page_size()
            clamp_scroll()
            start = state["scroll_offset"]
            end = min(total_lines, start + visible_count)
            fragments: list[
                tuple[str, str] | tuple[str, str, Callable[[MouseEvent], object]]
            ] = []

            status = "expanded" if entry.expanded else "collapsed"
            header = (
                f"● Tool output {state['index'] + 1}/{len(entries)}"
                f" · {entry.tool_name}"
            )
            if entry.origin_label:
                header += f" · {entry.origin_label}"
            if entry.duration is not None:
                header += f" ({entry.duration:.1f}s)"
            header += f" [{status}]"
            if entry.sequence:
                header += f" · call #{entry.sequence}"
            fragments.append((theme.selection_color, f"{header}\n"))
            fragments.append(
                (
                    "class:muted",
                    "Left/Right: switch tool call | Up/Down/PageUp/PageDown/Home/End: scroll | Click or Ctrl+O/Enter: expand/collapse | Esc: close\n",
                )
            )
            fragments.append(
                (
                    "class:muted",
                    f"Showing lines {start + 1}-{end} of {total_lines}\n\n",
                )
            )

            for line in visible_lines[start:end]:
                fragments.append(("", line, on_message_click))
                fragments.append(("", "\n", on_message_click))

            if end < total_lines:
                fragments.append(
                    (
                        "class:muted",
                        f"\n... {total_lines - end} more line(s) below",
                    )
                )

            return FormattedText(fragments)

        text_control = FormattedTextControl(
            text=build_text,
            focusable=True,
            show_cursor=False,
        )

        kb = KeyBindings()

        @kb.add(Keys.ControlO)
        @kb.add(Keys.Enter)
        def _(event) -> None:
            toggle()
            event.app.invalidate()

        @kb.add(Keys.Left)
        def _(event) -> None:
            move(-1)
            event.app.invalidate()

        @kb.add(Keys.Right)
        def _(event) -> None:
            move(1)
            event.app.invalidate()

        @kb.add(Keys.Up)
        def _(event) -> None:
            scroll(-1)
            event.app.invalidate()

        @kb.add(Keys.Down)
        def _(event) -> None:
            scroll(1)
            event.app.invalidate()

        @kb.add(Keys.PageUp)
        def _(event) -> None:
            scroll(-_PAGE_STEP_LINES)
            event.app.invalidate()

        @kb.add(Keys.PageDown)
        def _(event) -> None:
            scroll(_PAGE_STEP_LINES)
            event.app.invalidate()

        @kb.add(Keys.Home)
        def _(event) -> None:
            state["scroll_offset"] = 0
            event.app.invalidate()

        @kb.add(Keys.End)
        def _(event) -> None:
            state["scroll_offset"] = len(body_lines())
            clamp_scroll()
            event.app.invalidate()

        @kb.add(Keys.Escape)
        @kb.add(Keys.ControlC)
        def _(event) -> None:
            event.app.exit()

        app = create_selector_application(
            context=context,
            content_window=Window(content=text_control, wrap_lines=False),
            key_bindings=kb,
            header_windows=create_instruction(
                "Tool output viewer",
                spacer=True,
            ),
            body_windows=[Window(height=1, char=" ")],
            full_screen=True,
            mouse_support=True,
        )

        await app.run_async()

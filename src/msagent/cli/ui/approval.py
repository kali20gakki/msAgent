"""Interactive approval menu for human-in-the-loop tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.widgets import Frame

from msagent.cli.theme import console, theme
from msagent.cli.ui.shared import NoConsoleScreenBufferError, create_prompt_style
from msagent.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ApprovalChoice:
    """A selected approval option and optional rejection feedback."""

    value: str
    message: str | None = None


@dataclass(frozen=True)
class ApprovalOption:
    """One visible approval menu row."""

    value: str
    label: str
    hotkey: str | None = None


_OPTION_PRESENTATION = {
    "approve": ("Approve", "y"),
    "always_approve": ("Always approve this call", "a"),
    "always_allow_project_script": ("Always allow this project script", "s"),
    "reject": ("Reject", "n"),
    "always_reject": ("Always reject this call", "r"),
}


class ApprovalMenu:
    """A compact deepagents-code-style approval selector."""

    def __init__(
        self,
        *,
        context: Any,
        title: str,
        details: str,
        options: list[str],
        metadata: list[str] | None = None,
        notes: list[str] | None = None,
        option_labels: dict[str, str] | None = None,
    ) -> None:
        self.context = context
        self.title = title
        self.details = details
        self.metadata = metadata or []
        self.notes = notes or []
        self.options = self._build_options(options, option_labels=option_labels)
        self.selected = 0
        self.feedback_active = False
        self.reason_buffer = Buffer(multiline=False)
        self.menu_control = FormattedTextControl(
            text=self._format_menu,
            focusable=True,
            show_cursor=False,
        )
        self.menu_window = Window(
            content=self.menu_control,
            wrap_lines=True,
            dont_extend_height=True,
        )

    @staticmethod
    def _build_options(options: list[str], *, option_labels: dict[str, str] | None = None) -> list[ApprovalOption]:
        unique = list(dict.fromkeys(options))
        priority = {
            "approve": 0,
            "always_approve": 1,
            "always_allow_project_script": 2,
            "reject": 3,
            "always_reject": 4,
        }
        unique.sort(key=lambda value: priority.get(value, len(priority)))
        result = []
        for value in unique:
            label, hotkey = _OPTION_PRESENTATION.get(
                value,
                (value.replace("_", " ").title(), None),
            )
            if option_labels and value in option_labels:
                label = option_labels[value]
            result.append(ApprovalOption(value=value, label=label, hotkey=hotkey))
        return result

    def _format_menu(self) -> FormattedText:
        fragments: list[tuple[str, str]] = [
            ("class:approval.title", f">>> {self.title} <<<\n"),
            ("class:approval.details", self.details.rstrip()),
        ]
        for item in self.metadata:
            fragments.extend([("", "\n"), ("class:approval.metadata", item)])
        for item in self.notes:
            fragments.extend([("", "\n"), ("class:approval.note", item)])
        fragments.append(("", "\n\n"))
        for index, option in enumerate(self.options):
            hotkey = f" ({option.hotkey})" if option.hotkey else ""
            if index == self.selected:
                fragments.append(
                    (
                        "class:approval.selected",
                        f" › {index + 1}. {option.label}{hotkey} ",
                    )
                )
            else:
                fragments.append(("class:approval.option", f"   {index + 1}. {option.label}{hotkey}"))
            if index < len(self.options) - 1:
                fragments.append(("", "\n"))
        return FormattedText(fragments)

    def _format_help(self) -> FormattedText:
        if self.feedback_active:
            return FormattedText(
                [
                    (
                        "class:approval.help",
                        " Enter submit • Esc cancel • leave blank to reject without feedback",
                    )
                ]
            )
        quick_keys = "/".join(option.hotkey for option in self.options if option.hotkey)
        return FormattedText(
            [
                (
                    "class:approval.help",
                    f" ↑/↓ navigate • Enter select • {quick_keys} quick keys • Tab reject with feedback • Esc reject",
                )
            ]
        )

    def _reject_index(self) -> int | None:
        return next(
            (i for i, option in enumerate(self.options) if option.value == "reject"),
            None,
        )

    def _choice(self, index: int, *, message: str | None = None) -> ApprovalChoice:
        return ApprovalChoice(self.options[index].value, message=message)

    def create_application(self) -> Application[ApprovalChoice | None]:
        """Create the prompt-toolkit application used by :meth:`run_async`."""
        kb = KeyBindings()
        feedback = Condition(lambda: self.feedback_active)
        menu_mode = ~feedback

        @kb.add(Keys.Up, filter=menu_mode)
        @kb.add("k", filter=menu_mode)
        def _move_up(event) -> None:
            self.selected = (self.selected - 1) % len(self.options)
            event.app.invalidate()

        @kb.add(Keys.Down, filter=menu_mode)
        @kb.add("j", filter=menu_mode)
        def _move_down(event) -> None:
            self.selected = (self.selected + 1) % len(self.options)
            event.app.invalidate()

        @kb.add(Keys.Enter, filter=menu_mode)
        def _select(event) -> None:
            event.app.exit(result=self._choice(self.selected))

        for index in range(min(9, len(self.options))):

            @kb.add(str(index + 1), filter=menu_mode)
            def _select_position(event, index: int = index) -> None:
                event.app.exit(result=self._choice(index))

        def _bind_hotkey(hotkey: str, index: int) -> None:
            """Register one key without capturing the loop's option object."""

            @kb.add(hotkey, filter=menu_mode)
            def _select_hotkey(event) -> None:
                event.app.exit(result=self._choice(index))

        for index, option in enumerate(self.options):
            if option.hotkey:
                _bind_hotkey(option.hotkey, index)

        reason_control = BufferControl(buffer=self.reason_buffer)

        @kb.add(Keys.Tab, filter=menu_mode)
        def _feedback(event) -> None:
            reject_index = self._reject_index()
            if reject_index is None:
                return
            self.selected = reject_index
            self.feedback_active = True
            self.reason_buffer.text = ""
            event.app.layout.focus(reason_control)
            event.app.invalidate()

        @kb.add(Keys.Enter, filter=feedback)
        def _submit_feedback(event) -> None:
            reject_index = self._reject_index()
            if reject_index is not None:
                message = self.reason_buffer.text.strip() or None
                event.app.exit(result=self._choice(reject_index, message=message))

        @kb.add(Keys.Escape)
        def _escape(event) -> None:
            if self.feedback_active:
                self.feedback_active = False
                event.app.layout.focus(self.menu_control)
                event.app.invalidate()
                return
            reject_index = self._reject_index()
            result = self._choice(reject_index) if reject_index is not None else None
            event.app.exit(result=result)

        @kb.add(Keys.ControlC)
        @kb.add(Keys.ControlD)
        def _cancel(event) -> None:
            event.app.exit(result=None)

        reason_row = ConditionalContainer(
            Window(
                content=reason_control,
                height=1,
                style="class:approval.reason",
                get_line_prefix=lambda _line, _wrap: FormattedText([("class:approval.reason.label", " Feedback: ")]),
            ),
            filter=feedback,
        )
        body = HSplit(
            [
                self.menu_window,
                reason_row,
                Window(
                    content=FormattedTextControl(self._format_help),
                    height=1,
                    dont_extend_height=True,
                ),
            ]
        )
        root = Frame(body=body, style="class:approval.frame")
        style = merge_styles(
            [
                create_prompt_style(self.context, bash_mode=False),
                Style.from_dict(
                    {
                        "approval.frame": f"{theme.warning_color}",
                        "approval.title": f"{theme.warning_color} bold",
                        "approval.details": "default",
                        "approval.metadata": f"{theme.muted_text}",
                        "approval.note": f"{theme.muted_text} italic",
                        "approval.option": "default",
                        "approval.selected": f"{theme.background} bg:{theme.selection_color} bold",
                        "approval.help": f"{theme.muted_text} italic",
                        "approval.reason": f"{theme.primary_text} bg:{theme.background_light}",
                        "approval.reason.label": f"{theme.muted_text} bg:{theme.background_light}",
                    }
                ),
            ]
        )
        layout = Layout(root, focused_element=self.menu_control)
        return Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            erase_when_done=True,
        )

    async def run_async(self) -> ApprovalChoice | None:
        """Display the menu and wait for a decision."""
        try:
            return await self.create_application().run_async()
        except NoConsoleScreenBufferError:
            logger.exception("Unable to open the approval menu without a console screen buffer")
            console.print_warning("Approval cancelled because the terminal has no console screen buffer.")
            return None


def format_approval_details(*, tool_name: str, tool_args: dict[str, Any], description: str | None) -> str:
    """Format the compact tool preview displayed above the menu rows."""
    lines = []
    if description:
        lines.append(str(description))
    if tool_name == "execute" and "command" in tool_args:
        lines.append(str(tool_args.get("command", "")))
    else:
        lines.append(json.dumps(tool_args, ensure_ascii=False, indent=2))
    return "\n".join(line for line in lines if line)

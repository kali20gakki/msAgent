"""Prompt-toolkit session and input handling."""

import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle

from msagent.cli.completers import CompleterRouter
from msagent.cli.core.context import Context
from msagent.cli.ui.shared import create_bottom_toolbar, create_prompt_style
from msagent.core.constants import CONFIG_HISTORY_FILE_NAME
from msagent.core.logging import get_logger
from msagent.core.settings import settings

logger = get_logger(__name__)


class InteractivePrompt:
    """Interactive prompt interface using prompt-toolkit."""

    def __init__(self, context: Context, commands: list[str], session=None):
        self.context = context
        self.commands = commands
        self.session = session
        history_file = Path(context.working_dir) / CONFIG_HISTORY_FILE_NAME
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(history_file))
        self.prompt_session: PromptSession[str]
        self.completer: CompleterRouter
        self.mode_change_callback = None
        self.bash_mode_toggle_callback = None
        self._last_ctrl_c_time: float | None = None
        self._ctrl_c_timeout = 0.30  # 300ms window for double-press detection
        self._show_quit_message = False
        self.hotkeys: dict[str, str] = {}
        self._setup_session()

    def _reset_ctrl_c_state(self) -> None:
        """Clear Ctrl+C timing and banner state."""
        self._last_ctrl_c_time = None
        self._show_quit_message = False

    def reset_interrupt_state(self) -> None:
        """Public wrapper to clear Ctrl+C timing/banner flags."""
        self._reset_ctrl_c_state()

    @staticmethod
    def _format_key_name(key) -> str:
        """Format key enum to human-readable string."""
        key_str = str(key)
        replacements = {
            "Keys.Control": "Ctrl+",
            "Keys.Back": "Shift+",
            "Keys.": "",
        }
        for old, new in replacements.items():
            key_str = key_str.replace(old, new)
        return key_str

    def _setup_session(self) -> None:
        """Set up the prompt session with all configurations."""
        kb = self._create_key_bindings()
        style = create_prompt_style(self.context, bash_mode=self.context.bash_mode)

        self.completer = CompleterRouter(
            commands=self.commands,
            working_dir=Path(self.context.working_dir),
            max_suggestions=settings.cli.max_autocomplete_suggestions,
        )

        self.prompt_session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            complete_style=CompleteStyle.COLUMN,
            key_bindings=kb,
            style=style,
            multiline=False,
            prompt_continuation=lambda width, line_number, is_soft_wrap: " "
            * len(settings.cli.prompt_style),
            wrap_lines=settings.cli.enable_word_wrap,
            mouse_support=False,
            complete_while_typing=True,
            complete_in_thread=False,
            placeholder=self._get_placeholder,
            bottom_toolbar=self._get_bottom_toolbar,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings."""
        kb = KeyBindings()
        self.hotkeys.clear()

        @kb.add(Keys.ControlC)
        def _(event):
            """Ctrl-C: Clear input if text exists, or quit on double-press."""
            self._handle_ctrl_c(event.app, event.current_buffer)

        @kb.add(Keys.ControlJ)
        def _(event):
            """Ctrl-J: Insert newline for multiline input."""
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.BackTab)
        def _(event):
            """Shift-Tab: Cycle approval mode."""
            if self.mode_change_callback:
                self.mode_change_callback()

        @kb.add(Keys.ControlB)
        def _(event):
            """Ctrl-B: Toggle bash mode."""
            if self.bash_mode_toggle_callback:
                self.bash_mode_toggle_callback()

        @kb.add(Keys.ControlK)
        def _(event):
            """Ctrl-K: Show keyboard shortcuts."""
            buffer = event.current_buffer
            buffer.text = "/hotkeys"
            buffer.validate_and_handle()

        @kb.add(Keys.Enter, filter=completion_is_selected)
        def _(event):
            """Enter when completion is selected: apply completion."""
            buffer = event.current_buffer
            if buffer.complete_state:
                current_completion = buffer.complete_state.current_completion
                buffer.apply_completion(current_completion)

                if buffer.text.lstrip().startswith("/"):
                    buffer.validate_and_handle()
                else:
                    buffer.insert_text(" ")

        @kb.add(Keys.Tab)
        def _(event):
            """Tab: apply first completion immediately."""
            buffer = event.current_buffer

            if buffer.complete_state and buffer.complete_state.current_completion:
                current_completion = buffer.complete_state.current_completion
                buffer.apply_completion(current_completion)

                if not buffer.text.lstrip().startswith("/"):
                    buffer.insert_text(" ")
            else:
                buffer.start_completion(select_first=True)
                if buffer.complete_state and buffer.complete_state.current_completion:
                    current_completion = buffer.complete_state.current_completion
                    buffer.apply_completion(current_completion)

                    if not buffer.text.lstrip().startswith("/"):
                        buffer.insert_text(" ")

        self.hotkeys = {
            self._format_key_name(Keys.ControlC): "Clear input (press twice to quit)",
            self._format_key_name(Keys.ControlJ): "Insert newline for multiline input",
            self._format_key_name(Keys.BackTab): "Cycle approval mode",
            self._format_key_name(Keys.ControlB): "Toggle bash mode",
            self._format_key_name(Keys.ControlK): "Show keyboard shortcuts",
            "Tab": "Apply first completion",
            "Enter": "Apply selected completion or submit",
        }

        return kb

    def _handle_ctrl_c(self, app, buffer) -> None:
        """Apply Ctrl+C behavior consistently for key bindings and SIGINT."""
        current_time = time.time()

        if buffer.text.strip():
            buffer.text = ""
            self._reset_ctrl_c_state()
            app.invalidate()
            return

        if self._last_ctrl_c_time is not None:
            time_since_last = current_time - self._last_ctrl_c_time
            # If the quit banner is showing, treat the next Ctrl+C as quit even
            # if the nominal timeout has elapsed; _show_quit_message keeps the
            # window open until the scheduled hide runs.
            if time_since_last < self._ctrl_c_timeout or self._show_quit_message:
                self._reset_ctrl_c_state()
                app.exit(exception=EOFError())
                return

        self._last_ctrl_c_time = current_time
        self._show_quit_message = True
        self._schedule_hide_message(app)
        app.invalidate()

    def handle_external_sigint(self) -> bool:
        """Handle SIGINT while prompt-toolkit is idle and waiting for input."""
        prompt_session = getattr(self, "prompt_session", None)
        if prompt_session is None:
            return False

        app = getattr(prompt_session, "app", None)
        buffer = getattr(app, "current_buffer", None) if app is not None else None
        if app is None or buffer is None or not getattr(app, "is_running", False):
            return False

        self._handle_ctrl_c(app, buffer)
        return True

    def set_mode_change_callback(self, callback):
        """Set callback for mode change events."""
        self.mode_change_callback = callback

    def set_bash_mode_toggle_callback(self, callback):
        """Set callback for bash mode toggle events."""
        self.bash_mode_toggle_callback = callback

    def _get_placeholder(self) -> HTML:
        """Generate placeholder text shown inside the input box."""
        return HTML(f"<placeholder>{self._build_placeholder_text()}</placeholder>")

    def _build_placeholder_text(self) -> str:
        """Build placeholder copy shown inside the input box."""
        return "尽管问msAgent，@ 引用文件，/ 使用命令"

    def _get_bottom_toolbar(self) -> HTML:
        """Generate bottom toolbar text with version, working directory and approval mode."""
        if self._show_quit_message:
            return HTML("<muted> Ctrl+C again to quit</muted>")

        return create_bottom_toolbar(
            self.context,
            str(self.context.working_dir),
            bash_mode=self.context.bash_mode,
        )

    def _schedule_hide_message(self, app):
        """Schedule hiding the quit message after timeout."""

        def hide():
            self._reset_ctrl_c_state()
            app.invalidate()

        try:
            app.loop.call_later(self._ctrl_c_timeout, hide)
        except Exception:
            logger.debug("Hide message timer failed", exc_info=True)
            self._reset_ctrl_c_state()

    def refresh_style(self) -> None:
        """Refresh the prompt style after approval mode change."""
        if self.prompt_session:
            self.prompt_session.style = create_prompt_style(
                self.context, bash_mode=self.context.bash_mode
            )

    async def get_input(self) -> tuple[str, bool]:
        """Get user input asynchronously."""
        try:
            prompt_text = [
                ("class:prompt", settings.cli.prompt_style),
            ]

            default_text = ""
            if self.session and self.session.prefilled_text:
                default_text = self.session.prefilled_text
                self.session.prefilled_text = None

            result = await self.prompt_session.prompt_async(
                prompt_text,
                default=default_text,
                # Let our Ctrl+C key binding handle clear/quit behavior instead of
                # prompt_toolkit turning it into an immediate KeyboardInterrupt.
                handle_sigint=False,
            )  # type: ignore
            print()

            content = result.strip()

            is_command = False
            if content.startswith("/"):
                first_word = content.split()[0] if content.split() else content
                if first_word in self.commands:
                    is_command = True
                elif "/" not in content[1:]:
                    is_command = True

            return content, is_command

        except (KeyboardInterrupt, EOFError):
            raise

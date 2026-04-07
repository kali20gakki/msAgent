"""Interactive chat session management."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from types import FrameType
from typing import TYPE_CHECKING, Any

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.dispatchers import CommandDispatcher, MessageDispatcher
from msagent.cli.handlers.bash import BashDispatcher
from msagent.cli.core.tool_output import ToolOutputEntry
from msagent.cli.theme import console, theme
from msagent.cli.ui.prompt import InteractivePrompt
from msagent.cli.ui.renderer import Renderer
from msagent.core.logging import get_logger
from msagent.utils.version import check_for_updates

if TYPE_CHECKING:
    from msagent.cli.core.context import Context

SignalHandler = Callable[[int, FrameType | None], Any] | int | None

logger = get_logger(__name__)


class Session:
    """Main CLI session manager for interactive chat."""

    def __init__(
        self,
        context: Context,
    ):
        self.context = context
        self.renderer = Renderer(context)
        self.command_dispatcher = CommandDispatcher(self)
        self.message_dispatcher = MessageDispatcher(self)
        self.bash_dispatcher = BashDispatcher(self)
        self.prompt = self._create_prompt_with_fallback()

        if hasattr(self.prompt, "set_mode_change_callback"):
            self.prompt.set_mode_change_callback(self._handle_approval_mode_change)
        if hasattr(self.prompt, "set_bash_mode_toggle_callback"):
            self.prompt.set_bash_mode_toggle_callback(self._handle_bash_mode_toggle)

        # Session state
        self.graph: Any | None = None
        self.graph_context: AbstractAsyncContextManager[Any] | None = None
        self.running = False
        self.needs_reload = False
        self.prefilled_text: str | None = None
        self.prefilled_reference_mapping: dict[str, str] = {}
        self.current_stream_task: asyncio.Task | None = None
        self._sigint_registered = False
        self._previous_sigint: SignalHandler = None
        self._sigint_handler: SignalHandler = None
        self.tool_outputs: list[ToolOutputEntry] = []
        self.latest_tool_output: ToolOutputEntry | None = None

    def _create_prompt_with_fallback(self) -> InteractivePrompt | SimpleNamespace:
        try:
            return InteractivePrompt(
                self.context,
                list(self.command_dispatcher.commands.keys()),
                session=self,
            )
        except Exception:
            logger.debug(
                "Prompt initialization failed, falling back to non-interactive stub",
                exc_info=True,
            )
            return SimpleNamespace(
                hotkeys={},
                handle_external_sigint=lambda: False,
                refresh_style=lambda: None,
                reset_interrupt_state=lambda: None,
                get_input=self._unsupported_get_input,
            )

    async def _unsupported_get_input(self) -> tuple[str, bool]:
        raise RuntimeError("Interactive prompt is unavailable in this environment")

    async def start(self, show_welcome: bool = True) -> None:
        """Start the interactive session."""
        try:
            self.graph_context = initializer.get_graph(
                agent=self.context.agent,
                model=self.context.model,
                working_dir=self.context.working_dir,
            )

            self._register_sigint_handler()

            with console.console.status(
                f"[{theme.spinner_color}]Loading...[/{theme.spinner_color}]"
            ) as status:
                async with self.graph_context as graph:
                    self.graph = graph
                    status.stop()

                    if show_welcome:
                        console.print("")
                        self.renderer.show_welcome(self.context)

                        # Check for updates in background
                        update_task = asyncio.create_task(
                            self._check_updates_background()
                        )
                        await update_task

                    await self._main_loop()
                    status.start()
                    status.update(
                        f"[{theme.spinner_color}]Cleaning...[/{theme.spinner_color}]"
                    )
        finally:
            self._restore_sigint()

    async def _main_loop(self) -> None:
        """Main interactive loop."""
        logger.info("Session started")
        self.running = True

        while self.running:
            try:
                content, is_slash_command = await self.prompt.get_input()

                if not content:
                    continue

                if self.context.bash_mode:
                    await self.bash_dispatcher.dispatch(content)
                    continue

                if is_slash_command:
                    await self.command_dispatcher.dispatch(content)
                    continue

                await self.message_dispatcher.dispatch(content)

            except EOFError:
                break
            except Exception as e:
                console.print_error(f"Error processing input: {e}")
                console.print("")
                logger.exception("Input processing error")

        logger.info("Session ended")

    async def send(self, message: str) -> int:
        """Send a single message in one-shot mode (non-interactive)."""
        try:
            self.graph_context = initializer.get_graph(
                agent=self.context.agent,
                model=self.context.model,
                working_dir=self.context.working_dir,
            )

            self._register_sigint_handler()

            async with self.graph_context as graph:
                self.graph = graph

                await self.message_dispatcher.dispatch(message)
                return 0

        except KeyboardInterrupt:
            return 0
        except Exception as e:
            console.print_error(f"Error sending message: {e}")
            console.print("")
            logger.exception("CLI message error")
            return 1
        finally:
            self._restore_sigint()

    def update_context(self, **kwargs) -> None:
        """Update context fields dynamically.

        Args:
            **kwargs: Context fields to update (thread_id, agent, model,
                     current_input_tokens, current_output_tokens, context_window, etc.)
        """
        # Fields that trigger reload
        if "agent" in kwargs or "model" in kwargs:
            self.needs_reload = True
            self.running = False

        # Update all fields
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    def remember_tool_output(self, entry: ToolOutputEntry) -> None:
        """Store or refresh an expandable tool output for the viewer."""
        for index, existing in enumerate(self.tool_outputs):
            if existing.tool_call_id and existing.tool_call_id == entry.tool_call_id:
                entry.sequence = existing.sequence
                self.tool_outputs[index] = entry
                self.latest_tool_output = entry
                return

        entry.sequence = len(self.tool_outputs) + 1
        self.tool_outputs.append(entry)
        self.latest_tool_output = entry

    def clear_tool_output(self) -> None:
        """Drop remembered expandable tool outputs."""
        self.tool_outputs = []
        self.latest_tool_output = None

    def _handle_approval_mode_change(self) -> None:
        """Handle approval mode cycling from keyboard shortcut."""
        self.context.cycle_approval_mode()
        # Refresh the prompt style to reflect the new mode
        self.prompt.refresh_style()

    def _handle_bash_mode_toggle(self) -> None:
        """Handle bash mode toggle from keyboard shortcut."""
        self.context.toggle_bash_mode()
        # Refresh the prompt style to reflect the new mode
        self.prompt.refresh_style()

    def _register_sigint_handler(self) -> None:
        """Install SIGINT handler that cancels the active stream before exit.

        Contract: first Ctrl+C cancels any in-flight stream task; subsequent
        Ctrl+C follows the previous handler (which, in interactive mode, is the
        prompt's double-press-to-quit logic). One-shot and interactive paths
        share this handler to keep behavior consistent.
        """
        if self._sigint_registered and self._sigint_handler is not None:
            try:
                if signal.getsignal(signal.SIGINT) == self._sigint_handler:
                    return
            except Exception:
                return

        try:
            self._previous_sigint = signal.getsignal(signal.SIGINT)

            def _handle_sigint(signum, frame):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if (
                    loop
                    and self.current_stream_task
                    and not self.current_stream_task.done()
                ):
                    loop.call_soon_threadsafe(self.current_stream_task.cancel)
                    return

                if self.prompt.handle_external_sigint():
                    return

                if callable(self._previous_sigint):
                    self._previous_sigint(signum, frame)
                    return

                if self._previous_sigint == signal.SIG_IGN:
                    return

                raise KeyboardInterrupt()

            self._sigint_handler = _handle_sigint
            signal.signal(signal.SIGINT, self._sigint_handler)
            self._sigint_registered = True
        except Exception as e:
            logger.exception("Failed to register SIGINT handler", exc_info=e)

    def _restore_sigint(self) -> None:
        """Restore previous SIGINT handler if we overrode it."""
        if not self._sigint_registered:
            return

        try:
            if (
                self._sigint_handler is not None
                and signal.getsignal(signal.SIGINT) != self._sigint_handler
            ):
                return

            signal.signal(
                signal.SIGINT,
                (
                    self._previous_sigint
                    if self._previous_sigint is not None
                    else signal.SIG_DFL
                ),
            )
        except Exception:
            pass
        finally:
            self._sigint_registered = False
            self._previous_sigint = None
            self._sigint_handler = None

    async def _check_updates_background(self) -> None:
        """Check for updates in background without blocking prompt."""
        try:
            updates = await asyncio.to_thread(check_for_updates)
            if updates:
                latest_version, upgrade_command = updates
                if latest_version and upgrade_command:
                    console.print_warning(
                        f"[muted]New version available ({latest_version}). Upgrade with: [muted.bold]{upgrade_command}[/muted.bold][/muted]"
                    )
                    console.print("")
        except Exception:
            pass

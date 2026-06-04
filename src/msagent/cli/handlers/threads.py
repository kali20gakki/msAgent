"""Browse and resume previous conversation threads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console, theme
from msagent.cli.ui.shared import (
    SelectorState,
    create_instruction,
    create_selector_application,
)
from msagent.core.logging import get_logger
from msagent.core.settings import settings
from msagent.utils.time import format_relative_time

logger = get_logger(__name__)


@dataclass(slots=True)
class ThreadEntry:
    """Minimal thread metadata needed by the selector."""

    thread_id: str
    preview: str
    timestamp: str
    pending_interrupts: list[Any]
    offloaded_path: str | None = None


class ThreadsHandler:
    """Handles browsing and restoring prior conversation threads."""

    def __init__(self, session) -> None:
        self.session = session

    async def handle(self) -> None:
        """Browse previous threads and restore the selected one."""
        try:
            entries = await self._load_thread_entries()
            if not entries:
                console.print_warning("No previous conversation threads found")
                console.print("")
                return

            selected_thread_id = await self._select_thread(entries)
            if not selected_thread_id:
                return

            await self._load_thread(selected_thread_id, render_history=True)
        except Exception as exc:
            console.print_error(f"Error browsing threads: {exc}")
            console.print("")
            logger.debug("Thread browser failed", exc_info=True)

    async def _load_thread_entries(self) -> list[ThreadEntry]:
        """Read the latest checkpoint for each prior thread."""
        ctx = self.session.context
        entries: list[ThreadEntry] = []

        async with initializer.get_checkpointer(ctx.agent, ctx.working_dir) as checkpointer:
            conn = getattr(checkpointer, "conn", None)
            if conn is not None:
                # Fast path: one GROUP BY query to get unique thread IDs, then a
                # targeted aget_tuple per thread instead of streaming all rows.
                await checkpointer.setup()
                async with conn.execute(
                    "SELECT thread_id FROM checkpoints "
                    "WHERE checkpoint_ns = '' AND thread_id != ? "
                    "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC",
                    (ctx.thread_id,),
                ) as cursor:
                    thread_ids = [row[0] async for row in cursor]

                for thread_id in thread_ids:
                    entry = await self._build_entry_from_tuple(checkpointer, thread_id)
                    if entry is not None:
                        entries.append(entry)
            else:
                # Fallback for InMemorySaver (no raw connection available).
                seen_thread_ids: set[str] = set()
                async for checkpoint_tuple in checkpointer.alist(None):
                    thread_id = str(checkpoint_tuple.config.get("configurable", {}).get("thread_id", ""))
                    if not thread_id or thread_id == ctx.thread_id or thread_id in seen_thread_ids:
                        continue
                    entry = self._entry_from_checkpoint_tuple(checkpoint_tuple)
                    if entry is not None:
                        entries.append(entry)
                    seen_thread_ids.add(thread_id)

        return entries

    async def _build_entry_from_tuple(self, checkpointer, thread_id: str) -> "ThreadEntry | None":
        """Fetch the latest checkpoint tuple for a thread and build a ThreadEntry."""
        checkpoint_tuple = await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))
        if checkpoint_tuple is None:
            return None
        return self._entry_from_checkpoint_tuple(checkpoint_tuple)

    @staticmethod
    def _entry_from_checkpoint_tuple(checkpoint_tuple) -> "ThreadEntry | None":
        """Build a ThreadEntry from a CheckpointTuple, or None if it has no messages."""
        thread_id = str(checkpoint_tuple.config.get("configurable", {}).get("thread_id", ""))
        if not thread_id:
            return None

        checkpoint = checkpoint_tuple.checkpoint or {}
        channel_values = checkpoint.get("channel_values", {})
        messages = list(channel_values.get("messages", []) or [])
        if not messages:
            return None

        summarization_event = channel_values.get("_summarization_event")
        offloaded_path: str | None = None
        if isinstance(summarization_event, dict):
            offloaded_path = summarization_event.get("file_path") or None

        return ThreadEntry(
            thread_id=thread_id,
            preview=ThreadsHandler._build_preview(messages),
            timestamp=str(checkpoint.get("ts", "")),
            pending_interrupts=ThreadsHandler._extract_interrupts(checkpoint_tuple.pending_writes or []),
            offloaded_path=offloaded_path,
        )

    async def _select_thread(self, entries: list[ThreadEntry]) -> str:
        """Show a compact keyboard-driven selector for previous threads."""
        if not entries:
            return ""

        state = SelectorState(window_size=8)
        selected = [False]

        text_control = FormattedTextControl(
            text=lambda: self._format_thread_list(
                entries,
                selected_index=state.index,
                scroll_offset=state.scroll_offset,
                window_size=state.window_size or 8,
            ),
            focusable=True,
            show_cursor=False,
        )

        kb = KeyBindings()

        @kb.add(Keys.Up)
        def _(_event) -> None:
            state.move_linear(-1, size=len(entries))

        @kb.add(Keys.Down)
        def _(_event) -> None:
            state.move_linear(1, size=len(entries))

        @kb.add(Keys.Enter)
        def _(event) -> None:
            selected[0] = True
            event.app.exit()

        @kb.add(Keys.Escape)
        @kb.add(Keys.ControlC)
        def _(event) -> None:
            event.app.exit()

        context = self.session.context
        app = create_selector_application(
            context=context,
            text_control=text_control,
            key_bindings=kb,
            header_windows=create_instruction(
                "Select a thread to restore. Use Up/Down and press Enter.",
                spacer=True,
            ),
            body_windows=[Window(height=1, char=" ")],
        )

        try:
            await app.run_async()
        except (KeyboardInterrupt, EOFError):
            return ""

        if selected[0]:
            return entries[state.index].thread_id
        return ""

    async def _load_thread(self, thread_id: str, *, render_history: bool) -> None:
        """Restore a thread into the current interactive session."""
        ctx = self.session.context

        async with initializer.get_checkpointer(ctx.agent, ctx.working_dir) as checkpointer:
            checkpoint_tuple = await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))

        if checkpoint_tuple is None:
            console.print_error("No conversation history found for that thread")
            console.print("")
            return

        checkpoint = checkpoint_tuple.checkpoint or {}
        channel_values = checkpoint.get("channel_values", {})
        messages = list(channel_values.get("messages", []) or [])

        self.session.update_context(
            thread_id=thread_id,
            current_input_tokens=channel_values.get("current_input_tokens"),
            current_output_tokens=channel_values.get("current_output_tokens"),
        )
        clear_tool_output = getattr(self.session, "clear_tool_output", None)
        if callable(clear_tool_output):
            clear_tool_output()
        logger.info("Thread ID: %s", thread_id)

        if render_history:
            console.clear()
            for message in messages:
                self.session.renderer.render_message(message)
            console.print_success(f"Restored thread {thread_id}")
            console.print("")

        pending_interrupts = self._extract_interrupts(checkpoint_tuple.pending_writes or [])
        if pending_interrupts:
            await self.session.message_dispatcher.resume_from_interrupt(thread_id, pending_interrupts)

    @staticmethod
    def _build_preview(messages: list[AnyMessage]) -> str:
        """Choose a stable, human-friendly preview for a thread row."""
        preferred_message: AnyMessage | None = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                preferred_message = message
                break

        selected_message = preferred_message or messages[-1]
        raw_preview = (
            getattr(selected_message, "short_content", None)
            or getattr(selected_message, "text", None)
            or str(getattr(selected_message, "content", ""))
        )
        preview = str(raw_preview).replace("\n", " ").strip()
        if len(preview) > 72:
            preview = f"{preview[:69]}..."
        return preview or "No preview available"

    @staticmethod
    def _extract_interrupts(pending_writes: list[tuple[Any, ...]]) -> list[Any]:
        """Extract pending interrupt payloads from checkpoint writes."""
        interrupts: list[Any] = []
        for pending_write in pending_writes:
            if len(pending_write) < 3:
                continue
            _task_id, channel, value = pending_write[:3]
            if channel != "__interrupt__":
                continue
            if isinstance(value, list):
                interrupts.extend(value)
            else:
                interrupts.append(value)
        return interrupts

    @staticmethod
    def _format_thread_list(
        entries: list[ThreadEntry],
        *,
        selected_index: int,
        scroll_offset: int,
        window_size: int,
    ) -> FormattedText:
        """Render the visible portion of the thread selector."""
        prompt_symbol = settings.cli.prompt_style.strip()
        lines: list[tuple[str, str]] = []
        visible_entries = entries[scroll_offset : scroll_offset + window_size]

        for idx, entry in enumerate(visible_entries):
            actual_index = scroll_offset + idx
            relative_time = format_relative_time(entry.timestamp) if entry.timestamp else "unknown"
            suffix = " [pending approval]" if entry.pending_interrupts else ""
            if entry.offloaded_path:
                suffix += " [history offloaded]"
            display_text = f"[{relative_time}] {entry.preview}{suffix}"

            if actual_index == selected_index:
                lines.append((theme.selection_color, f"{prompt_symbol} {display_text}"))
            else:
                lines.append(("", f"  {display_text}"))

            if idx < len(visible_entries) - 1:
                lines.append(("", "\n"))

        return FormattedText(lines)

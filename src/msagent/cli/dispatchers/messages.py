"""Message handling for chat sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from json_repair import loads as repair_loads
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Overwrite
from rich.console import Group, RenderableType
from rich.live import Live
from rich.measure import Measurement
from rich.spinner import Spinner
from rich.text import Text

from msagent.agents.context import AgentContext, RetryNotice
from msagent.agents.local_context import build_local_environment_context
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.builders import MessageContentBuilder
from msagent.cli.core.tool_output import ToolOutputEntry
from msagent.cli.handlers import CompressionHandler, InterruptHandler
from msagent.cli.theme import console, theme
from msagent.cli.ui.renderer import Renderer
from msagent.core.constants import OS_VERSION, PLATFORM
from msagent.core.logging import get_logger
from msagent.middlewares.token_cost import extract_usage_counts
from msagent.utils.compression import should_auto_compress
from msagent.utils.render import TOOL_TIMING_RESPONSE_METADATA_KEY

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

    from langgraph.types import Interrupt

logger = get_logger(__name__)

_TOOL_RUNNING_DOT = "●"
_TOOL_ACTIVITY_CYCLE_S = 0.75
_TOOL_DOT_ON_STYLE = "indicator"
_TOOL_DOT_OFF_STYLE = "muted"
_TOOL_SWEEP_WIDTH = 8
_TOOL_SWEEP_EDGE_STYLE = "secondary"
_TOOL_SWEEP_CORE_STYLE = "accent.secondary bold"
_TOOL_PREFIX_STYLE = "accent"
_TOOL_NAME_STYLE = "primary"
_TOOL_ARG_KEY_STYLE = "muted"
_TOOL_ARG_VALUE_STYLE = "primary"
_TOOL_ARG_SEPARATOR_STYLE = "muted"
_TOOL_LIVE_SUMMARY_VALUE_MAX = 72


@dataclass(frozen=True, slots=True)
class ToolActivityCall:
    """Compact preview of a tool call for the live activity area."""

    name: str
    args: dict[str, Any]
    call_id: str | None = None
    start_time: float = field(default_factory=time.monotonic)

    def __eq__(self, other: object) -> bool:
        """Compare ToolActivityCall instances, ignoring start_time."""
        if not isinstance(other, ToolActivityCall):
            return NotImplemented
        return (
            self.name == other.name
            and self.args == other.args
            and self.call_id == other.call_id
        )

    def __hash__(self) -> int:
        """Hash based on name, args, and call_id only."""
        return hash((self.name, self.call_id, tuple(sorted(self.args.items()))))


@dataclass(frozen=True, slots=True)
class DeferredToolHeader:
    """Cached tool call header rendered when the matching tool result arrives."""

    tool_call: dict[str, Any]
    indent_level: int
    origin_label: str | None
    started_at: float


@dataclass
class ToolActivityIndicator:
    """A tool activity line with a blinking dot and looping sweep highlight."""

    text: Text
    details: Text | None = None
    cycle_seconds: float = _TOOL_ACTIVITY_CYCLE_S
    sweep_width: int = _TOOL_SWEEP_WIDTH
    glyph: str = _TOOL_RUNNING_DOT
    start_time: float | None = None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self.render(console.get_time())

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        return Measurement.get(console, options, self.render(console.get_time()))

    def render(self, time: float) -> RenderableType:
        """Render the indicator at a given time."""
        if self.start_time is None:
            self.start_time = time

        elapsed = max(time - self.start_time, 0.0)
        phase = (elapsed % self.cycle_seconds) / self.cycle_seconds
        dot_style = _TOOL_DOT_ON_STYLE if phase < 0.5 else _TOOL_DOT_OFF_STYLE
        dot = Text(self.glyph, style=dot_style)

        label = self.text.copy()
        self._apply_sweep(label, phase)
        
        # Add elapsed time (Claude Code style)
        elapsed_text = Text(f" ({elapsed:.1f}s)", style="muted")
        
        header = Text.assemble(dot, " ", label, elapsed_text)
        if self.details:
            return Group(header, self.details)
        return header

    def _apply_sweep(self, label: Text, phase: float) -> None:
        """Apply a moving highlight band across the label text."""
        plain = label.plain
        content_start = len(plain) - len(plain.lstrip(" "))
        content_length = len(plain) - content_start
        if content_length <= 0:
            return

        travel = content_length + self.sweep_width
        band_start = content_start + int(phase * travel) - self.sweep_width
        band_end = band_start + self.sweep_width

        highlight_start = max(content_start, band_start)
        highlight_end = min(len(plain), band_end)
        if highlight_start >= highlight_end:
            return

        label.stylize(_TOOL_SWEEP_EDGE_STYLE, highlight_start, highlight_end)

        core_inset = max(1, self.sweep_width // 4)
        core_start = max(content_start, band_start + core_inset)
        core_end = min(len(plain), band_end - core_inset)
        if core_start < core_end:
            label.stylize(_TOOL_SWEEP_CORE_STYLE, core_start, core_end)


class MessageDispatcher:
    """Dispatch user message processing and AI response streaming."""

    _MAX_LOG_VALUE_LENGTH = 400
    _SUBAGENT_ORIGIN_LABEL = "Subagent"
    
    # Tool names that should be hidden from activity display (shown via other means)
    _HIDDEN_ACTIVITY_TOOLS: set[str] = {"write_todos"}

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session
        self.interrupt_handler = InterruptHandler(session=session)
        self.message_builder = MessageContentBuilder(Path(session.context.working_dir))
        self._pending_compression = False
        self._pending_tool_headers: dict[str, DeferredToolHeader] = {}

    async def dispatch(self, content: str) -> None:
        """Dispatch user message and get AI response."""
        try:
            reference_mapping = self.session.prefilled_reference_mapping.copy()
            self.session.prefilled_reference_mapping.clear()

            message_content, image_refs = self.message_builder.build(content)

            reference_mapping.update(image_refs)

            human_message = HumanMessage(
                content=message_content,
                short_content=content,
                additional_kwargs={"reference_mapping": reference_mapping},
            )
            ctx = self.session.context
            agent_context = await self._build_agent_context()

            graph_config = RunnableConfig(
                configurable={"thread_id": ctx.thread_id},
                recursion_limit=ctx.recursion_limit,
            )

            if ctx.stream_output:
                await self._stream_response(
                    {"messages": [human_message]},
                    graph_config,
                    agent_context,
                )
            else:
                await self._invoke_without_stream(
                    {"messages": [human_message]},
                    graph_config,
                    agent_context,
                )

        except Exception as e:
            error_msg = self._format_console_error(e)
            console.print_error(f"Error processing message: {error_msg}")
            console.print("")
            await self._log_processing_error(e)

    async def _build_agent_context(self) -> AgentContext:
        """Build runtime context injected into prompt templates and middleware."""
        ctx = self.session.context
        now = datetime.now(timezone.utc).astimezone()
        user_memory = await initializer.load_user_memory(ctx.working_dir)
        return AgentContext(
            approval_mode=ctx.approval_mode,
            working_dir=ctx.working_dir,
            platform=PLATFORM,
            os_version=OS_VERSION,
            current_date_time_zoned=now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            local_environment_context=build_local_environment_context(
                ctx.working_dir,
                now=now,
            ),
            mcp_servers=(
                ", ".join(initializer.cached_mcp_server_names)
                if initializer.cached_mcp_server_names
                else "None"
            ),
            user_memory=user_memory,
            tool_catalog=cast(list[BaseTool], initializer.cached_tools_in_catalog),
            skill_catalog=initializer.cached_agent_skills,
            tool_output_max_tokens=ctx.tool_output_max_tokens,
        )

    async def _stream_response(
        self,
        input_data: dict[str, Any] | Command,
        config: RunnableConfig,
        context: AgentContext,
    ) -> None:
        """Stream with automatic interrupt handling loop."""
        self._pending_compression = False
        current_input: dict[str, Any] | Command = input_data
        rendered_messages: set[str] = set()
        streaming_states: dict[tuple, dict[str, Any]] = {}
        active_tools: dict[tuple, list[ToolActivityCall]] = {}
        thinking_previews: dict[tuple, list[str]] = {}
        status: Live | None = None

        def handle_retry_notice(notice: RetryNotice) -> None:
            self._render_retry_notice(notice, live=status)

        self.session.current_stream_task = asyncio.current_task()
        context.retry_notice_handler = handle_retry_notice

        try:
            while True:
                interrupted = False
                cancelled = False
                status = None

                try:
                    with Live(
                        self._build_activity_renderable(active_tools, thinking_previews),
                        console=console.console,
                        transient=True,
                        refresh_per_second=24,
                    ) as status:
                        async for chunk in self.session.graph.astream(
                            current_input,
                            config,
                            context=context,
                            stream_mode=["messages", "updates"],
                            subgraphs=True,
                        ):
                            interrupts = self._extract_interrupts(chunk)
                            if interrupts:
                                # Clear all active streaming states on interrupt
                                for state in streaming_states.values():
                                    self._clear_preview(state)
                                active_tools.clear()
                                thinking_previews.clear()
                                status.stop()
                                resume_value = await self.interrupt_handler.handle(
                                    interrupts
                                )
                                # Sync approval mode from session context in case it changed during interrupt
                                context.approval_mode = (
                                    self.session.context.approval_mode
                                )

                                if isinstance(resume_value, dict):
                                    current_input = Command(resume=resume_value)
                                else:
                                    current_input = Command(
                                        resume={interrupts[0].id: resume_value}
                                    )
                                interrupted = True
                                break

                            namespace, mode, data = chunk

                            if mode == "messages":
                                await self._process_message_chunk(
                                    data,
                                    namespace,
                                    streaming_states,
                                    status,
                                    rendered_messages,
                                    active_tools,
                                    thinking_previews,
                                )
                            elif mode == "updates":
                                await self._finalize_streaming(
                                    namespace,
                                    streaming_states,
                                    status,
                                    rendered_messages,
                                    active_tools,
                                    thinking_previews,
                                )
                                await self._process_update_chunk(
                                    data,
                                    namespace,
                                    rendered_messages,
                                    status,
                                    active_tools,
                                    thinking_previews,
                                )

                except (asyncio.CancelledError, KeyboardInterrupt):
                    if status:
                        status.stop()
                    self.session.prompt.reset_interrupt_state()
                    await self._finalize_all_streaming(
                        streaming_states,
                        status,
                        rendered_messages,
                        active_tools,
                        thinking_previews,
                    )
                    cancelled = True

                if cancelled:
                    break

                if not interrupted:
                    await self._finalize_all_streaming(
                        streaming_states,
                        status,
                        rendered_messages,
                        active_tools,
                        thinking_previews,
                    )
                    break

            if self._pending_compression and not cancelled:
                self._pending_compression = False
                try:
                    await self._execute_compression()
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
        finally:
            context.retry_notice_handler = None
            self.session.current_stream_task = None

    async def _invoke_without_stream(
        self,
        input_data: dict[str, Any] | Command,
        config: RunnableConfig,
        context: AgentContext,
    ) -> None:
        """Run a request without token-by-token rendering."""
        context.retry_notice_handler = self._render_retry_notice
        try:
            result = await self.session.graph.ainvoke(input_data, config, context=context)
        finally:
            context.retry_notice_handler = None

        if not isinstance(result, dict):
            return

        await self._update_token_tracking(result)

        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, (AIMessage, ToolMessage)):
                self.session.renderer.render_message(message)
                if isinstance(message, ToolMessage):
                    self._remember_expandable_tool_output(message, indent_level=0)
                break

    async def _log_processing_error(self, error: Exception) -> None:
        """Log a message-processing failure with LLM and HTTP context."""
        log_fields = [
            f"thread_id={self.session.context.thread_id}",
            f"agent={self.session.context.agent}",
            f"model_alias={self.session.context.model}",
        ]
        log_fields.extend(self._resolve_error_log_fields(error))
        log_fields.extend(await self._resolve_llm_log_fields())
        log_fields.extend(self._resolve_http_log_fields(error))

        exception_chain = self._format_exception_chain(error)
        if exception_chain:
            log_fields.append(f"exception_chain={exception_chain}")

        logger.error(
            "Message processing error [%s]",
            ", ".join(log_fields),
            exc_info=error,
        )

    @classmethod
    def _resolve_error_log_fields(cls, error: BaseException) -> list[str]:
        """Extract message-specific error details for verbose logs."""
        fields = [
            f"console_error={cls._format_console_error(error)}",
            f"exception_type={type(error).__name__}",
        ]

        message = (str(error) or type(error).__name__).strip()
        if message:
            fields.append(f"exception_message={cls._truncate_log_value(message)}")

        error_repr = repr(error).strip()
        if error_repr and error_repr != message:
            fields.append(f"exception_repr={cls._truncate_log_value(error_repr)}")

        return fields

    async def _resolve_llm_log_fields(self) -> list[str]:
        """Resolve the current model configuration for error logging."""
        try:
            llm_config = await initializer.load_llm_config(
                self.session.context.model,
                Path(self.session.context.working_dir),
            )
        except Exception:
            logger.debug("Failed to resolve LLM config for error logging", exc_info=True)
            return []

        fields = [
            f"provider={llm_config.provider.value}",
            f"resolved_model={llm_config.model}",
        ]
        if llm_config.base_url:
            fields.append(f"base_url={llm_config.base_url}")
        return fields

    @classmethod
    def _resolve_http_log_fields(cls, error: BaseException) -> list[str]:
        """Extract request/response context from the exception chain."""
        fields: list[str] = []
        request = cls._find_exception_attr(error, "request")
        if request is not None:
            method = getattr(request, "method", None)
            url = getattr(request, "url", None)
            if method and url:
                fields.append(f"request={method} {url}")
            elif url:
                fields.append(f"request={url}")

        status_code = cls._find_exception_attr(error, "status_code")
        if status_code is not None:
            fields.append(f"status_code={status_code}")

        response_body = cls._extract_response_body(error)
        if response_body:
            fields.append(f"response_body={response_body}")

        return fields

    @classmethod
    def _extract_response_body(cls, error: BaseException) -> str | None:
        """Extract and normalize an API response body from the exception chain."""
        body = cls._find_exception_attr(error, "body")
        if body is not None and body != "":
            return cls._truncate_log_value(repr(body))

        response = cls._find_exception_attr(error, "response")
        if response is not None:
            text = getattr(response, "text", None)
            if text:
                return cls._truncate_log_value(str(text))

        return None

    @classmethod
    def _format_console_error(cls, error: BaseException) -> str:
        """Build a concise terminal-friendly error message."""
        message = (str(error) or type(error).__name__).strip()
        if message != "Connection error.":
            return message

        chain = list(cls._walk_exception_chain(error))
        for cause in chain[1:]:
            cause_message = str(cause).strip()
            if cause_message and cause_message != message:
                return cls._truncate_log_value(
                    f"{message} Cause: {type(cause).__name__}: {cause_message}",
                    limit=200,
                )
        return message

    @classmethod
    def _format_exception_chain(cls, error: BaseException) -> str:
        """Format the causal exception chain into a compact single line."""
        parts: list[str] = []
        for current in cls._walk_exception_chain(error):
            message = (str(current) or type(current).__name__).strip()
            parts.append(
                f"{type(current).__name__}: "
                f"{cls._truncate_log_value(message, limit=160)}"
            )
        return " <- ".join(parts)

    @staticmethod
    def _walk_exception_chain(
        error: BaseException, max_depth: int = 8
    ) -> list[BaseException]:
        """Collect the exception and its direct causes without looping forever."""
        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = error

        while current is not None and id(current) not in seen and len(chain) < max_depth:
            chain.append(current)
            seen.add(id(current))
            current = current.__cause__ or current.__context__

        return chain

    @classmethod
    def _find_exception_attr(cls, error: BaseException, attr: str) -> Any | None:
        """Return the first non-empty attribute found in the exception chain."""
        for current in cls._walk_exception_chain(error):
            value = getattr(current, attr, None)
            if value is not None and value != "":
                return value
        return None

    @classmethod
    def _truncate_log_value(cls, value: str, limit: int | None = None) -> str:
        """Keep log fields compact and single-line."""
        max_length = limit or cls._MAX_LOG_VALUE_LENGTH
        normalized = " ".join(value.split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[: max_length - 3]}..."

    @staticmethod
    def _format_retry_delay(delay: float) -> str:
        """Format retry delays for compact TUI feedback."""
        rounded = round(delay, 1)
        if rounded.is_integer():
            return f"{int(rounded)}s"
        return f"{rounded:.1f}s"

    @classmethod
    def _format_retry_notice_text(cls, notice: RetryNotice) -> str:
        """Render a short human-friendly retry hint."""
        delay_text = cls._format_retry_delay(notice.delay)
        if notice.scope == "tool":
            target_name = notice.target_name or "unknown"
            return (
                f"Tool {target_name} 重试 {notice.attempt}/{notice.max_retries}，"
                f"{delay_text} 后重试"
            )
        return f"LLM 重试 {notice.attempt}/{notice.max_retries}，{delay_text} 后重试"

    @classmethod
    def _render_retry_notice(
        cls,
        notice: RetryNotice,
        live: Live | None = None,
    ) -> None:
        """Show a visible retry prompt in the TUI."""
        if notice.phase != "scheduled":
            return

        text = Text()
        text.append("⚠︎", style="warning")
        text.append(" ")
        text.append(cls._format_retry_notice_text(notice), style="warning")

        if live is not None:
            live.console.print(text)
            return

        console.print_warning(cls._format_retry_notice_text(notice))

    @staticmethod
    def _extract_interrupts(chunk) -> list[Interrupt] | None:
        """Extract interrupt data from chunk."""
        if isinstance(chunk, tuple) and len(chunk) == 3:
            _namespace, _mode, data = chunk
            if isinstance(data, dict):
                return data.get("__interrupt__")
        elif isinstance(chunk, dict):
            return chunk.get("__interrupt__")
        return None

    @staticmethod
    def _unwrap_overwrite(value: Any) -> Any:
        """Unwrap LangGraph Overwrite wrapper values when present."""
        if isinstance(value, Overwrite):
            return value.value
        return value

    @staticmethod
    def _get_stable_message_id(message: AnyMessage) -> str:
        """Get a stable ID for deduplication, even when message.id is None.

        Returns a base ID without type suffix. Caller should append type if needed.
        """
        if message.id:
            return message.id

        content_str = str(message.content) if message.content else ""
        stable_key = hashlib.sha256(
            f"{content_str}:{message.type}".encode()
        ).hexdigest()[:8]
        return stable_key

    def _get_streaming_state(
        self, namespace: tuple, streaming_states: dict[tuple, dict[str, Any]]
    ) -> dict[str, Any]:
        """Get or create streaming state for namespace."""
        if namespace not in streaming_states:
            streaming_states[namespace] = {
                "active": False,
                "message_id": None,
                "preview_lines": [""],
                "chunks": [],
                "namespace": namespace,
            }
        return streaming_states[namespace]

    @staticmethod
    def _extract_tool_name(tool_call: Any) -> str | None:
        """Extract a tool name from a tool call payload or chunk."""
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

            function = tool_call.get("function")
            if isinstance(function, dict):
                function_name = function.get("name")
                if isinstance(function_name, str) and function_name.strip():
                    return function_name.strip()

        name = getattr(tool_call, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()

        function = getattr(tool_call, "function", None)
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str) and function_name.strip():
                return function_name.strip()
        else:
            function_name = getattr(function, "name", None)
            if isinstance(function_name, str) and function_name.strip():
                return function_name.strip()

        return None

    @staticmethod
    def _extract_tool_args(tool_call: Any) -> dict[str, Any]:
        """Extract normalized args from a tool call payload."""
        args: Any = None
        if isinstance(tool_call, dict):
            args = tool_call.get("args")
            if args is None:
                function = tool_call.get("function")
                if isinstance(function, dict):
                    args = function.get("arguments")
        else:
            args = getattr(tool_call, "args", None)
            if args is None:
                function = getattr(tool_call, "function", None)
                if isinstance(function, dict):
                    args = function.get("arguments")
                else:
                    args = getattr(function, "arguments", None)

        if isinstance(args, dict):
            return args

        if isinstance(args, str):
            stripped = args.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    parsed = repair_loads(stripped)
                except Exception:
                    return {}
            return parsed if isinstance(parsed, dict) else {}

        return {}

    @staticmethod
    def _extract_tool_call_id(tool_call: Any) -> str | None:
        """Extract a stable tool call identifier when present."""
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
        else:
            call_id = getattr(tool_call, "id", None)
        return str(call_id) if call_id else None

    @classmethod
    def _extract_tool_call_preview(
        cls, tool_call: Any, start_time: float | None = None
    ) -> ToolActivityCall | None:
        """Build a live preview entry from a tool call payload."""
        name = cls._extract_tool_name(tool_call)
        if not name:
            return None
        return ToolActivityCall(
            name=name,
            args=cls._extract_tool_args(tool_call),
            call_id=cls._extract_tool_call_id(tool_call),
            start_time=start_time if start_time is not None else time.monotonic(),
        )

    @classmethod
    def _extract_tool_call_names(cls, message: AIMessage | AIMessageChunk) -> list[str]:
        """Collect tool names from streamed or final AI messages."""
        return [preview.name for preview in cls._extract_tool_call_previews(message)]

    @classmethod
    def _extract_tool_call_previews(
        cls, message: AIMessage | AIMessageChunk
    ) -> list[ToolActivityCall]:
        """Collect live previews for streamed or final tool calls."""
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_call_chunks = getattr(message, "tool_call_chunks", None) or []

        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        raw_tool_calls = additional_kwargs.get("tool_calls")
        raw_tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []

        previews: list[ToolActivityCall] = []
        for source in (tool_calls, raw_tool_calls, tool_call_chunks):
            for candidate in source:
                preview = cls._extract_tool_call_preview(candidate)
                if preview is None:
                    continue
                duplicate_index = next(
                    (
                        index
                        for index, existing in enumerate(previews)
                        if cls._tool_activity_calls_match(existing, preview)
                    ),
                    None,
                )
                if duplicate_index is None:
                    previews.append(preview)
                    continue

                existing = previews[duplicate_index]
                existing_score = (len(existing.args), bool(existing.call_id))
                preview_score = (len(preview.args), bool(preview.call_id))
                preferred = preview if preview_score > existing_score else existing
                previews[duplicate_index] = ToolActivityCall(
                    name=preferred.name,
                    args=cls._merge_tool_args(existing.args, preview.args),
                    call_id=preferred.call_id or existing.call_id or preview.call_id,
                    start_time=max(existing.start_time, preview.start_time),
                )

        return previews

    @staticmethod
    def _summarize_tool_names(tool_names: list[str]) -> str:
        """Compact tool call labels for the live activity line."""
        if not tool_names:
            return "Calling tool..."
        if len(tool_names) == 1:
            return tool_names[0]
        return f"{tool_names[0]} +{len(tool_names) - 1}"

    @classmethod
    def _build_tool_activity_label(
        cls,
        tool_call: ToolActivityCall,
        indent_level: int = 0,
        origin_label: str | None = None,
    ) -> Text:
        """Build the live tool label with a fixed action prefix."""
        text = Text("  " * indent_level)
        if origin_label:
            text.append("[", style="muted")
            text.append(origin_label, style="secondary")
            text.append("] ", style="muted")
        text.append("Use tool ", style=_TOOL_PREFIX_STYLE)
        text.append(tool_call.name, style=_TOOL_NAME_STYLE)
        return text

    @staticmethod
    def _sorted_activity_items(
        activity: dict[tuple, Any]
    ) -> list[tuple[tuple, Any]]:
        """Sort per-namespace activity for stable live rendering."""
        return sorted(
            activity.items(),
            key=lambda item: (len(item[0]), tuple(str(part) for part in item[0])),
        )

    @staticmethod
    def _stringify_tool_arg(value: Any, max_length: int) -> str:
        """Convert tool args to a compact single-line string."""
        if isinstance(value, str):
            text = value.replace("\r\n", "\n").replace("\n", " | ")
        elif isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        if max_length <= 0 or len(text) <= max_length:
            return text

        suffix = f"... ({len(text)} chars)"
        keep = max(8, max_length - len(suffix))
        if keep >= len(text):
            return text
        return f"{text[:keep]}{suffix}"

    @classmethod
    def _resolve_origin_label(cls, namespace: tuple | None = None) -> str | None:
        """Mark nested namespace output as coming from a subagent."""
        if namespace:
            return cls._SUBAGENT_ORIGIN_LABEL
        return None

    @classmethod
    def _build_tool_arg_items(
        cls, tool_args: dict[str, Any], *, max_value_length: int
    ) -> list[tuple[str, str]]:
        """Prepare compact key/value pairs for live tool arg rendering."""
        return [
            (str(key), cls._stringify_tool_arg(value, max_value_length))
            for key, value in tool_args.items()
        ]

    @classmethod
    def _build_tool_activity_args(
        cls, tool_call: ToolActivityCall, indent_level: int = 0
    ) -> Text | None:
        """Build a Claude Code-like compact arg block for live tool activity."""
        if not tool_call.args:
            return None

        arg_items = cls._build_tool_arg_items(
            tool_call.args,
            max_value_length=_TOOL_LIVE_SUMMARY_VALUE_MAX,
        )

        base_indent = "  " * indent_level
        details = Text()
        for line_index, (key, value) in enumerate(arg_items):
            if line_index > 0:
                details.append("\n")
            details.append(f"{base_indent}  ")
            details.append(key, style=_TOOL_ARG_KEY_STYLE)
            details.append(": ", style=_TOOL_ARG_SEPARATOR_STYLE)
            details.append(value, style=_TOOL_ARG_VALUE_STYLE)
        return details

    @staticmethod
    def _merge_tool_args(
        existing_args: dict[str, Any], incoming_args: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge partial tool args without clobbering richer existing values."""
        merged = dict(existing_args)
        for key, value in incoming_args.items():
            if value in ("", None, [], {}) and key in merged:
                continue
            merged[key] = value
        return merged

    @staticmethod
    def _tool_args_are_incremental(
        left_args: dict[str, Any], right_args: dict[str, Any]
    ) -> bool:
        """Detect arg snapshots that look like progressive fills of one call."""
        shared_keys = set(left_args) & set(right_args)
        if any(
            not MessageDispatcher._tool_arg_values_are_compatible(
                left_args[key], right_args[key]
            )
            for key in shared_keys
        ):
            return False

        left_keys = set(left_args)
        right_keys = set(right_args)
        return left_keys.issubset(right_keys) or right_keys.issubset(left_keys)

    @staticmethod
    def _tool_arg_values_are_compatible(left_value: Any, right_value: Any) -> bool:
        """Treat streamed partial values as compatible with their fuller versions."""
        if left_value in ("", None, [], {}):
            return True
        if right_value in ("", None, [], {}):
            return True
        if left_value == right_value:
            return True

        if isinstance(left_value, str) and isinstance(right_value, str):
            return left_value.startswith(right_value) or right_value.startswith(left_value)

        if isinstance(left_value, dict) and isinstance(right_value, dict):
            shared_keys = set(left_value) & set(right_value)
            return all(
                MessageDispatcher._tool_arg_values_are_compatible(
                    left_value[key], right_value[key]
                )
                for key in shared_keys
            )

        if isinstance(left_value, list) and isinstance(right_value, list):
            shorter, longer = (
                (left_value, right_value)
                if len(left_value) <= len(right_value)
                else (right_value, left_value)
            )
            return all(
                MessageDispatcher._tool_arg_values_are_compatible(
                    shorter[index], longer[index]
                )
                for index in range(len(shorter))
            )

        return False

    @classmethod
    def _tool_activity_calls_match(
        cls,
        left: ToolActivityCall,
        right: ToolActivityCall,
    ) -> bool:
        """Heuristically decide whether two previews represent one tool call."""
        if left.call_id and right.call_id and left.call_id == right.call_id:
            return True

        if left.name != right.name:
            return False

        if not left.call_id or not right.call_id:
            return True

        return cls._tool_args_are_incremental(left.args, right.args)

    @classmethod
    def _merge_tool_activity_calls(
        cls,
        existing_calls: list[ToolActivityCall],
        incoming_calls: list[ToolActivityCall],
    ) -> list[ToolActivityCall]:
        """Merge streamed tool previews so args remain visible across chunks."""
        merged = list(existing_calls)

        for incoming in incoming_calls:
            match_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if cls._tool_activity_calls_match(existing, incoming)
                ),
                None,
            )

            if match_index is None:
                merged.append(incoming)
                continue

            existing = merged[match_index]
            merged[match_index] = ToolActivityCall(
                name=incoming.name or existing.name,
                args=cls._merge_tool_args(existing.args, incoming.args),
                call_id=existing.call_id or incoming.call_id,
                start_time=max(existing.start_time, incoming.start_time),
            )

        return merged

    @staticmethod
    def _is_same_tool_activity_call(
        left: ToolActivityCall, right: ToolActivityCall
    ) -> bool:
        """Check whether two live tool previews refer to the same tool call."""
        return MessageDispatcher._tool_activity_calls_match(left, right)

    @classmethod
    def _dedupe_tool_activity_namespaces(
        cls,
        active_tools: dict[tuple, list[ToolActivityCall]],
        target_namespace: tuple,
    ) -> None:
        """Keep each live tool call in only one namespace to avoid duplicate rows."""
        target_calls = active_tools.get(target_namespace, [])
        if not target_calls:
            return

        for namespace in list(active_tools):
            if namespace == target_namespace:
                continue

            calls = active_tools.get(namespace, [])
            if not calls:
                continue

            remaining_calls: list[ToolActivityCall] = []
            merged_target_calls = target_calls
            for call in calls:
                matching_target = next(
                    (
                        target_call
                        for target_call in merged_target_calls
                        if cls._is_same_tool_activity_call(target_call, call)
                    ),
                    None,
                )
                if matching_target is None:
                    remaining_calls.append(call)
                    continue

                merged_target_calls = cls._merge_tool_activity_calls(
                    merged_target_calls,
                    [call],
                )

            if remaining_calls:
                active_tools[namespace] = remaining_calls
            else:
                active_tools.pop(namespace, None)

            target_calls = merged_target_calls

        active_tools[target_namespace] = target_calls

    @classmethod
    def _build_activity_renderable(
        cls,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
    ) -> RenderableType:
        """Build the transient live activity area shown while streaming."""
        renderables: list[RenderableType] = []

        for namespace, tool_calls in cls._sorted_activity_items(active_tools):
            origin_label = cls._resolve_origin_label(namespace)
            for tool_call in tool_calls:
                # Skip hidden tools (e.g., write_todos shown via panel)
                if tool_call.name in cls._HIDDEN_ACTIVITY_TOOLS:
                    continue
                renderables.append(
                    ToolActivityIndicator(
                        cls._build_tool_activity_label(
                            tool_call,
                            indent_level=len(namespace),
                            origin_label=origin_label,
                        ),
                        details=cls._build_tool_activity_args(
                            tool_call, indent_level=len(namespace)
                        ),
                        start_time=tool_call.start_time,
                    )
                )

        for namespace, preview_lines in cls._sorted_activity_items(thinking_previews):
            indent = "  " * len(namespace)
            renderables.append(
                Spinner(
                    "dots",
                    Text(f"{indent}Thinking...", style="indicator"),
                    style="indicator",
                )
            )
            preview_text = "\n".join(f"{indent}{line}" for line in preview_lines[-3:])
            if preview_text:
                renderables.append(Text(preview_text, style="dim"))

        if not renderables:
            renderables.append(
                Spinner(
                    "dots",
                    Text("Thinking...", style="indicator"),
                    style="indicator",
                )
            )

        return Group(*renderables)

    @classmethod
    def _refresh_activity_live(
        cls,
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
        *,
        refresh: bool = False,
    ) -> None:
        """Refresh the live activity block with current tool/thinking state."""
        if live is None:
            return
        live.update(
            cls._build_activity_renderable(active_tools, thinking_previews),
            refresh=refresh,
        )

    @classmethod
    def _set_tool_activity(
        cls,
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
        namespace: tuple,
        tool_calls: list[ToolActivityCall],
    ) -> None:
        """Track an active tool call and refresh the live area."""
        active_tools[namespace] = cls._merge_tool_activity_calls(
            active_tools.get(namespace, []),
            tool_calls,
        )
        cls._dedupe_tool_activity_namespaces(active_tools, namespace)
        thinking_previews.pop(namespace, None)
        cls._refresh_activity_live(live, active_tools, thinking_previews)

    @classmethod
    def _clear_tool_activity(
        cls,
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
        namespace: tuple,
        *,
        refresh: bool = False,
    ) -> None:
        """Remove an active tool line and refresh the live area."""
        if namespace in active_tools:
            active_tools.pop(namespace, None)
            cls._refresh_activity_live(
                live,
                active_tools,
                thinking_previews,
                refresh=refresh,
            )

    @classmethod
    def _set_thinking_preview(
        cls,
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
        namespace: tuple,
        preview_lines: list[str],
    ) -> None:
        """Track the latest streaming preview for a namespace."""
        active_tools.pop(namespace, None)
        thinking_previews[namespace] = preview_lines[-3:]
        cls._refresh_activity_live(live, active_tools, thinking_previews)

    @classmethod
    def _clear_thinking_preview(
        cls,
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
        namespace: tuple,
        *,
        refresh: bool = False,
    ) -> None:
        """Remove preview text for a namespace and refresh the live area."""
        if namespace in thinking_previews:
            thinking_previews.pop(namespace, None)
            cls._refresh_activity_live(
                live,
                active_tools,
                thinking_previews,
                refresh=refresh,
            )

    async def _process_message_chunk(
        self,
        data: tuple[AnyMessage, dict],
        namespace: tuple,
        streaming_states: dict[tuple, dict[str, Any]],
        live: Live | None,
        rendered_messages: set[str],
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
    ) -> None:
        """Process message chunk for token-by-token streaming preview."""
        message_chunk, _metadata = data

        streaming_state = self._get_streaming_state(namespace, streaming_states)

        if isinstance(message_chunk, AIMessageChunk):
            message_id = self._get_stable_message_id(message_chunk)

            if (
                not streaming_state["active"]
                or streaming_state["message_id"] != message_id
            ):
                await self._finalize_streaming(
                    namespace,
                    streaming_states,
                    live,
                    rendered_messages,
                    active_tools,
                    thinking_previews,
                )
                streaming_state["active"] = True
                streaming_state["message_id"] = message_id
                streaming_state["preview_lines"] = [""]
                streaming_state["chunks"] = []

            streaming_state["chunks"].append(message_chunk)

            tool_previews = self._extract_tool_call_previews(message_chunk)
            if tool_previews:
                self._set_tool_activity(
                    live,
                    active_tools,
                    thinking_previews,
                    namespace,
                    tool_previews,
                )
                return

            content = self._extract_chunk_content(message_chunk)
            if content:
                lines = content.split("\n")
                if len(lines) == 1:
                    streaming_state["preview_lines"][-1] += lines[0]
                else:
                    streaming_state["preview_lines"][-1] += lines[0]
                    for new_line in lines[1:]:
                        streaming_state["preview_lines"].append(new_line)
                    if len(streaming_state["preview_lines"]) > 4:
                        streaming_state["preview_lines"] = streaming_state[
                            "preview_lines"
                        ][-4:]

                self._set_thinking_preview(
                    live,
                    active_tools,
                    thinking_previews,
                    namespace,
                    streaming_state["preview_lines"],
                )

    async def _process_update_chunk(
        self,
        data: dict,
        namespace: tuple,
        rendered_messages: set[str],
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
    ) -> None:
        """Process update chunk for tools/state (batch mode)."""
        for _node_name, node_data in data.items():
            if not isinstance(node_data, dict):
                continue

            await self._update_token_tracking(node_data)
            last_message = self._extract_last_update_message(node_data)
            if last_message is None:
                continue

            indent_level = len(namespace)
            self._update_activity_for_message(
                last_message,
                namespace=namespace,
                live=live,
                active_tools=active_tools,
                thinking_previews=thinking_previews,
            )
            self._render_new_update_message(
                last_message,
                indent_level=indent_level,
                rendered_messages=rendered_messages,
            )

    def _extract_last_update_message(self, node_data: dict[str, Any]) -> AnyMessage | None:
        """Return the newest message from an update payload when present."""
        messages_value = self._unwrap_overwrite(node_data.get("messages"))
        if isinstance(messages_value, tuple):
            messages_value = list(messages_value)
        if not isinstance(messages_value, list) or not messages_value:
            return None

        last_message = messages_value[-1]
        if not isinstance(last_message, BaseMessage):
            return None
        return cast(AnyMessage, last_message)

    def _update_activity_for_message(
        self,
        message: AnyMessage,
        *,
        namespace: tuple,
        live: Live | None,
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
    ) -> None:
        """Update transient live activity state based on a final update message."""
        if isinstance(message, AIMessage):
            tool_previews = self._extract_tool_call_previews(message)
            if tool_previews:
                self._set_tool_activity(
                    live,
                    active_tools,
                    thinking_previews,
                    namespace,
                    tool_previews,
                )
            return

        if isinstance(message, ToolMessage):
            self._clear_tool_activity(
                live,
                active_tools,
                thinking_previews,
                namespace,
                refresh=True,
            )
            self._clear_thinking_preview(
                live,
                active_tools,
                thinking_previews,
                namespace,
                refresh=True,
            )

    def _render_new_update_message(
        self,
        message: AnyMessage,
        *,
        indent_level: int,
        rendered_messages: set[str],
    ) -> None:
        """Render the final message from an update chunk once per stable message id."""
        message_id = f"{self._get_stable_message_id(message)}_{message.type}"
        if message_id in rendered_messages:
            return
        rendered_messages.add(message_id)

        if isinstance(message, AIMessage):
            self._render_assistant_with_deferred_tools(
                message,
                indent_level=indent_level,
            )
            return

        if isinstance(message, ToolMessage):
            self._render_pending_tool_header(
                message,
                indent_level=indent_level,
            )
            self.session.renderer.render_tool_message(
                message, indent_level=indent_level
            )
            self._remember_expandable_tool_output(
                message, indent_level=indent_level
            )

    @staticmethod
    def _extract_chunk_content(chunk: AIMessageChunk) -> str:
        """Extract text content from AI message chunk."""
        if isinstance(chunk.content, str):
            return chunk.content
        elif isinstance(chunk.content, list):
            texts = []
            for block in chunk.content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "".join(texts)
        return ""

    @staticmethod
    def _clear_preview(streaming_state: dict) -> None:
        """Clear preview without rendering final message."""
        if streaming_state["active"]:
            streaming_state["active"] = False
            streaming_state["message_id"] = None
            streaming_state["preview_lines"] = [""]
            streaming_state["chunks"] = []

    @staticmethod
    def _extract_tool_execution_duration(message: ToolMessage) -> float | None:
        """Prefer exact wrapped tool runtime when present on the ToolMessage."""
        response_metadata = getattr(message, "response_metadata", {}) or {}
        timing = response_metadata.get(TOOL_TIMING_RESPONSE_METADATA_KEY)
        if not isinstance(timing, dict):
            return None

        duration = timing.get("duration_seconds")
        if not isinstance(duration, (int, float)):
            return None

        return max(float(duration), 0.0)

    def _remember_tool_headers(self, message: AIMessage, indent_level: int) -> None:
        """Cache tool call headers so they can be rendered with the tool result."""
        origin_label = (
            self._SUBAGENT_ORIGIN_LABEL if indent_level > 0 else None
        )
        for tool_call in message.tool_calls:
            call_id = tool_call.get("id")
            if not call_id:
                continue

            # Skip hidden tools (e.g., write_todos shown via panel)
            tool_name = tool_call.get("name", "")
            if tool_name in self._HIDDEN_ACTIVITY_TOOLS:
                continue

            self._pending_tool_headers[str(call_id)] = DeferredToolHeader(
                tool_call=dict(tool_call),
                indent_level=indent_level,
                origin_label=origin_label,
                started_at=time.time(),
            )

    def _render_assistant_with_deferred_tools(
        self, message: AIMessage, indent_level: int
    ) -> None:
        """Render assistant content now and defer tool call headers until results arrive."""
        if message.tool_calls:
            self._remember_tool_headers(message, indent_level)
            self.session.renderer.render_assistant_message(
                message,
                indent_level=indent_level,
                show_tool_calls=False,
            )
            return

        self.session.renderer.render_assistant_message(
            message,
            indent_level=indent_level,
        )

    def _render_pending_tool_header(self, message: ToolMessage, indent_level: int) -> None:
        """Render the deferred tool header immediately before its result."""
        tool_call_id = getattr(message, "tool_call_id", None)
        if not tool_call_id:
            return

        pending = self._pending_tool_headers.pop(str(tool_call_id), None)
        if pending is None:
            return

        if isinstance(pending, tuple):
            pending = DeferredToolHeader(
                tool_call=dict(pending[0]),
                indent_level=int(pending[1]),
                origin_label=None,
                started_at=float(pending[2]),
            )

        tool_call = pending.tool_call

        # Skip hidden tools (e.g., write_todos shown via panel)
        tool_name = tool_call.get("name", "")
        if tool_name in self._HIDDEN_ACTIVITY_TOOLS:
            return

        duration = self._extract_tool_execution_duration(message)
        if duration is None:
            duration = time.time() - pending.started_at if pending.started_at else None
        self.session.renderer.render_tool_call(
            tool_call,
            indent_level=pending.indent_level,
            duration=duration,
            origin_label=pending.origin_label,
        )

    def _remember_expandable_tool_output(
        self, message: ToolMessage, *, indent_level: int
    ) -> None:
        """Store the latest tool output that has a richer full view."""
        remember = getattr(self.session, "remember_tool_output", None)
        if not callable(remember):
            return

        display = Renderer._build_tool_message_display(message)
        if display is None or display.is_todo_panel or not display.can_expand:
            return

        tool_call_id = str(
            getattr(message, "tool_call_id", None)
            or getattr(message, "id", None)
            or ""
        )
        tool_name = str(getattr(message, "name", None) or "tool")
        remember(
            ToolOutputEntry(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                preview_content=display.preview_content,
                full_content=display.full_content,
                indent_level=indent_level,
                origin_label=(
                    self._SUBAGENT_ORIGIN_LABEL if indent_level > 0 else None
                ),
                duration=self._extract_tool_execution_duration(message),
            )
        )

    async def _finalize_streaming(
        self,
        namespace: tuple,
        streaming_states: dict[tuple, dict[str, Any]],
        live: Live | None,
        rendered_messages: set[str],
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
    ) -> None:
        """Finalize active streaming message and render final version."""
        streaming_state = self._get_streaming_state(namespace, streaming_states)

        if streaming_state["active"]:
            if streaming_state["chunks"]:
                final_message = self._merge_chunks(streaming_state["chunks"])
                await self._update_token_tracking({"messages": [final_message]})
                indent_level = len(namespace)
                self._render_assistant_with_deferred_tools(
                    final_message, indent_level=indent_level
                )
                message_id = f"{streaming_state['message_id']}_{final_message.type}"
                rendered_messages.add(message_id)

            self._clear_thinking_preview(
                live,
                active_tools,
                thinking_previews,
                namespace,
            )
            self._clear_preview(streaming_state)

    async def _finalize_all_streaming(
        self,
        streaming_states: dict[tuple, dict[str, Any]],
        live: Live | None,
        rendered_messages: set[str],
        active_tools: dict[tuple, list[ToolActivityCall]],
        thinking_previews: dict[tuple, list[str]],
    ) -> None:
        """Finalize all active streaming messages."""
        for namespace in streaming_states:
            await self._finalize_streaming(
                namespace,
                streaming_states,
                live,
                rendered_messages,
                active_tools,
                thinking_previews,
            )

    @staticmethod
    def _merge_chunks(chunks: list[AIMessageChunk]) -> AIMessage:
        """Merge message chunks into final AIMessage, preserving all attributes."""
        if not chunks:
            return AIMessage(content="")

        merged = chunks[0]
        for chunk in chunks[1:]:
            merged = merged + chunk

        return AIMessage(
            content=merged.content,
            additional_kwargs=merged.additional_kwargs,
            response_metadata=merged.response_metadata,
            usage_metadata=merged.usage_metadata,
            tool_calls=merged.tool_calls,
            id=merged.id,
            name=merged.name,
        )

    async def _update_token_tracking(self, node_data: dict[str, Any]) -> None:
        """Update session context with token tracking data if present in node."""
        token_fields = {
            "current_input_tokens",
            "current_output_tokens",
        }

        updates = {
            field: self._unwrap_overwrite(node_data.get(field))
            for field in token_fields
            if field in node_data
        }

        if len(updates) < len(token_fields):
            messages = self._unwrap_overwrite(node_data.get("messages")) or []
            if isinstance(messages, tuple):
                messages = list(messages)
            if not isinstance(messages, list):
                messages = []
            latest_message = messages[-1] if messages else None
            if isinstance(latest_message, AIMessage):
                usage_counts = extract_usage_counts(latest_message)
                if usage_counts is not None:
                    input_tokens, output_tokens = usage_counts
                    updates.setdefault("current_input_tokens", input_tokens)
                    updates.setdefault("current_output_tokens", output_tokens)

        if updates:
            self.session.update_context(**updates)
            # Check if auto-compression should be triggered after token update
            await self._check_auto_compression()

    async def _check_auto_compression(self) -> None:
        """Check if auto-compression should be triggered."""
        try:
            ctx = self.session.context
            config_data = await initializer.load_agents_config(ctx.working_dir)
            agent_config = config_data.get_agent_config(ctx.agent)

            if (
                agent_config
                and agent_config.compression
                and agent_config.compression.auto_compress_enabled
                and should_auto_compress(
                    ctx.current_input_tokens or 0,
                    ctx.context_window,
                    agent_config.compression.auto_compress_threshold,
                )
            ):
                self._pending_compression = True

        except Exception as e:
            logger.warning(f"Auto-compression check failed: {e}", exc_info=True)

    async def _execute_compression(self) -> None:
        """Execute compression after streaming completes."""
        ctx = self.session.context
        usage_pct = int(
            (ctx.current_input_tokens or 0) / ctx.context_window * 100
            if ctx.context_window
            else 0
        )

        with console.console.status(
            f"[{theme.spinner_color}]Context at {usage_pct}%, auto-compacting conversation in place...[/{theme.spinner_color}]"
        ):
            await CompressionHandler(self.session).handle()

    async def resume_from_interrupt(
        self, thread_id: str, interrupts: list[Interrupt]
    ) -> None:
        """Resume graph from pending interrupts.

        Shows approval panel and resumes graph execution.
        """
        # Show approval panel
        resume_value = await self.interrupt_handler.handle(interrupts)
        if resume_value is None:
            return

        # Build Command
        cmd: Command
        if isinstance(resume_value, dict):
            cmd = Command(resume=resume_value)
        else:
            cmd = Command(resume={interrupts[0].id: resume_value})

        # Reuse existing context creation + streaming
        ctx = self.session.context
        agent_context = await self._build_agent_context()

        graph_config = RunnableConfig(
            configurable={"thread_id": thread_id},
            recursion_limit=ctx.recursion_limit,
        )

        await self._stream_response(cmd, graph_config, agent_context)

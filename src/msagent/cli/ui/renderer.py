"""Rich-based UI rendering and message formatting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

import mdformat
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from rich import box
from rich.align import Align
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, NewLine, RenderableType
from rich.markdown import CodeBlock, Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax, SyntaxTheme
from rich.table import Table
from rich.text import Text

from msagent.cli.core.context import Context
from msagent.cli.theme import console
from msagent.cli.ui.markdown import wrap_html_in_code_blocks
from msagent.cli.ui.shared import build_agent_prompt
from msagent.core.constants import UNKNOWN
from msagent.core.settings import settings
from msagent.agents.state import Todo
from msagent.tools.internal.todo import parse_todos_for_panel, render_todos_panel
from msagent.utils.version import get_version

try:
    from pyfiglet import Figlet
except (
    ImportError
):  # pragma: no cover - graceful fallback when dependency isn't installed yet
    Figlet = None  # type: ignore[misc,assignment]

WELCOME_TITLE = "* Welcome to msAgent"
WELCOME_ASCII_FONT = "ansi_shadow"
WELCOME_ASCII_PALETTE = [
    "#0b1f5e",
    "#123b8d",
    "#1d4ed8",
    "#2563eb",
    "#3b82f6",
    "#4f8ff7",
    "#6ea8fb",
    "#8bb8f8",
]


def _render_welcome_ascii(
    text: str = "msAgent",
    font: str = WELCOME_ASCII_FONT,
    gradient: str = "dark_to_light",
) -> Text:
    """Render ASCII art welcome text in the langchain-code style."""

    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _lerp(a: int, b: int, t: float) -> int:
        return int(a + (b - a) * t)

    def _interpolate_palette(palette: list[str], width: int) -> list[str]:
        if width <= 1:
            return [palette[0]]
        out: list[str] = []
        steps_total = width - 1
        for x in range(width):
            pos = x / steps_total
            seg = min(int(pos * (len(palette) - 1)), len(palette) - 2)
            seg_start = seg / (len(palette) - 1)
            seg_end = (seg + 1) / (len(palette) - 1)
            local_t = (pos - seg_start) / (seg_end - seg_start + 1e-9)
            c1 = _hex_to_rgb(palette[seg])
            c2 = _hex_to_rgb(palette[seg + 1])
            rgb = tuple(_lerp(a, b, local_t) for a, b in zip(c1, c2))
            out.append("#{:02x}{:02x}{:02x}".format(*rgb))
        return out

    if Figlet is None:
        return Text(text, style="accent")

    try:
        lines = Figlet(font=font).renderText(text).rstrip("\n").splitlines()
    except (
        Exception
    ):  # pragma: no cover - invalid font or partial install should not break startup
        return Text(text, style="accent")

    while lines and not lines[-1].strip():
        lines.pop()

    palette = WELCOME_ASCII_PALETTE
    if gradient == "light_to_dark":
        palette = list(reversed(palette))

    width = max(len(line) for line in lines) if lines else 0
    ramp = _interpolate_palette(palette, width)

    result = Text()
    for line in lines:
        padded = line.ljust(width)
        for j, ch in enumerate(padded):
            if ch == " ":
                result.append(ch)
            else:
                result.append(ch, Style(color=ramp[j], bold=True))
        result.append("\n")
    return result


THINKING_STYLE = Style(italic=True, dim=True)
LOW_PRIORITY_STYLE = console.console.get_style("muted") + Style(dim=True)
TOOL_PREFIX_STYLE = "accent"
TOOL_NAME_STYLE = "primary"
TOOL_ARG_KEY_STYLE = "muted"
TOOL_ARG_VALUE_STYLE = "primary"
TOOL_ARG_SEPARATOR_STYLE = "muted"
TOOL_SUMMARY_VALUE_MAX = 72
TOOL_MESSAGE_MAX_DISPLAY_CHARS = 200
TOOL_MESSAGE_TOGGLE_HINT = "Ctrl+O /tool-output"
SUBAGENT_ORIGIN_LABEL = "Subagent"
TODO_PANEL_ACTION_LABEL = "Update TODOs"


@dataclass(slots=True)
class ToolMessageDisplay:
    """Prepared display state for a tool result."""

    preview_content: str
    full_content: str
    display_content: str
    can_expand: bool
    is_error: bool
    is_todo_panel: bool = False
    todos: list[Todo] | None = None


def _truncate_preview_text(text: str, max_length: int) -> str:
    """Keep tool arg previews compact while preserving original length context."""
    if max_length <= 0 or len(text) <= max_length:
        return text

    suffix = f"... ({len(text)} chars)"
    keep = max(8, max_length - len(suffix))
    if keep >= len(text):
        return text
    return f"{text[:keep]}{suffix}"


def _fix_escaped_code_fences(content: str) -> str:
    """Fix escaped code fence delimiters that should be proper markdown.

    Some LLMs escape closing backticks (``` -> \\`\\`\\`) which prevents
    proper code block rendering. This must run before mdformat.
    """
    content = re.sub(
        r"```\n(.*?)\n\\`\\`\\`", r"```\n\1\n```", content, flags=re.DOTALL
    )
    content = re.sub(
        r"```([^\n]*)\n(.*?)\n\\`\\`\\`",
        r"```\1\n\2\n```",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"\\`\\`\\`([^\n]*)\n(.*?)\n\\`\\`\\`",
        r"```\1\n\2\n```",
        content,
        flags=re.DOTALL,
    )
    return content


class TransparentSyntax(Syntax):
    """Syntax highlighter with transparent background."""

    @classmethod
    def get_theme(cls, name):
        """Wrap theme to strip background colors from all token styles."""
        base_theme = super().get_theme(name)

        class TransparentThemeWrapper(SyntaxTheme):
            def __init__(self, base):
                self.base = base

            def get_style_for_token(self, token_type):
                style = self.base.get_style_for_token(token_type)
                return Style(
                    color=style.color,
                    bold=style.bold,
                    italic=style.italic,
                    underline=style.underline,
                )

            def get_background_style(self):
                return Style()

        return TransparentThemeWrapper(base_theme)


class TransparentCodeBlock(CodeBlock):
    """Code block with transparent background."""

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        yield TransparentSyntax(
            str(self.text).rstrip(), self.lexer_name, theme=self.theme
        )


class TransparentMarkdown(Markdown):
    """Markdown with transparent code blocks."""

    elements = {
        **Markdown.elements,
        "code_block": TransparentCodeBlock,
        "fence": TransparentCodeBlock,
    }


class PrefixedMarkdown:
    """Markdown with a styled prefix on the first line."""

    def __init__(
        self,
        prefix: str,
        content: str,
        prefix_style: str = "success",
        code_theme: str = "dracula",
        indent_level: int = 0,
    ):
        self.prefix = prefix
        self.prefix_style = prefix_style
        self.content = content
        self.code_theme = code_theme
        self.indent_level = indent_level

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        """Render markdown with prefix on first line and indent all subsequent lines."""
        # Calculate base indentation for subagents
        base_indent_str = "  " * self.indent_level
        base_indent_width = len(base_indent_str)

        # Calculate indentation width using visual cell width
        indent_width = cell_len(self.prefix) + base_indent_width

        # Adjust rendering width to account for prefix/indent
        adjusted_options = options.update_width(options.max_width - indent_width)

        # Render all content as markdown with adjusted width
        markdown = TransparentMarkdown(self.content, code_theme=self.code_theme)
        segments = list(console.render(markdown, adjusted_options))

        if not segments:
            return

        # Get prefix style from console theme
        prefix_style = console.get_style(self.prefix_style)

        # Create base indent and prefix segment for first line
        base_indent_segment = Segment(base_indent_str)
        prefix_segment = Segment(self.prefix, prefix_style)

        # Create full indentation segment for subsequent lines
        indent = " " * indent_width
        indent_segment = Segment(indent)

        # Track whether we've added the prefix yet
        prefix_added = False
        at_line_start = True

        for segment in segments:
            # Skip completely empty segments
            if not segment.text:
                continue

            # Split segment by newlines to handle indentation per line
            lines = segment.text.split("\n")

            for i, line in enumerate(lines):
                # Add base indent + prefix to first line content
                if not prefix_added and line:
                    yield base_indent_segment
                    yield prefix_segment
                    prefix_added = True
                    at_line_start = False
                # Add full indentation to subsequent lines
                elif at_line_start and line:
                    yield indent_segment
                    at_line_start = False

                # Yield the line content
                if line or i < len(lines) - 1:  # Include empty lines except trailing
                    yield Segment(line, segment.style)

                # Handle newlines between lines
                if i < len(lines) - 1:
                    yield Segment("\n")
                    at_line_start = True

        # Handle case where no content was found
        if not prefix_added:
            yield base_indent_segment
            yield prefix_segment


@dataclass(slots=True)
class ChatWelcomeBanner:
    """Welcome banner shown when msagent starts."""

    agent_name: str
    agent_description: str | None = None
    model_label: str | None = None
    mcp_servers: list[str] | None = None
    loaded_skills: list[str] | None = None

    @staticmethod
    def _meta_line(label: str, value: str) -> Text:
        line = Text()
        line.append(label, style="accent")
        line.append(": ", style="muted")
        line.append(value, style="secondary")
        return line

    @staticmethod
    def _agent_label(agent_name: str, agent_description: str | None) -> str:
        description = (agent_description or "").strip()
        if not description:
            return agent_name
        return f"{agent_name} - {description}"

    @staticmethod
    def _normalize_items(items: list[str] | None) -> list[str]:
        return [item for item in items or [] if item]

    @classmethod
    def _section_lines(cls, label: str, items: list[str] | None) -> list[Text]:
        normalized = cls._normalize_items(items)
        header = Text()
        header.append(label, style="accent")
        header.append(f" ({len(normalized)})", style="muted")

        if not normalized:
            empty_line = Text()
            empty_line.append("  - ", style="muted")
            empty_line.append("none", style="secondary")
            return [header, empty_line]

        lines = [header]
        for item in normalized:
            line = Text()
            line.append("  - ", style="muted")
            line.append(item, style="secondary")
            lines.append(line)
        return lines

    def compose(self) -> list[RenderableType]:
        model_label = self.model_label or UNKNOWN

        ascii_art = _render_welcome_ascii()

        welcome = Text()
        welcome.append("msAgent", style="accent")
        welcome.append(
            " 是 MindStudio 一站式调试调优 Agent，支持性能、精度、算子等场景问题定位。",
            style="secondary",
        )

        return [
            Align.center(ascii_art),
            welcome,
            self._meta_line(
                "Agent",
                self._agent_label(
                    self.agent_name,
                    getattr(self, "agent_description", None),
                ),
            ),
            self._meta_line("Model", model_label),
            *self._section_lines("MCP", self.mcp_servers),
            *self._section_lines("Skills", self.loaded_skills),
        ]

    def render(self) -> RenderableType:
        return Panel(
            Group(*self.compose()),
            title=f"[accent]{WELCOME_TITLE} v{get_version()}[/accent]",
            border_style="border",
            padding=(1, 2),
            box=box.ROUNDED,
            expand=False,
        )


class Renderer:
    """Handles rendering of UI elements using Rich."""

    TOOL_MESSAGE_MAX_DISPLAY_CHARS = TOOL_MESSAGE_MAX_DISPLAY_CHARS

    def __init__(self, context: Context | None = None) -> None:
        self.context = context

    @staticmethod
    def show_welcome(context: Context) -> None:
        """Display the msAgent welcome banner with legacy TUI styling."""
        from msagent.cli.bootstrap.initializer import initializer

        banner = ChatWelcomeBanner(
            agent_name=context.agent,
            agent_description=context.agent_description,
            model_label=context.model_display or context.model,
            mcp_servers=initializer.cached_mcp_server_names,
            loaded_skills=[skill.name for skill in initializer.cached_agent_skills],
        )

        console.print(banner.render())
        console.print("")

    def render_user_message(self, message: HumanMessage) -> None:
        """Render user message."""
        content = getattr(message, "short_content", None) or message.text

        # Calculate the visual width of the prompt prefix
        prompt_prefix = (
            build_agent_prompt(self.context)
            if self.context is not None
            else settings.cli.prompt_style
        )
        prefix_width = cell_len(prompt_prefix)
        indent = " " * prefix_width

        # Split content into lines and add indentation
        lines = content.split("\n")
        formatted_lines = []
        for i, line in enumerate(lines):
            if i == 0:
                formatted_lines.append(f"[prompt]{prompt_prefix}[/prompt]{line}")
            else:
                formatted_lines.append(f"{indent}{line}")

        console.print("\n".join(formatted_lines))
        console.print("")

    @staticmethod
    def _extract_thinking_from_metadata(message: AIMessage) -> str | None:
        """Extract thinking from message metadata (e.g., Bedrock stores it here)."""
        if not hasattr(message, "additional_kwargs"):
            return None
        if not isinstance(message.additional_kwargs, dict):
            return None
        thinking_data = message.additional_kwargs.get("thinking")
        if isinstance(thinking_data, dict):
            return thinking_data.get("text")
        return None

    @staticmethod
    def _extract_thinking_and_text_from_blocks(
        blocks: list[str | dict],
    ) -> tuple[list[str], list[str]]:
        """Extract text and thinking blocks separately.

        Returns:
            Tuple of (text_parts, thinking_parts)
        """
        texts = []
        thinking_blocks = []

        for block in blocks:
            if isinstance(block, str):
                text = block.strip(" ")
                if text and text[-1] != "\n":
                    text += "\n"
                texts.append(text)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "").strip(" ")
                    if text:
                        if text[-1] != "\n":
                            text += "\n"
                        texts.append(text)
                elif block_type == "thinking":
                    thinking_blocks.append(block.get("thinking", ""))
                elif block_type == "reasoning":
                    summary = block.get("summary", [])
                    if isinstance(summary, list):
                        summary_texts = [
                            s.get("text", "") for s in summary if isinstance(s, dict)
                        ]
                        if summary_texts:
                            thinking_blocks.append("\n".join(summary_texts))
                elif block_type == "reasoning_content":
                    reasoning_text = block.get("reasoning_content", "")
                    if reasoning_text:
                        thinking_blocks.append(reasoning_text)

        return texts, thinking_blocks

    @staticmethod
    def _stringify_tool_arg(value: Any, max_length: int) -> str:
        """Convert tool args to a compact single-line string."""
        if isinstance(value, str):
            text = value.replace("\r\n", "\n").replace("\n", " | ")
        elif isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        return _truncate_preview_text(text, max_length)

    @staticmethod
    def _resolve_origin_label(
        indent_level: int, origin_label: str | None = None
    ) -> str | None:
        """Resolve a human-friendly origin tag for nested subagent output."""
        if origin_label:
            return origin_label
        if indent_level > 0:
            return SUBAGENT_ORIGIN_LABEL
        return None

    @staticmethod
    def _append_origin_label(result: Text, label: str | None) -> None:
        """Append a compact origin badge before the tool action."""
        if not label:
            return
        result.append("[", style="muted")
        result.append(label, style="secondary")
        result.append("] ", style="muted")

    @staticmethod
    def _append_tool_arg_block(
        result: Text,
        arg_items: list[tuple[str, str]],
        base_indent: str,
    ) -> None:
        """Append a vertically stacked tool arg block."""
        if not arg_items:
            result.append("\n")
            return

        result.append("\n")
        for index, (key, value) in enumerate(arg_items):
            if index > 0:
                result.append("\n")
            result.append(f"{base_indent}  ")
            result.append(key, style=TOOL_ARG_KEY_STYLE)
            result.append(": ", style=TOOL_ARG_SEPARATOR_STYLE)
            result.append(value, style=TOOL_ARG_VALUE_STYLE)
        result.append("\n")

    @staticmethod
    def _strip_frontmatter_fences(content: str) -> str:
        """Remove leading Markdown frontmatter fences while preserving metadata."""
        stripped_leading = content.lstrip()
        if not stripped_leading.startswith("---\n"):
            return content

        lines = stripped_leading.splitlines()
        try:
            closing_index = lines.index("---", 1)
        except ValueError:
            return "\n".join(lines[1:]).strip() or content

        stripped_lines = lines[1:closing_index] + lines[closing_index + 1 :]
        stripped = "\n".join(stripped_lines).strip()
        return stripped or content

    @staticmethod
    def _format_tool_call(
        tool_call: dict[str, Any],
        indent_level: int = 0,
        duration: float | None = None,
        origin_label: str | None = None,
    ) -> Text:
        """Format a single tool call with improved readability."""
        tool_name = tool_call.get("name", UNKNOWN)
        tool_args = cast(dict[str, Any], tool_call.get("args", {}))

        base_indent = "  " * indent_level
        resolved_origin_label = Renderer._resolve_origin_label(
            indent_level, origin_label
        )

        # Build the text with formatting
        result = Text()
        result.append(base_indent)
        result.append("● ", style="indicator")
        Renderer._append_origin_label(result, resolved_origin_label)
        result.append("Use tool ", style=TOOL_PREFIX_STYLE)
        result.append(tool_name, style=TOOL_NAME_STYLE)

        # Add duration if provided (Claude Code style)
        if duration is not None:
            result.append(" ", style=TOOL_PREFIX_STYLE)
            result.append(f"({duration:.1f}s)", style="muted")

        if tool_args:
            summary_arg_items = [
                (
                    str(key),
                    Renderer._stringify_tool_arg(value, TOOL_SUMMARY_VALUE_MAX),
                )
                for key, value in tool_args.items()
            ]
            Renderer._append_tool_arg_block(
                result,
                summary_arg_items,
                base_indent=base_indent,
            )
        else:
            result.append("\n")

        return result

    @staticmethod
    def _should_skip_tool_call(tool_call: dict[str, Any]) -> bool:
        """Check if tool call should be skipped from display.

        write_todos is rendered separately as a TODO panel from tool payload.
        """
        return tool_call.get("name") == "write_todos"

    @staticmethod
    def render_assistant_message(
        message: AIMessage,
        indent_level: int = 0,
        *,
        show_tool_calls: bool = True,
    ) -> None:
        """Render an assistant message with optional tool calls."""
        if not message.content and not message.tool_calls:
            return

        content: str | list[str | dict] = message.content
        tool_calls = [dict(tc) for tc in message.tool_calls] if show_tool_calls else []
        is_error = getattr(message, "is_error", False)

        # Filter out write_todos tool calls (they are shown via ToolMessage panel)
        tool_calls = [
            tc for tc in tool_calls if not Renderer._should_skip_tool_call(tc)
        ]

        if not content:
            # Only tool calls, no content
            if tool_calls:
                for i, tool_call in enumerate(tool_calls):
                    console.print(
                        Renderer._format_tool_call(tool_call, indent_level=indent_level)
                    )
                    if i < len(tool_calls) - 1:
                        console.print("")
            return

        if is_error:
            indent = "  " * indent_level
            console.print(Text(f"{indent}{cast(str, content)}", style="error"))
            return

        # Extract thinking from all sources
        thinking_parts = []

        # 1. Check metadata first (Bedrock, etc.)
        metadata_thinking = Renderer._extract_thinking_from_metadata(message)
        if metadata_thinking:
            thinking_parts.append(metadata_thinking)

        # 2. Extract from content blocks
        if isinstance(content, list):
            text_parts, block_thinking = (
                Renderer._extract_thinking_and_text_from_blocks(content)
            )
            thinking_parts.extend(block_thinking)
            content = "".join(text_parts)

        # 3. Extract XML-style thinking tags
        if isinstance(content, str):
            content, xml_thinking = Renderer._extract_thinking_tags(content)
            if xml_thinking:
                thinking_parts.append(xml_thinking)

        # Render thinking first if present
        parts: list[RenderableType] = []
        if thinking_parts:
            parts.append(Text("\n\n".join(thinking_parts), style=THINKING_STYLE))
            parts.append(NewLine())

        # Render main content
        if isinstance(content, str):
            content = _fix_escaped_code_fences(content)
            try:
                content = mdformat.text(content)
            except Exception:
                pass

            content = wrap_html_in_code_blocks(content)
        if content:
            if indent_level > 0:
                prefix = "● "
            else:
                prefix = "● "

            parts.append(
                PrefixedMarkdown(
                    prefix,
                    content,
                    prefix_style="indicator",
                    code_theme="dracula",
                    indent_level=indent_level,
                )
            )

        # Print content if any
        if parts:
            console.print(Group(*parts))

        # Print tool calls with separator if we had content
        if tool_calls:
            if parts:
                console.print(NewLine())
            for tool_call in tool_calls:
                console.print(
                    Renderer._format_tool_call(tool_call, indent_level=indent_level)
                )
        elif parts:
            console.print("")

    @staticmethod
    def render_tool_call(
        tool_call: dict[str, Any],
        indent_level: int = 0,
        duration: float | None = None,
        origin_label: str | None = None,
    ) -> None:
        """Render a single tool call header."""
        console.print(
            Renderer._format_tool_call(
                tool_call,
                indent_level=indent_level,
                duration=duration,
                origin_label=origin_label,
            )
        )

    @staticmethod
    def render_tool_message(message: ToolMessage, indent_level: int = 0) -> None:
        """Render a tool execution message with Rich markup support."""
        display = Renderer._build_tool_message_display(message)
        if display is None:
            return

        if display.is_todo_panel and display.todos is not None:
            Renderer._render_todo_panel(
                display.todos,
                indent_level,
                origin_label=Renderer._resolve_origin_label(indent_level),
            )
            return

        if not display.display_content.strip():
            return

        base_indent = "  " * indent_level
        formatted_lines = []
        for i, line in enumerate(display.display_content.split("\n")):
            if i == 0:
                formatted_lines.append(f"{base_indent}  - {line}")
            else:
                formatted_lines.append(f"{base_indent}    {line}")

        if display.can_expand:
            formatted_lines.append(
                (
                    f"{base_indent}    "
                    f"... press {TOOL_MESSAGE_TOGGLE_HINT} to browse full tool outputs"
                )
            )

        formatted_content = "\n".join(formatted_lines)

        if display.is_error:
            console.print(Text(formatted_content, style="error"))
        else:
            console.print(Text(formatted_content, style=LOW_PRIORITY_STYLE))

        console.print("")

    @staticmethod
    def _build_tool_message_display(
        message: ToolMessage,
        *,
        expanded: bool = False,
    ) -> ToolMessageDisplay | None:
        """Normalize a tool result into preview/full display content."""
        short_content = getattr(message, "short_content", None)
        content = message.text

        preview_source = short_content if short_content is not None else content
        preview_text = str(preview_source or "")
        full_text = str(content or "")

        if not preview_text.strip() and not full_text.strip():
            return None

        parsed_todos = parse_todos_for_panel(preview_text) or parse_todos_for_panel(
            full_text
        )
        if parsed_todos:
            return ToolMessageDisplay(
                preview_content="",
                full_content="",
                display_content="",
                can_expand=False,
                is_error=False,
                is_todo_panel=True,
                todos=parsed_todos,
            )

        preview_text = Renderer._strip_frontmatter_fences(preview_text)
        full_text = Renderer._strip_frontmatter_fences(full_text)
        preview_display = Renderer._truncate_tool_content_for_display(preview_text)
        can_expand = bool(full_text.strip()) and preview_display != full_text

        is_error = (
            getattr(message, "is_error", False)
            or getattr(message, "status", None) == "error"
        )

        return ToolMessageDisplay(
            preview_content=preview_display,
            full_content=full_text,
            display_content=full_text if expanded and can_expand else preview_display,
            can_expand=can_expand,
            is_error=is_error,
        )

    @staticmethod
    def _truncate_tool_content_for_display(content: str) -> str:
        """Truncate long tool output for terminal readability.

        Includes the original character length so users can tell how much was omitted.
        """
        max_chars = Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS
        if max_chars <= 0 or len(content) <= max_chars:
            return content
        return (
            f"{content[:max_chars]}\n"
            f"... (truncated for display, original length: {len(content)} chars)"
        )

    @staticmethod
    def _render_todo_panel(
        todos: list[Todo],
        indent_level: int = 0,
        origin_label: str | None = None,
    ) -> None:
        """Render todo items in a bordered panel."""
        base_indent = "  " * indent_level
        header = Text()
        header.append(base_indent)
        header.append("● ", style="indicator")
        Renderer._append_origin_label(header, origin_label)
        header.append(TODO_PANEL_ACTION_LABEL, style=TOOL_NAME_STYLE)
        console.print(header)

        panel = render_todos_panel(
            cast(list[dict[str, Any]], todos),
            max_items=50,
            max_completed=50,
            show_completed_indicator=False,
        )
        panel_padding = len(base_indent) + 2
        console.console.print(Padding(panel, (0, 0, 0, panel_padding)))

        console.print("")

    @staticmethod
    def render_help(commands_dict: dict[str, Any]) -> None:
        """Render help information dynamically from registered commands."""
        content_parts = []

        # Commands section
        commands_table = Table.grid(padding=(0, 2))
        commands_table.add_column(style="command", justify="left", width=20)
        commands_table.add_column(style="secondary")

        # Dynamic generation from registered commands
        for command_name, command_func in commands_dict.items():
            # Extract description from docstring
            description = "No description available"
            if command_func.__doc__:
                description = command_func.__doc__.strip()

            commands_table.add_row(command_name, description)

        content_parts.append(commands_table)

        help_panel = Panel(
            Group(*content_parts),
            title="[accent]Help[/accent]",
            border_style="border",
            padding=(1, 2),
        )

        console.print(help_panel)
        console.print("")

    @staticmethod
    def render_hotkeys(hotkeys: dict[str, str]) -> None:
        """Render keyboard shortcuts."""
        hotkeys_table = Table.grid(padding=(0, 2))
        hotkeys_table.add_column(style="command", justify="left", width=20)
        hotkeys_table.add_column(style="secondary")

        for shortcut, description in hotkeys.items():
            hotkeys_table.add_row(shortcut, description)

        hotkeys_panel = Panel(
            Group(hotkeys_table),
            title="[accent]Keyboard Shortcuts[/accent]",
            border_style="border",
            padding=(1, 2),
        )

        console.print(hotkeys_panel)
        console.print("")

    @staticmethod
    def _extract_thinking_tags(content: str) -> tuple[str, str | None]:
        """Extract thinking content from XML-style tags like <think>...</think>.

        Only extracts if <think> appears at the start of content (provider pattern).
        If <think> tags appear mid-content, they're treated as literal text.

        Args:
            content: The content that may contain thinking tags

        Returns:
            Tuple of (cleaned_content, thinking_content)
        """
        content_stripped = content.lstrip()

        # Only extract if content starts with <think> tag (provider-generated pattern)
        if not content_stripped.startswith("<think>"):
            return content, None

        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, content, re.DOTALL)

        if matches:
            thinking_content = "\n\n".join(match.strip() for match in matches)
            cleaned_content = re.sub(
                think_pattern, "", content, flags=re.DOTALL
            ).strip()
            return cleaned_content, thinking_content

        return content, None

    def render_message(self, message: AnyMessage, indent_level: int = 0) -> None:
        """Render any message."""
        if isinstance(message, HumanMessage):
            self.render_user_message(message)

        elif isinstance(message, AIMessage):
            Renderer.render_assistant_message(message, indent_level=indent_level)

        elif isinstance(message, ToolMessage):
            Renderer.render_tool_message(message, indent_level=indent_level)

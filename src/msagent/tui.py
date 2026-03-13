""" Prompt-toolkit TUI """

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style as PromptStyle
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, NewLine, RenderableType
from rich.markup import escape
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax, SyntaxTheme
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .agent import Agent
from .application import ChatApplicationService
from .interfaces import AgentBackend, AgentStatus

def _supports_unicode_output(sample: str) -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        sample.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


_USE_UNICODE_UI = _supports_unicode_output("\u276f\u25cf\u25cb\u2699\u2715")

PROMPT_STYLE = "\u276f " if _USE_UNICODE_UI else "> "
ASSISTANT_PREFIX = "\u25cf " if _USE_UNICODE_UI else "* "
SUBAGENT_PREFIX = "\u25cb " if _USE_UNICODE_UI else "- "
TOOL_PREFIX = "\u2699" if _USE_UNICODE_UI else "tool"
ERROR_PREFIX = "\u2715" if _USE_UNICODE_UI else "x"
WELCOME_TITLE = "\u273b Welcome to msAgent" if _USE_UNICODE_UI else "* Welcome to msAgent"
STREAM_SPINNER_TEXT = "Thinking..."

_TOKYO_NIGHT_THEME = Theme(
    {
        "default": Style(color="#c0caf5"),
        "primary": Style(color="#c0caf5"),
        "secondary": Style(color="#9aa5ce"),
        "muted": Style(color="#565f89"),
        "muted.bold": Style(color="#565f89", bold=True),
        "accent": Style(color="#7aa2f7", bold=True),
        "accent.primary": Style(color="#7aa2f7"),
        "accent.secondary": Style(color="#7dcfff"),
        "accent.tertiary": Style(color="#bb9af7"),
        "success": Style(color="#8be4e1"),
        "warning": Style(color="#e4e38b"),
        "error": Style(color="#e48be4"),
        "info": Style(color="#7aa2f7"),
        "border": Style(color="#414868"),
        "prompt": Style(color="#7aa2f7", bold=True),
        "command": Style(color="#bb9af7"),
        "option": Style(color="#7dcfff"),
        "indicator": Style(color="#8be4e1"),
        "timestamp": Style(color="#565f89", italic=True),
    }
)
_RICH_THEME = _TOKYO_NIGHT_THEME


def _fix_escaped_code_fences(content: str) -> str:
    return content.replace("\\`\\`\\`", "```")


class TransparentSyntax(Syntax):
    """Syntax highlighter without opaque backgrounds."""

    @classmethod
    def get_theme(cls, name: str):
        base_theme = super().get_theme(name)

        class TransparentThemeWrapper(SyntaxTheme):
            def __init__(self, base: Any) -> None:
                self.base = base

            def get_style_for_token(self, token_type: Any) -> Style:
                style = self.base.get_style_for_token(token_type)
                return Style(
                    color=style.color,
                    bold=style.bold,
                    italic=style.italic,
                    underline=style.underline,
                )

            def get_background_style(self) -> Style:
                return Style()

        return TransparentThemeWrapper(base_theme)


class TransparentCodeBlock(CodeBlock):
    """Code block using transparent syntax highlighting."""

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        yield TransparentSyntax(
            str(self.text).rstrip(),
            self.lexer_name or "text",
            theme=self.theme,
        )


class TransparentMarkdown(Markdown):
    """Markdown renderer with transparent fenced blocks."""

    elements = {
        **Markdown.elements,
        "code_block": TransparentCodeBlock,
        "fence": TransparentCodeBlock,
    }


def _format_duration_text(duration_s: float | None) -> str:
    if duration_s is None or duration_s < 0:
        return ""
    if duration_s < 1:
        return f"took {duration_s * 1000:.0f}ms"
    if duration_s < 60:
        precision = 1 if duration_s < 10 else 0
        return f"took {duration_s:.{precision}f}s"
    minutes, seconds = divmod(duration_s, 60)
    rounded_seconds = int(round(seconds))
    whole_minutes = int(minutes)
    if rounded_seconds == 60:
        whole_minutes += 1
        rounded_seconds = 0
    return f"took {whole_minutes}m {rounded_seconds:02d}s"


def _format_token_count(value: int | None) -> str:
    if value is None or value <= 0:
        return "0"
    if value >= 1_000_000:
        return _format_token_unit(value / 1_000_000, "M")
    if value >= 100_000:
        return f"{int(round(value / 1_000))}K"
    if value >= 1_000:
        return _format_token_unit(value / 1_000, "K")
    return str(value)


def _format_token_unit(value: float, suffix: str) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def _format_prompt_status_text(status: AgentStatus) -> str:
    cumulative = status.cumulative_usage
    prompt_tokens = cumulative.prompt_tokens if cumulative is not None else 0
    completion_tokens = cumulative.completion_tokens if cumulative is not None else 0
    total_tokens = cumulative.total_tokens if cumulative is not None else 0
    return (
        f"Ctx {_format_token_count(status.context_tokens)}"
        f" | Total {_format_token_count(total_tokens)}"
        f" | In {_format_token_count(prompt_tokens)}"
        f" | Out {_format_token_count(completion_tokens)}"
    )


def _format_tool_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except Exception:
            return {"value": stripped}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if value is None:
        return {}
    return {"value": value}


def _extract_active_at_token(value: str, cursor: int) -> tuple[str, int, int] | None:
    if cursor < 0 or cursor > len(value):
        return None
    at_pos = value.rfind("@", 0, cursor)
    if at_pos < 0:
        return None
    if at_pos > 0 and not value[at_pos - 1].isspace():
        return None
    token = value[at_pos + 1 : cursor]
    if any(ch.isspace() for ch in token):
        return None
    return (token, at_pos, cursor)


def _extract_active_slash_token(value: str, cursor: int) -> tuple[str, int, int] | None:
    if cursor < 0 or cursor > len(value) or "\n" in value or not value:
        return None
    start = 0
    while start < len(value) and value[start].isspace():
        start += 1
    if start >= len(value) or value[start] != "/":
        return None
    if value[:start].strip() or value[cursor:].strip():
        return None
    query = value[start:cursor] or "/"
    if query.endswith(" "):
        return None
    return (query, start, len(value))


def _append_preview_lines(preview_lines: list[str], chunk: str) -> list[str]:
    lines = chunk.split("\n")
    if len(lines) == 1:
        preview_lines[-1] += lines[0]
    else:
        preview_lines[-1] += lines[0]
        preview_lines.extend(lines[1:])
    if len(preview_lines) > 4:
        preview_lines = preview_lines[-4:]
    return preview_lines


class PrefixedMarkdown:
    """Markdown with a styled prefix on the first line."""

    def __init__(
        self,
        prefix: str,
        content: str,
        *,
        prefix_style: str = "success",
        code_theme: str = "dracula",
        indent_level: int = 0,
    ) -> None:
        self.prefix = prefix
        self.content = content
        self.prefix_style = prefix_style
        self.code_theme = code_theme
        self.indent_level = indent_level

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        base_indent = "  " * self.indent_level
        indent_width = cell_len(self.prefix) + len(base_indent)
        adjusted_options = options.update_width(max(20, options.max_width - indent_width))

        markdown = TransparentMarkdown(
            _fix_escaped_code_fences(self.content),
            code_theme=self.code_theme,
        )
        segments = list(console.render(markdown, adjusted_options))
        if not segments:
            return

        prefix_style = console.get_style(self.prefix_style)
        base_indent_segment = Segment(base_indent)
        prefix_segment = Segment(self.prefix, prefix_style)
        indent_segment = Segment(" " * indent_width)

        prefix_added = False
        at_line_start = True

        for segment in segments:
            if not segment.text:
                continue
            lines = segment.text.split("\n")
            for index, line in enumerate(lines):
                if not prefix_added and line:
                    yield base_indent_segment
                    yield prefix_segment
                    prefix_added = True
                    at_line_start = False
                elif at_line_start and line:
                    yield indent_segment
                    at_line_start = False

                if line or index < len(lines) - 1:
                    yield Segment(line, segment.style)

                if index < len(lines) - 1:
                    yield Segment("\n")
                    at_line_start = True

        if not prefix_added:
            yield base_indent_segment
            yield prefix_segment

@dataclass(slots=True)
class ChatWelcomeBanner:
    """Welcome banner shown when the TUI starts."""

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

    def compose(self) -> Iterable[RenderableType]:
        servers = ", ".join(self.mcp_servers or []) or "none"
        skills = ", ".join(self.loaded_skills or []) or "none"
        model_label = self.model_label or "unknown"

        welcome = Text()
        welcome.append("msAgent", style="accent")
        welcome.append(
            " 是面向 Ascend NPU Profiling 的性能分析助手，基于真实数据定位瓶颈、解释根因并给出可执行优化方案。",
            style="secondary",
        )
        yield welcome
        yield self._meta_line("Model", model_label)
        yield self._meta_line("MCP", servers)
        yield self._meta_line("Skills", skills)

    def render(self) -> RenderableType:
        return Panel(
            Group(*self.compose()),
            title=f"[accent]{WELCOME_TITLE}[/accent]",
            border_style="border",
            padding=(1, 2),
        )


class Renderer:
    """Rich-based renderer mirroring langrepl's CLI look."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def show_welcome(self, status: AgentStatus) -> None:
        banner = ChatWelcomeBanner(
            model_label=f"{status.provider}:{status.model}",
            mcp_servers=list(status.connected_servers),
            loaded_skills=list(status.loaded_skills),
        )
        self.console.print(banner.render())
        self.console.print("")

    def render_user_message(self, content: str, *, status_text: str | None = None) -> None:
        indent = " " * cell_len(PROMPT_STYLE)
        lines = content.split("\n")
        rendered_lines: list[str] = []
        for index, line in enumerate(lines):
            if index == 0:
                first_line = f"[prompt]{escape(PROMPT_STYLE)}[/prompt]{escape(line)}"
                if status_text:
                    plain_left = f"{PROMPT_STYLE}{line}"
                    gap = self.console.width - cell_len(plain_left) - cell_len(status_text)
                    spacer = " " * gap if gap >= 2 else "  "
                    first_line += f"[muted]{escape(spacer + status_text)}[/muted]"
                rendered_lines.append(first_line)
            else:
                rendered_lines.append(f"{indent}{escape(line)}")
        self.console.print("\n".join(rendered_lines), markup=True)
        self.console.print("")

    def render_empty_prompt_submit(self, prompt_text: str) -> None:
        self.console.print(f"[prompt]{prompt_text}[/prompt]", markup=True)

    def build_stream_placeholder(self) -> RenderableType:
        return Group(Text(STREAM_SPINNER_TEXT, style="indicator"))

    def build_stream_preview(self, preview_lines: list[str], *, indent_level: int = 0) -> RenderableType:
        indent = "  " * indent_level
        preview_text = "\n".join(f"{indent}{line}" for line in preview_lines[-3:] if line)
        if preview_text:
            return Group(
                Text(f"{indent}{STREAM_SPINNER_TEXT}", style="indicator"),
                Text(preview_text, style="muted"),
            )
        return Group(Text(f"{indent}{STREAM_SPINNER_TEXT}", style="indicator"))

    def build_assistant_renderable(self, content: str, *, indent_level: int = 0) -> RenderableType:
        prefix = SUBAGENT_PREFIX if indent_level > 0 else ASSISTANT_PREFIX
        return PrefixedMarkdown(
            prefix,
            content,
            prefix_style="indicator",
            code_theme="dracula",
            indent_level=indent_level,
        )

    def render_assistant_message(self, content: str, *, indent_level: int = 0) -> None:
        if not content.strip():
            return
        self.console.print(self.build_assistant_renderable(content, indent_level=indent_level))
        self.console.print("")

    def build_tool_call(self, name: str, payload: Any, *, indent_level: int = 0) -> RenderableType:
        tool_args = _format_tool_payload(payload)
        base_indent = "  " * indent_level
        text = Text()
        text.append(base_indent)
        text.append(TOOL_PREFIX, style="indicator")
        text.append("   ")
        text.append(name, style="bold")
        text.append("\n")
        for key, value in tool_args.items():
            value_text = str(value)
            if len(value_text) > 200:
                value_text = value_text[:200] + "..."
            text.append(f"{base_indent}  {key} : ")
            text.append(value_text)
            text.append("\n")
        return text

    def render_tool_call(self, name: str, payload: Any, *, indent_level: int = 0) -> None:
        self.console.print(self.build_tool_call(name, payload, indent_level=indent_level))
        self.console.print("")

    def render_system_message(
        self,
        message: str,
        *,
        title: str = "Info",
        border_style: str = "border",
    ) -> None:
        self.console.print(
            Panel(message, title=f"[accent]{title}[/accent]", border_style=border_style, padding=(1, 2))
        )
        self.console.print("")

    def render_error(self, message: str) -> None:
        self.console.print(f"[error]{ERROR_PREFIX}[/error] {message}", markup=True)
        self.console.print("")

    def render_help(self, commands: dict[str, Any] | list[tuple[str, str]]) -> None:
        rows: list[tuple[str, str]] = []
        if isinstance(commands, dict):
            for command_name, handler in commands.items():
                description = getattr(handler, "__doc__", None) or "No description available"
                rows.append((command_name, str(description).strip()))
        else:
            rows.extend(commands)

        table = Table.grid(padding=(0, 2))
        table.add_column(style="command", justify="left", width=20)
        table.add_column(style="secondary")
        for command, description in rows:
            table.add_row(command, description)
        self.console.print(
            Panel(Group(table), title="[accent]Help[/accent]", border_style="border", padding=(1, 2))
        )
        self.console.print("")

    def render_hotkeys(self, hotkeys: dict[str, str]) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="command", justify="left", width=20)
        table.add_column(style="secondary")
        for shortcut, description in hotkeys.items():
            table.add_row(shortcut, description)
        self.console.print(
            Panel(
                Group(table),
                title="[accent]Keyboard Shortcuts[/accent]",
                border_style="border",
                padding=(1, 2),
            )
        )
        self.console.print("")


class ToolBrowser:
    """Interactive tool list mirroring langrepl's selector behavior."""

    def __init__(self, prompt_style: str) -> None:
        self._prompt_style = prompt_style

    async def show(
        self,
        tools: list[dict[str, Any]],
        *,
        style: Any,
    ) -> None:
        if not tools:
            return

        current_index = 0
        expanded_indices: set[int] = set()
        scroll_offset = 0
        window_size = 10

        text_control = FormattedTextControl(
            text=lambda: self._format_tool_list(
                tools,
                selected_index=current_index,
                expanded_indices=expanded_indices,
                scroll_offset=scroll_offset,
                window_size=window_size,
            ),
            focusable=True,
            show_cursor=False,
        )

        instructions = FormattedTextControl(
            lambda: FormattedText(
                [
                    ("class:muted", "Enter: expand/collapse"),
                    ("", "  "),
                    ("class:muted", "Ctrl+C: close"),
                ]
            )
        )

        key_bindings = KeyBindings()

        @key_bindings.add(Keys.Up)
        def handle_up(event: Any) -> None:
            nonlocal current_index, scroll_offset
            if current_index > 0:
                current_index -= 1
                if current_index < scroll_offset:
                    scroll_offset = current_index

        @key_bindings.add(Keys.Down)
        def handle_down(event: Any) -> None:
            nonlocal current_index, scroll_offset
            if current_index < len(tools) - 1:
                current_index += 1
                if current_index >= scroll_offset + window_size:
                    scroll_offset = current_index - window_size + 1

        @key_bindings.add(Keys.Enter)
        def handle_enter(event: Any) -> None:
            if current_index in expanded_indices:
                expanded_indices.remove(current_index)
            else:
                expanded_indices.add(current_index)

        @key_bindings.add(Keys.ControlC)
        def handle_ctrl_c(event: Any) -> None:
            event.app.exit()

        app = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(height=1, content=instructions),
                        Window(height=1, char=" "),
                        Window(content=text_control),
                    ]
                )
            ),
            key_bindings=key_bindings,
            full_screen=False,
            style=style,
            erase_when_done=True,
        )

        try:
            await app.run_async()
        except (KeyboardInterrupt, EOFError):
            return

    def _format_tool_list(
        self,
        tools: list[dict[str, Any]],
        *,
        selected_index: int,
        expanded_indices: set[int],
        scroll_offset: int,
        window_size: int,
    ) -> FormattedText:
        prompt_symbol = self._prompt_style.strip()
        lines: list[tuple[str, str]] = []
        visible_tools = tools[scroll_offset : scroll_offset + window_size]

        for idx, tool in enumerate(visible_tools):
            actual_index = scroll_offset + idx
            name, description = self._tool_name_and_description(tool)
            if actual_index == selected_index:
                lines.append(("class:selected", f"{prompt_symbol} {name}"))
            else:
                lines.append(("", f"  {name}"))

            if actual_index in expanded_indices:
                lines.append(("", "\n"))
                expanded_lines = self._expanded_lines(tool, description)
                for desc_idx, line in enumerate(expanded_lines):
                    lines.append(("class:auto-suggestion", f"    {line}"))
                    if desc_idx < len(expanded_lines) - 1:
                        lines.append(("", "\n"))

            if idx < len(visible_tools) - 1:
                lines.append(("", "\n"))

        return FormattedText(lines)

    def _expanded_lines(self, tool: dict[str, Any], description: str) -> list[str]:
        lines = self._wrap_text(description or "No description", width=max(20, self._terminal_width() - 6))
        parameters = self._tool_parameters(tool)
        if parameters:
            lines.append("")
            lines.append("Parameters:")
            for parameter in parameters:
                lines.extend(self._wrap_text(f"- {parameter}", width=max(20, self._terminal_width() - 6)))
        return lines

    @staticmethod
    def _tool_name_and_description(tool: dict[str, Any]) -> tuple[str, str]:
        function = tool.get("function")
        if not isinstance(function, dict):
            return ("Unknown", "No description")
        name = function.get("name")
        description = function.get("description")
        return (
            name if isinstance(name, str) and name else "Unknown",
            description if isinstance(description, str) and description else "No description",
        )

    @staticmethod
    def _tool_parameters(tool: dict[str, Any]) -> list[str]:
        function = tool.get("function")
        if not isinstance(function, dict):
            return []
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return []
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return []

        required = set(parameters.get("required") or [])
        lines: list[str] = []
        for name, schema in properties.items():
            type_name = ""
            if isinstance(schema, dict):
                schema_type = schema.get("type")
                if isinstance(schema_type, str):
                    type_name = schema_type
                description = schema.get("description")
                description_text = description if isinstance(description, str) else ""
            else:
                description_text = ""
            suffix = " required" if name in required else " optional"
            label = f"{name} ({type_name or 'any'}, {suffix.strip()})"
            if description_text:
                label += f": {description_text}"
            lines.append(label)
        return lines

    @staticmethod
    def _wrap_text(text: str, *, width: int) -> list[str]:
        if not text:
            return [""]
        width = max(10, width)
        output: list[str] = []
        for paragraph in text.splitlines() or [""]:
            if not paragraph:
                output.append("")
                continue
            words = paragraph.split()
            if not words:
                output.append("")
                continue
            current = words[0]
            for word in words[1:]:
                if len(current) + len(word) + 1 <= width:
                    current += f" {word}"
                else:
                    output.append(current)
                    current = word
            output.append(current)
        return output

    @staticmethod
    def _terminal_width() -> int:
        try:
            import shutil

            return shutil.get_terminal_size((100, 24)).columns
        except Exception:
            return 100


class InteractivePrompt:
    """Prompt-toolkit session aligned with langrepl's prompt behavior."""

    _CTRL_C_TIMEOUT_S = 0.30

    def __init__(
        self,
        service: ChatApplicationService,
        *,
        history_file: Path | None = None,
        prompt_text: str = PROMPT_STYLE,
    ) -> None:
        self.service = service
        self.history_file = history_file
        self.prompt_text = prompt_text
        self._last_ctrl_c_time: float | None = None
        self._show_quit_message = False
        self.hotkeys: dict[str, str] = {}
        self.prompt_session = self._create_session()

    def _create_style(self) -> Any:
        return PromptStyle.from_dict(
            {
                "prompt": "#7aa2f7 bold",
                "prompt.arg": "#7dcfff",
                "": "#c0caf5",
                "text": "#c0caf5",
                "completion-menu.completion": "#c0caf5 bg:#24283b",
                "completion-menu.completion.current": "#1a1b26 bg:#7aa2f7",
                "completion-menu.meta.completion": "#565f89 bg:#24283b",
                "completion-menu.meta.completion.current": "#c0caf5 bg:#7aa2f7",
                "auto-suggestion": "#565f89 italic",
                "placeholder": "#414868 italic",
                "muted": "#565f89",
                "selected": "#8be4e1",
                "rprompt.label": "#414868",
                "rprompt.value": "#565f89 italic",
                "rprompt.sep": "#414868",
            }
        )

    def _reset_ctrl_c_state(self) -> None:
        self._last_ctrl_c_time = None
        self._show_quit_message = False

    @staticmethod
    def _format_key_name(key: Any) -> str:
        key_str = str(key)
        replacements = {
            "Keys.Control": "Ctrl+",
            "Keys.Back": "Shift+",
            "Keys.": "",
        }
        for old, new in replacements.items():
            key_str = key_str.replace(old, new)
        return key_str

    def _schedule_hide_message(self, app: Any) -> None:
        def hide() -> None:
            self._reset_ctrl_c_state()
            app.invalidate()

        try:
            app.loop.call_later(self._CTRL_C_TIMEOUT_S, hide)
        except Exception:
            self._reset_ctrl_c_state()

    def _create_key_bindings(self) -> Any:
        key_bindings = KeyBindings()
        self.hotkeys.clear()

        @key_bindings.add(Keys.ControlC)
        def handle_ctrl_c(event: Any) -> None:
            buffer = event.current_buffer
            now = time.time()

            if buffer.text.strip():
                buffer.text = ""
                self._reset_ctrl_c_state()
                return

            if self._last_ctrl_c_time is not None:
                elapsed = now - self._last_ctrl_c_time
                if elapsed < self._CTRL_C_TIMEOUT_S or self._show_quit_message:
                    self._reset_ctrl_c_state()
                    event.app.exit(exception=EOFError())
                    return

            self._last_ctrl_c_time = now
            self._show_quit_message = True
            self._schedule_hide_message(event.app)

        @key_bindings.add(Keys.ControlJ)
        def handle_ctrl_j(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add(Keys.ControlK)
        def handle_ctrl_k(event: Any) -> None:
            event.current_buffer.text = "/hotkeys"
            event.current_buffer.validate_and_handle()

        @key_bindings.add(Keys.Enter, filter=completion_is_selected)
        def handle_enter_with_completion(event: Any) -> None:
            buffer = event.current_buffer
            completion = buffer.complete_state.current_completion
            buffer.apply_completion(completion)
            if buffer.text.lstrip().startswith("/"):
                buffer.validate_and_handle()
            else:
                buffer.insert_text(" ")

        @key_bindings.add(Keys.Tab)
        def handle_tab(event: Any) -> None:
            buffer = event.current_buffer
            if buffer.complete_state and buffer.complete_state.current_completion:
                completion = buffer.complete_state.current_completion
                buffer.apply_completion(completion)
                if not buffer.text.lstrip().startswith("/"):
                    buffer.insert_text(" ")
                return

            buffer.start_completion(select_first=True)
            if buffer.complete_state and buffer.complete_state.current_completion:
                completion = buffer.complete_state.current_completion
                buffer.apply_completion(completion)
                if not buffer.text.lstrip().startswith("/"):
                    buffer.insert_text(" ")

        self.hotkeys = {
            self._format_key_name(Keys.ControlC): "Clear input (press twice to quit)",
            self._format_key_name(Keys.ControlJ): "Insert newline for multiline input",
            self._format_key_name(Keys.ControlK): "Show keyboard shortcuts",
            "Tab": "Apply first completion",
            "Enter": "Apply selected completion or submit",
        }

        return key_bindings

    def _build_completer(self) -> Any:
        service = self.service

        class RouterCompleter(Completer):
            def get_completions(self, document: Any, complete_event: Any):
                text = document.text
                cursor = document.cursor_position

                at_token = _extract_active_at_token(text, cursor)
                if at_token is not None:
                    query, start, _ = at_token
                    for path in service.find_local_files(query, limit=20):
                        yield Completion(
                            f"@{path}",
                            start_position=start - cursor,
                            display=path,
                            display_meta="file",
                        )
                    return

                slash_token = _extract_active_slash_token(text, cursor)
                if slash_token is not None:
                    query, start, _ = slash_token
                    for command, detail in service.find_commands(query, limit=20):
                        yield Completion(
                            command,
                            start_position=start - cursor,
                            display=command,
                            display_meta=detail,
                        )

        return RouterCompleter()

    def _get_placeholder(self) -> Any:
        return FormattedText(
            [
                ("class:placeholder", "尽管问 msAgent，/ 命令，@ 关联文件"),
            ]
        )
    def _get_rprompt(self) -> Any:
        try:
            status = self.service.get_status()
        except Exception:
            return FormattedText([])
        status_text = _format_prompt_status_text(status)
        return FormattedText([("class:rprompt.value", status_text)])

    def _create_session(self) -> Any:
        if self.history_file is not None:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(self.history_file))
        else:
            history = InMemoryHistory()

        return PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._build_completer(),
            complete_style=CompleteStyle.COLUMN,
            key_bindings=self._create_key_bindings(),
            style=self._create_style(),
            multiline=False,
            mouse_support=False,
            complete_while_typing=True,
            complete_in_thread=False,
            wrap_lines=True,
            prompt_continuation=lambda width, line_number, is_soft_wrap: " " * len(self.prompt_text),
            placeholder=self._get_placeholder,
            rprompt=self._get_rprompt,
            erase_when_done=True,
        )

    def get_status_text(self) -> str | None:
        try:
            return _format_prompt_status_text(self.service.get_status())
        except Exception:
            return None

    async def get_input(self) -> tuple[str, bool]:
        prompt_tokens = [("class:prompt", self.prompt_text)]
        try:
            raw = await self.prompt_session.prompt_async(prompt_tokens)
        except (KeyboardInterrupt, EOFError):
            raise EOFError
        content = raw.strip()
        if not content:
            return ("", False)

        is_command = False
        if content.startswith("/"):
            first_word = content.split()[0] if content.split() else content
            if first_word in {command for command, _ in self.service.find_commands("", limit=64)}:
                is_command = True
            elif "/" not in content[1:]:
                is_command = True
        return (content, is_command)


class MSAgentApp:
    """Prompt-driven TUI application."""

    def __init__(
        self,
        backend: AgentBackend | None = None,
        service: ChatApplicationService | None = None,
        *,
        console: Console | None = None,
        history_file: Path | None = None,
    ) -> None:
        self.service = service or ChatApplicationService(backend or Agent())
        self.console = console or Console(
            theme=_TOKYO_NIGHT_THEME,
            force_terminal=True,
            legacy_windows=False,
            highlight=False,
            soft_wrap=True,
        )
        self.renderer = Renderer(self.console)
        self.history_file = history_file
        self.is_processing = False

    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        initialized = await self.service.initialize()
        if not initialized:
            self.renderer.render_error(
                self.service.get_status().error_message or "Failed to initialize msAgent."
            )
            return

        try:
            self.renderer.show_welcome(self.service.get_status())
            prompt = InteractivePrompt(
                self.service,
                history_file=self.history_file,
            )

            while True:
                try:
                    raw_input, _ = await prompt.get_input()
                except EOFError:
                    self.renderer.render_system_message("Goodbye.", title="Session")
                    break

                if not raw_input:
                    self.renderer.render_empty_prompt_submit(prompt.prompt_text)
                    continue

                intent = self.service.resolve_user_input(raw_input)
                if intent.type == "ignore":
                    continue
                if intent.type == "help":
                    self.renderer.render_help(self.service.find_commands("", limit=64))
                    continue
                if intent.type == "hotkeys":
                    self.renderer.render_hotkeys(prompt.hotkeys)
                    continue
                if intent.type == "tools":
                    await self._show_tools_dialog(prompt)
                    continue
                if intent.type == "exit":
                    self.renderer.render_system_message("Goodbye.", title="Session")
                    break
                if intent.type == "clear":
                    self.service.clear_history()
                    self.renderer.render_system_message("Chat history cleared.", title="Session")
                    continue
                if intent.type == "new_session":
                    session_number = self.service.start_new_session()
                    self.renderer.render_system_message(
                        f"Started session #{session_number}.",
                        title="Session",
                    )
                    continue
                if intent.type == "backend_status":
                    self.renderer.render_system_message(
                        self.service.get_status_message(),
                        title="Backend",
                    )
                    continue
                if intent.type == "backend_switch":
                    self.renderer.render_system_message(
                        self.service.switch_deepagents_backend(intent.message),
                        title="Backend",
                    )
                    continue

                self.renderer.render_user_message(
                    intent.message,
                    status_text=prompt.get_status_text(),
                )
                await self._run_chat_turn(intent.message)
        finally:
            await self.service.shutdown()

    async def _show_tools_dialog(self, prompt: InteractivePrompt) -> None:
        tools = self.service.get_available_tools()
        if not tools:
            self.renderer.render_system_message("No tools available.", title="Tools")
            return

        browser = ToolBrowser(prompt.prompt_text)
        await browser.show(
            tools,
            style=prompt.prompt_session.style,
        )

    async def _run_chat_turn(self, user_input: str) -> None:
        self.is_processing = True
        response_chunks: list[str] = []
        preview_lines = [""]

        try:
            with self.console.status("[indicator]Thinking...[/indicator]", spinner="dots") as status:
                async for event in self.service.stream_chat_events(user_input):
                    if event.type == "text" and event.content:
                        response_chunks.append(event.content)
                        preview_lines = _append_preview_lines(preview_lines, event.content)
                        status.update(self.renderer.build_stream_preview(preview_lines))
                        continue

                    if event.type == "tool_call":
                        status.stop()
                        tool_name = event.full_name or event.tool or "unknown_tool"
                        self.renderer.render_tool_call(tool_name, event.payload)
                        status.start()
                        status.update(self.renderer.build_stream_preview(preview_lines))
                        continue

                    if event.type == "tool_result":
                        continue

                    if event.type == "error" and event.content:
                        status.stop()
                        self.renderer.render_error(event.content)
                        status.start()
                        continue

                    if event.type == "done":
                        break

            final_text = "".join(response_chunks).strip()
            if final_text:
                self.renderer.render_assistant_message(final_text)
        finally:
            self.is_processing = False

def run_tui(
    backend: AgentBackend | None = None,
    service: ChatApplicationService | None = None,
) -> None:
    """Run the TUI application."""
    if backend is None and service is None:
        app = MSAgentApp()
    else:
        app = MSAgentApp(backend=backend, service=service)
    app.run()

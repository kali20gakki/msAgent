"""TUI interface for msagent using Textual."""

import asyncio
import json
import time
from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import (
    Input,
    Label,
    LoadingIndicator,
    Static,
    TextArea,
)

from .agent import Agent
from .application import ChatApplicationService
from .interfaces import AgentBackend, AgentStatus


class MessageWidget(Container):
    """Widget to display a chat message."""

    _STREAM_MARKDOWN_INTERVAL_S = 0.08
    
    DEFAULT_CSS = """
    MessageWidget {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }
    
    .gutter {
        width: 3;
        color: $accent;
        text-style: bold;
        padding-top: 1;
        content-align: center top;
    }

    .content-container {
        width: 1fr;
        height: auto;
    }

    .tool-message .role-label {
        color: #8f9bb3;
    }

    .tool-message .content-area {
        color: #aeb8cc;
    }

    .tool-message .selectable-text {
        color: #aeb8cc;
    }

    .header-row {
        layout: horizontal;
        height: 1;
        width: 100%;
        margin-bottom: 0;
    }

    .role-label {
        color: $text-muted;
        text-style: bold;
        width: 1fr;
    }

    .actions {
        width: auto;
    }
    
    /* Removed hover logic to make buttons always visible */

    .action-btn {
        background: #4c566a;
        border: none;
        color: #eceff4;
        height: 1;
        min-width: 6;
        padding: 0 1;
        margin-left: 1;
        text-align: center;
    }
    
    .action-btn:hover {
        background: #88c0d0;
        color: #2e3440;
        text-style: bold;
    }

    .content-area {
        height: auto;
        min-height: 1;
        color: #eceff4;
        padding-top: 0;
    }

    .message-duration {
        height: 1;
        color: #7f8aa3;
        text-style: dim;
        margin-top: 1;
    }

    .tool-input-wrap {
        height: auto;
        margin-top: 0;
        padding: 0;
    }

    .tool-input-toggle {
        color: #9eacc7;
        text-style: dim;
        background: transparent;
        height: 1;
        min-height: 1;
        padding: 0;
        margin: 0;
    }

    .tool-input-area {
        margin-top: 1;
        border: round #4c566a;
        background: #1f232b;
        padding: 0 1;
        height: auto;
        max-height: 12;
        color: #93a0b8;
    }

    .tool-output-wrap {
        height: auto;
        margin-top: 0;
        padding: 0;
    }

    .tool-output-toggle {
        color: #9eacc7;
        text-style: dim;
        background: transparent;
        height: 1;
        min-height: 1;
        padding: 0;
        margin: 0;
    }

    .tool-output-area {
        margin-top: 1;
        border: round #4c566a;
        background: #191d24;
        padding: 0 1;
        height: auto;
        max-height: 18;
        color: #b5c0d5;
    }
    
    /* 可选择的文本区域 */
    .selectable-text {
        height: auto;
        border: none;
        background: transparent;
        padding: 0;
        color: #eceff4;
    }
    
    .selectable-text:focus {
        border: solid #88c0d0;
        background: #2e3440;
    }
    
    .hidden {
        display: none;
    }

    /* Role specific styles */
    .user-gutter { color: $accent; }
    .assistant-gutter { color: $success; }
    .tool-gutter { color: $warning; }
    .system-gutter { color: $secondary; }
    """
    
    def __init__(
        self,
        role: str,
        content: str,
        *,
        duration_s: float | None = None,
        tool_input_text: str | None = None,
        tool_output_text: str | None = None,
        tool_output_truncated: bool = False,
        **kwargs: Any,
    ):
        self.role = role
        self.content = content
        self.duration_s = duration_s
        self.tool_input_text = tool_input_text
        self.tool_output_text = tool_output_text
        self.tool_output_truncated = tool_output_truncated
        self._stream_markdown_enabled = False
        self._last_markdown_render_ts = 0.0
        super().__init__(**kwargs)
    
    def compose(self) -> ComposeResult:
        # Determine icon
        icon = "✦"
        if self.role == "user":
            icon = "❯"
        elif self.role == "tool":
            icon = "🛠"
        elif self.role == "system":
            icon = "ℹ"
            
        gutter_class = f"{self.role}-gutter"
        
        yield Label(icon, classes=f"gutter {gutter_class}")
        
        content_classes = "content-container tool-message" if self.role == "tool" else "content-container"
        with Container(classes=content_classes):
            # Header with actions
            with Horizontal(classes="header-row"):
                yield Static(" ", classes="role-label")
                with Horizontal(classes="actions"):
                    yield Label("复制", id="copy-btn", classes="action-btn")
                    yield Label("渲染", id="raw-btn", classes="action-btn")
            
            # Markdown 渲染视图（默认隐藏，流式输出完成后显示）
            yield Static(RichMarkdown(self.content), id="render-md", classes="content-area hidden")
            
            # 纯文本视图（默认显示，用于流式输出和复制）
            yield CopyableTextArea(
                self.content, 
                id="content-text", 
                read_only=True, 
                classes="selectable-text",
                show_line_numbers=False
            )
            if self.role == "tool":
                if self.tool_input_text:
                    with Container(classes="tool-input-wrap"):
                        yield Label(
                            "▶ 输入参数（点击展开）",
                            id="tool-input-toggle",
                            classes="tool-input-toggle",
                        )
                        yield CopyableTextArea(
                            self.tool_input_text,
                            id="tool-input-text",
                            read_only=True,
                            classes="selectable-text tool-input-area hidden",
                            show_line_numbers=False,
                        )

                output_wrap_class = "tool-output-wrap"
                if not self.tool_output_text:
                    output_wrap_class = "tool-output-wrap hidden"
                with Container(id="tool-output-wrap", classes=output_wrap_class):
                    yield Label(
                        self._tool_output_toggle_label(expanded=False),
                        id="tool-output-toggle",
                        classes="tool-output-toggle",
                    )
                    yield CopyableTextArea(
                        self.tool_output_text or "",
                        id="tool-output-text",
                        read_only=True,
                        classes="selectable-text tool-output-area hidden",
                        show_line_numbers=False,
                    )
            if self.role == "assistant":
                duration_classes = "message-duration"
                if self.duration_s is None:
                    duration_classes += " hidden"
                yield Static(
                    self._format_duration_label(),
                    id="message-duration",
                    classes=duration_classes,
                )

    def update_content(self, content: str) -> None:
        """Update the message content."""
        self.content = content
        try:
            self.query_one("#content-text", CopyableTextArea).text = content
            self._render_markdown(force=True)
        except NoMatches:
            # Widget may be removed during async updates.
            return
    
    def update_content_fast(self, content: str, *, render_markdown: bool = False) -> None:
        """快速更新内容（流式输出时使用）"""
        self.content = content
        try:
            self.query_one("#content-text", CopyableTextArea).text = content
            if render_markdown or self._stream_markdown_enabled:
                self._render_markdown(force=False)
        except NoMatches:
            return

    def start_stream_markdown(self) -> None:
        """Enable markdown rendering while streaming."""
        self._stream_markdown_enabled = True
        self._last_markdown_render_ts = 0.0
        try:
            self._set_render_mode("markdown")
            self._render_markdown(force=True)
        except NoMatches:
            return
    
    def finalize_content(self) -> None:
        """流式输出完成，切换到美观的 Markdown 渲染模式"""
        self._stream_markdown_enabled = False
        try:
            self._render_markdown(force=True)
            text_widget = self.query_one("#content-text", CopyableTextArea)
            if "hidden" in text_widget.classes:
                self._set_render_mode("markdown")
        except NoMatches:
            return

    @staticmethod
    def _format_duration_text(duration_s: float | None) -> str:
        if duration_s is None or duration_s < 0:
            return ""
        if duration_s < 1:
            return f"耗时 {duration_s * 1000:.0f}ms"
        if duration_s < 60:
            precision = 1 if duration_s < 10 else 0
            return f"耗时 {duration_s:.{precision}f}s"
        minutes, seconds = divmod(duration_s, 60)
        rounded_seconds = int(round(seconds))
        whole_minutes = int(minutes)
        if rounded_seconds == 60:
            whole_minutes += 1
            rounded_seconds = 0
        return f"耗时 {whole_minutes}m {rounded_seconds:02d}s"

    def _format_duration_label(self) -> str:
        return self._format_duration_text(self.duration_s)

    def set_duration(self, duration_s: float | None) -> None:
        if self.role != "assistant":
            return
        self.duration_s = duration_s
        try:
            duration_widget = self.query_one("#message-duration", Static)
        except NoMatches:
            return

        duration_label = self._format_duration_label()
        duration_widget.update(duration_label)
        if duration_label:
            duration_widget.remove_class("hidden")
            return
        duration_widget.add_class("hidden")

    def _render_markdown(self, *, force: bool) -> None:
        if not force:
            now = time.monotonic()
            if now - self._last_markdown_render_ts < self._STREAM_MARKDOWN_INTERVAL_S:
                return
            self._last_markdown_render_ts = now
        else:
            self._last_markdown_render_ts = time.monotonic()
        self.query_one("#render-md", Static).update(RichMarkdown(self.content))

    def _set_render_mode(self, mode: str) -> None:
        md_widget = self.query_one("#render-md", Static)
        text_widget = self.query_one("#content-text", CopyableTextArea)
        raw_btn = self.query_one("#raw-btn", Label)
        if mode == "raw":
            text_widget.remove_class("hidden")
            md_widget.add_class("hidden")
            raw_btn.update("渲染")
            return
        text_widget.add_class("hidden")
        md_widget.remove_class("hidden")
        raw_btn.update("原文")

    def _tool_output_toggle_label(self, *, expanded: bool) -> str:
        suffix = "（已截断）" if self.tool_output_truncated else ""
        action = "收起" if expanded else "展开"
        arrow = "▼" if expanded else "▶"
        return f"{arrow} 输出{suffix}（点击{action}）"

    def update_tool_output(self, content: str | None, *, truncated: bool = False) -> None:
        if self.role != "tool":
            return
        self.tool_output_truncated = truncated
        self.tool_output_text = content
        try:
            output_wrap = self.query_one("#tool-output-wrap", Container)
            output_toggle = self.query_one("#tool-output-toggle", Label)
            output_text = self.query_one("#tool-output-text", CopyableTextArea)
        except NoMatches:
            return

        if not content:
            output_wrap.add_class("hidden")
            output_text.text = ""
            output_text.add_class("hidden")
            output_toggle.update(self._tool_output_toggle_label(expanded=False))
            return

        output_wrap.remove_class("hidden")
        output_text.text = content
        output_text.add_class("hidden")
        output_toggle.update(self._tool_output_toggle_label(expanded=False))

    def on_click(self, event: events.Click) -> None:
        """Handle click events."""
        if event.widget.id == "copy-btn":
            try:
                import pyperclip
                pyperclip.copy(self.content)
                self.app.notify("已复制到剪贴板", severity="information")
            except Exception:
                self.app.copy_to_clipboard(self.content)
                self.app.notify("已复制（备用方式）", severity="information")
        elif event.widget.id == "raw-btn":
            text_widget = self.query_one("#content-text", CopyableTextArea)
            if "hidden" in text_widget.classes:
                self._set_render_mode("raw")
            else:
                self._set_render_mode("markdown")
        elif event.widget.id == "tool-input-toggle":
            try:
                input_widget = self.query_one("#tool-input-text", CopyableTextArea)
            except NoMatches:
                return
            btn = event.widget
            if "hidden" in input_widget.classes:
                input_widget.remove_class("hidden")
                btn.update("▼ 输入参数（点击收起）")
            else:
                input_widget.add_class("hidden")
                btn.update("▶ 输入参数（点击展开）")
        elif event.widget.id == "tool-output-toggle":
            try:
                output_widget = self.query_one("#tool-output-text", CopyableTextArea)
            except NoMatches:
                return
            btn = event.widget
            if "hidden" in output_widget.classes:
                output_widget.remove_class("hidden")
                btn.update(self._tool_output_toggle_label(expanded=True))
            else:
                output_widget.add_class("hidden")
                btn.update(self._tool_output_toggle_label(expanded=False))


class CopyableTextArea(TextArea):
    """TextArea with copy support."""
    
    BINDINGS = [
        Binding("ctrl+c", "copy_selection", "复制", show=False),
    ]
    
    def action_copy_selection(self) -> None:
        """Copy selected text to clipboard."""
        if self.selected_text:
            try:
                import pyperclip
                pyperclip.copy(self.selected_text)
                self.app.notify("已复制选中文本", severity="information")
            except Exception:
                self.app.copy_to_clipboard(self.selected_text)
                self.app.notify("已复制选中文本", severity="information")
        else:
            # If nothing selected, maybe quit? No, better safe than sorry.
            self.app.notify("未选择文本", severity="warning")


class ChatWelcomeBanner(Vertical):
    """Small welcome banner in chat."""
    
    DEFAULT_CSS = """
    ChatWelcomeBanner {
        border: solid $accent;
        padding: 1 2;
        margin: 1 0 2 0;
        background: $surface;
        height: auto;
        width: 100%;
    }
    
    .welcome-message {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .mcp-status {
        color: $success;
        padding-top: 1;
        border-top: solid #d8dee9;
        width: 100%;
    }

    .skills-status {
        color: $accent;
        padding-top: 1;
        border-top: solid #d8dee9;
        width: 100%;
    }
    """

    def __init__(
        self,
        *,
        mcp_servers: list[str] | None = None,
        loaded_skills: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mcp_servers = mcp_servers
        self._loaded_skills = loaded_skills or []

    def compose(self) -> ComposeResult:
        yield Label("✱ msAgent 已就绪，我可以帮你做什么？", classes="welcome-message")

        servers = self._mcp_servers or []
        if servers:
            server_str = ", ".join(servers)
            yield Label(f"🔌 已连接 MCP 服务器：{server_str}", classes="mcp-status")
        if self._loaded_skills:
            skills_str = ", ".join(self._loaded_skills)
            yield Label(f"🧠 已加载SKILLS：{skills_str}", classes="skills-status")

class CustomFooter(Static):
    """Custom footer with shortcuts."""
    
    DEFAULT_CSS = """
    CustomFooter {
        dock: bottom;
        height: 1;
        width: 100%;
        background: $surface;
        color: $text-muted;
        padding: 0 1; 
    }
    """
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._session_status = "会话: #1"
        self._model_status = "模型: unknown"
        self._context_status = "提示词: N/A"
        self._token_status = "Token: 0"

    def set_session_status(self, status: str) -> None:
        self._session_status = status
        self.refresh()

    def set_model_status(self, status: str) -> None:
        self._model_status = status
        self.refresh()

    def set_token_status(self, status: str) -> None:
        self._token_status = status
        self.refresh()

    def set_context_status(self, status: str) -> None:
        self._context_status = status
        self.refresh()

    def render(self) -> str:
        return (
            f"{self._session_status} • {self._model_status} • "
            f"{self._context_status} • {self._token_status}"
        )


class ChatArea(VerticalScroll):
    """Area to display chat messages."""
    
    DEFAULT_CSS = """
    ChatArea {
        scrollbar-gutter: stable;
    }
    """
    
    def compose(self) -> ComposeResult:
        # We start empty now, message added upon initialization
        yield from []
    
    async def add_message(
        self,
        role: str,
        content: str,
        *,
        duration_s: float | None = None,
        tool_input_text: str | None = None,
        tool_output_text: str | None = None,
        tool_output_truncated: bool = False,
    ) -> MessageWidget:
        """Add a message to the chat area."""
        widget = MessageWidget(
            role,
            content,
            duration_s=duration_s,
            tool_input_text=tool_input_text,
            tool_output_text=tool_output_text,
            tool_output_truncated=tool_output_truncated,
        )
        await self.mount(widget)
        widget.scroll_visible()
        return widget


class SendButton(Static):
    """Compact send/stop control with deterministic centered text."""

    def __init__(self, text: str = "发送", **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._text = text

    def set_text(self, text: str) -> None:
        self._text = text
        self.refresh()

    def render(self) -> RenderableType:
        inner_width = max((self.size.width or 7) - 2, len(self._text))
        diff = max(inner_width - len(self._text), 0)
        # For odd extra space, bias one cell to the left padding to avoid visual left-lean.
        left_pad = (diff + 1) // 2
        right_pad = diff - left_pad
        return Text((" " * left_pad) + self._text + (" " * right_pad))


class InputArea(Container):
    """Area for user input."""
    
    DEFAULT_CSS = """
    InputArea {
        height: 3;
        min-height: 3;
        max-height: 3;
        margin: 0 0 1 0;
        border: none;
        background: #1f232b;
        padding: 0;
        align-vertical: middle;
    }
    
    InputArea:focus-within {
        background: #242a33;
    }
    
    .input-row {
        layout: horizontal;
        align-vertical: middle;
        height: 3;
        margin: 0 1 0 0;
        width: 100%;
    }
    
    .prompt-label {
        width: 2;
        height: 3;
        padding: 0;
        color: #81a1c1;
        text-style: none;
        content-align: center middle;
    }
    
    #message-input {
        width: 1fr;
        min-width: 20;
        background: #1a1e26;
        border: round #5a6478;
        color: #eceff4;
        padding: 0 1;
        height: 3;
        margin: 0 1 0 0;
    }
    
    #message-input:focus {
        border: round #88c0d0;
    }

    #send-btn {
        width: 7;
        min-width: 7;
        max-width: 7;
        height: 3;
        margin: 0;
        background: #5e81ac;
        color: #eceff4;
        border: round #88c0d0;
        padding: 0;
        text-style: bold;
        text-align: center;
        content-align: center middle;
    }

    #send-btn.processing {
        background: #bf616a;
        color: #ffffff;
        border: round #d08770;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="input-row"):
            yield Label(">", classes="prompt-label")
            yield Input(
                placeholder="向msAgent提问（@ 引用文件，/ 命令补全）",
                id="message-input",
            )
            yield SendButton("发送", id="send-btn")

class WelcomeScreen(Screen):
    """Full screen welcome page."""
    
    CSS = """
    WelcomeScreen {
        align: center middle;
        background: $background;
    }
    
    .welcome-container {
        width: 63;
        height: auto;
    }
    
    .welcome-box {
        border: solid $accent;
        padding: 1 2;
        width: auto;
        color: $text;
        background: $surface;
        margin-bottom: 2;
    }
    
    .ascii-art {
        color: $accent;
        text-align: left;
        margin-bottom: 2;
        width: 100%;
    }
    
    .continue-text {
        color: $text-muted;
        text-align: left;
        width: 100%;
    }
    
    .status-text {
        color: $warning;
        text-align: left;
        margin-top: 1;
        width: 100%;
    }
    
    LoadingIndicator {
        height: 1;
        width: 100%;
        margin: 1 0;
        color: $accent;
    }
    
    .hidden {
        display: none;
    }
    """
    
    BINDINGS = [
        Binding("enter", "continue", "继续"),
    ]
    
    def compose(self) -> ComposeResult:
        ascii_text = r"""
███╗   ███╗███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗
████╗ ████║██╔════╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██╔████╔██║███████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██║╚██╔╝██║╚════██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██║ ╚═╝ ██║███████║██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
"""
        
        with Vertical(classes="welcome-container"):
            yield Static(ascii_text, classes="ascii-art")
            
            # Loading state components
            yield LoadingIndicator(id="loading")
            yield Label("正在初始化 Agent 与 MCP 工具...", id="status-text", classes="status-text")
            
            # Ready state component (initially hidden)
            t = Text.from_markup("按 [bold white]Enter[/bold white] 继续")
            yield Label(t, id="continue-text", classes="continue-text hidden")
            
    async def on_mount(self) -> None:
        """Start initialization when screen mounts."""
        self.run_worker(self._monitor_agent_init(), exclusive=True)
        
    async def _monitor_agent_init(self) -> None:
        """Monitor agent initialization status."""
        try:
            # Wait for agent to be initialized by the App worker
            while True:
                status = self.app.service.get_status()
                if status.is_initialized or status.error_message:
                    break
                await asyncio.sleep(0.1)
            
            if status.error_message:
                # Show error
                self.query_one("#loading").add_class("hidden")
                self.query_one("#status-text").update(f"❌ 错误：{status.error_message}")
            else:
                # Update UI
                self.query_one("#loading").add_class("hidden")
                self.query_one("#status-text").add_class("hidden")
                self.query_one("#continue-text").remove_class("hidden")
                
                self.is_ready = True
            
        except Exception as e:
            # Show error
            self.query_one("#loading").add_class("hidden")
            self.query_one("#status-text").update(f"❌ 错误：{e}")
            
    def action_continue(self) -> None:
        if getattr(self, "is_ready", False):
            self.app.push_screen(ChatScreen())


class ChatScreen(Screen):
    """Main chat interface."""
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", show=False),
        Binding("ctrl+l", "clear", "清空对话", show=False),
        Binding("ctrl+n", "new_session", "新会话", show=False),
    ]
    _MAX_TOOL_OUTPUT_CHARS = 4000
    _LOADING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _THINKING_STATUS_TEXT = "思考中..."
    _WAITING_MCP_STATUS_TEXT = "等待 MCP 工具结果..."

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._completion_matches: list[tuple[str, str]] = []
        self._completion_target_range: tuple[int, int] | None = None
        self._completion_selected_index = 0
        self._current_worker: Any | None = None
        self._completion_refresh_task: asyncio.Task[None] | None = None
    
    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            with ChatArea(id="chat-area"):
                pass
            yield VerticalScroll(id="at-suggestions", classes="hidden")
            yield InputArea()
        yield CustomFooter()

    async def on_mount(self) -> None:
        """Called when screen is mounted."""
        status = self._get_status()
        chat_area = self.query_one("#chat-area", ChatArea)

        if status.is_initialized:
            self._mount_welcome_banner(chat_area, status)
        else:
            await chat_area.add_message(
                "system",
                status.error_message or "Agent 尚未初始化",
            )

        # Focus input
        self.query_one("#message-input", Input).focus()
        self._update_footer_model()
        self._update_footer_session()
        self._update_footer_context()
        self._update_footer_tokens()
        self._render_completion_suggestions([])
        self._set_send_button_state(False)

    def on_unmount(self) -> None:
        task = self._completion_refresh_task
        if task is not None and not task.done():
            task.cancel()

    def on_click(self, event: events.Click) -> None:
        if event.widget.id != "send-btn":
            return
        if self.app.is_processing:
            self.interrupt_message()
            return
        self.send_message()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update completion suggestions as user types."""
        if event.input.id != "message-input":
            return
        self._schedule_completion_refresh(event.input)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理输入提交事件"""
        if event.input.id != "message-input":
            return
        intent = self.app.service.resolve_user_input(event.input.value)
        if intent.type != "chat":
            self.send_message()
            return
        if self._try_apply_selected_completion(event.input):
            return
        # 不使用 await，让 UI 立即响应
        self.send_message()

    def on_key(self, event: events.Key) -> None:
        input_widget = self.query_one("#message-input", Input)
        if self.focused is not input_widget:
            return
        if not self._completion_matches or not self._completion_target_range:
            return

        if event.key == "up":
            self._completion_selected_index = (
                self._completion_selected_index - 1
            ) % len(self._completion_matches)
            self._render_completion_suggestions(self._completion_matches)
            event.stop()
            event.prevent_default()
            return

        if event.key == "down":
            self._completion_selected_index = (
                self._completion_selected_index + 1
            ) % len(self._completion_matches)
            self._render_completion_suggestions(self._completion_matches)
            event.stop()
            event.prevent_default()
            return

        if event.key != "tab":
            return

        self._try_apply_selected_completion(input_widget)
        event.stop()
        event.prevent_default()
            
    def send_message(self) -> None:
        """发送消息（同步启动，异步执行）"""
        app: MSAgentApp = self.app
        if app.is_processing:
            return
            
        input_widget = self.query_one("#message-input", Input)
        intent = app.service.resolve_user_input(input_widget.value)
        if intent.type == "ignore":
            return
        if intent.type == "exit":
            app.exit()
            return
        if intent.type == "clear":
            self.action_clear()
            input_widget.value = ""
            return
        if intent.type == "new_session":
            self.action_new_session()
            input_widget.value = ""
            return
        if intent.type != "chat":
            return
        message = intent.message
        
        # 立即清空输入框
        input_widget.value = ""
        
        # 标记正在处理
        app.is_processing = True
        self._set_send_button_state(True)
        
        # 使用 run_worker 在后台执行，UI 立即更新
        self._current_worker = self.run_worker(self._process_message(message), exclusive=True)

    def interrupt_message(self) -> None:
        app: MSAgentApp = self.app
        if not app.is_processing:
            return
        worker = self._current_worker
        if worker is not None:
            worker.cancel()
        self._set_send_button_state(True)
    
    def _loading_text(self, status_text: str, frame: str | None = None) -> str:
        prefix = frame or self._LOADING_FRAMES[0]
        return f"{prefix} {status_text}"

    async def _animate_loading(
        self,
        widget: MessageWidget,
        stop_event: asyncio.Event,
        status_text: str,
    ) -> None:
        """动态加载动画"""
        frame_idx = 0
        
        while not stop_event.is_set():
            if not widget.is_mounted:
                break
            widget.update_content_fast(
                self._loading_text(status_text, self._LOADING_FRAMES[frame_idx])
            )
            frame_idx = (frame_idx + 1) % len(self._LOADING_FRAMES)
            await asyncio.sleep(0.1)  # 100ms 更新一次
    
    async def _process_message(self, message: str) -> None:
        """后台处理消息的 worker"""
        app: MSAgentApp = self.app
        chat_area = self.query_one("#chat-area", ChatArea)
        
        try:
            # 1. 立即显示用户输入
            await chat_area.add_message("user", message)
            chat_area.scroll_end(animate=False)
            
            status = self._get_status()
            if not status.is_initialized:
                await chat_area.add_message(
                    "system",
                    status.error_message or "Agent 尚未初始化",
                )
                return
            
            # 2. 创建加载消息并启动动画
            loading_widget = await chat_area.add_message(
                "assistant",
                self._loading_text(self._THINKING_STATUS_TEXT),
            )
            chat_area.scroll_end(animate=False)
            
            # 启动加载动画
            stop_animation = asyncio.Event()
            animation_task = asyncio.create_task(
                self._animate_loading(
                    loading_widget,
                    stop_animation,
                    self._THINKING_STATUS_TEXT,
                )
            )
            
            # 3. 流式接收并实时更新
            response_widget = loading_widget
            response_text = ""
            response_duration_s: float | None = None
            first_chunk_received = False
            pending_tool_widgets: list[MessageWidget] = []
            
            try:
                async for event in app.service.stream_chat_events(message):
                    event_type = event.type

                    if event_type == "tool_call":
                        # 当前 assistant 段落结束，后续回答应显示在 tool 提示下方
                        stop_animation.set()
                        if not animation_task.done():
                            await animation_task
                        if first_chunk_received:
                            response_widget.finalize_content()
                        else:
                            # 还未收到文本时，移除占位 thinking，避免界面残留“卡住”提示
                            await response_widget.remove()

                        server = event.server or "unknown"
                        tool = event.tool or "unknown_tool"
                        tool_input = self._format_tool_input(event.payload)
                        tool_widget = await chat_area.add_message(
                            "tool",
                            f"调用 MCP 工具：`{server}__{tool}`",
                            tool_input_text=tool_input,
                        )
                        pending_tool_widgets.append(tool_widget)
                        response_widget = await chat_area.add_message(
                            "assistant",
                            self._loading_text(self._WAITING_MCP_STATUS_TEXT),
                        )
                        response_text = ""
                        response_duration_s = None
                        first_chunk_received = False
                        stop_animation = asyncio.Event()
                        animation_task = asyncio.create_task(
                            self._animate_loading(
                                response_widget,
                                stop_animation,
                                self._WAITING_MCP_STATUS_TEXT,
                            )
                        )
                        chat_area.scroll_end(animate=False)
                        await asyncio.sleep(0)
                        continue

                    if event_type == "tool_result":
                        tool_output, output_truncated = self._format_tool_output(event.payload)
                        if pending_tool_widgets:
                            pending_tool_widgets.pop(0).update_tool_output(
                                tool_output,
                                truncated=output_truncated,
                            )
                        if not first_chunk_received:
                            stop_animation.set()
                            if not animation_task.done():
                                await animation_task
                            response_widget.update_content(
                                self._loading_text(self._THINKING_STATUS_TEXT)
                            )
                            stop_animation = asyncio.Event()
                            animation_task = asyncio.create_task(
                                self._animate_loading(
                                    response_widget,
                                    stop_animation,
                                    self._THINKING_STATUS_TEXT,
                                )
                            )
                        chat_area.scroll_end(animate=False)
                        await asyncio.sleep(0)
                        continue

                    if event_type == "done":
                        response_duration_s = event.duration_s
                        continue

                    if event_type == "error":
                        if event.content:
                            if not first_chunk_received:
                                stop_animation.set()
                                if not animation_task.done():
                                    await animation_task
                                first_chunk_received = True
                                response_text = event.content
                                response_widget.start_stream_markdown()
                            else:
                                response_text += event.content
                            response_widget.update_content_fast(
                                response_text, render_markdown=True
                            )
                            chat_area.scroll_end(animate=False)
                            await asyncio.sleep(0)
                        continue

                    if event_type != "text":
                        continue

                    chunk = event.content
                    if not chunk:
                        continue

                    if not first_chunk_received:
                        # 停止加载动画
                        stop_animation.set()
                        if not animation_task.done():
                            await animation_task

                        # 收到第一个 chunk，开始显示内容
                        first_chunk_received = True
                        response_text = chunk
                        response_widget.start_stream_markdown()
                        response_widget.update_content_fast(
                            response_text, render_markdown=True
                        )
                    else:
                        # 追加内容并立即更新
                        response_text += chunk
                        response_widget.update_content_fast(
                            response_text, render_markdown=True
                        )

                    # 滚动到底部（不触发全局刷新）
                    chat_area.scroll_end(animate=False)

                    # 让出控制权
                    await asyncio.sleep(0)

                # 循环结束后确保动画停止，避免残留 spinner
                stop_animation.set()
                if not animation_task.done():
                    await animation_task

                # 流式输出完成后，渲染最终的 Markdown
                if first_chunk_received:
                    response_widget.finalize_content()
                    response_widget.set_duration(response_duration_s)
                
            except asyncio.CancelledError:
                # Worker cancelled (e.g. quit screen); stop background animation quietly.
                stop_animation.set()
                if not animation_task.done():
                    await animation_task
                return
            except Exception as stream_error:
                # 确保停止动画
                stop_animation.set()
                if not animation_task.done():
                    await animation_task
                raise stream_error
            
            # 如果没有收到任何内容
            if not first_chunk_received:
                stop_animation.set()
                if not animation_task.done():
                    await animation_task
                response_widget.update_content("_未收到回复_")
                 
        except asyncio.CancelledError:
            return
        except Exception as e:
            await chat_area.add_message("system", f"❌ 错误：{str(e)}")
        finally:
            app.is_processing = False
            self._current_worker = None
            if self.is_mounted:
                self._set_send_button_state(False)
                self._update_footer_tokens()
                chat_area.scroll_end(animate=False)

    def _update_footer_tokens(self) -> None:
        usage = self._get_status().usage
        total_tokens = usage.total_tokens if usage is not None else None
        token_text = (
            f"Token: {self._format_token_count(total_tokens)}"
            if total_tokens is not None
            else "Token: N/A"
        )
        footer = self._query_footer()
        if footer is None:
            return
        footer.set_token_status(token_text)
        self._update_footer_context()

    def _update_footer_model(self) -> None:
        status = self._get_status()
        footer = self._query_footer()
        if footer is None:
            return
        footer.set_model_status(f"模型: {status.provider}/{status.model}")

    def _update_footer_session(self) -> None:
        footer = self._query_footer()
        if footer is None:
            return
        footer.set_session_status(f"会话: #{self._get_status().session_number}")

    def _update_footer_context(self) -> None:
        usage = self._get_status().usage
        prompt_tokens = usage.prompt_tokens if usage is not None else None

        if prompt_tokens is None:
            footer = self._query_footer()
            if footer is None:
                return
            footer.set_context_status("提示词: N/A")
            return
        footer = self._query_footer()
        if footer is None:
            return
        footer.set_context_status(f"提示词: {self._format_token_count(prompt_tokens)}")

    def _get_status(self) -> AgentStatus:
        return self.app.service.get_status()

    def _mount_welcome_banner(self, chat_area: ChatArea, status: AgentStatus) -> None:
        chat_area.mount(
            ChatWelcomeBanner(
                mcp_servers=list(status.connected_servers),
                loaded_skills=list(status.loaded_skills),
            )
        )

    def _query_footer(self) -> CustomFooter | None:
        try:
            return self.query_one(CustomFooter)
        except NoMatches:
            return None

    def _format_token_count(self, count: int) -> str:
        if count < 1_000:
            return str(count)
        if count < 1_000_000:
            value = count / 1_000
            return f"{value:.1f}K" if value < 10 else f"{value:.0f}K"
        value = count / 1_000_000
        return f"{value:.1f}M" if value < 10 else f"{value:.0f}M"

    def action_clear(self) -> None:
        if self.app.is_processing:
            self.notify("请先停止当前回复，再清空对话。", severity="warning")
            return
        self.app.service.clear_history()
        self._reset_chat_area("对话历史已清空。")
        self._update_footer_tokens()
        self._update_footer_session()

    def action_new_session(self) -> None:
        if self.app.is_processing:
            self.notify("请先停止当前回复，再开始新会话。", severity="warning")
            return
        new_session_number = self.app.service.start_new_session()
        self._reset_chat_area(f"已开始新会话 #{new_session_number}，上下文已清空。")
        self._update_footer_tokens()
        self._update_footer_session()

    def _reset_chat_area(self, system_message: str) -> None:
        chat_area = self.query_one("#chat-area", ChatArea)
        chat_area.remove_children()
        self._mount_welcome_banner(chat_area, self._get_status())
        chat_area.run_worker(self._add_system_message(chat_area, system_message))

    def _refresh_completion_candidates(self, input_widget: Input) -> None:
        cursor = getattr(input_widget, "cursor_position", len(input_widget.value))
        at_token = self._extract_active_at_token(input_widget.value, cursor)
        if at_token is not None:
            query, start, end = at_token
            file_matches = self.app.service.find_local_files(query, limit=30)
            self._completion_matches = [(f"@{path}", "") for path in file_matches]
            self._completion_target_range = (start, end)
            self._completion_selected_index = 0
            self._render_completion_suggestions(self._completion_matches)
            return

        slash_token = self._extract_active_slash_token(input_widget.value, cursor)
        if slash_token is not None:
            query, start, end = slash_token
            command_matches = self.app.service.find_commands(query, limit=30)
            self._completion_matches = list(command_matches)
            self._completion_target_range = (start, end)
            self._completion_selected_index = 0
            self._render_completion_suggestions(self._completion_matches)
            return

        self._completion_matches = []
        self._completion_target_range = None
        self._completion_selected_index = 0
        self._render_completion_suggestions([])

    def _schedule_completion_refresh(self, input_widget: Input) -> None:
        task = self._completion_refresh_task
        if task is not None and not task.done():
            task.cancel()
        self._completion_refresh_task = asyncio.create_task(
            self._debounced_refresh_completion(input_widget)
        )

    async def _debounced_refresh_completion(self, input_widget: Input) -> None:
        try:
            await asyncio.sleep(0.06)
        except asyncio.CancelledError:
            return
        if not self.is_mounted:
            return
        self._refresh_completion_candidates(input_widget)

    def _extract_active_at_token(
        self, value: str, cursor: int
    ) -> tuple[str, int, int] | None:
        if cursor < 0 or cursor > len(value):
            return None
        at_pos = value.rfind("@", 0, cursor)
        if at_pos < 0:
            return None
        if at_pos > 0 and not value[at_pos - 1].isspace():
            return None
        token = value[at_pos + 1 : cursor]
        if not token:
            return ("", at_pos, cursor)
        if any(ch.isspace() for ch in token):
            return None
        return (token, at_pos, cursor)

    def _extract_active_slash_token(
        self, value: str, cursor: int
    ) -> tuple[str, int, int] | None:
        if cursor < 0 or cursor > len(value):
            return None
        if not value:
            return None
        if "\n" in value:
            return None

        start = 0
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value):
            return None
        if value[start] != "/":
            return None

        if value[:start].strip():
            return None
        if value[cursor:].strip():
            return None

        query = value[start:cursor]
        if not query:
            query = "/"
        if query[-1].isspace():
            return None
        return (query, start, len(value))

    def _render_completion_suggestions(self, items: list[tuple[str, str]]) -> None:
        container = self.query_one("#at-suggestions", VerticalScroll)
        container.remove_children()
        if not items:
            container.add_class("hidden")
            return
        container.remove_class("hidden")
        for idx, (value, detail) in enumerate(items):
            cls = (
                "suggestion-item selected"
                if idx == self._completion_selected_index
                else "suggestion-item"
            )
            text = value if not detail else f"{value}  {detail}"
            container.mount(Label(text, classes=cls))
        selected_nodes = list(container.query(".suggestion-item.selected"))
        if selected_nodes:
            selected_nodes[0].scroll_visible(animate=False)

    def _try_apply_selected_completion(self, input_widget: Input) -> bool:
        if not self._completion_matches or not self._completion_target_range:
            return False
        cursor = getattr(input_widget, "cursor_position", len(input_widget.value))
        if (
            self._extract_active_at_token(input_widget.value, cursor) is None
            and self._extract_active_slash_token(input_widget.value, cursor) is None
        ):
            return False

        start, end = self._completion_target_range
        replacement = self._completion_matches[self._completion_selected_index][0]
        value = input_widget.value
        updated = value[:start] + replacement + value[end:]
        new_cursor = start + len(replacement)
        if new_cursor == len(updated) or not updated[new_cursor].isspace():
            updated = updated[:new_cursor] + " " + updated[new_cursor:]
            new_cursor += 1

        input_widget.value = updated
        if hasattr(input_widget, "cursor_position"):
            input_widget.cursor_position = new_cursor
        self._refresh_completion_candidates(input_widget)
        return True

    async def _add_system_message(self, chat_area: ChatArea, message: str) -> None:
        await chat_area.add_message("system", message)

    def _format_tool_input(self, value: Any) -> str | None:
        return self._format_tool_payload(value)

    def _format_tool_output(self, value: Any) -> tuple[str | None, bool]:
        payload = self._format_tool_payload(value)
        if payload is None:
            return (None, False)
        return self._truncate_for_tool_output(payload)

    def _format_tool_payload(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except Exception:
                return stripped
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                return str(value)
        return str(value)

    def _truncate_for_tool_output(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._MAX_TOOL_OUTPUT_CHARS:
            return (text, False)

        kept = text[: self._MAX_TOOL_OUTPUT_CHARS].rstrip()
        omitted = len(text) - self._MAX_TOOL_OUTPUT_CHARS
        suffix = f"\n\n...[省略 {omitted} 个字符]..."
        return (f"{kept}{suffix}", True)

    def _set_send_button_state(self, processing: bool) -> None:
        try:
            btn = self.query_one("#send-btn", SendButton)
        except NoMatches:
            return
        if processing:
            btn.set_text("停止")
            btn.add_class("processing")
            return
        btn.set_text("发送")
        btn.remove_class("processing")


class MSAgentApp(App):
    """msagent TUI Application."""
    
    CSS = """
    /* Theme Variables */
    $accent: #88c0d0;
    $success: #a3be8c;
    $warning: #ebcb8b;
    $secondary: #81a1c1;
    $background: #121212;
    $surface: #2e3440;
    $text: #eceff4;
    $text-muted: #d8dee9;
    
    Screen {
        background: $background;
        color: $text;
    }
    
    #main-container {
        width: 100%;
        height: 1fr;
        padding: 1 2 2 2;
    }
    
    #chat-area {
        height: 1fr;
        margin-bottom: 1;
        background: $background;
    }

    #at-suggestions {
        height: auto;
        max-height: 8;
        margin: 0 0 1 0;
        padding: 0 1;
        background: #1f232b;
        border: round #3b4252;
    }

    #at-suggestions.hidden {
        display: none;
    }

    .suggestion-item {
        color: #81a1c1;
        height: 1;
        padding: 0 1;
    }

    .suggestion-item.selected {
        background: #3b4252;
        color: #eceff4;
        text-style: bold;
    }
    """
    
    def __init__(
        self,
        backend: AgentBackend | None = None,
        service: ChatApplicationService | None = None,
        **kwargs: Any,
    ):
        if service is not None:
            self._service = service
        else:
            self._service = ChatApplicationService(backend or Agent())
        self.is_processing = False
        super().__init__(**kwargs)

    @property
    def service(self) -> ChatApplicationService:
        return self._service

    @service.setter
    def service(self, value: ChatApplicationService) -> None:
        self._service = value

    @property
    def backend(self) -> AgentBackend:
        return self._service.backend

    @backend.setter
    def backend(self, value: AgentBackend) -> None:
        self._service = ChatApplicationService(value)

    @property
    def agent(self) -> AgentBackend:
        """Backward-compatible alias."""
        return self._service.backend

    @agent.setter
    def agent(self, value: AgentBackend) -> None:
        self._service = ChatApplicationService(value)
        
    async def on_mount(self) -> None:
        # Start the agent lifecycle worker immediately
        self.run_worker(self._connection_worker(), name="agent_lifecycle")
        
        # Push welcome screen
        self.push_screen(WelcomeScreen())

    async def _connection_worker(self) -> None:
        """Manages the agent connection lifecycle."""
        try:
            # Connect
            await self.service.initialize()
            
            # Wait until cancelled (app shutdown)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            # Disconnect in the same task
            await self.service.shutdown()

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

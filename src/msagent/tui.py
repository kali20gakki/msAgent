"""TUI interface for msagent using Textual."""

import asyncio
import time
from typing import Any

from rich.markdown import Markdown as RichMarkdown
from rich.align import Align
from rich.console import RenderableType
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Static,
    TextArea,
)
from textual.binding import Binding

from .agent import Agent
from .config import config_manager


class MessageWidget(Container):
    """Widget to display a chat message."""
    
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
    
    def __init__(self, role: str, content: str, **kwargs: Any):
        self.role = role
        self.content = content
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
        
        with Container(classes="content-container"):
            # Header with actions
            with Horizontal(classes="header-row"):
                yield Static(" ", classes="role-label")
                with Horizontal(classes="actions"):
                    yield Label("Copy", id="copy-btn", classes="action-btn")
                    yield Label("Raw", id="raw-btn", classes="action-btn")
            
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

    def update_content(self, content: str) -> None:
        """Update the message content."""
        self.content = content
        self.query_one("#content-text", CopyableTextArea).text = content
        # 同时更新 Markdown 内容，以备切换
        self.query_one("#render-md", Static).update(RichMarkdown(content))
    
    def update_content_fast(self, content: str) -> None:
        """快速更新内容（流式输出时使用）"""
        self.content = content
        self.query_one("#content-text", CopyableTextArea).text = content
    
    def finalize_content(self) -> None:
        """流式输出完成，切换到美观的 Markdown 渲染模式"""
        # 更新 Markdown 视图
        self.query_one("#render-md", Static).update(RichMarkdown(self.content))
        
        # 切换视图：隐藏文本框，显示 Markdown
        self.query_one("#content-text", CopyableTextArea).add_class("hidden")
        self.query_one("#render-md", Static).remove_class("hidden")

    def on_click(self, event: events.Click) -> None:
        """Handle click events."""
        if event.widget.id == "copy-btn":
            try:
                import pyperclip
                pyperclip.copy(self.content)
                self.app.notify("Copied to clipboard", severity="information")
            except Exception:
                self.app.copy_to_clipboard(self.content)
                self.app.notify("Copied (fallback)", severity="information")
        elif event.widget.id == "raw-btn":
            md_widget = self.query_one("#render-md", Static)
            text_widget = self.query_one("#content-text", CopyableTextArea)
            btn = event.widget
            
            if "hidden" in text_widget.classes:
                # 切换到纯文本模式（可选择）
                text_widget.remove_class("hidden")
                md_widget.add_class("hidden")
                # Label 不支持直接修改 label 属性，使用 update
                btn.update("View")
            else:
                # 切换回 Markdown 渲染模式
                text_widget.add_class("hidden")
                md_widget.remove_class("hidden")
                btn.update("Raw")


class CopyableTextArea(TextArea):
    """TextArea with copy support."""
    
    BINDINGS = [
        Binding("ctrl+c", "copy_selection", "Copy", show=False),
    ]
    
    def action_copy_selection(self) -> None:
        """Copy selected text to clipboard."""
        if self.selected_text:
            try:
                import pyperclip
                pyperclip.copy(self.selected_text)
                self.app.notify("Selection copied", severity="information")
            except Exception:
                self.app.copy_to_clipboard(self.selected_text)
                self.app.notify("Selection copied", severity="information")
        else:
            # If nothing selected, maybe quit? No, better safe than sorry.
            self.app.notify("No text selected", severity="warning")


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
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Label("✱ msAgent initialized. How can I help you?", classes="welcome-message")
        
        from .mcp_client import mcp_manager
        servers = mcp_manager.get_connected_servers()
        if servers:
            server_str = ", ".join(servers)
            yield Label(f"🔌 Connected MCP Servers: {server_str}", classes="mcp-status")

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
        self._base = "/ for commands"
        self._model_status = "model: unknown"
        self._token_status = "tokens: 0"

    def set_model_status(self, status: str) -> None:
        self._model_status = status
        self.refresh()

    def set_token_status(self, status: str) -> None:
        self._token_status = status
        self.refresh()

    def render(self) -> str:
        return f"{self._base} • {self._model_status} • {self._token_status}"


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
    
    async def add_message(self, role: str, content: str) -> MessageWidget:
        """Add a message to the chat area."""
        widget = MessageWidget(role, content)
        await self.mount(widget)
        widget.scroll_visible()
        return widget


class InputArea(Container):
    """Area for user input."""
    
    DEFAULT_CSS = """
    InputArea {
        height: auto;
        min-height: 3;
        margin: 1 0;
        border: solid #d8dee9;
        background: #2e3440;
        padding: 0;
    }
    
    InputArea:focus-within {
        border: solid #88c0d0;
    }
    
    .input-row {
        align-vertical: middle;
        height: auto;
        margin: 0;
    }
    
    .prompt-label {
        width: 3;
        padding-left: 1;
        color: #88c0d0;
        text-style: bold;
        content-align: center middle;
    }
    
    Input {
        width: 1fr;
        background: #2e3440;
        border: none;
        color: #eceff4;
        padding: 0 1;
        height: 1;
    }
    
    Input:focus {
        border: none;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="input-row"):
            yield Label("❯", classes="prompt-label")
            yield Input(placeholder='Type your message...', id="message-input")

class WelcomeScreen(Screen):
    """Full screen welcome page."""
    
    CSS = """
    WelcomeScreen {
        align: center middle;
        background: $background;
    }
    
    .welcome-container {
        width: 80%;
        height: auto;
        align: center middle;
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
        text-align: center;
        margin-bottom: 4;
        width: 100%;
    }
    
    .continue-text {
        color: $text-muted;
        text-align: left;
    }
    
    .status-text {
        color: $warning;
        text-align: center;
        margin-top: 1;
    }
    
    LoadingIndicator {
        height: 1;
        margin: 1 0;
        color: $accent;
    }
    
    .hidden {
        display: none;
    }
    """
    
    BINDINGS = [
        Binding("enter", "continue", "Continue"),
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
            yield Label("✱ Welcome to msAgent", classes="welcome-box")
            yield Static(ascii_text, classes="ascii-art")
            
            # Loading state components
            yield LoadingIndicator(id="loading")
            yield Label("Initializing agent and MCP tools...", id="status-text", classes="status-text")
            
            # Ready state component (initially hidden)
            t = Text.from_markup("Press [bold white]Enter[/bold white] to continue")
            yield Label(t, id="continue-text", classes="continue-text hidden")
            
    async def on_mount(self) -> None:
        """Start initialization when screen mounts."""
        self.run_worker(self._monitor_agent_init(), exclusive=True)
        
    async def _monitor_agent_init(self) -> None:
        """Monitor agent initialization status."""
        try:
            # Wait for agent to be initialized by the App worker
            while not self.app.agent.is_initialized and not self.app.agent.error_message:
                await asyncio.sleep(0.1)
            
            if self.app.agent.error_message:
                # Show error
                self.query_one("#loading").add_class("hidden")
                self.query_one("#status-text").update(f"❌ Error: {self.app.agent.error_message}")
            else:
                # Update UI
                self.query_one("#loading").add_class("hidden")
                self.query_one("#status-text").add_class("hidden")
                self.query_one("#continue-text").remove_class("hidden")
                
                self.is_ready = True
            
        except Exception as e:
            # Show error
            self.query_one("#loading").add_class("hidden")
            self.query_one("#status-text").update(f"❌ Error: {e}")
            
    def action_continue(self) -> None:
        if getattr(self, "is_ready", False):
            self.app.push_screen(ChatScreen())


class ChatScreen(Screen):
    """Main chat interface."""
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear Chat", show=False),
    ]
    
    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            with ChatArea(id="chat-area"):
                pass
            yield InputArea()
        yield CustomFooter()

    async def on_mount(self) -> None:
        """Called when screen is mounted."""
        # Check if agent is already initialized in app
        agent = self.app.agent
        
        # We can trigger a small welcome message in the chat
        chat_area = self.query_one("#chat-area", ChatArea)
        
        if agent.is_initialized:
            chat_area.mount(ChatWelcomeBanner())
        else:
            # If not initialized (should not happen if we init in app.on_mount, but just in case)
            await chat_area.add_message("system", agent.error_message)

        # Focus input
        self.query_one("#message-input", Input).focus()
        self._update_footer_model()
        self._update_footer_tokens()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理输入提交事件"""
        if event.input.id == "message-input":
            # 不使用 await，让 UI 立即响应
            self.send_message()
            
    def send_message(self) -> None:
        """发送消息（同步启动，异步执行）"""
        app: MSAgentApp = self.app
        if app.is_processing:
            return
            
        input_widget = self.query_one("#message-input", Input)
        message = input_widget.value.strip()
        
        if not message:
            return
            
        # Commands
        if message.lower() in ["/exit", "/quit", ":q"]:
            app.exit()
            return
        if message.lower() == "/clear":
            self.action_clear()
            input_widget.value = ""
            return
        
        # 立即清空输入框
        input_widget.value = ""
        
        # 标记正在处理
        app.is_processing = True
        
        # 使用 run_worker 在后台执行，UI 立即更新
        self._current_worker = self.run_worker(self._process_message(message), exclusive=True)
    
    async def _animate_loading(self, widget: MessageWidget, stop_event: asyncio.Event) -> None:
        """动态加载动画"""
        loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = 0
        
        while not stop_event.is_set():
            widget.update_content(f"{loading_frames[frame_idx]} Thinking...")
            frame_idx = (frame_idx + 1) % len(loading_frames)
            await asyncio.sleep(0.1)  # 100ms 更新一次
    
    async def _process_message(self, message: str) -> None:
        """后台处理消息的 worker"""
        app: MSAgentApp = self.app
        chat_area = self.query_one("#chat-area", ChatArea)
        
        try:
            # 1. 立即显示用户输入
            await chat_area.add_message("user", message)
            chat_area.scroll_end(animate=False)
            
            if not app.agent.is_initialized:
                await chat_area.add_message("system", app.agent.error_message or "Agent not initialized")
                return
            
            # 2. 创建加载消息并启动动画
            loading_widget = await chat_area.add_message("assistant", "⠋ Thinking...")
            chat_area.scroll_end(animate=False)
            
            # 启动加载动画
            stop_animation = asyncio.Event()
            animation_task = asyncio.create_task(self._animate_loading(loading_widget, stop_animation))
            
            # 3. 流式接收并实时更新
            response_text = ""
            first_chunk_received = False
            chunk_count = 0
            
            try:
                async for chunk in app.agent.chat_stream(message):
                    chunk_count += 1
                    
                    if not first_chunk_received:
                        # 停止加载动画
                        stop_animation.set()
                        await animation_task
                        
                        # 收到第一个 chunk，开始显示内容
                        first_chunk_received = True
                        response_text = chunk
                        loading_widget.update_content_fast(response_text)  # 使用快速更新
                    else:
                        # 追加内容并立即更新
                        response_text += chunk
                        loading_widget.update_content_fast(response_text)  # 使用快速更新
                    
                    # 滚动到底部（不触发全局刷新）
                    chat_area.scroll_end(animate=False)
                    
                    # 让出控制权
                    await asyncio.sleep(0)
                
                # 流式输出完成后，渲染最终的 Markdown
                if first_chunk_received:
                    loading_widget.finalize_content()
                
            except Exception as stream_error:
                # 确保停止动画
                stop_animation.set()
                if not animation_task.done():
                    await animation_task
                raise stream_error
            
            # 如果没有收到任何内容
            if not first_chunk_received:
                stop_animation.set()
                await animation_task
                loading_widget.update_content("_No response received_")
                 
        except Exception as e:
            await chat_area.add_message("system", f"❌ Error: {str(e)}")
        finally:
            app.is_processing = False
            self._update_footer_tokens()
            chat_area.scroll_end(animate=False)

    def _update_footer_tokens(self) -> None:
        usage = getattr(self.app.agent.llm_client, "last_usage", None)
        total_tokens: int | None = None
        if isinstance(usage, dict):
            val = usage.get("total_tokens")
            if isinstance(val, int):
                total_tokens = val
        token_text = (
            f"tokens: {self._format_token_count(total_tokens)}"
            if total_tokens is not None
            else "tokens: N/A"
        )
        self.query_one(CustomFooter).set_token_status(token_text)

    def _update_footer_model(self) -> None:
        llm_cfg = self.app.agent.config.llm
        provider = (llm_cfg.provider or "unknown").strip()
        model = (llm_cfg.model or "unknown").strip()
        self.query_one(CustomFooter).set_model_status(f"model: {provider}/{model}")

    def _format_token_count(self, count: int) -> str:
        if count < 1_000:
            return str(count)
        if count < 1_000_000:
            value = count / 1_000
            return f"{value:.1f}K" if value < 10 else f"{value:.0f}K"
        value = count / 1_000_000
        return f"{value:.1f}M" if value < 10 else f"{value:.0f}M"

    def action_clear(self) -> None:
        self.app.agent.clear_history()
        chat_area = self.query_one("#chat-area", ChatArea)
        chat_area.remove_children()
        chat_area.mount(ChatWelcomeBanner())
        chat_area.run_worker(self._add_system_message(chat_area, "Chat history cleared."))

    async def _add_system_message(self, chat_area: ChatArea, message: str) -> None:
        await chat_area.add_message("system", message)


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
        padding: 1 2;
    }
    
    #chat-area {
        height: 1fr;
        margin-bottom: 1;
        background: $background;
    }
    """
    
    def __init__(self, **kwargs: Any):
        self.agent = Agent()
        self.is_processing = False
        super().__init__(**kwargs)
        
    async def on_mount(self) -> None:
        # Start the agent lifecycle worker immediately
        self.run_worker(self._connection_worker(), name="agent_lifecycle")
        
        # Push welcome screen
        self.push_screen(WelcomeScreen())

    async def _connection_worker(self) -> None:
        """Manages the agent connection lifecycle."""
        try:
            # Connect
            await self.agent.initialize()
            
            # Wait until cancelled (app shutdown)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            # Disconnect in the same task
            if self.agent.is_initialized:
                await self.agent.shutdown()

def run_tui() -> None:
    """Run the TUI application."""
    app = MSAgentApp()
    app.run()

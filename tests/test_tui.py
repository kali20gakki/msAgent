"""Tests for the prompt-based TUI module."""

from __future__ import annotations

import pytest
from rich.console import Console

import msagent.tui as tui_module
from msagent.tui import (
    ChatWelcomeBanner,
    MSAgentApp,
    Renderer,
    ToolBrowser,
    run_tui,
)


def test_chat_welcome_banner_compose_shows_model_mcp_and_skills() -> None:
    widgets = list(
        ChatWelcomeBanner(
            model_label="openai:gpt-5-mini",
            mcp_servers=["filesystem"],
            loaded_skills=["code-review", "devops"],
        ).compose()
    )

    assert len(widgets) == 4
    assert (
        widgets[0].plain
        == "msAgent 是面向 Ascend NPU Profiling 的性能分析助手，基于真实数据定位瓶颈、解释根因并给出可执行优化方案。"
    )
    assert widgets[1].plain == "Model: openai:gpt-5-mini"
    assert widgets[2].plain == "MCP: filesystem"
    assert widgets[3].plain == "Skills: code-review, devops"


def test_chat_welcome_banner_compose_shows_none_for_empty_status() -> None:
    widgets = list(ChatWelcomeBanner().compose())

    assert len(widgets) == 4
    assert widgets[1].plain == "Model: unknown"
    assert widgets[2].plain == "MCP: none"
    assert widgets[3].plain == "Skills: none"


def test_welcome_banner_uses_claude_style_title() -> None:
    banner = ChatWelcomeBanner()
    console = Console(record=True, width=100, theme=tui_module._RICH_THEME)

    console.print(banner.render())
    output = console.export_text()

    assert "Welcome to msAgent" in output


def test_run_tui_creates_app_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"run": False}

    class FakeApp:
        def run(self) -> None:
            called["run"] = True

    monkeypatch.setattr(tui_module, "MSAgentApp", FakeApp)
    run_tui()
    assert called["run"] is True


def test_msagent_app_uses_in_memory_history_by_default() -> None:
    app = MSAgentApp()
    assert app.history_file is None


def test_format_duration_text() -> None:
    assert tui_module._format_duration_text(None) == ""
    assert tui_module._format_duration_text(0.42) == "took 420ms"
    assert tui_module._format_duration_text(3.2) == "took 3.2s"
    assert tui_module._format_duration_text(18.7) == "took 19s"
    assert tui_module._format_duration_text(65.2) == "took 1m 05s"


def test_renderer_build_assistant_renderable_uses_streaming_prefix() -> None:
    console = Console(record=True, width=100, theme=tui_module._RICH_THEME)
    renderer = Renderer(console)

    console.print(renderer.build_assistant_renderable("**hello**"))
    output = console.export_text()

    assert "hello" in output
    assert "hello" in output


def test_renderer_build_stream_placeholder_contains_thinking_text() -> None:
    console = Console(record=True, width=100, theme=tui_module._RICH_THEME)
    renderer = Renderer(console)

    console.print(renderer.build_stream_placeholder())
    output = console.export_text()

    assert "Thinking..." in output


def test_renderer_render_help_accepts_command_mapping() -> None:
    console = Console(record=True, width=100, theme=tui_module._RICH_THEME)
    renderer = Renderer(console)

    def cmd_help() -> None:
        """Show help information."""

    renderer.render_help({"/help": cmd_help})
    output = console.export_text()

    assert "/help" in output
    assert "Show help information." in output


def test_renderer_build_tool_call_has_space_between_icon_and_name() -> None:
    console = Console(record=True, width=100, theme=tui_module._RICH_THEME)
    renderer = Renderer(console)

    console.print(renderer.build_tool_call("ls", {"path": "/"}))
    output = console.export_text()

    assert f"{tui_module.TOOL_PREFIX}   ls" in output
    assert "path : /" in output


def test_tool_browser_format_tool_list_marks_selection_and_expands_parameters() -> None:
    browser = ToolBrowser(prompt_style="> ")
    tool = {
        "function": {
            "name": "read_file",
            "description": "Read a file from the current workspace.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Path to read"},
                    "encoding": {"type": "string", "description": "Optional encoding"},
                },
            },
        }
    }

    lines = browser._format_tool_list(
        [tool],
        selected_index=0,
        expanded_indices={0},
        scroll_offset=0,
        window_size=10,
    )

    assert lines[0] == ("class:selected", "> read_file")
    assert ("class:auto-suggestion", "    Parameters:") in lines
    assert any("path (string, required)" in text for _, text in lines)


@pytest.mark.asyncio
async def test_interactive_prompt_erases_submitted_input(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePromptSession:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        async def prompt_async(self, *args, **kwargs):
            self.kwargs = kwargs
            return "hello"

    class FakeService:
        def find_commands(self, _query: str, limit: int = 64):
            return [("/help", "help")][:limit]

    prompt = tui_module.InteractivePrompt.__new__(tui_module.InteractivePrompt)
    prompt.service = FakeService()
    prompt.prompt_text = "> "
    prompt.prompt_session = FakePromptSession()

    content, is_command = await tui_module.InteractivePrompt.get_input(prompt)

    assert content == "hello"
    assert is_command is False
    assert prompt.prompt_session.kwargs == {}


def test_interactive_prompt_configures_session_to_erase_when_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakePromptSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeFileHistory:
        def __init__(self, _path: str) -> None:
            pass

    class FakeInMemoryHistory:
        pass

    class FakeAutoSuggestFromHistory:
        pass

    class FakeCompleteStyle:
        COLUMN = "column"

    prompt = tui_module.InteractivePrompt.__new__(tui_module.InteractivePrompt)
    prompt.history_file = None
    prompt.prompt_text = "> "
    prompt._build_completer = lambda: "completer"
    prompt._create_key_bindings = lambda: "keys"
    prompt._create_style = lambda: "style"
    prompt._get_placeholder = lambda: "placeholder"

    monkeypatch.setattr(tui_module, "PromptSession", FakePromptSession)
    monkeypatch.setattr(tui_module, "FileHistory", FakeFileHistory)
    monkeypatch.setattr(tui_module, "InMemoryHistory", FakeInMemoryHistory)
    monkeypatch.setattr(tui_module, "AutoSuggestFromHistory", FakeAutoSuggestFromHistory)
    monkeypatch.setattr(tui_module, "CompleteStyle", FakeCompleteStyle)

    tui_module.InteractivePrompt._create_session(prompt)

    assert captured["erase_when_done"] is True


def test_interactive_prompt_placeholder_shows_command_and_file_hints() -> None:
    prompt = tui_module.InteractivePrompt.__new__(tui_module.InteractivePrompt)

    placeholder = tui_module.InteractivePrompt._get_placeholder(prompt)

    assert list(placeholder) == [("class:placeholder", "尽管问 msAgent，/ 命令，@ 关联文件")]


@pytest.mark.asyncio
async def test_run_async_initializes_and_shuts_down_on_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeService:
        def __init__(self) -> None:
            self.initialize_called = False
            self.shutdown_called = False

        async def initialize(self) -> bool:
            self.initialize_called = True
            return True

        async def shutdown(self) -> None:
            self.shutdown_called = True

        def get_status(self):
            return tui_module.AgentStatus(
                is_initialized=True,
                error_message="",
                session_number=1,
                provider="openai",
                model="gpt",
                backend_mode="filesystem",
                connected_servers=(),
                loaded_skills=(),
                usage=None,
            )

    class FakePrompt:
        hotkeys: dict[str, str] = {}

        def __init__(self, service, *, history_file=None, prompt_text=tui_module.PROMPT_STYLE):
            self.service = service
            self.history_file = history_file
            self.prompt_text = prompt_text

        async def get_input(self):
            raise EOFError

    service = FakeService()
    app = MSAgentApp(service=service, console=Console(record=True, width=100, theme=tui_module._RICH_THEME))
    monkeypatch.setattr(tui_module, "InteractivePrompt", FakePrompt)

    await app.run_async()

    assert service.initialize_called is True
    assert service.shutdown_called is True

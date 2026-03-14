from types import SimpleNamespace
from pathlib import Path

from rich.console import Console

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.cli.theme import theme
from msagent.cli.ui import renderer as renderer_module
from msagent.configs import ApprovalMode


class _CaptureConsole:
    def __init__(self) -> None:
        self.console = Console(record=True, width=120, theme=theme.rich_theme)

    def print(self, *args, **kwargs) -> None:
        self.console.print(*args, **kwargs)


def test_show_welcome_uses_legacy_banner(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)
    monkeypatch.setattr(initializer, "cached_mcp_server_names", ["msprof-mcp"])
    monkeypatch.setattr(
        initializer,
        "cached_agent_skills",
        [SimpleNamespace(name="profiling-skill")],
    )

    context = Context(
        agent="general",
        model="default",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        recursion_limit=80,
    )

    renderer_module.Renderer.show_welcome(context)
    output = capture.console.export_text()

    assert "Welcome to msAgent" in output
    assert "面向 Ascend NPU Profiling 的性能分析助手" in output
    assert "Model: default" in output
    assert "MCP: msprof-mcp" in output
    assert "Skills: profiling-skill" in output


def test_show_welcome_prefers_resolved_model_display(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)
    monkeypatch.setattr(initializer, "cached_mcp_server_names", ["msprof-mcp"])
    monkeypatch.setattr(
        initializer,
        "cached_agent_skills",
        [SimpleNamespace(name="profiling-skill")],
    )

    context = Context(
        agent="general",
        model="default",
        model_display="deepseek-chat (openai)",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        recursion_limit=80,
    )

    renderer_module.Renderer.show_welcome(context)
    output = capture.console.export_text()

    assert "Model: deepseek-chat (openai)" in output

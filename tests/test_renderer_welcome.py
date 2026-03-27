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


def test_format_tool_call_uses_dot_prefix() -> None:
    text = renderer_module.Renderer._format_tool_call(
        {"name": "run_command", "args": {"command": "ls"}}
    )

    assert text.plain.startswith("● Use tool run_command\n  command: ls\n")
    assert "⚙" not in text.plain
    assert [span.style for span in text.spans[:3]] == [
        "indicator",
        "accent",
        "primary",
    ]
    assert "muted" in [span.style for span in text.spans]


def test_format_tool_call_wraps_long_args_into_compact_second_line() -> None:
    text = renderer_module.Renderer._format_tool_call(
        {
            "name": "run_command",
            "args": {
                "command": "python -c \"print('this is a very long command that should not stay inline')\"",
                "cwd": "/tmp/project",
            },
        }
    )

    assert text.plain.startswith("● Use tool run_command\n")
    assert "\n  command: " in text.plain
    assert "\n  cwd: /tmp/project" in text.plain


def test_format_tool_call_keeps_full_long_args() -> None:
    text = renderer_module.Renderer._format_tool_call(
        {"name": "read_file", "args": {"range": "a" * 80}}
    )

    assert f"range: {'a' * 80}" in text.plain
    assert "original length" not in text.plain


def test_strip_frontmatter_fences_removes_yaml_markers() -> None:
    content = (
        "---\n"
        "name: cluster-fast-slow-rank-detector\n"
        "description: profiler skill\n"
        "---\n"
        "\n"
        "# Body\n"
    )

    stripped = renderer_module.Renderer._strip_frontmatter_fences(content)

    assert stripped.startswith("name: cluster-fast-slow-rank-detector")
    assert "\n---\n" not in stripped
    assert "# Body" in stripped


def test_strip_frontmatter_fences_removes_leading_opening_marker_without_closing_fence() -> None:
    content = (
        "\n"
        "---\n"
        "name: cluster-fast-slow-rank-detector\n"
        "description: profiler skill\n"
        "技能正文\n"
    )

    stripped = renderer_module.Renderer._strip_frontmatter_fences(content)

    assert not stripped.startswith("---")
    assert stripped.startswith("name: cluster-fast-slow-rank-detector")
    assert "技能正文" in stripped

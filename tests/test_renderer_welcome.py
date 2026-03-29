from types import SimpleNamespace
from pathlib import Path

from langchain_core.messages import ToolMessage
from rich.console import Console

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.cli.theme import theme
from msagent.cli.ui import renderer as renderer_module
from msagent.configs import ApprovalMode
from msagent.utils.version import get_version


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
        agent_description="General-purpose assistant",
        model="default",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        recursion_limit=80,
    )

    renderer_module.Renderer.show_welcome(context)
    output = capture.console.export_text()

    assert f"Welcome to msAgent v{get_version()}" in output
    assert "Agent: general - General-purpose assistant" in output
    assert "MindStudio 一站式调试调优 Agent，支持性能、精度、算子等场景问题定位" in output
    assert "Model: default" in output
    assert "MCP (1)" in output
    assert "msprof-mcp" in output
    assert "Skills (1)" in output
    assert "profiling-skill" in output


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
        agent_description="General-purpose assistant",
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


def test_format_tool_call_marks_subagent_origin() -> None:
    text = renderer_module.Renderer._format_tool_call(
        {"name": "run_command", "args": {"command": "ls"}},
        indent_level=1,
    )

    assert text.plain.startswith("  ● [Subagent] Use tool run_command\n")


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


def test_format_tool_call_truncates_long_args_for_compact_preview() -> None:
    text = renderer_module.Renderer._format_tool_call(
        {"name": "read_file", "args": {"range": "a" * 80}}
    )

    assert "range: " in text.plain
    assert "aaaaaaaa" in text.plain
    assert "(80 chars)" in text.plain
    assert f"range: {'a' * 80}" not in text.plain


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


def test_strip_frontmatter_fences_removes_leading_opening_marker_without_closing_fence() -> (
    None
):
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


def test_truncate_tool_content_for_display_reports_original_length() -> None:
    original = "x" * 50
    old_limit = renderer_module.Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS
    renderer_module.Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS = 20
    try:
        truncated = renderer_module.Renderer._truncate_tool_content_for_display(
            original
        )
    finally:
        renderer_module.Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS = old_limit

    assert truncated.startswith("x" * 20)
    assert "truncated for display" in truncated
    assert "original length: 50 chars" in truncated


def test_build_tool_message_display_supports_expanded_view() -> None:
    message = ToolMessage(
        name="run_command",
        content="line 1\nline 2\nline 3",
        short_content="line 1\n... (truncated, original length: 20)",
        tool_call_id="call-1",
    )

    collapsed = renderer_module.Renderer._build_tool_message_display(message)
    expanded = renderer_module.Renderer._build_tool_message_display(
        message, expanded=True
    )

    assert collapsed is not None
    assert expanded is not None
    assert collapsed.can_expand is True
    assert collapsed.display_content == "line 1\n... (truncated, original length: 20)"
    assert expanded.display_content == "line 1\nline 2\nline 3"


def test_render_tool_message_truncates_long_content(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)

    old_limit = renderer_module.Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS
    renderer_module.Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS = 30
    try:
        message = ToolMessage(
            name="fetch_skills",
            content="abcdefghijklmnopqrstuvwxyz0123456789",
            tool_call_id="call-1",
        )
        renderer_module.Renderer.render_tool_message(message)
    finally:
        renderer_module.Renderer.TOOL_MESSAGE_MAX_DISPLAY_CHARS = old_limit

    output = capture.console.export_text()
    assert "truncated for display" in output
    assert "original length: 36 chars" in output
    assert "abcdefghijklmnopqrstuvwxyz0123456789" not in output


def test_render_tool_message_shows_toggle_hint_for_expandable_output(
    monkeypatch,
) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(renderer_module, "console", capture)

    message = ToolMessage(
        name="run_command",
        content="full output\nwith more lines",
        short_content="full output",
        tool_call_id="call-1",
    )
    renderer_module.Renderer.render_tool_message(message)

    output = capture.console.export_text()
    assert "press Ctrl+O /tool-output to browse full tool outputs" in output

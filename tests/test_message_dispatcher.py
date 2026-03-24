from __future__ import annotations

import logging
from types import MethodType, SimpleNamespace
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from openai import APIConnectionError
from rich.console import Console

from msagent.cli.dispatchers import messages as message_module
from msagent.cli.dispatchers.messages import MessageDispatcher
from msagent.cli.theme import theme
from msagent.configs import ApprovalMode, LLMConfig, LLMProvider
from msagent.core.constants import CONFIG_LOG_DIR
from msagent.core.logging import configure_logging


def _build_session(tmp_path: Path) -> SimpleNamespace:
    session = SimpleNamespace(
        prefilled_reference_mapping={},
        current_stream_task=None,
        context=SimpleNamespace(
            approval_mode=ApprovalMode.ACTIVE,
            working_dir=tmp_path,
            thread_id="thread-1",
            recursion_limit=80,
            tool_output_max_tokens=None,
            stream_output=False,
            agent="msagent",
            model="default",
            current_input_tokens=None,
            current_output_tokens=None,
            context_window=128000,
        ),
        graph=SimpleNamespace(),
        prompt=SimpleNamespace(reset_interrupt_state=lambda: None),
        renderer=SimpleNamespace(
            render_assistant_message=lambda *args, **kwargs: None,
            render_tool_call=lambda *args, **kwargs: None,
            render_tool_message=lambda *args, **kwargs: None,
        ),
    )

    def update_context(**kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(session.context, key, value)

    session.update_context = update_context
    return session


def _patch_dispatch_to_raise_connection_error(
    dispatcher: MessageDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dispatcher.message_builder,
        "build",
        lambda content: (content, {}),
    )

    async def fake_load_user_memory(_working_dir: Path) -> str:
        return ""

    async def fake_load_llm_config(_model: str, _working_dir: Path) -> LLMConfig:
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="deepseek-chat",
            alias="default",
            base_url="https://api.deepseek.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

    async def fake_invoke_without_stream(self, *_args, **_kwargs) -> None:
        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/v1/chat/completions",
        )
        try:
            raise httpx.ConnectError("all connection attempts failed", request=request)
        except httpx.ConnectError as err:
            raise APIConnectionError(request=request) from err

    monkeypatch.setattr(
        message_module.initializer,
        "load_user_memory",
        fake_load_user_memory,
    )
    monkeypatch.setattr(
        message_module.initializer,
        "load_llm_config",
        fake_load_llm_config,
    )
    monkeypatch.setattr(
        dispatcher,
        "_invoke_without_stream",
        MethodType(fake_invoke_without_stream, dispatcher),
    )


@pytest.mark.asyncio
async def test_dispatch_logs_detailed_connection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)
    printed_errors: list[str] = []
    _patch_dispatch_to_raise_connection_error(dispatcher, monkeypatch)
    monkeypatch.setattr(message_module.console, "print_error", printed_errors.append)
    monkeypatch.setattr(message_module.console, "print", lambda *args, **kwargs: None)

    with caplog.at_level(logging.ERROR, logger=message_module.logger.name):
        await dispatcher.dispatch("hello")

    assert printed_errors == [
        "Error processing message: Connection error. Cause: ConnectError: all connection attempts failed"
    ]
    assert "Message processing error [thread_id=thread-1" in caplog.text
    assert (
        "console_error=Connection error. Cause: ConnectError: all connection attempts failed"
        in caplog.text
    )
    assert "exception_type=APIConnectionError" in caplog.text
    assert "exception_message=Connection error." in caplog.text
    assert "exception_repr=APIConnectionError('Connection error.')" in caplog.text
    assert "provider=openai" in caplog.text
    assert "resolved_model=deepseek-chat" in caplog.text
    assert "base_url=https://api.deepseek.com/v1" in caplog.text
    assert "request=POST https://api.deepseek.com/v1/chat/completions" in caplog.text
    assert (
        "exception_chain=APIConnectionError: Connection error. <- "
        "ConnectError: all connection attempts failed"
    ) in caplog.text


@pytest.mark.asyncio
async def test_dispatch_writes_detailed_processing_errors_to_verbose_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        configure_logging(show_logs=True, working_dir=tmp_path)

        session = _build_session(tmp_path)
        dispatcher = MessageDispatcher(session)
        _patch_dispatch_to_raise_connection_error(dispatcher, monkeypatch)
        monkeypatch.setattr(message_module.console, "print_error", lambda *args: None)
        monkeypatch.setattr(message_module.console, "print", lambda *args, **kwargs: None)

        await dispatcher.dispatch("hello")

        for handler in root_logger.handlers:
            flush = getattr(handler, "flush", None)
            if callable(flush):
                flush()

        log_path = tmp_path / CONFIG_LOG_DIR / "app.log"
        assert log_path.exists()

        log_text = log_path.read_text(encoding="utf-8")
        assert "Message processing error [thread_id=thread-1" in log_text
        assert (
            "console_error=Connection error. Cause: ConnectError: "
            "all connection attempts failed" in log_text
        )
        assert "exception_type=APIConnectionError" in log_text
        assert "exception_message=Connection error." in log_text
        assert (
            "exception_chain=APIConnectionError: Connection error. <- "
            "ConnectError: all connection attempts failed" in log_text
        )
        assert "Traceback (most recent call last):" in log_text
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            close = getattr(handler, "close", None)
            if callable(close):
                close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_extract_tool_call_names_handles_chunks_and_raw_payloads(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "run_command",
                "args": '{"command":"ls","cwd":"/tmp"}',
                "id": "call-1",
                "index": 0,
            }
        ],
        additional_kwargs={
            "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "run_command"}},
            ]
        },
    )

    assert dispatcher._extract_tool_call_names(chunk) == ["run_command", "read_file"]
    previews = dispatcher._extract_tool_call_previews(chunk)
    assert len(previews) == 2
    assert message_module.ToolActivityCall(
        name="read_file",
        args={},
        call_id=None,
    ) in previews
    assert message_module.ToolActivityCall(
        name="run_command",
        args={"command": "ls", "cwd": "/tmp"},
        call_id="call-1",
    ) in previews


def test_extract_tool_call_previews_merges_same_tool_with_conflicting_source_ids(
    tmp_path: Path,
) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    chunk = AIMessageChunk(
        content="",
        tool_calls=[
            {
                "name": "msprof-mcp__msprof_analyze_advisor",
                "args": {"profiler_data_dir": "/tmp/profile"},
                "id": "normalized-call-1",
            }
        ],
        tool_call_chunks=[
            {
                "name": "msprof-mcp__msprof_analyze_advisor",
                "args": '{"profiler_data_dir":"/tmp/profile","mode":"all"',
                "id": "chunk-call-1",
                "index": 0,
            }
        ],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "raw-call-1",
                    "function": {
                        "name": "msprof-mcp__msprof_analyze_advisor",
                        "arguments": (
                            '{"profiler_data_dir":"/tmp/profile","mode":"all"}'
                        ),
                    },
                }
            ]
        },
    )

    previews = dispatcher._extract_tool_call_previews(chunk)

    assert len(previews) == 1
    assert previews[0].name == "msprof-mcp__msprof_analyze_advisor"
    assert previews[0].args == {
        "profiler_data_dir": "/tmp/profile",
        "mode": "all",
    }
    assert previews[0].call_id in {"raw-call-1", "chunk-call-1"}


def test_extract_tool_call_previews_merges_progressively_longer_string_args(
    tmp_path: Path,
) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "msprof-mcp__msprof_analyze_advisor",
                "args": {"profiler_data_dir": "/Users/weizhang/Down"},
                "id": "call-1",
            },
            {
                "name": "msprof-mcp__msprof_analyze_advisor",
                "args": {
                    "profiler_data_dir": (
                        "/Users/weizhang/Downloads/kv_cache_type_page_seqlen_1024"
                    ),
                    "mode": "all",
                },
                "id": "call-2",
            },
        ],
    )

    previews = dispatcher._extract_tool_call_previews(message)

    assert len(previews) == 1
    assert previews[0].name == "msprof-mcp__msprof_analyze_advisor"
    assert previews[0].args == {
        "profiler_data_dir": "/Users/weizhang/Downloads/kv_cache_type_page_seqlen_1024",
        "mode": "all",
    }


def test_extract_tool_call_previews_keeps_distinct_same_name_calls_with_conflicting_args(
    tmp_path: Path,
) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "run_command", "args": {"command": "ls"}, "id": "call-1"},
            {"name": "run_command", "args": {"command": "pwd"}, "id": "call-2"},
        ],
    )

    assert dispatcher._extract_tool_call_previews(message) == [
        message_module.ToolActivityCall(
            name="run_command",
            args={"command": "ls"},
            call_id="call-1",
        ),
        message_module.ToolActivityCall(
            name="run_command",
            args={"command": "pwd"},
            call_id="call-2",
        ),
    ]


def test_merge_tool_activity_calls_keeps_args_visible_across_chunks(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    existing = [
        message_module.ToolActivityCall(
            name="msprof-mcp__msprof_analyze_advisor",
            args={"profiler_data_dir": "/tmp/profile", "mode": ""},
            call_id="call-2",
        )
    ]
    incoming = [
        message_module.ToolActivityCall(
            name="msprof-mcp__msprof_analyze_advisor",
            args={"mode": "all"},
            call_id="call-2",
        )
    ]

    merged = dispatcher._merge_tool_activity_calls(existing, incoming)

    assert merged == [
        message_module.ToolActivityCall(
            name="msprof-mcp__msprof_analyze_advisor",
            args={"profiler_data_dir": "/tmp/profile", "mode": "all"},
            call_id="call-2",
        )
    ]


def test_set_tool_activity_dedupes_same_call_across_namespaces(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)
    active_tools: dict[tuple, list[message_module.ToolActivityCall]] = {
        ("parent",): [
            message_module.ToolActivityCall(
                name="msprof-mcp__msprof_analyze_advisor",
                args={"profiler_data_dir": "/tmp/profile"},
                call_id="call-2",
            )
        ]
    }
    thinking_previews: dict[tuple, list[str]] = {}

    dispatcher._set_tool_activity(
        None,
        active_tools,
        thinking_previews,
        ("parent", "child"),
        [
            message_module.ToolActivityCall(
                name="msprof-mcp__msprof_analyze_advisor",
                args={"mode": "all"},
                call_id="call-2",
            )
        ],
    )

    assert ("parent",) not in active_tools
    assert active_tools == {
        ("parent", "child"): [
            message_module.ToolActivityCall(
                name="msprof-mcp__msprof_analyze_advisor",
                args={"profiler_data_dir": "/tmp/profile", "mode": "all"},
                call_id="call-2",
            )
        ]
    }


def test_refresh_activity_live_defers_terminal_flush_to_live_auto_refresh(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    captured: list[tuple[object, bool]] = []

    class _FakeLive:
        def update(self, renderable, *, refresh: bool = False) -> None:
            captured.append((renderable, refresh))

    dispatcher._refresh_activity_live(
        _FakeLive(),
        {
            (): [
                message_module.ToolActivityCall(
                    name="run_command",
                    args={"command": "ls"},
                    call_id="call-1",
                )
            ]
        },
        {},
    )

    assert len(captured) == 1
    assert captured[0][1] is False


def test_clear_tool_activity_can_force_flush_before_static_render(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    captured: list[tuple[object, bool]] = []

    class _FakeLive:
        def update(self, renderable, *, refresh: bool = False) -> None:
            captured.append((renderable, refresh))

    active_tools = {
        (): [
            message_module.ToolActivityCall(
                name="msprof-mcp__msprof_analyze_advisor",
                args={"profiler_data_dir": "/tmp/profile", "mode": "all"},
                call_id="call-1",
            )
        ]
    }

    dispatcher._clear_tool_activity(
        _FakeLive(),
        active_tools,
        {},
        (),
        refresh=True,
    )

    assert active_tools == {}
    assert len(captured) == 1
    assert captured[0][1] is True


def test_extract_tool_args_repairs_partial_json_strings(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    args = dispatcher._extract_tool_args(
        {
            "name": "get_skill",
            "args": '{"name":"cluster-fast-slow-rank-',
            "id": "call-3",
        }
    )

    assert args == {"name": "cluster-fast-slow-rank-"}


def test_extract_tool_call_names_handles_final_ai_messages(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "run_command", "args": {"command": "ls"}, "id": "call-1"},
            {"name": "read_file", "args": {"file_path": "README.md"}, "id": "call-2"},
        ],
    )

    assert dispatcher._extract_tool_call_names(message) == [
        "run_command",
        "read_file",
    ]
    assert dispatcher._summarize_tool_names(["run_command", "read_file"]) == (
        "run_command +1"
    )
    label = dispatcher._build_tool_activity_label(
        message_module.ToolActivityCall(name="run_command", args={})
    )
    assert label.plain == "Use tool run_command"
    assert [span.style for span in label.spans] == ["accent", "primary"]


def test_render_assistant_with_deferred_tools_hides_header_until_result(
    tmp_path: Path,
) -> None:
    session = _build_session(tmp_path)
    rendered: list[tuple[str, Any]] = []
    session.renderer = SimpleNamespace(
        render_assistant_message=lambda message, indent_level=0, show_tool_calls=True: rendered.append(
            ("assistant", indent_level, show_tool_calls, message)
        ),
        render_tool_call=lambda tool_call, indent_level=0, duration=None: rendered.append(
            ("tool_call", indent_level, tool_call, duration)
        ),
        render_tool_message=lambda message, indent_level=0: rendered.append(
            ("tool_message", indent_level, message)
        ),
    )
    dispatcher = MessageDispatcher(session)

    message = AIMessage(
        content="planning",
        tool_calls=[
            {"name": "run_command", "args": {"command": "ls"}, "id": "call-1"}
        ],
    )

    dispatcher._render_assistant_with_deferred_tools(message, indent_level=1)

    assert rendered == [("assistant", 1, False, message)]
    assert "call-1" in dispatcher._pending_tool_headers
    pending = dispatcher._pending_tool_headers["call-1"]
    assert pending[0] == {
        "name": "run_command",
        "args": {"command": "ls"},
        "id": "call-1",
        "type": "tool_call",
    }
    assert pending[1] == 1
    assert isinstance(pending[2], float)  # start_time


def test_render_pending_tool_header_uses_deferred_header_before_result(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    rendered: list[tuple[str, Any]] = []
    session.renderer = SimpleNamespace(
        render_assistant_message=lambda *args, **kwargs: None,
        render_tool_call=lambda tool_call, indent_level=0, duration=None: rendered.append(
            ("tool_call", indent_level, tool_call, duration)
        ),
        render_tool_message=lambda message, indent_level=0: rendered.append(
            ("tool_message", indent_level, message)
        ),
    )
    dispatcher = MessageDispatcher(session)
    dispatcher._pending_tool_headers["call-1"] = (
        {
            "name": "run_command",
            "args": {"command": "ls"},
            "id": "call-1",
            "type": "tool_call",
        },
        2,
        1234567890.0,  # start_time
    )
    tool_message = message_module.ToolMessage(
        content="done",
        tool_call_id="call-1",
        name="run_command",
    )

    dispatcher._render_pending_tool_header(tool_message, indent_level=0)
    session.renderer.render_tool_message(tool_message, indent_level=2)

    assert len(rendered) == 2
    assert rendered[0][0] == "tool_call"
    assert rendered[0][1] == 2
    assert rendered[0][2] == {
        "name": "run_command",
        "args": {"command": "ls"},
        "id": "call-1",
        "type": "tool_call",
    }
    assert isinstance(rendered[0][3], float)  # duration
    assert rendered[1] == ("tool_message", 2, tool_message)
    assert dispatcher._pending_tool_headers == {}


def test_merge_chunks_preserves_usage_metadata() -> None:
    merged = MessageDispatcher._merge_chunks(
        [
            AIMessageChunk(content="Hello"),
            AIMessageChunk(
                content=" world",
                usage_metadata={
                    "input_tokens": 2048,
                    "output_tokens": 256,
                    "total_tokens": 2304,
                },
            ),
        ]
    )

    assert merged.content == "Hello world"
    assert merged.usage_metadata == {
        "input_tokens": 2048,
        "output_tokens": 256,
        "total_tokens": 2304,
    }


@pytest.mark.asyncio
async def test_update_token_tracking_falls_back_to_ai_message_usage_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)

    async def fake_check_auto_compression() -> None:
        return None

    monkeypatch.setattr(
        dispatcher,
        "_check_auto_compression",
        fake_check_auto_compression,
    )

    await dispatcher._update_token_tracking(
        {
            "messages": [
                AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 4096,
                        "output_tokens": 512,
                        "total_tokens": 4608,
                    },
                )
            ]
        }
    )

    assert session.context.current_input_tokens == 4096
    assert session.context.current_output_tokens == 512


@pytest.mark.asyncio
async def test_finalize_streaming_updates_context_from_usage_only_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session(tmp_path)
    rendered: list[AIMessage] = []
    session.renderer = SimpleNamespace(
        render_assistant_message=lambda message, **kwargs: rendered.append(message),
        render_tool_call=lambda *args, **kwargs: None,
        render_tool_message=lambda *args, **kwargs: None,
    )
    dispatcher = MessageDispatcher(session)

    async def fake_check_auto_compression() -> None:
        return None

    monkeypatch.setattr(
        dispatcher,
        "_check_auto_compression",
        fake_check_auto_compression,
    )

    streaming_states = {
        (): {
            "active": True,
            "message_id": "msg-1",
            "preview_lines": ["Hello world"],
            "chunks": [
                AIMessageChunk(content="Hello"),
                AIMessageChunk(content=" world"),
                AIMessageChunk(
                    content="",
                    usage_metadata={
                        "input_tokens": 2048,
                        "output_tokens": 256,
                        "total_tokens": 2304,
                    },
                ),
            ],
        }
    }

    await dispatcher._finalize_streaming(
        (),
        streaming_states,
        None,
        set(),
        {},
        {},
    )

    assert session.context.current_input_tokens == 2048
    assert session.context.current_output_tokens == 256
    assert len(rendered) == 1
    assert rendered[0].usage_metadata == {
        "input_tokens": 2048,
        "output_tokens": 256,
        "total_tokens": 2304,
    }


def test_build_activity_renderable_keeps_tool_line_separate(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    dispatcher = MessageDispatcher(session)
    renderable = dispatcher._build_activity_renderable(
        {
            ("subagent",): [
                message_module.ToolActivityCall(
                    name="run_command",
                    args={"command": "ls", "cwd": "/tmp"},
                )
            ]
        },
        {(): ["preview line"]},
    )

    capture = Console(record=True, width=120, force_terminal=True, theme=theme.rich_theme)
    capture.print(renderable)
    output = capture.export_text()

    assert "run_command" in output
    assert "Use tool" in output
    assert "\n    command: ls" in output
    assert "\n    cwd: /tmp" in output
    assert "Thinking..." in output
    assert "preview line" in output


def test_tool_activity_indicator_blinks_dot_and_moves_sweep() -> None:
    indicator = message_module.ToolActivityIndicator(
        message_module.MessageDispatcher._build_tool_activity_label(
            message_module.ToolActivityCall(name="run_command", args={})
        )
    )

    first = indicator.render(0.0)
    second = indicator.render(0.32)
    third = indicator.render(0.64)

    # Check format includes elapsed time (Claude Code style)
    assert "● Use tool run_command" in first.plain
    assert "(0.0s)" in first.plain
    assert "● Use tool run_command" in second.plain
    assert "(0.3s)" in second.plain
    assert "● Use tool run_command" in third.plain
    assert "(0.6s)" in third.plain
    assert second.spans[0].style != third.spans[0].style
    sweep_second = next(span for span in second.spans if span.style == "secondary")
    sweep_third = next(span for span in third.spans if span.style == "secondary")
    assert sweep_second.start < sweep_third.start

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from msagent.cli.handlers import threads as threads_module
from msagent.configs import ApprovalMode


def _build_session(tmp_path: Path):
    session = SimpleNamespace()
    session.context = SimpleNamespace(
        agent="msagent",
        working_dir=tmp_path,
        approval_mode=ApprovalMode.ACTIVE,
        bash_mode=False,
        thread_id="current-thread",
        current_input_tokens=None,
        current_output_tokens=None,
    )
    session.renderer = SimpleNamespace(render_message=lambda message: rendered.append(message))
    session.message_dispatcher = SimpleNamespace(resume_from_interrupt=resume_from_interrupt)
    session.update_context = lambda **kwargs: session.context.__dict__.update(kwargs)
    return session


rendered: list[object] = []
resumed_interrupts: list[tuple[str, list[object]]] = []


async def resume_from_interrupt(thread_id: str, interrupts: list[object]) -> None:
    resumed_interrupts.append((thread_id, interrupts))


@pytest.fixture(autouse=True)
def _reset_globals() -> None:
    rendered.clear()
    resumed_interrupts.clear()


def _checkpoint_tuple(
    *,
    thread_id: str,
    messages: list[object],
    timestamp: str = "2026-04-02T12:00:00+00:00",
    pending_writes: list[tuple[object, ...]] | None = None,
):
    return SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={
            "ts": timestamp,
            "channel_values": {
                "messages": messages,
                "current_input_tokens": 321,
                "current_output_tokens": 123,
            },
        },
        pending_writes=pending_writes or [],
    )


@pytest.mark.asyncio
async def test_threads_handler_lists_unique_previous_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _build_session(tmp_path)
    handler = threads_module.ThreadsHandler(session)

    tuples = [
        _checkpoint_tuple(
            thread_id="thread-b",
            messages=[HumanMessage(content="newer duplicate")],
            timestamp="2026-04-02T13:00:00+00:00",
        ),
        _checkpoint_tuple(
            thread_id="thread-b",
            messages=[HumanMessage(content="older duplicate")],
        ),
        _checkpoint_tuple(
            thread_id="thread-a",
            messages=[
                AIMessage(content="assistant only"),
                HumanMessage(content="latest human prompt"),
            ],
            pending_writes=[("task-1", "__interrupt__", ["approval"])],
        ),
        _checkpoint_tuple(
            thread_id="current-thread",
            messages=[HumanMessage(content="current thread should be skipped")],
        ),
    ]

    @asynccontextmanager
    async def fake_get_checkpointer(_agent: str, _working_dir: Path):
        class _Checkpointer:
            async def alist(self, _config):
                for item in tuples:
                    yield item

        yield _Checkpointer()

    monkeypatch.setattr(threads_module.initializer, "get_checkpointer", fake_get_checkpointer)

    entries = await handler._load_thread_entries()

    assert [entry.thread_id for entry in entries] == ["thread-b", "thread-a"]
    assert entries[0].preview == "newer duplicate"
    assert entries[1].preview == "latest human prompt"
    assert entries[1].pending_interrupts == ["approval"]


@pytest.mark.asyncio
async def test_threads_handler_loads_history_and_resumes_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session(tmp_path)
    handler = threads_module.ThreadsHandler(session)
    checkpoint_tuple = _checkpoint_tuple(
        thread_id="thread-restore",
        messages=[
            HumanMessage(content="hello"),
            AIMessage(content="world"),
        ],
        pending_writes=[("task-1", "__interrupt__", ["approve-me"])],
    )

    @asynccontextmanager
    async def fake_get_checkpointer(_agent: str, _working_dir: Path):
        class _Checkpointer:
            async def aget_tuple(self, _config):
                return checkpoint_tuple

        yield _Checkpointer()

    cleared: list[bool] = []
    success_messages: list[str] = []

    monkeypatch.setattr(threads_module.initializer, "get_checkpointer", fake_get_checkpointer)
    monkeypatch.setattr(threads_module.console, "clear", lambda: cleared.append(True))
    monkeypatch.setattr(threads_module.console, "print_success", success_messages.append)
    monkeypatch.setattr(threads_module.console, "print", lambda *_args, **_kwargs: None)

    await handler._load_thread("thread-restore", render_history=True)

    assert session.context.thread_id == "thread-restore"
    assert session.context.current_input_tokens == 321
    assert session.context.current_output_tokens == 123
    assert cleared == [True]
    assert rendered == checkpoint_tuple.checkpoint["channel_values"]["messages"]
    assert resumed_interrupts == [("thread-restore", ["approve-me"])]
    assert success_messages == ["Restored thread thread-restore"]


def test_build_preview_prefers_latest_human_message() -> None:
    messages = [
        HumanMessage(content="first question"),
        AIMessage(content="answer"),
        HumanMessage(content="follow-up question that is quite long and needs to be truncated eventually"),
    ]
    preview = threads_module.ThreadsHandler._build_preview(messages)
    assert "follow-up question" in preview


def test_build_preview_truncates_long_preview() -> None:
    messages = [HumanMessage(content="a" * 200)]
    preview = threads_module.ThreadsHandler._build_preview(messages)
    assert len(preview) <= 75
    assert preview.endswith("...") or len(preview) <= 72


def test_build_preview_falls_back_to_last_message_when_no_human() -> None:
    messages = [AIMessage(content="assistant only")]
    preview = threads_module.ThreadsHandler._build_preview(messages)
    assert "assistant only" in preview


def test_build_preview_returns_no_preview_for_empty_content() -> None:
    messages = [HumanMessage(content="")]
    preview = threads_module.ThreadsHandler._build_preview(messages)
    assert preview == "No preview available"


def test_extract_interrupts_skips_non_interrupt_channel() -> None:
    writes = [("task-1", "messages", "some content")]
    result = threads_module.ThreadsHandler._extract_interrupts(writes)
    assert result == []


def test_extract_interrupts_collects_interrupt_values() -> None:
    writes = [
        ("task-1", "__interrupt__", ["approval_request"]),
        ("task-2", "__interrupt__", "single_interrupt"),
    ]
    result = threads_module.ThreadsHandler._extract_interrupts(writes)
    assert "approval_request" in result
    assert "single_interrupt" in result


def test_extract_interrupts_skips_short_pending_writes() -> None:
    writes = [("task-1", "channel")]
    result = threads_module.ThreadsHandler._extract_interrupts(writes)
    assert result == []


def test_format_thread_list_shows_pending_approval_tag() -> None:
    entries = [
        threads_module.ThreadEntry(
            thread_id="thread-a",
            preview="test message",
            timestamp="2026-04-02T13:00:00+00:00",
            pending_interrupts=["approval"],
        ),
    ]
    formatted = threads_module.ThreadsHandler._format_thread_list(
        entries, selected_index=0, scroll_offset=0, window_size=10
    )
    text = "".join(fragment[1] for fragment in formatted)
    assert "[pending approval]" in text


def test_format_thread_list_hides_pending_tag_when_no_interrupts() -> None:
    entries = [
        threads_module.ThreadEntry(
            thread_id="thread-a",
            preview="clean thread",
            timestamp="2026-04-02T13:00:00+00:00",
            pending_interrupts=[],
        ),
    ]
    formatted = threads_module.ThreadsHandler._format_thread_list(
        entries, selected_index=0, scroll_offset=0, window_size=10
    )
    text = "".join(fragment[1] for fragment in formatted)
    assert "[pending approval]" not in text


@pytest.mark.asyncio
async def test_threads_handler_reports_no_threads_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(threads_module.console, "print_warning", warnings.append)
    monkeypatch.setattr(threads_module.console, "print", lambda *_args, **_kwargs: None)

    session = _build_session(tmp_path)
    handler = threads_module.ThreadsHandler(session)

    @asynccontextmanager
    async def fake_get_checkpointer(_agent: str, _working_dir: Path):
        class _Checkpointer:
            async def alist(self, _config):
                return
                yield

        yield _Checkpointer()

    monkeypatch.setattr(threads_module.initializer, "get_checkpointer", fake_get_checkpointer)

    await handler.handle()
    assert "No previous conversation threads found" in warnings


@pytest.mark.asyncio
async def test_threads_handler_skips_selection_when_none_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    handler = threads_module.ThreadsHandler(session)

    tuples = [
        _checkpoint_tuple(
            thread_id="thread-a",
            messages=[HumanMessage(content="hello")],
        ),
    ]

    @asynccontextmanager
    async def fake_get_checkpointer(_agent: str, _working_dir: Path):
        class _Checkpointer:
            async def alist(self, _config):
                for item in tuples:
                    yield item

        yield _Checkpointer()

    monkeypatch.setattr(threads_module.initializer, "get_checkpointer", fake_get_checkpointer)

    async def fake_select_thread(_entries):
        return ""

    monkeypatch.setattr(handler, "_select_thread", fake_select_thread)

    load_called: list[str] = []

    async def fake_load_thread(thread_id, *, render_history):
        load_called.append(thread_id)

    monkeypatch.setattr(handler, "_load_thread", fake_load_thread)

    await handler.handle()
    assert load_called == []


@pytest.mark.asyncio
async def test_threads_handler_handles_exception_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(threads_module.console, "print_error", errors.append)
    monkeypatch.setattr(threads_module.console, "print", lambda *_args, **_kwargs: None)

    async def failing_load_entries():
        raise RuntimeError("checkpoint failure")

    session = _build_session(tmp_path)
    handler = threads_module.ThreadsHandler(session)
    monkeypatch.setattr(handler, "_load_thread_entries", failing_load_entries)

    await handler.handle()
    assert any("Error browsing threads" in e for e in errors)


@pytest.mark.asyncio
async def test_threads_handler_load_thread_skips_render_when_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _build_session(tmp_path)
    handler = threads_module.ThreadsHandler(session)

    @asynccontextmanager
    async def fake_get_checkpointer(_agent: str, _working_dir: Path):
        class _Checkpointer:
            async def aget_tuple(self, _config):
                return None

        yield _Checkpointer()

    monkeypatch.setattr(threads_module.initializer, "get_checkpointer", fake_get_checkpointer)
    errors: list[str] = []
    monkeypatch.setattr(threads_module.console, "print_error", errors.append)
    monkeypatch.setattr(threads_module.console, "print", lambda *_args, **_kwargs: None)

    await handler._load_thread("missing-thread", render_history=True)
    assert "No conversation history found for that thread" in errors[0]

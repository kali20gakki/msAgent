#!/usr/bin/python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from msagent.utils import offload as offload_module


class _FakeBackend:
    def __init__(self, *, write_error: str | None = None) -> None:
        self.storage: dict[str, str] = {}
        self.write_error = write_error

    async def adownload_files(self, paths: list[str]):
        path = paths[0]
        content = self.storage.get(path)
        return [
            SimpleNamespace(
                content=content.encode("utf-8") if content is not None else None,
                error=None if content is not None else "file_not_found",
            )
        ]

    async def awrite(self, path: str, content: str):
        if self.write_error is not None:
            return SimpleNamespace(error=self.write_error)
        self.storage[path] = content
        return SimpleNamespace(error=None)

    async def aedit(self, path: str, _old: str, new: str):
        if self.write_error is not None:
            return SimpleNamespace(error=self.write_error)
        self.storage[path] = new
        return SimpleNamespace(error=None)


class _FakeSummarizationMiddleware:
    last_summary_prompt: str | None = None

    def __init__(
        self,
        *,
        model,
        backend,
        keep,
        trim_tokens_to_summarize=None,
        summary_prompt: str,
    ) -> None:
        del backend, trim_tokens_to_summarize
        self.model = model
        self.keep = keep
        self.summary_prompt = summary_prompt
        type(self).last_summary_prompt = summary_prompt

    def _filter_summary_messages(self, messages):
        return [message for message in messages if message.additional_kwargs.get("lc_source") != "summarization"]

    def _apply_event_to_messages(self, messages, event):
        if event is None:
            return list(messages)
        return [event["summary_message"], *messages[event["cutoff_index"] :]]

    def _determine_cutoff_index(self, messages):
        return max(0, len(messages) - int(self.keep[1]))

    def _partition_messages(self, messages, cutoff):
        return messages[:cutoff], messages[cutoff:]

    async def _acreate_summary(self, messages):
        return " | ".join(message.text for message in messages)

    def _build_new_messages_with_path(self, summary: str, file_path: str | None):
        return [
            HumanMessage(
                content=f"summary={summary};path={file_path}",
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    def _compute_state_cutoff(self, event, cutoff: int) -> int:
        if event is None:
            return cutoff
        return event["cutoff_index"] + cutoff - 1

    def token_counter(self, messages) -> int:
        total = 0
        for message in messages:
            content = message.content
            total += len(content) if isinstance(content, str) else 1
        return total


def _make_middleware(*, backend, keep, summary_prompt) -> _FakeSummarizationMiddleware:
    return _FakeSummarizationMiddleware(
        model=SimpleNamespace(),
        backend=backend,
        keep=keep,
        summary_prompt=summary_prompt,
    )


@pytest.mark.asyncio
async def test_perform_conversation_offload_summarizes_and_persists_history() -> None:
    backend = _FakeBackend()
    middleware = _make_middleware(
        backend=backend,
        keep=("messages", 1),
        summary_prompt="Summarize {conversation}",
    )
    result = await offload_module.perform_conversation_offload(
        middleware=middleware,
        messages=[
            HumanMessage(content="user-1"),
            AIMessage(content="assistant-1"),
            HumanMessage(content="user-2"),
        ],
        prior_event=None,
        thread_id="thread-1",
        backend=backend,
    )

    assert result is not None
    assert result.messages_offloaded == 2
    assert result.messages_kept == 1
    assert result.new_event["cutoff_index"] == 2
    assert result.new_event["file_path"] == "/conversation_history/thread-1.md"
    assert "assistant-1" in backend.storage["/conversation_history/thread-1.md"]
    assert "Offloaded 2 messages" in result.new_event["summary_message"].content
    assert _FakeSummarizationMiddleware.last_summary_prompt == "Summarize {conversation}"


@pytest.mark.asyncio
async def test_perform_conversation_offload_warns_when_backend_write_fails() -> None:
    backend = _FakeBackend(write_error="permission denied")
    middleware = _make_middleware(
        backend=backend,
        keep=("messages", 1),
        summary_prompt="Summarize",
    )
    result = await offload_module.perform_conversation_offload(
        middleware=middleware,
        messages=[
            HumanMessage(content="user-1"),
            AIMessage(content="assistant-1"),
            HumanMessage(content="user-2"),
        ],
        prior_event=None,
        thread_id="thread-2",
        backend=backend,
    )

    assert result is not None
    assert result.new_event["file_path"] is None
    assert result.offload_warning is not None


@pytest.mark.asyncio
async def test_perform_conversation_offload_skips_degenerate_chained_compaction() -> None:
    backend = _FakeBackend()
    middleware = _make_middleware(
        backend=backend,
        keep=("messages", 2),
        summary_prompt="Summarize",
    )
    prior_event = {
        "cutoff_index": 4,
        "summary_message": HumanMessage(content="prior-summary"),
        "file_path": None,
    }
    result = await offload_module.perform_conversation_offload(
        middleware=middleware,
        messages=[
            HumanMessage(content="m0"),
            AIMessage(content="m1"),
            HumanMessage(content="m2"),
            AIMessage(content="m3"),
            HumanMessage(content="m4"),
            AIMessage(content="m5"),
        ],
        prior_event=prior_event,
        thread_id="thread-4",
        backend=backend,
    )

    assert result is None


def test_event_cutoff_rejects_bool_cutoff_index() -> None:
    event = {
        "cutoff_index": True,  # bool is an int subclass but must not count as 1
        "summary_message": HumanMessage(content="summary"),
        "file_path": None,
    }
    assert offload_module._event_cutoff(event) == 0


def test_event_cutoff_returns_cutoff_index() -> None:
    event = {
        "cutoff_index": 3,
        "summary_message": HumanMessage(content="summary"),
        "file_path": None,
    }
    assert offload_module._event_cutoff(event) == 3
    assert offload_module._event_cutoff(None) == 0


def test_render_compression_summary_prompt_should_return_none_when_prompt_is_empty() -> None:
    assert offload_module.render_compression_summary_prompt(None, Path(".")) is None
    assert offload_module.render_compression_summary_prompt("", Path(".")) is None
    assert offload_module.render_compression_summary_prompt([], Path(".")) is None


def test_render_compression_summary_prompt_should_replace_conversation_with_messages_when_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        offload_module,
        "build_local_environment_context",
        lambda working_dir, now=None: "ENV",
    )
    rendered = offload_module.render_compression_summary_prompt(
        "Summarize: {conversation}",
        tmp_path,
    )
    assert rendered is not None
    assert "Summarize: {messages}" in rendered
    assert "{conversation}" not in rendered
    assert "ENV" in rendered


class _AppendingBackend:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {"/conversation_history/thread-x.md": "OLD"}
        self.edits: list[tuple[str, str, str]] = []

    async def adownload_files(self, paths: list[str]):
        path = paths[0]
        content = self.storage.get(path)
        if content is None:
            return [SimpleNamespace(content=None, error="file_not_found")]
        return [SimpleNamespace(content=content.encode("utf-8"), error=None)]

    async def awrite(self, path: str, content: str):
        self.storage[path] = content
        return SimpleNamespace(error=None)

    async def aedit(self, path: str, old: str, new: str):
        self.edits.append((path, old, new))
        self.storage[path] = new
        return SimpleNamespace(error=None)


@pytest.mark.asyncio
async def test_offload_messages_to_backend_should_append_when_history_exists() -> None:
    backend = _AppendingBackend()
    middleware = _make_middleware(
        backend=backend,
        keep=("messages", 1),
        summary_prompt="Summarize",
    )
    path = await offload_module.offload_messages_to_backend(
        [HumanMessage(content="new-message")],
        middleware,
        thread_id="thread-x",
        backend=backend,
    )

    assert path == "/conversation_history/thread-x.md"
    updated = backend.storage["/conversation_history/thread-x.md"]
    assert updated.startswith("OLD")
    assert "new-message" in updated
    assert len(backend.edits) == 1
    assert backend.edits[0][0] == "/conversation_history/thread-x.md"
    assert backend.edits[0][1] == "OLD"

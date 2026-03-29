from __future__ import annotations

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
        del model, backend, trim_tokens_to_summarize
        self.keep = keep
        self.summary_prompt = summary_prompt
        type(self).last_summary_prompt = summary_prompt

    def _filter_summary_messages(self, messages):
        return [
            message
            for message in messages
            if message.additional_kwargs.get("lc_source") != "summarization"
        ]

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


def _fake_token_count(messages, _model) -> int:
    total = 0
    for message in messages:
        content = message.content
        total += len(content) if isinstance(content, str) else 1
    return total


@pytest.mark.asyncio
async def test_perform_conversation_offload_summarizes_and_persists_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        offload_module,
        "SummarizationMiddleware",
        _FakeSummarizationMiddleware,
    )
    monkeypatch.setattr(
        offload_module,
        "calculate_message_tokens",
        _fake_token_count,
    )

    backend = _FakeBackend()
    result = await offload_module.perform_conversation_offload(
        messages=[
            HumanMessage(content="user-1"),
            AIMessage(content="assistant-1"),
            HumanMessage(content="user-2"),
        ],
        prior_event=None,
        thread_id="thread-1",
        model=SimpleNamespace(),
        backend=backend,
        keep_messages=1,
        summary_prompt="Summarize {conversation}",
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
async def test_perform_conversation_offload_warns_when_backend_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        offload_module,
        "SummarizationMiddleware",
        _FakeSummarizationMiddleware,
    )
    monkeypatch.setattr(
        offload_module,
        "calculate_message_tokens",
        _fake_token_count,
    )

    backend = _FakeBackend(write_error="permission denied")
    result = await offload_module.perform_conversation_offload(
        messages=[
            HumanMessage(content="user-1"),
            AIMessage(content="assistant-1"),
            HumanMessage(content="user-2"),
        ],
        prior_event=None,
        thread_id="thread-2",
        model=SimpleNamespace(),
        backend=backend,
        keep_messages=1,
    )

    assert result is not None
    assert result.new_event["file_path"] is None
    assert result.offload_warning is not None


@pytest.mark.asyncio
async def test_perform_conversation_offload_supports_zero_keep_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        offload_module,
        "SummarizationMiddleware",
        _FakeSummarizationMiddleware,
    )
    monkeypatch.setattr(
        offload_module,
        "calculate_message_tokens",
        _fake_token_count,
    )

    backend = _FakeBackend()
    result = await offload_module.perform_conversation_offload(
        messages=[
            HumanMessage(content="user-1"),
            AIMessage(content="assistant-1"),
        ],
        prior_event=None,
        thread_id="thread-3",
        model=SimpleNamespace(),
        backend=backend,
        keep_messages=0,
    )

    assert result is not None
    assert result.messages_offloaded == 2
    assert result.messages_kept == 0
    assert result.new_event["cutoff_index"] == 2

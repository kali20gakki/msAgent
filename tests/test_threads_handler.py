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
    session.renderer = SimpleNamespace(
        render_message=lambda message: rendered.append(message)
    )
    session.message_dispatcher = SimpleNamespace(
        resume_from_interrupt=resume_from_interrupt
    )
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

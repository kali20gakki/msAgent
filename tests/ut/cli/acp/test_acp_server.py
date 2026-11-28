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

"""Unit tests for the ACP server agent and CLI registration."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import acp
import pytest
from acp.schema import ConfigOptionUpdate, LoadSessionResponse, ListSessionsResponse, PromptResponse

from msagent.cli.acp.server import (
    AcpSession,
    MsagentAcpAgent,
    _build_config_options,
    _build_modes,
    _resolve_llm_alias,
    _SessionEntry,
    _write_session_index,
)
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.bootstrap.legacy import (
    create_legacy_parser,
    dispatch_legacy_command,
    normalize_argv,
)
from msagent.cli.core.context import Context
from msagent.configs import ApprovalMode, ToolDecisionRule


@pytest.fixture
def patch_state_dir(monkeypatch):
    """Point project state lookups at the test state directory."""

    def patch_project_paths(path: Path) -> None:
        monkeypatch.setattr(
            initializer,
            "get_project_paths",
            lambda _working_dir: SimpleNamespace(root=path),
        )

    return patch_project_paths


class _FakeGraphContext:
    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self) -> "_FakeGraphContext":
        self.enter_count += 1
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        self.exit_count += 1
        return False


class _FakeRunner:
    instances: list["_FakeRunner"] = []

    def __init__(self, conn: Any, session_id: str, session: Any, graph: Any) -> None:
        self.conn = conn
        self.session_id = session_id
        self.session = session
        self.graph = graph
        self.received_text: str | None = None
        _FakeRunner.instances.append(self)

    async def run(self, user_text: str) -> None:
        self.received_text = user_text


def _make_context(tmp_path: Path, thread_id: str = "thread-1", model: str = "test-model") -> Context:
    return Context(
        agent="Profiler",
        model=model,
        thread_id=thread_id,
        working_dir=tmp_path,
        state_dir=tmp_path,
        approval_mode=ApprovalMode.ACTIVE,
        recursion_limit=25,
        audit_log_enabled=False,
    )


def _make_args(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "agent": None,
        "model": None,
        "approval_mode": "active",
        "execute_approval_mode": None,
        "trace_jsonl": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_initialize_should_advertise_load_and_list_session_capabilities() -> None:
    agent = MsagentAcpAgent(_make_args())

    response = await agent.initialize(acp.PROTOCOL_VERSION)

    assert response.protocol_version == acp.PROTOCOL_VERSION
    assert response.agent_capabilities.load_session is True
    assert response.agent_capabilities.session_capabilities.list is not None
    assert response.agent_capabilities.session_capabilities.close is not None


@pytest.mark.asyncio
async def test_new_session_and_prompt_dispatch_turn_runner(monkeypatch, tmp_path) -> None:
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )
    monkeypatch.setattr("msagent.cli.acp.server.AcpTurnRunner", _FakeRunner)
    _FakeRunner.instances.clear()

    async def fake_create(cls, **kwargs: Any) -> Context:
        return Context(
            agent="Profiler",
            model=kwargs["model"] or "test-model",
            thread_id="thread-1",
            working_dir=Path(kwargs["working_dir"]),
            state_dir=tmp_path,
            approval_mode=kwargs["approval_mode"],
            recursion_limit=25,
            audit_log_enabled=False,
        )

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    agent = MsagentAcpAgent(_make_args(agent="Profiler", approval_mode="semi-active"))
    new_session_response = await agent.new_session(str(tmp_path))
    session_id = new_session_response.session_id
    assert session_id in agent._sessions
    created_index = json.loads((tmp_path / "acp_sessions.json").read_text(encoding="utf-8"))
    assert created_index["thread-1"]["cwd"] == str(tmp_path.resolve())

    prompt_response = await agent.prompt(
        session_id,
        [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
    )

    assert isinstance(prompt_response, PromptResponse)
    assert prompt_response.stop_reason == "end_turn"
    runner = _FakeRunner.instances[0]
    assert runner.received_text == "hello\nworld"
    assert graph_context.enter_count == 1

    await agent.prompt(session_id, [{"type": "text", "text": "again"}])
    assert graph_context.enter_count == 1
    assert _FakeRunner.instances[1].received_text == "again"

    index = json.loads((tmp_path / "acp_sessions.json").read_text(encoding="utf-8"))
    assert index["thread-1"]["title"] == "hello"
    assert index["thread-1"]["cwd"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_record_session_should_keep_concurrent_updates(tmp_path) -> None:
    agent = MsagentAcpAgent(_make_args())

    async def record(thread_id: str) -> None:
        context = _make_context(tmp_path, thread_id)
        entry = _SessionEntry(session=AcpSession(context), context=context)
        await agent._record_session(entry, user_text=f"question {thread_id}")

    await asyncio.gather(record("thread-a"), record("thread-b"))

    index = json.loads((tmp_path / "acp_sessions.json").read_text(encoding="utf-8"))
    assert set(index) == {"thread-a", "thread-b"}


@pytest.mark.asyncio
async def test_recorded_model_should_be_restored_on_load(monkeypatch, tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: _FakeGraphContext(),
    )

    async def fake_create(cls, **kwargs: Any) -> Context:
        return _make_context(
            Path(kwargs["working_dir"]),
            thread_id=kwargs["thread_id"] or "generated",
            model=kwargs["model"] or "test-model",
        )

    async def fake_read_messages(self: Any, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        return False, []

    async def fake_replay(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))
    monkeypatch.setattr(MsagentAcpAgent, "_read_checkpoint_messages", fake_read_messages)
    monkeypatch.setattr("msagent.cli.acp.server.replay_session_history", fake_replay)

    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path, model="alias-a")
    entry = _SessionEntry(session=AcpSession(context), context=context)
    await agent._record_session(entry, user_text="hello")

    await agent.load_session(str(tmp_path), "thread-1")

    assert agent._sessions["thread-1"].context.model == "alias-a"


@pytest.mark.asyncio
async def test_prompt_should_reject_closed_session(tmp_path) -> None:
    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path)
    entry = _SessionEntry(session=AcpSession(context), context=context)
    entry.closed = True
    agent._sessions["s1"] = entry

    with pytest.raises(acp.RequestError):
        await agent.prompt("s1", [{"type": "text", "text": "hello"}])


class _BlockingRunner:
    started_event: asyncio.Event

    def __init__(self, *, session: Any, **_kwargs: Any) -> None:
        self._session = session

    async def run(self, user_text: str) -> None:
        self._session.current_stream_task = asyncio.current_task()
        type(self).started_event.set()
        try:
            await asyncio.sleep(3600)
        finally:
            self._session.current_stream_task = None


@pytest.mark.asyncio
async def test_prompt_should_return_cancelled_when_cancel_requested(monkeypatch, tmp_path) -> None:
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )
    monkeypatch.setattr("msagent.cli.acp.server.AcpTurnRunner", _BlockingRunner)
    _BlockingRunner.started_event = asyncio.Event()

    async def fake_create(cls, **kwargs: Any) -> Context:
        return _make_context(Path(kwargs["working_dir"]))

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    agent = MsagentAcpAgent(_make_args(agent="Profiler"))
    session_id = (await agent.new_session(str(tmp_path))).session_id
    prompt_task = asyncio.create_task(agent.prompt(session_id, [{"type": "text", "text": "long task"}]))

    await _BlockingRunner.started_event.wait()
    await agent.cancel(session_id)
    response = await asyncio.wait_for(prompt_task, timeout=5)

    assert isinstance(response, PromptResponse)
    assert response.stop_reason == "cancelled"


class _AbortTrackingRunner(_BlockingRunner):
    aborted = False

    async def abort_pending_tool_calls(self) -> None:
        _AbortTrackingRunner.aborted = True


@pytest.mark.asyncio
async def test_cancelled_prompt_should_report_pending_tool_calls(monkeypatch, tmp_path) -> None:
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )
    monkeypatch.setattr("msagent.cli.acp.server.AcpTurnRunner", _AbortTrackingRunner)
    _AbortTrackingRunner.started_event = asyncio.Event()
    _AbortTrackingRunner.aborted = False

    async def fake_create(cls, **kwargs: Any) -> Context:
        return _make_context(Path(kwargs["working_dir"]))

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    agent = MsagentAcpAgent(_make_args(agent="Profiler"))
    session_id = (await agent.new_session(str(tmp_path))).session_id
    prompt_task = asyncio.create_task(agent.prompt(session_id, [{"type": "text", "text": "long task"}]))

    await _AbortTrackingRunner.started_event.wait()
    await agent.cancel(session_id)
    response = await asyncio.wait_for(prompt_task, timeout=5)

    assert response.stop_reason == "cancelled"
    assert _AbortTrackingRunner.aborted is True


class _BlockingGraphContext:
    def __init__(self) -> None:
        self.enter_started = asyncio.Event()

    async def __aenter__(self) -> "_BlockingGraphContext":
        self.enter_started.set()
        await asyncio.sleep(3600)
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


@pytest.mark.asyncio
async def test_prompt_should_return_cancelled_when_cancelled_before_graph_ready(monkeypatch, tmp_path) -> None:
    graph_context = _BlockingGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )

    async def fake_create(cls, **kwargs: Any) -> Context:
        return _make_context(Path(kwargs["working_dir"]))

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    agent = MsagentAcpAgent(_make_args(agent="Profiler"))
    session_id = (await agent.new_session(str(tmp_path))).session_id
    prompt_task = asyncio.create_task(agent.prompt(session_id, [{"type": "text", "text": "long task"}]))

    await graph_context.enter_started.wait()
    await agent.cancel(session_id)
    response = await asyncio.wait_for(prompt_task, timeout=5)

    assert isinstance(response, PromptResponse)
    assert response.stop_reason == "cancelled"
    assert agent._sessions[session_id].graph_stack is None


@pytest.mark.asyncio
async def test_load_session_should_rebuild_context_and_replay_history(monkeypatch, tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )

    async def fake_create(cls, **kwargs: Any) -> Context:
        return Context(
            agent="Profiler",
            model="test-model",
            thread_id=kwargs["thread_id"] or "generated",
            working_dir=Path(kwargs["working_dir"]),
            state_dir=tmp_path,
            approval_mode=kwargs["approval_mode"],
            recursion_limit=25,
            audit_log_enabled=False,
        )

    messages = ["history"]
    replay_call: dict[str, Any] = {}

    async def fake_read_messages(self: Any, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        return True, messages

    async def fake_replay(*, conn: Any, session_id: str, messages: list[Any]) -> None:
        replay_call.update({"conn": conn, "session_id": session_id, "messages": messages})

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))
    monkeypatch.setattr(MsagentAcpAgent, "_read_checkpoint_messages", fake_read_messages)
    monkeypatch.setattr("msagent.cli.acp.server.replay_session_history", fake_replay)

    _write_session_index(tmp_path, {"thread-1": {"cwd": str(tmp_path.resolve()), "title": "hello"}})
    agent = MsagentAcpAgent(_make_args())
    agent._conn = object()

    load_response = await agent.load_session(str(tmp_path), "thread-1")

    assert isinstance(load_response, LoadSessionResponse)
    entry = agent._sessions["thread-1"]
    assert entry.context.thread_id == "thread-1"
    assert graph_context.enter_count == 1
    assert replay_call["session_id"] == "thread-1"
    assert replay_call["messages"] == messages
    assert replay_call["conn"] is agent._conn


@pytest.mark.asyncio
async def test_load_session_should_reject_unknown_thread(monkeypatch, tmp_path) -> None:
    async def fake_read_messages(self: Any, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        return False, []

    monkeypatch.setattr(MsagentAcpAgent, "_read_checkpoint_messages", fake_read_messages)

    agent = MsagentAcpAgent(_make_args())

    with pytest.raises(acp.RequestError):
        await agent.load_session(str(tmp_path), "missing")


@pytest.mark.asyncio
async def test_load_session_should_allow_indexed_session_without_checkpoint(
    monkeypatch, tmp_path, patch_state_dir
) -> None:
    """A session listed in the index but without checkpoints must still load."""
    patch_state_dir(tmp_path)
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )

    async def fake_create(cls, **kwargs: Any) -> Context:
        return Context(
            agent="Profiler",
            model="test-model",
            thread_id=kwargs["thread_id"] or "generated",
            working_dir=Path(kwargs["working_dir"]),
            state_dir=tmp_path,
            approval_mode=kwargs["approval_mode"],
            recursion_limit=25,
            audit_log_enabled=False,
        )

    async def fake_read_messages(self: Any, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        return False, []

    replay_call: dict[str, Any] = {}

    async def fake_replay(*, conn: Any, session_id: str, messages: list[Any]) -> None:
        replay_call.update({"session_id": session_id, "messages": messages})

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))
    monkeypatch.setattr(MsagentAcpAgent, "_read_checkpoint_messages", fake_read_messages)
    monkeypatch.setattr("msagent.cli.acp.server.replay_session_history", fake_replay)

    _write_session_index(tmp_path, {"thread-1": {"cwd": str(tmp_path.resolve()), "title": "hello"}})
    agent = MsagentAcpAgent(_make_args())

    load_response = await agent.load_session(str(tmp_path), "thread-1")

    assert isinstance(load_response, LoadSessionResponse)
    assert "thread-1" in agent._sessions
    assert replay_call == {"session_id": "thread-1", "messages": []}


@pytest.mark.asyncio
async def test_load_session_should_reject_cwd_mismatch(tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    _write_session_index(tmp_path, {"thread-1": {"cwd": str(other_dir.resolve())}})
    agent = MsagentAcpAgent(_make_args())

    with pytest.raises(acp.RequestError):
        await agent.load_session(str(tmp_path), "thread-1")


@pytest.mark.asyncio
async def test_list_sessions_should_return_newest_first(tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    _write_session_index(
        tmp_path,
        {
            "old": {
                "cwd": str(tmp_path.resolve()),
                "title": "old",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "new": {
                "cwd": str(tmp_path.resolve()),
                "title": "new",
                "updated_at": "2026-02-01T00:00:00+00:00",
            },
        },
    )
    agent = MsagentAcpAgent(_make_args())

    list_response = await agent.list_sessions(cwd=str(tmp_path))

    assert isinstance(list_response, ListSessionsResponse)
    assert [info.session_id for info in list_response.sessions] == ["new", "old"]
    assert list_response.sessions[0].title == "new"
    assert list_response.sessions[0].cwd == str(tmp_path.resolve())
    assert list_response.next_cursor is None


@pytest.mark.asyncio
async def test_list_sessions_should_filter_by_cwd(tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    _write_session_index(
        tmp_path,
        {
            "here": {
                "cwd": str(tmp_path.resolve()),
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "elsewhere": {
                "cwd": str(other_dir.resolve()),
                "updated_at": "2026-02-01T00:00:00+00:00",
            },
        },
    )
    agent = MsagentAcpAgent(_make_args())

    list_response = await agent.list_sessions(cwd=str(tmp_path))

    assert [info.session_id for info in list_response.sessions] == ["here"]


@pytest.mark.asyncio
async def test_list_sessions_should_page_without_duplicates(monkeypatch, tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    monkeypatch.setattr("msagent.cli.acp.server._SESSION_PAGE_SIZE", 2)
    _write_session_index(
        tmp_path,
        {
            "s1": {
                "cwd": str(tmp_path.resolve()),
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "s2": {
                "cwd": str(tmp_path.resolve()),
                "updated_at": "2026-02-01T00:00:00+00:00",
            },
            "s3": {
                "cwd": str(tmp_path.resolve()),
                "updated_at": "2026-03-01T00:00:00+00:00",
            },
        },
    )
    agent = MsagentAcpAgent(_make_args())

    first_page = await agent.list_sessions(cwd=str(tmp_path))
    assert [info.session_id for info in first_page.sessions] == ["s3", "s2"]
    assert first_page.next_cursor is not None

    second_page = await agent.list_sessions(cwd=str(tmp_path), cursor=first_page.next_cursor)
    assert [info.session_id for info in second_page.sessions] == ["s1"]
    assert second_page.next_cursor is None


@pytest.mark.asyncio
async def test_list_sessions_should_ignore_malformed_cursor(tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    _write_session_index(
        tmp_path,
        {
            "s1": {
                "cwd": str(tmp_path.resolve()),
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        },
    )
    agent = MsagentAcpAgent(_make_args())

    list_response = await agent.list_sessions(cwd=str(tmp_path), cursor="not-a-cursor")

    assert [info.session_id for info in list_response.sessions] == ["s1"]


@pytest.mark.asyncio
async def test_prompt_unknown_session_raises_request_error() -> None:
    agent = MsagentAcpAgent(_make_args())

    with pytest.raises(acp.RequestError):
        await agent.prompt("missing", [{"type": "text", "text": "hello"}])


@pytest.mark.asyncio
async def test_cancel_cancels_running_stream_task(tmp_path) -> None:
    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path)
    entry = _SessionEntry(session=AcpSession(context), context=context)
    agent._sessions["s1"] = entry

    task = asyncio.create_task(asyncio.sleep(30))
    entry.active_task = task
    try:
        await agent.cancel("s1")
        await asyncio.sleep(0)
        assert task.cancelled()
        assert entry.cancelled_task is task
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_close_session_should_dispose_graph_and_forget_entry(monkeypatch, tmp_path) -> None:
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )
    monkeypatch.setattr("msagent.cli.acp.server.AcpTurnRunner", _FakeRunner)
    _FakeRunner.instances.clear()

    async def fake_create(cls, **kwargs: Any) -> Context:
        return _make_context(Path(kwargs["working_dir"]))

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    agent = MsagentAcpAgent(_make_args(agent="Profiler"))
    session_id = (await agent.new_session(str(tmp_path))).session_id
    await agent.prompt(session_id, [{"type": "text", "text": "hello"}])

    await agent.close_session(session_id)

    assert session_id not in agent._sessions
    assert graph_context.enter_count == 1
    assert graph_context.exit_count == 1


class _FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def setup(self) -> None:
        return None

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)

    async def __aenter__(self) -> "_FakeCheckpointer":
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


@pytest.mark.asyncio
async def test_delete_session_should_remove_index_and_checkpoints(monkeypatch, tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    checkpointer = _FakeCheckpointer()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_checkpointer",
        lambda *_args, **_kwargs: checkpointer,
    )
    context = _make_context(tmp_path, thread_id="thread-1")
    agent = MsagentAcpAgent(_make_args())
    agent._sessions["thread-1"] = _SessionEntry(
        session=AcpSession(context),
        context=context,
    )
    _write_session_index(tmp_path, {"thread-1": {"cwd": str(tmp_path.resolve()), "title": "hello"}})

    await agent.delete_session("thread-1")

    assert "thread-1" not in agent._sessions
    assert json.loads((tmp_path / "acp_sessions.json").read_text(encoding="utf-8")) == {}
    assert checkpointer.deleted == ["thread-1"]


@pytest.mark.asyncio
async def test_delete_session_should_clear_index_without_loaded_entry(monkeypatch, tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    checkpointer = _FakeCheckpointer()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_checkpointer",
        lambda *_args, **_kwargs: checkpointer,
    )
    project_state = tmp_path / "project-a"
    project_state.mkdir()
    _write_session_index(project_state, {"thread-9": {"cwd": str(tmp_path), "title": "old"}})
    monkeypatch.setattr("msagent.cli.acp.server._iter_project_state_dirs", lambda: [project_state])

    agent = MsagentAcpAgent(_make_args())
    await agent.delete_session("thread-9")

    assert json.loads((project_state / "acp_sessions.json").read_text(encoding="utf-8")) == {}
    assert checkpointer.deleted == ["thread-9"]


@pytest.mark.asyncio
async def test_prompt_should_reject_empty_text(tmp_path) -> None:
    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path)
    agent._sessions["s1"] = _SessionEntry(session=AcpSession(context), context=context)

    with pytest.raises(acp.RequestError):
        await agent.prompt("s1", [{"type": "image", "data": "aGk=", "mimeType": "image/png"}])

    assert not (tmp_path / "acp_sessions.json").exists()


def test_extract_prompt_text_should_join_blocks_with_newline() -> None:
    text = MsagentAcpAgent._extract_prompt_text([{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}])

    assert text == "hello\nworld"


@pytest.mark.asyncio
async def test_list_sessions_should_ignore_non_string_index_fields(tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    (tmp_path / "acp_sessions.json").write_text(
        json.dumps({"bad": {"cwd": 123, "title": 5, "updated_at": "2026-01-01T00:00:00+00:00"}}),
        encoding="utf-8",
    )
    agent = MsagentAcpAgent(_make_args())

    list_response = await agent.list_sessions(cwd=str(tmp_path))

    assert [info.session_id for info in list_response.sessions] == ["bad"]
    assert list_response.sessions[0].cwd == str(tmp_path)


@pytest.mark.asyncio
async def test_load_session_should_not_refresh_updated_at(monkeypatch, tmp_path, patch_state_dir) -> None:
    patch_state_dir(tmp_path)
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )

    async def fake_create(cls, **kwargs: Any) -> Context:
        return Context(
            agent="Profiler",
            model="test-model",
            thread_id=kwargs["thread_id"] or "generated",
            working_dir=Path(kwargs["working_dir"]),
            state_dir=tmp_path,
            approval_mode=kwargs["approval_mode"],
            recursion_limit=25,
            audit_log_enabled=False,
        )

    async def fake_read_messages(self: Any, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        return True, []

    async def fake_replay(*, conn: Any, session_id: str, messages: list[Any]) -> None:
        return None

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))
    monkeypatch.setattr(MsagentAcpAgent, "_read_checkpoint_messages", fake_read_messages)
    monkeypatch.setattr("msagent.cli.acp.server.replay_session_history", fake_replay)

    _write_session_index(
        tmp_path,
        {
            "thread-1": {
                "cwd": str(tmp_path.resolve()),
                "title": "hello",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        },
    )
    agent = MsagentAcpAgent(_make_args())

    await agent.load_session(str(tmp_path), "thread-1")

    index = json.loads((tmp_path / "acp_sessions.json").read_text(encoding="utf-8"))
    assert index["thread-1"]["updated_at"] == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_dispatch_legacy_command_routes_acp(monkeypatch) -> None:
    captured_call: dict[str, Any] = {}

    async def fake_handle(args: argparse.Namespace) -> int:
        captured_call["args"] = args
        return 7

    monkeypatch.setattr("msagent.cli.acp.server.handle_acp_command", fake_handle)

    args = argparse.Namespace(
        version=False,
        cli_command="acp",
        working_dir=".",
        agent=None,
        model=None,
        approval_mode="active",
        execute_approval_mode=None,
        trace_jsonl=None,
    )
    result = await dispatch_legacy_command(args)

    assert result == 7
    assert captured_call["args"].agent is None


def test_acp_command_registered_in_legacy_parser() -> None:
    parser = create_legacy_parser()

    args = parser.parse_args(["acp", "-a", "Profiler", "--approval-mode", "semi-active"])
    assert args.cli_command == "acp"
    assert args.agent == "Profiler"
    assert args.approval_mode == "semi-active"

    assert normalize_argv(["acp", "-a", "Profiler"]) == ["acp", "-a", "Profiler"]


def _fake_llms(*aliases: str) -> SimpleNamespace:
    return SimpleNamespace(
        llms=[SimpleNamespace(alias=alias, model=f"{alias}-model", provider="openai") for alias in aliases]
    )


@pytest.mark.asyncio
async def test_build_config_options_should_hide_builtin_models_when_user_config_exists(monkeypatch, tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.llms.yml").write_text(
        "llms:\n- alias: mine\n  provider: openai\n  model: my-model\n  max_tokens: 1024\n  temperature: 0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_llms("mine", "builtin-a", "builtin-b")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)
    context = _make_context(tmp_path, thread_id="thread-1", model="builtin-b")

    options = await _build_config_options(context)

    assert [option.value for option in options[0].options] == [
        "mine-model",
        "builtin-b-model",
    ]
    assert [option.name for option in options[0].options] == [
        "mine-model",
        "builtin-b-model",
    ]
    assert options[0].current_value == "builtin-b-model"


@pytest.mark.asyncio
async def test_build_config_options_should_list_all_models_without_user_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_llms("mine", "builtin-a")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)
    context = _make_context(tmp_path, thread_id="thread-1")

    options = await _build_config_options(context)

    assert [option.value for option in options[0].options] == [
        "mine-model",
        "builtin-a-model",
    ]
    assert options[0].current_value == "test-model"


def test_resolve_llm_alias_should_map_model_name_and_alias_to_alias() -> None:
    llms = _fake_llms("mine", "other").llms

    assert _resolve_llm_alias(llms, "mine") == "mine"
    assert _resolve_llm_alias(llms, "other-model") == "other"
    assert _resolve_llm_alias(llms, "missing") == "missing"


@pytest.mark.asyncio
async def test_set_config_option_should_keep_session_approval_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "msagent.cli.acp.server._build_graph_context",
        lambda _context: _FakeGraphContext(),
    )

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_llms("alias-a", "alias-b")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)

    agent = MsagentAcpAgent(_make_args())
    old_context = _make_context(tmp_path, model="alias-a")
    old_entry = agent._build_entry(old_context)
    old_entry.session.approval_session_rules.append(ToolDecisionRule(name="execute", decision="always_approve"))
    old_entry.session.execute_approval_mode = "convenience"
    agent._sessions[old_context.thread_id] = old_entry

    new_context = _make_context(tmp_path, model="alias-b")

    async def fake_create(*_args: Any, **_kwargs: Any) -> Context:
        return new_context

    async def fake_ensure(_entry: _SessionEntry) -> None:
        return None

    monkeypatch.setattr(agent, "_create_context", fake_create)
    monkeypatch.setattr(agent, "_ensure_graph", fake_ensure)

    response = await agent.set_config_option(old_context.thread_id, "model", "alias-b-model")

    entry = agent._sessions[old_context.thread_id]
    assert response.config_options[0].current_value == "alias-b-model"
    # The switch returns before the graph is rebuilt so the client feels instant.
    assert entry.pending_model == "alias-b"
    assert entry.context.model == "alias-a"

    assert entry.rebuild_task is not None
    await entry.rebuild_task

    assert entry.pending_model is None
    assert entry.context.model == "alias-b"
    assert [rule.name for rule in entry.session.approval_session_rules] == ["execute"]
    assert entry.session.execute_approval_mode == "convenience"


@pytest.mark.asyncio
async def test_prompt_should_finish_pending_model_switch_first(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )
    monkeypatch.setattr("msagent.cli.acp.server.AcpTurnRunner", _FakeRunner)
    _FakeRunner.instances.clear()

    async def fake_create(cls, **kwargs: Any) -> Context:
        return Context(
            agent="Profiler",
            model=kwargs["model"] or "alias-a",
            thread_id=kwargs["thread_id"] or "generated",
            working_dir=Path(kwargs["working_dir"]),
            state_dir=tmp_path,
            approval_mode=kwargs["approval_mode"],
            recursion_limit=25,
            audit_log_enabled=False,
        )

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_llms("alias-a", "alias-b")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)

    agent = MsagentAcpAgent(_make_args(agent="Profiler"))
    session_id = (await agent.new_session(str(tmp_path))).session_id

    await agent.set_config_option(session_id, "model", "alias-b-model")
    await agent.prompt(session_id, [{"type": "text", "text": "hi"}])

    runner = _FakeRunner.instances[0]
    assert runner.session.context.model == "alias-b"


@pytest.mark.asyncio
async def test_set_config_option_should_reject_while_prompt_is_running(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_llms("alias-a", "alias-b")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)

    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path, model="alias-a")
    entry = _SessionEntry(session=AcpSession(context), context=context)
    agent._sessions[context.thread_id] = entry

    await entry.prompt_lock.acquire()
    try:
        with pytest.raises(acp.RequestError):
            await agent.set_config_option(context.thread_id, "model", "alias-b-model")
    finally:
        entry.prompt_lock.release()

    assert entry.pending_model is None


@pytest.mark.asyncio
async def test_set_config_option_should_report_config_load_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        raise RuntimeError("boom")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)

    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path, model="alias-a")
    entry = _SessionEntry(session=AcpSession(context), context=context)
    agent._sessions[context.thread_id] = entry

    with pytest.raises(acp.RequestError):
        await agent.set_config_option(context.thread_id, "model", "whatever")

    assert entry.pending_model is None


def _write_two_provider_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.llms.yml").write_text(
        "llms:\n"
        "- alias: ds\n"
        "  provider: openai\n"
        "  model: ds-model\n"
        "  max_tokens: 1024\n"
        "  temperature: 0.1\n"
        "- alias: claude\n"
        "  provider: anthropic\n"
        "  model: claude-model\n"
        "  max_tokens: 1024\n"
        "  temperature: 0.1\n",
        encoding="utf-8",
    )


def _fake_two_provider_llms() -> SimpleNamespace:
    return SimpleNamespace(
        llms=[
            SimpleNamespace(alias="ds", model="ds-model", provider="openai"),
            SimpleNamespace(alias="claude", model="claude-model", provider="anthropic"),
        ]
    )


@pytest.mark.asyncio
async def test_build_config_options_should_hide_models_without_api_key(monkeypatch, tmp_path) -> None:
    _write_two_provider_config(tmp_path)
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_two_provider_llms()

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)
    context = _make_context(tmp_path, thread_id="thread-1", model="ds")

    options = await _build_config_options(context)

    assert [option.value for option in options[0].options] == ["ds-model"]


@pytest.mark.asyncio
async def test_build_config_options_should_keep_current_model_without_api_key(monkeypatch, tmp_path) -> None:
    _write_two_provider_config(tmp_path)
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_two_provider_llms()

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)
    context = _make_context(tmp_path, thread_id="thread-1", model="claude")

    options = await _build_config_options(context)

    assert [option.value for option in options[0].options] == [
        "ds-model",
        "claude-model",
    ]
    claude = options[0].options[1]
    assert "missing ANTHROPIC_API_KEY" in claude.description


@pytest.mark.asyncio
async def test_build_config_options_should_report_config_load_failure(monkeypatch, tmp_path) -> None:
    async def failing_load(_working_dir: Path) -> SimpleNamespace:
        raise RuntimeError("boom")

    monkeypatch.setattr(initializer, "load_llms_config", failing_load)

    with pytest.raises(acp.RequestError):
        await _build_config_options(_make_context(tmp_path))


def test_build_modes_should_not_expose_ineffective_semi_active_mode(tmp_path) -> None:
    context = _make_context(tmp_path)
    context.approval_mode = ApprovalMode.SEMI_ACTIVE

    modes = _build_modes(context)

    assert modes.current_mode_id == ApprovalMode.ACTIVE.value
    assert [mode.id for mode in modes.available_modes] == [
        ApprovalMode.ACTIVE.value,
        ApprovalMode.AGGRESSIVE.value,
    ]


@pytest.mark.asyncio
async def test_set_session_mode_should_reject_semi_active_mode(tmp_path) -> None:
    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path)
    agent._sessions["s1"] = _SessionEntry(session=AcpSession(context), context=context)

    with pytest.raises(acp.RequestError):
        await agent.set_session_mode("s1", ApprovalMode.SEMI_ACTIVE.value)


class _QueueingRunner:
    started: asyncio.Event
    release: asyncio.Event
    received: list[str]

    def __init__(self, *, session: Any, **_kwargs: Any) -> None:
        self._session = session

    async def run(self, user_text: str) -> None:
        _QueueingRunner.received.append(user_text)
        _QueueingRunner.started.set()
        if user_text == "first task":
            await _QueueingRunner.release.wait()


@pytest.mark.asyncio
async def test_prompt_should_reject_concurrent_prompt_for_same_session(monkeypatch, tmp_path) -> None:
    graph_context = _FakeGraphContext()
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: graph_context,
    )
    monkeypatch.setattr("msagent.cli.acp.server.AcpTurnRunner", _QueueingRunner)
    _QueueingRunner.started = asyncio.Event()
    _QueueingRunner.release = asyncio.Event()
    _QueueingRunner.received = []

    async def fake_create(cls, **kwargs: Any) -> Context:
        return _make_context(Path(kwargs["working_dir"]))

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))

    agent = MsagentAcpAgent(_make_args(agent="Profiler"))
    session_id = (await agent.new_session(str(tmp_path))).session_id
    first_prompt = asyncio.create_task(agent.prompt(session_id, [{"type": "text", "text": "first task"}]))
    await _QueueingRunner.started.wait()

    with pytest.raises(acp.RequestError):
        await agent.prompt(session_id, [{"type": "text", "text": "second task"}])

    _QueueingRunner.release.set()
    first_response = await asyncio.wait_for(first_prompt, timeout=5)

    assert _QueueingRunner.received == ["first task"]
    assert first_response.stop_reason == "end_turn"


class _FakeGraphStack:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_close_session_should_defer_dispose_when_turn_ignores_cancel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("msagent.cli.acp.server._SESSION_CLOSE_TIMEOUT_S", 0.01)
    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path)
    entry = _SessionEntry(session=AcpSession(context), context=context)
    stack = _FakeGraphStack()
    entry.graph_stack = stack
    agent._sessions["s1"] = entry

    started = asyncio.Event()

    async def stubborn_turn() -> None:
        async with entry.prompt_lock:
            entry.active_task = asyncio.current_task()
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)

    turn = asyncio.create_task(stubborn_turn())
    await started.wait()

    await agent.close_session("s1")

    assert stack.closed is False
    await turn
    await asyncio.sleep(0)
    assert entry.dispose_task is not None
    await entry.dispose_task
    assert stack.closed is True


@pytest.mark.asyncio
async def test_dispose_all_should_wait_for_cancelled_turn_before_dispose(tmp_path) -> None:
    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path)
    entry = _SessionEntry(session=AcpSession(context), context=context)
    stack = _FakeGraphStack()
    entry.graph_stack = stack
    agent._sessions["s1"] = entry
    started = asyncio.Event()
    finished = asyncio.Event()

    async def turn() -> None:
        async with entry.prompt_lock:
            entry.active_task = asyncio.current_task()
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                finished.set()

    task = asyncio.create_task(turn())
    await started.wait()

    await agent._dispose_all()

    assert finished.is_set()
    assert stack.closed is True
    assert agent._sessions == {}
    task.cancel()


@pytest.mark.asyncio
async def test_load_session_should_close_replaced_entry(monkeypatch, tmp_path, patch_state_dir) -> None:
    """A prompt queued on the replaced entry must not keep running afterwards."""
    patch_state_dir(tmp_path)
    monkeypatch.setattr(
        "msagent.cli.bootstrap.initializer.initializer.get_graph",
        lambda **_kwargs: _FakeGraphContext(),
    )

    async def fake_create(cls, **kwargs: Any) -> Context:
        return Context(
            agent="Profiler",
            model="test-model",
            thread_id=kwargs["thread_id"] or "generated",
            working_dir=Path(kwargs["working_dir"]),
            state_dir=tmp_path,
            approval_mode=kwargs["approval_mode"],
            recursion_limit=25,
            audit_log_enabled=False,
        )

    async def fake_read_messages(self: Any, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        return True, []

    async def fake_replay(*, conn: Any, session_id: str, messages: list[Any]) -> None:
        return None

    monkeypatch.setattr("msagent.cli.core.context.Context.create", classmethod(fake_create))
    monkeypatch.setattr(MsagentAcpAgent, "_read_checkpoint_messages", fake_read_messages)
    monkeypatch.setattr("msagent.cli.acp.server.replay_session_history", fake_replay)

    agent = MsagentAcpAgent(_make_args())
    agent._conn = object()
    old_entry = agent._build_entry(_make_context(tmp_path, thread_id="thread-1"))
    agent._sessions["thread-1"] = old_entry

    await agent.load_session(str(tmp_path), "thread-1")

    assert old_entry.closed is True
    assert agent._sessions["thread-1"] is not old_entry


@pytest.mark.asyncio
async def test_set_config_option_should_roll_back_when_rebuild_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MSAGENT_HOME", str(tmp_path))

    async def fake_load(_working_dir: Path) -> SimpleNamespace:
        return _fake_llms("alias-a", "alias-b")

    monkeypatch.setattr(initializer, "load_llms_config", fake_load)

    updates: list[Any] = []

    class _Conn:
        async def session_update(self, session_id: str, update: Any, **_kwargs: Any) -> None:
            updates.append(update)

    agent = MsagentAcpAgent(_make_args())
    context = _make_context(tmp_path, model="alias-a")
    entry = agent._build_entry(context)
    agent._sessions[context.thread_id] = entry
    agent._conn = _Conn()

    async def broken_create(*_args: Any, **_kwargs: Any) -> Context:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(agent, "_create_context", broken_create)

    await agent.set_config_option(context.thread_id, "model", "alias-b-model")
    assert entry.rebuild_task is not None
    with pytest.raises(RuntimeError):
        await entry.rebuild_task
    for _ in range(20):
        await asyncio.sleep(0)
        if updates:
            break

    assert entry.pending_model is None
    assert entry.context.model == "alias-a"
    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, ConfigOptionUpdate)
    assert update.config_options[0].current_value == "alias-a-model"

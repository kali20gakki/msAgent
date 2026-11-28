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

"""Unit tests for the ACP graph-to-ACP stream mapping."""

from __future__ import annotations

from typing import Any

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.configs import ApprovalMode
from msagent.cli.acp.mapping import AcpTurnRunner, replay_session_history
from msagent.cli.acp.server import AcpSession


class _FakeConn:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append(update)


class _FakeGraph:
    def __init__(self, script: list[list[tuple]]) -> None:
        self.script = script
        self.inputs: list[Any] = []

    async def astream(self, current_input: Any, config: Any, context: Any = None, **_kwargs: Any):
        self.inputs.append(current_input)
        index = len(self.inputs) - 1
        for chunk in self.script[index]:
            yield chunk


@pytest.fixture
def stub_initializer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def fake_refresh(**_kwargs: Any) -> list:
        return []

    async def fake_load_user_memory(_working_dir: Any) -> str:
        return ""

    monkeypatch.setattr(initializer, "refresh_cached_skills", fake_refresh)
    monkeypatch.setattr(initializer, "load_user_memory", fake_load_user_memory)
    monkeypatch.setattr(initializer, "cached_mcp_server_names", [])
    monkeypatch.setattr(initializer, "cached_tools_in_catalog", [])
    monkeypatch.setattr(initializer, "cached_agent_skills", [])


def _make_session(tmp_path) -> AcpSession:
    context = Context(
        agent="Profiler",
        model="test-model",
        thread_id="thread-1",
        working_dir=tmp_path,
        state_dir=tmp_path,
        approval_mode=ApprovalMode.ACTIVE,
        recursion_limit=25,
        audit_log_enabled=False,
    )
    return AcpSession(context)


@pytest.mark.asyncio
async def test_text_and_tool_calls_forwarded_to_client(stub_initializer, tmp_path) -> None:
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "execute", "args": '{"command": "ls"}', "id": "t1", "index": 0}],
    )
    graph = _FakeGraph(
        [
            [
                ("", "messages", (tool_call_chunk, {})),
                ("", "messages", (AIMessageChunk(content="done"), {})),
                (
                    "",
                    "updates",
                    {
                        "node": {
                            "messages": [ToolMessage(content="ok", tool_call_id="t1")],
                            "x": 1,
                        }
                    },
                ),
            ]
        ]
    )
    conn = _FakeConn()
    session = _make_session(tmp_path)
    runner = AcpTurnRunner(conn=conn, session_id="s1", session=session, graph=graph)

    await runner.run("hello")

    assert isinstance(conn.updates[0], ToolCallStart)
    assert conn.updates[0].tool_call_id == "t1"
    assert conn.updates[0].name == "execute"
    assert conn.updates[0].status == "pending"
    assert isinstance(conn.updates[1], AgentMessageChunk)
    assert conn.updates[1].content.text == "done"
    assert isinstance(conn.updates[2], ToolCallProgress)
    assert conn.updates[2].tool_call_id == "t1"
    assert conn.updates[2].status == "completed"


@pytest.mark.asyncio
async def test_cancelled_turn_should_report_pending_tool_calls_as_failed(stub_initializer, tmp_path) -> None:
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "execute", "args": '{"command": "ls"}', "id": "t9", "index": 0}],
    )
    graph = _FakeGraph([[("", "messages", (tool_call_chunk, {}))]])
    conn = _FakeConn()
    runner = AcpTurnRunner(conn=conn, session_id="s1", session=_make_session(tmp_path), graph=graph)

    await runner.run("hello")
    await runner.abort_pending_tool_calls()
    await runner.abort_pending_tool_calls()

    assert isinstance(conn.updates[0], ToolCallStart)
    assert isinstance(conn.updates[1], ToolCallProgress)
    assert conn.updates[1].tool_call_id == "t9"
    assert conn.updates[1].status == "failed"
    assert len(conn.updates) == 2


@pytest.mark.asyncio
async def test_hidden_tool_completion_not_forwarded_to_client(stub_initializer, tmp_path) -> None:
    todo_call = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "write_todos", "args": "{}", "id": "t2", "index": 0}],
    )
    graph = _FakeGraph(
        [
            [
                ("", "messages", (todo_call, {})),
                (
                    "",
                    "updates",
                    {"node": {"messages": [ToolMessage(content="saved", tool_call_id="t2")]}},
                ),
            ]
        ]
    )
    conn = _FakeConn()
    runner = AcpTurnRunner(conn=conn, session_id="s1", session=_make_session(tmp_path), graph=graph)

    await runner.run("hello")

    assert conn.updates == []


@pytest.mark.asyncio
async def test_interrupt_resumes_with_command(stub_initializer, monkeypatch, tmp_path) -> None:
    interrupt = Interrupt(value={"question": "Approve?", "options": ["approve", "reject"]}, id="int-1")
    graph = _FakeGraph(
        [
            [("", "updates", {"__interrupt__": [interrupt]})],
            [("", "messages", (AIMessageChunk(content="after"), {}))],
        ]
    )

    class _FakeBackend:
        def __init__(self, conn: Any, session_id: str, **_kwargs: Any) -> None:
            self.conn = conn
            self.session_id = session_id

        async def ask(self, **_kwargs: Any) -> str:
            return "approve"

    monkeypatch.setattr("msagent.cli.acp.mapping.AcpApprovalBackend", _FakeBackend)

    conn = _FakeConn()
    session = _make_session(tmp_path)
    runner = AcpTurnRunner(conn=conn, session_id="s1", session=session, graph=graph)

    await runner.run("hello")

    assert isinstance(graph.inputs[1], Command)
    assert graph.inputs[1].resume == {"int-1": "approve"}
    assert conn.updates[0].content.text == "after"


@pytest.mark.asyncio
async def test_replay_session_history_should_map_messages_to_session_updates() -> None:
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "ls"},
                    "id": "t1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ok", tool_call_id="t1", name="execute"),
        AIMessage(content=[{"type": "text", "text": "done"}]),
    ]
    conn = _FakeConn()

    await replay_session_history(conn=conn, session_id="s1", messages=messages)

    assert isinstance(conn.updates[0], UserMessageChunk)
    assert conn.updates[0].content.text == "hi"
    assert isinstance(conn.updates[1], ToolCallStart)
    assert conn.updates[1].tool_call_id == "t1"
    assert conn.updates[1].status == "completed"
    assert isinstance(conn.updates[2], AgentMessageChunk)
    assert conn.updates[2].content.text == "done"


@pytest.mark.asyncio
async def test_replay_session_history_should_mark_failed_tool_calls() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "execute", "args": {}, "id": "t1", "type": "tool_call"}],
        ),
        ToolMessage(content="boom", tool_call_id="t1", name="execute", status="error"),
    ]
    conn = _FakeConn()

    await replay_session_history(conn=conn, session_id="s1", messages=messages)

    assert len(conn.updates) == 1
    assert isinstance(conn.updates[0], ToolCallStart)
    assert conn.updates[0].status == "failed"


@pytest.mark.asyncio
async def test_replay_session_history_should_mark_orphan_tool_calls_failed() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "execute", "args": {}, "id": "t2", "type": "tool_call"}],
        ),
    ]
    conn = _FakeConn()

    await replay_session_history(conn=conn, session_id="s1", messages=messages)

    assert len(conn.updates) == 1
    assert isinstance(conn.updates[0], ToolCallStart)
    assert conn.updates[0].status == "failed"


@pytest.mark.asyncio
async def test_replay_session_history_should_skip_hidden_tools_and_empty_text() -> None:
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "write_todos", "args": {}, "id": "t2", "type": "tool_call"}],
        ),
        ToolMessage(content="saved", tool_call_id="t2", name="write_todos"),
        AIMessage(content=""),
    ]
    conn = _FakeConn()

    await replay_session_history(conn=conn, session_id="s1", messages=messages)

    assert [type(update) for update in conn.updates] == [UserMessageChunk]


def test_extract_reasoning_delta_should_cover_provider_shapes() -> None:
    assert (
        AcpTurnRunner._extract_reasoning_delta(
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "deepseek"})
        )
        == "deepseek"
    )
    assert (
        AcpTurnRunner._extract_reasoning_delta(
            AIMessageChunk(content="", additional_kwargs={"thinking": {"text": "bedrock"}})
        )
        == "bedrock"
    )
    assert (
        AcpTurnRunner._extract_reasoning_delta(AIMessageChunk(content=[{"type": "thinking", "thinking": "claude"}]))
        == "claude"
    )
    assert (
        AcpTurnRunner._extract_reasoning_delta(
            AIMessageChunk(content=[{"type": "reasoning", "summary": [{"text": "a"}, {"text": "b"}]}])
        )
        == "a\nb"
    )
    assert (
        AcpTurnRunner._extract_reasoning_delta(
            AIMessageChunk(content=[{"type": "reasoning_content", "reasoning_content": "rc"}])
        )
        == "rc"
    )
    assert AcpTurnRunner._extract_reasoning_delta(AIMessageChunk(content="plain")) == ""


@pytest.mark.asyncio
async def test_reasoning_delta_forwarded_as_thought_chunk(stub_initializer, tmp_path) -> None:
    graph = _FakeGraph(
        [
            [
                (
                    "",
                    "messages",
                    (
                        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "hmm"}),
                        {},
                    ),
                ),
                ("", "messages", (AIMessageChunk(content="answer"), {})),
            ]
        ]
    )
    conn = _FakeConn()
    runner = AcpTurnRunner(conn=conn, session_id="s1", session=_make_session(tmp_path), graph=graph)

    await runner.run("hello")

    assert isinstance(conn.updates[0], AgentThoughtChunk)
    assert conn.updates[0].content.text == "hmm"
    assert isinstance(conn.updates[1], AgentMessageChunk)
    assert conn.updates[1].content.text == "answer"

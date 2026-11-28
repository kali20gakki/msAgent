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

"""Unit tests for the ACP approval backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome

from msagent.cli.acp.approval import AcpApprovalBackend
from msagent.cli.handlers.interrupts import ALWAYS_ALLOW_PROJECT_SCRIPT, SWITCH_TO_AUTO


class _FakeConn:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    async def request_permission(self, *, session_id: str, tool_call: object, options: list) -> object:
        self.calls.append({"session_id": session_id, "tool_call": tool_call, "options": options})
        return SimpleNamespace(outcome=self.outcome)


@pytest.mark.asyncio
async def test_allowed_outcome_returns_selected_option_id() -> None:
    conn = _FakeConn(AllowedOutcome(option_id="always_approve", outcome="selected"))
    backend = AcpApprovalBackend(conn=conn, session_id="s1")

    selected_option_id = await backend.ask(
        question="Tool execution requires approval.\nTool: execute",
        options=["approve", "always_approve", "reject", "always_reject"],
        tool_name="execute",
        tool_args={"command": "ls"},
    )

    assert selected_option_id == "always_approve"
    call = conn.calls[0]
    assert call["session_id"] == "s1"
    kinds = {option.option_id: option.kind for option in call["options"]}
    assert kinds == {
        "approve": "allow_once",
        "always_approve": "allow_always",
        "reject": "reject_once",
        "always_reject": "reject_always",
    }
    assert call["tool_call"].name == "execute"
    assert call["tool_call"].raw_input["tool_args"] == {"command": "ls"}


@pytest.mark.asyncio
async def test_denied_outcome_returns_none() -> None:
    conn = _FakeConn(DeniedOutcome(outcome="cancelled"))
    backend = AcpApprovalBackend(conn=conn, session_id="s1")

    selected_option_id = await backend.ask(question="Approval required", options=["approve", "reject"])

    assert selected_option_id is None


@pytest.mark.asyncio
async def test_option_labels_override_default_names() -> None:
    conn = _FakeConn(AllowedOutcome(option_id="always_approve", outcome="selected"))
    backend = AcpApprovalBackend(conn=conn, session_id="s1")

    await backend.ask(
        question="External directory",
        options=["approve", "always_approve"],
        option_labels={"always_approve": "Always allow this directory for this project"},
    )

    names = [option.name for option in conn.calls[0]["options"]]
    assert names == ["Allow", "Always allow this directory for this project"]


@pytest.mark.asyncio
async def test_non_policy_options_default_to_allow_once() -> None:
    conn = _FakeConn(AllowedOutcome(option_id=SWITCH_TO_AUTO, outcome="selected"))
    backend = AcpApprovalBackend(conn=conn, session_id="s1")

    selected_option_id = await backend.ask(question="Choose mode", options=[SWITCH_TO_AUTO])

    assert selected_option_id == SWITCH_TO_AUTO
    option = conn.calls[0]["options"][0]
    assert option.kind == "allow_once"
    assert option.name == "Switch to Auto Mode"


@pytest.mark.asyncio
async def test_always_allow_project_script_should_map_to_allow_always() -> None:
    conn = _FakeConn(AllowedOutcome(option_id=ALWAYS_ALLOW_PROJECT_SCRIPT, outcome="selected"))
    backend = AcpApprovalBackend(conn=conn, session_id="s1")

    selected_option_id = await backend.ask(question="Choose", options=[ALWAYS_ALLOW_PROJECT_SCRIPT])

    assert selected_option_id == ALWAYS_ALLOW_PROJECT_SCRIPT
    assert conn.calls[0]["options"][0].kind == "allow_always"


@pytest.mark.asyncio
async def test_permission_request_should_reuse_pending_tool_call_id() -> None:
    conn = _FakeConn(AllowedOutcome(option_id="approve", outcome="selected"))
    backend = AcpApprovalBackend(
        conn=conn,
        session_id="s1",
        resolve_tool_call_id=lambda name: "call-1" if name == "execute" else None,
    )

    await backend.ask(question="Approval required", options=["approve", "reject"], tool_name="execute")

    assert conn.calls[0]["tool_call"].tool_call_id == "call-1"

    unknown_conn = _FakeConn(AllowedOutcome(option_id="approve", outcome="selected"))
    unknown_backend = AcpApprovalBackend(
        conn=unknown_conn,
        session_id="s1",
        resolve_tool_call_id=lambda _name: None,
    )

    await unknown_backend.ask(question="Approval required", options=["approve", "reject"], tool_name="execute")

    assert unknown_conn.calls[0]["tool_call"].tool_call_id.startswith("permission-")


@pytest.mark.asyncio
async def test_long_approval_payload_should_be_truncated() -> None:
    conn = _FakeConn(AllowedOutcome(option_id="approve", outcome="selected"))
    backend = AcpApprovalBackend(conn=conn, session_id="s1")
    content = "x" * 10000
    question = "write_file Requires Approval\n" + "y" * 10000

    await backend.ask(
        question=question,
        options=["approve", "reject"],
        tool_name="write_file",
        tool_args={"path": "a.py", "content": content},
    )

    tool_call = conn.calls[0]["tool_call"]
    assert tool_call.title == "write_file Requires Approval"
    raw_input = tool_call.raw_input
    assert raw_input["tool_args"]["path"] == "a.py"
    assert len(raw_input["tool_args"]["content"]) < len(content)
    assert "truncated" in raw_input["tool_args"]["content"]
    assert len(raw_input["question"]) < len(question)

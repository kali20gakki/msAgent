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

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.handlers import interrupts as interrupts_module
from msagent.cli.handlers.interrupts import InterruptHandler
from msagent.configs import ToolApprovalConfig


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
        prompt=SimpleNamespace(mode_change_callback=None),
    )


@pytest.mark.asyncio
async def test_hitl_uses_decision_rules_to_auto_approve_non_risky_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()

    prompt_called = False

    async def _never_prompt(**_kwargs):
        nonlocal prompt_called
        prompt_called = True
        return "reject"

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", lambda _cfg: None)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _never_prompt)

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [{"name": "execute", "args": {"command": "echo hello"}}],
            "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
        }
    )

    assert result == {"decisions": [{"type": "approve"}]}
    assert prompt_called is False


@pytest.mark.asyncio
async def test_hitl_persists_always_reject_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()
    saved = {"called": False}

    async def _prompt(**_kwargs):
        return "always_reject"

    def _save(updated: ToolApprovalConfig) -> None:
        saved["called"] = True
        assert updated.resolve_decision("execute", {"command": "rm -rf /tmp/demo"}) == "always_reject"

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", _save)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [{"name": "execute", "args": {"command": "rm -rf /tmp/demo"}}],
            "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
        }
    )

    assert result == {"decisions": [{"type": "reject", "message": "Rejected by local approval policy."}]}
    assert saved["called"] is True


@pytest.mark.asyncio
async def test_interrupt_handler_returns_none_for_empty_interrupt_list(
    tmp_path: Path,
) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    result = await handler.handle([])
    assert result is None


@pytest.mark.asyncio
async def test_interrupt_handler_returns_none_for_unknown_payload_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    monkeypatch.setattr(interrupts_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(interrupts_module.console, "print", lambda *_args, **_kwargs: None)

    fake_interrupt = SimpleNamespace(id="int-1", value="just a string")
    result = await handler._get_choice(fake_interrupt)
    assert result is None


@pytest.mark.asyncio
async def test_hitl_returns_none_when_action_requests_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    monkeypatch.setattr(handler, "_load_approval_config", lambda: ToolApprovalConfig())

    result = await handler._get_hitl_decisions({"action_requests": [], "review_configs": []})
    assert result is None


@pytest.mark.asyncio
async def test_hitl_persists_always_approve_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()
    saved = {"called": False}

    async def _prompt(**_kwargs):
        return "always_approve"

    def _save(updated: ToolApprovalConfig) -> None:
        saved["called"] = True

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", _save)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [{"name": "dangerous_tool", "args": {"target": "/etc/passwd"}}],
            "review_configs": [{"action_name": "dangerous_tool", "allowed_decisions": ["approve", "reject"]}],
        }
    )

    assert result == {"decisions": [{"type": "approve"}]}
    assert saved["called"] is True


@pytest.mark.asyncio
async def test_hitl_auto_rejects_when_policy_is_always_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()

    prompt_called = False

    async def _never_prompt(**_kwargs):
        nonlocal prompt_called
        prompt_called = True
        return "approve"

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", lambda _cfg: None)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _never_prompt)

    cfg.prepend_decision_rule(tool_name="execute", tool_args=None, decision="always_reject")

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [{"name": "execute", "args": {"command": "rm -rf /"}}],
            "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
        }
    )

    assert result == {"decisions": [{"type": "reject", "message": "Rejected by local approval policy."}]}
    assert prompt_called is False


def test_selection_to_decision_maps_approve_and_reject() -> None:
    assert InterruptHandler._selection_to_decision("approve") == {"type": "approve"}
    assert InterruptHandler._selection_to_decision("reject") == {"type": "reject"}
    assert InterruptHandler._selection_to_decision("unknown") == {"type": "approve"}


@pytest.mark.asyncio
async def test_hitl_handles_non_dict_tool_args_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()

    async def _prompt(**_kwargs):
        return "approve"

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", lambda _cfg: None)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [{"name": "search", "args": "not a dict"}],
            "review_configs": [{"action_name": "search", "allowed_decisions": ["approve", "reject"]}],
        }
    )

    assert result == {"decisions": [{"type": "approve"}]}


@pytest.mark.asyncio
async def test_hitl_prompt_returns_none_cancels_entire_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()

    async def _prompt(**_kwargs):
        return None

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", lambda _cfg: None)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [{"name": "dangerous_tool", "args": {"target": "/root"}}],
            "review_configs": [{"action_name": "dangerous_tool", "allowed_decisions": ["approve", "reject"]}],
        }
    )

    assert result is None


@pytest.mark.asyncio
async def test_interrupt_handler_returns_dict_for_multiple_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    monkeypatch.setattr(interrupts_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(interrupts_module.console, "print", lambda *_args, **_kwargs: None)

    async def _fake_get_choice(interrupt):
        return "approve"

    monkeypatch.setattr(handler, "_get_choice", _fake_get_choice)

    interrupts = [
        SimpleNamespace(id="int-1", value={"question": "Q1", "options": ["approve"]}),
        SimpleNamespace(id="int-2", value={"question": "Q2", "options": ["reject"]}),
    ]

    result = await handler.handle(interrupts)
    assert result == {"int-1": "approve", "int-2": "approve"}


@pytest.mark.asyncio
async def test_interrupt_handler_returns_none_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    monkeypatch.setattr(interrupts_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(interrupts_module.console, "print", lambda *_args, **_kwargs: None)

    async def _failing_get_choice(interrupt):
        raise RuntimeError("boom")

    monkeypatch.setattr(handler, "_get_choice", _failing_get_choice)

    result = await handler.handle([SimpleNamespace(id="int-1", value="something")])
    assert result is None

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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
            "action_requests": [
                {"name": "execute", "args": {"command": "echo hello"}}
            ],
            "review_configs": [
                {"action_name": "execute", "allowed_decisions": ["approve", "reject"]}
            ],
        }
    )

    assert result == {"decisions": [{"type": "approve"}]}
    assert prompt_called is False


@pytest.mark.asyncio
async def test_hitl_persists_always_reject_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = InterruptHandler(_build_session(tmp_path))
    cfg = ToolApprovalConfig()
    saved = {"called": False}

    async def _prompt(**_kwargs):
        return "always_reject"

    def _save(updated: ToolApprovalConfig) -> None:
        saved["called"] = True
        assert (
            updated.resolve_decision("execute", {"command": "rm -rf /tmp/demo"})
            == "always_reject"
        )

    monkeypatch.setattr(handler, "_load_approval_config", lambda: cfg)
    monkeypatch.setattr(handler, "_save_approval_config", _save)
    monkeypatch.setattr(handler, "_prompt_hitl_decision", _prompt)

    result = await handler._get_hitl_decisions(
        {
            "action_requests": [
                {"name": "execute", "args": {"command": "rm -rf /tmp/demo"}}
            ],
            "review_configs": [
                {"action_name": "execute", "allowed_decisions": ["approve", "reject"]}
            ],
        }
    )

    assert result == {
        "decisions": [
            {"type": "reject", "message": "Rejected by local approval policy."}
        ]
    }
    assert saved["called"] is True

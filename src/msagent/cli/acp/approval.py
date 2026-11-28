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

"""Approval backends bridging interrupt prompts to interactive clients."""

from __future__ import annotations

from typing import Any, Callable, Protocol, cast
from uuid import uuid4

from acp.schema import AllowedOutcome, PermissionOption, ToolCallUpdate

from msagent.cli.handlers.interrupts import ALWAYS_ALLOW_PROJECT_SCRIPT, SWITCH_TO_AUTO

_OPTION_KIND_MAP = {
    "approve": "allow_once",
    "always_approve": "allow_always",
    ALWAYS_ALLOW_PROJECT_SCRIPT: "allow_always",
    "reject": "reject_once",
    "always_reject": "reject_always",
}

_OPTION_NAME_MAP = {
    "approve": "Allow",
    "always_approve": "Always Allow",
    "reject": "Reject",
    "always_reject": "Always Reject",
    SWITCH_TO_AUTO: "Switch to Auto Mode",
}

_TEXT_LIMIT = 4000
_TITLE_LIMIT = 200


def _truncate_text(text: str, limit: int) -> str:
    """Clip approval payloads so permission cards stay renderable."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated, {len(text)} characters)"


def _truncate_args(value: Any) -> Any:
    """Recursively clip long strings inside tool arguments."""
    if isinstance(value, str):
        return _truncate_text(value, _TEXT_LIMIT)
    if isinstance(value, dict):
        return {key: _truncate_args(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_args(item) for item in value]
    return value


class ApprovalBackend(Protocol):
    """Collect one option selection from a human."""

    async def ask(
        self,
        *,
        question: str,
        options: list[str],
        option_labels: dict[str, str] | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> str | None:
        """Return the selected option string, or None when the request is cancelled."""
        ...


class AcpApprovalBackend:
    """Forward approval prompts to the ACP client via request_permission."""

    def __init__(
        self,
        conn: Any,
        session_id: str,
        resolve_tool_call_id: Callable[[str], str | None] | None = None,
    ) -> None:
        self._conn = conn
        self._session_id = session_id
        self._resolve_tool_call_id = resolve_tool_call_id

    async def ask(
        self,
        *,
        question: str,
        options: list[str],
        option_labels: dict[str, str] | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> str | None:
        labels = option_labels or {}
        permission_options = [
            PermissionOption(
                option_id=option,
                name=labels.get(option) or _OPTION_NAME_MAP.get(option, option),
                kind=cast(Any, _OPTION_KIND_MAP.get(option, "allow_once")),
            )
            for option in options
        ]
        tool_call_id = self._resolve_tool_call_id(tool_name) if tool_name and self._resolve_tool_call_id else None
        title = question.split("\n", 1)[0] if question else "Approval required"
        tool_call_update = ToolCallUpdate(
            tool_call_id=tool_call_id or f"permission-{uuid4().hex[:12]}",
            title=_truncate_text(title, _TITLE_LIMIT),
            name=tool_name,
            raw_input={
                "question": _truncate_text(question, _TEXT_LIMIT),
                "tool_args": _truncate_args(tool_args or {}),
            },
        )
        response = await self._conn.request_permission(
            session_id=self._session_id,
            tool_call=tool_call_update,
            options=permission_options,
        )
        if isinstance(response.outcome, AllowedOutcome):
            return response.outcome.option_id
        return None

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

"""Observe main-agent `task` tool calls and record subagent delegation outcomes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from msagent.audit.events import format_audit_timestamp
from msagent.audit.protocol import parse_completion_output, parse_delegation_input
from msagent.audit.writer import AuditWriter

TASK_TOOL_NAME = "task"
MAX_TASK_DESCRIPTION_LENGTH = 2000


@dataclass(slots=True)
class _PendingDelegation:
    subagent_type: str
    task_description: str | None
    start_time: str
    monotonic_started: float


class SubagentAuditTracker:
    """Record which subagent was delegated to and what result was returned."""

    def __init__(self, writer: AuditWriter) -> None:
        self._writer = writer
        self._run_id = ""
        self._pending: dict[str, _PendingDelegation] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def begin_run(
        self,
        run_id: str,
        *,
        user_message: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Start tracking a new user turn."""
        self._run_id = run_id
        self._pending.clear()
        self._writer.begin_run(run_id)
        if user_message:
            self._writer.emit_user_turn(message=user_message, prompt=prompt)

    def observe(self, message: Any, *, namespace: tuple[Any, ...]) -> None:
        """Inspect a graph message and emit audit events on the main agent thread."""
        if not self._writer.enabled or not self._run_id:
            return
        if namespace:
            return

        if isinstance(message, AIMessage):
            self._observe_ai_message(message)
        elif isinstance(message, ToolMessage):
            self._observe_tool_message(message)

    def _observe_ai_message(self, message: AIMessage) -> None:
        for tool_call in _iter_tool_calls(message):
            if _tool_name(tool_call) != TASK_TOOL_NAME:
                continue
            args = _tool_args(tool_call)
            delegation_id = _tool_call_id(tool_call)
            if not delegation_id:
                continue

            subagent_type = _coerce_str(args.get("subagent_type") or args.get("subagentType"))
            if not subagent_type:
                continue

            task_description = _truncate(
                _coerce_str(args.get("description")),
                MAX_TASK_DESCRIPTION_LENGTH,
            )
            self._pending[delegation_id] = _PendingDelegation(
                subagent_type=subagent_type,
                task_description=task_description,
                start_time=format_audit_timestamp(),
                monotonic_started=time.monotonic(),
            )

    def _observe_tool_message(self, message: ToolMessage) -> None:
        if _tool_message_name(message) != TASK_TOOL_NAME:
            return

        delegation_id = str(getattr(message, "tool_call_id", "") or "")
        if not delegation_id:
            return

        pending = self._pending.pop(delegation_id, None)
        subagent_type = pending.subagent_type if pending else "unknown"
        start_time = pending.start_time if pending else format_audit_timestamp()
        duration_ms = None
        if pending is not None:
            duration_ms = int((time.monotonic() - pending.monotonic_started) * 1000)

        task_description = pending.task_description if pending else None
        result_text = _extract_tool_message_text(message)
        status = _resolve_status(message, result_text)
        end_time = format_audit_timestamp()

        input_parse = parse_delegation_input(
            task_description,
            expected_subagent_type=subagent_type,
        )
        output_parse = parse_completion_output(
            result_text,
            expected_subagent_type=subagent_type,
        )

        self._writer.emit_delegation(
            delegation_id=delegation_id,
            subagent_type=subagent_type,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status=status,
            input_parse=input_parse,
            output_parse=output_parse,
            task_description_raw=task_description if not input_parse.valid else None,
            result_raw=result_text if not output_parse.valid else None,
        )


def _iter_tool_calls(message: AIMessage) -> list[Any]:
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    raw_tool_calls = additional_kwargs.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        tool_calls.extend(raw_tool_calls)
    return tool_calls


def _tool_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "")
    return str(getattr(tool_call, "name", "") or "")


def _tool_args(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        args = tool_call.get("args")
    else:
        args = getattr(tool_call, "args", None)
    return args if isinstance(args, dict) else {}


def _tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id")
    else:
        call_id = getattr(tool_call, "id", None)
    return str(call_id) if call_id else None


def _tool_message_name(message: ToolMessage) -> str:
    return str(getattr(message, "name", "") or "")


def _extract_tool_message_text(message: ToolMessage) -> str:
    content = message.content
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        text = "".join(parts).strip()
        if text:
            return text

    short_content = getattr(message, "short_content", None)
    if isinstance(short_content, str) and short_content.strip():
        return short_content.strip()
    return str(content or "").strip()


def _resolve_status(message: ToolMessage, result_text: str) -> str:
    status = getattr(message, "status", None)
    if status == "error":
        return "failed"
    lowered = result_text.lower()
    if lowered.startswith("we cannot invoke subagent"):
        return "failed"
    return "ok"


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate(value: str | None, max_length: int) -> str | None:
    if value is None:
        return None
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."

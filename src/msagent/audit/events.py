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

"""Structured audit events for session and subagent activity."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from msagent.audit.user_interaction import truncate_audit_text

_DELEGATION_FIELD_ORDER: tuple[str, ...] = (
    "agent_name",
    "event",
    "subagent_type",
    "run_id",
    "delegation_id",
    "start_time",
    "end_time",
    "duration_ms",
    "status",
    "input_valid",
    "output_valid",
    "input_errors",
    "output_errors",
    "input",
    "output",
    "error",
    "task_description_raw",
    "result_raw",
)

_USER_TURN_FIELD_ORDER: tuple[str, ...] = (
    "agent_name",
    "event",
    "run_id",
    "start_time",
    "prompt",
    "message",
)

_USER_RESPONSE_FIELD_ORDER: tuple[str, ...] = (
    "agent_name",
    "event",
    "run_id",
    "start_time",
    "kind",
    "prompt",
    "options",
    "response",
    "context",
)


class AuditEventType(StrEnum):
    """Supported audit event types."""

    USER_TURN = "user.turn"
    USER_RESPONSE = "user.response"
    SUBAGENT_DELEGATION = "subagent.delegation"


def format_audit_timestamp(*, now: datetime | None = None) -> str:
    """Format audit timestamps as ``YYYY-MM-DD HH:MM:SS`` in local time."""
    moment = now or datetime.now()
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def omit_null_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Serialize audit payloads, omitting null fields."""
    return {key: value for key, value in data.items() if value is not None}


def serialize_event(payload: dict[str, Any], *, field_order: tuple[str, ...]) -> dict[str, Any]:
    """Serialize one audit event with stable field order, omitting null fields."""
    ordered: dict[str, Any] = {}
    for key in field_order:
        value = payload.get(key)
        if value is not None:
            ordered[key] = value
    return ordered


class AuditEvent(BaseModel):
    """Append-only audit record for one subagent delegation lifecycle."""

    agent_name: str
    event: AuditEventType
    subagent_type: str
    run_id: str
    delegation_id: str
    start_time: str
    end_time: str
    duration_ms: int | None = None
    status: str
    input_valid: bool | None = None
    output_valid: bool | None = None
    input_errors: list[str] | None = None
    output_errors: list[str] | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    task_description_raw: str | None = None
    result_raw: str | None = None

    @classmethod
    def delegation(
        cls,
        *,
        run_id: str,
        agent_name: str,
        delegation_id: str,
        subagent_type: str,
        start_time: str,
        end_time: str,
        duration_ms: int | None,
        status: str,
        input_valid: bool | None = None,
        output_valid: bool | None = None,
        input_errors: list[str] | None = None,
        output_errors: list[str] | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error_data: dict[str, Any] | None = None,
        task_description_raw: str | None = None,
        result_raw: str | None = None,
    ) -> AuditEvent:
        """Build a merged ``subagent.delegation`` audit record."""
        return cls(
            agent_name=agent_name,
            event=AuditEventType.SUBAGENT_DELEGATION,
            subagent_type=subagent_type,
            run_id=run_id,
            delegation_id=delegation_id,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status=status,
            input_valid=input_valid,
            output_valid=output_valid,
            input_errors=input_errors or None,
            output_errors=output_errors or None,
            input=input_data,
            output=output_data,
            error=error_data,
            task_description_raw=task_description_raw,
            result_raw=result_raw,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for JSONL storage with stable field order, omitting null fields."""
        return serialize_event(self.model_dump(mode="json"), field_order=_DELEGATION_FIELD_ORDER)


class UserTurnEvent(BaseModel):
    """Append-only audit record for one user message that starts a run."""

    agent_name: str
    event: AuditEventType = AuditEventType.USER_TURN
    run_id: str
    start_time: str
    message: str
    prompt: str | None = None

    @classmethod
    def create(
        cls,
        *,
        agent_name: str,
        run_id: str,
        message: str,
        prompt: str | None = None,
        start_time: str | None = None,
    ) -> UserTurnEvent:
        return cls(
            agent_name=agent_name,
            run_id=run_id,
            start_time=start_time or format_audit_timestamp(),
            message=truncate_audit_text(message),
            prompt=prompt,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return serialize_event(self.model_dump(mode="json"), field_order=_USER_TURN_FIELD_ORDER)


class UserResponseEvent(BaseModel):
    """Append-only audit record for one in-run interrupt response."""

    agent_name: str
    event: AuditEventType = AuditEventType.USER_RESPONSE
    run_id: str
    start_time: str
    kind: str
    prompt: str
    response: Any
    options: list[str] | None = None
    context: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        agent_name: str,
        run_id: str,
        kind: str,
        prompt: str,
        response: Any,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
        start_time: str | None = None,
    ) -> UserResponseEvent:
        return cls(
            agent_name=agent_name,
            run_id=run_id,
            start_time=start_time or format_audit_timestamp(),
            kind=kind,
            prompt=prompt,
            response=response,
            options=options,
            context=context,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return serialize_event(self.model_dump(mode="json"), field_order=_USER_RESPONSE_FIELD_ORDER)

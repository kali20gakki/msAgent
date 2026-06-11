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

"""Append-only JSONL writer for subagent audit events."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from msagent.audit.events import AuditEvent, UserResponseEvent, UserTurnEvent
from msagent.audit.protocol import ProtocolParseResult
from msagent.core.constants import CONFIG_AUDIT_DIR


def resolve_audit_log_enabled(agent_config: object | None) -> bool:
    """Resolve per-agent audit logging from agent YAML."""
    if agent_config is None:
        return False
    cfg = getattr(agent_config, "audit_log", None)
    if cfg is None:
        return False
    return bool(getattr(cfg, "enabled", False))


def build_audit_filename(*, agent_name: str, thread_id: str) -> str:
    """Build ``{agent-prefix}_{thread_id}.jsonl`` audit file name."""
    prefix = _sanitize_agent_prefix(agent_name)
    return f"{prefix}_{thread_id}.jsonl"


def _sanitize_agent_prefix(agent_name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "-", agent_name.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "agent"


class AuditWriter:
    """Append flat audit events to per-session JSONL files."""

    def __init__(
        self,
        *,
        working_dir: Path,
        thread_id: str,
        agent_name: str,
        enabled: bool | None = None,
    ) -> None:
        self.working_dir = working_dir.resolve()
        self.thread_id = thread_id
        self.agent_name = agent_name
        self.enabled = False if enabled is None else enabled
        self._audit_dir = self.working_dir / CONFIG_AUDIT_DIR
        self._path = self._audit_dir / build_audit_filename(
            agent_name=self.agent_name,
            thread_id=self.thread_id,
        )
        self._current_run_id: str | None = None

    def rebind(self, *, thread_id: str, agent_name: str | None = None) -> None:
        """Point the writer at a different conversation thread or agent."""
        self.thread_id = thread_id
        if agent_name is not None:
            self.agent_name = agent_name
        self._path = self._audit_dir / build_audit_filename(
            agent_name=self.agent_name,
            thread_id=self.thread_id,
        )
        self._current_run_id = None

    @property
    def path(self) -> Path:
        return self._path

    def begin_run(self, run_id: str) -> None:
        """Start a new user turn within the current session audit file."""
        if not self.enabled:
            return
        self._current_run_id = run_id

    def emit(self, event: AuditEvent) -> None:
        """Append one subagent delegation audit event when auditing is enabled."""
        self._append(event.to_json_dict())

    def emit_user_turn(self, *, message: str, prompt: str | None = None) -> None:
        """Record the user message that started the active run."""
        if not self._current_run_id:
            return
        self._append(
            UserTurnEvent.create(
                agent_name=self.agent_name,
                run_id=self._current_run_id,
                message=message,
                prompt=prompt,
            ).to_json_dict()
        )

    def emit_user_response(
        self,
        *,
        kind: str,
        prompt: str,
        response: Any,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record one in-run interrupt response on the active run."""
        if not self._current_run_id:
            return
        self._append(
            UserResponseEvent.create(
                agent_name=self.agent_name,
                run_id=self._current_run_id,
                kind=kind,
                prompt=prompt,
                response=response,
                options=options,
                context=context,
            ).to_json_dict()
        )

    def emit_delegation(
        self,
        *,
        delegation_id: str,
        subagent_type: str,
        start_time: str,
        end_time: str,
        duration_ms: int | None,
        status: str,
        input_parse: ProtocolParseResult,
        output_parse: ProtocolParseResult,
        task_description_raw: str | None = None,
        result_raw: str | None = None,
    ) -> None:
        """Record one completed subagent delegation on the active run."""
        if not self._current_run_id:
            return

        self.emit(
            AuditEvent.delegation(
                run_id=self._current_run_id,
                agent_name=self.agent_name,
                delegation_id=delegation_id,
                subagent_type=subagent_type,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                status=status,
                input_valid=input_parse.valid if input_parse.parsed else False,
                output_valid=output_parse.valid if output_parse.parsed else False,
                input_errors=input_parse.errors or None,
                output_errors=output_parse.errors or None,
                input_data=input_parse.input_data,
                output_data=output_parse.output_data,
                error_data=output_parse.error_data,
                task_description_raw=task_description_raw,
                result_raw=result_raw,
            )
        )

    def _append(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self._audit_dir.mkdir(parents=True, exist_ok=True)
        block = json.dumps(payload, ensure_ascii=False, indent=2)
        with self._path.open("a", encoding="utf-8") as handle:
            if self._path.stat().st_size > 0:
                handle.write("\n")
            handle.write(block)
            handle.write("\n")

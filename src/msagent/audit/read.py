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

"""Read and summarize flat subagent audit JSONL logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from msagent.audit.events import AuditEventType
from msagent.core.constants import CONFIG_AUDIT_DIR


def iter_json_values(text: str) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from compact or pretty-printed audit log text."""
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)

    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break

        payload, end = decoder.raw_decode(text, index)
        index = end
        if isinstance(payload, dict):
            yield payload


def resolve_audit_path(*, working_dir: Path, thread_id: str) -> Path | None:
    """Resolve the audit file for a session thread id."""
    audit_dir = working_dir.resolve() / CONFIG_AUDIT_DIR
    if not audit_dir.is_dir():
        return None

    prefixed = sorted(audit_dir.glob(f"*_{thread_id}.jsonl"))
    if prefixed:
        return prefixed[0]

    legacy = audit_dir / f"{thread_id}.jsonl"
    if legacy.is_file():
        return legacy
    return None


class AuditReader:
    """Load audit events and delegation summaries from disk."""

    def __init__(self, *, working_dir: Path, thread_id: str) -> None:
        self.thread_id = thread_id
        self.path = resolve_audit_path(working_dir=working_dir, thread_id=thread_id)

    def exists(self) -> bool:
        return self.path is not None and self.path.is_file()

    def iter_events(self) -> Iterator[dict[str, Any]]:
        """Yield flat audit events from the log file."""
        if self.path is None or not self.path.is_file():
            return

        text = self.path.read_text(encoding="utf-8")
        for payload in iter_json_values(text):
            if not isinstance(payload, dict):
                continue
            event_type = payload.get("event")
            if event_type in {
                AuditEventType.USER_TURN,
                AuditEventType.USER_RESPONSE,
                AuditEventType.SUBAGENT_DELEGATION,
            }:
                yield _normalize_event(payload)

    def list_delegations(self) -> list[dict[str, Any]]:
        """Return delegation records keyed by delegation_id."""
        rows: list[dict[str, Any]] = []
        for event in self.iter_events():
            if event.get("event") != AuditEventType.SUBAGENT_DELEGATION:
                continue
            delegation_id = str(event.get("delegation_id") or "")
            if not delegation_id:
                continue
            rows.append(_delegation_summary_from_merged(event))
        return rows


def _delegation_summary_from_merged(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": event.get("run_id"),
        "delegation_id": event.get("delegation_id"),
        "subagent_type": event.get("subagent_type"),
        "task_description": event.get("task_description_raw"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "status": event.get("status"),
        "duration_ms": event.get("duration_ms"),
        "result": event.get("result_raw"),
        "input": event.get("input"),
        "output": event.get("output"),
        "error": event.get("error"),
        "input_valid": event.get("input_valid"),
        "output_valid": event.get("output_valid"),
        "input_errors": event.get("input_errors"),
        "output_errors": event.get("output_errors"),
    }


def _migrate_time_field(
    event: dict[str, Any],
    *,
    legacy_keys: tuple[str, ...],
    target: str,
) -> None:
    if target not in event:
        for key in legacy_keys:
            if key in event:
                event[target] = event.pop(key)
                break
    for key in legacy_keys:
        event.pop(key, None)


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    _migrate_time_field(
        normalized,
        legacy_keys=("delegated_at", "started_at"),
        target="start_time",
    )
    _migrate_time_field(
        normalized,
        legacy_keys=("completed_at", "ended_at"),
        target="end_time",
    )
    normalized.pop("thread_id", None)
    normalized.pop("timestamp", None)
    normalized.pop("protocol_version", None)
    normalized.pop("ts", None)
    return normalized

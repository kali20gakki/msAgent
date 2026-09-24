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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap and CLI helpers for msmodelslim skill scripts."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


def ensure_msmodelslim() -> None:
    import msmodelslim  # noqa: F401 — trigger Ascend / package patches


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def emit_result(result: dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, default=_json_default))
    if result.get("ok") is False:
        return 1
    if result.get("valid") is False:
        return 1
    return 0


def parse_optional_json(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    stripped = value.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("device_indices must be a JSON array")
        return [int(x) for x in parsed]
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def file_md5(path: str | Path) -> str:
    """Return hex MD5 of a file's bytes. Used by quant-tuning scripts that
    need a stable identifier for practice.yaml / history.yaml (orchestrator
    dedup)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def now_iso() -> str:
    """Return local-time timestamp at second precision (`YYYY-MM-DD HH:MM:SS`).

    Single canonical format so all skill scripts that emit `time` fields are
    comparable without re-parsing."""
    return _dt.datetime.now().isoformat(sep=" ", timespec="seconds")


def upsert_yaml_section(
    existing: dict[str, Any],
    section_name: str,
    record: dict[str, Any],
    *,
    key_field: str,
) -> dict[str, Any]:
    """Upsert ``record`` into ``existing[section_name]`` (a list keyed by ``key_field``).

    Returns the updated ``existing`` dict (mutates in place; the dict reference
    is also returned for chaining). If ``record[key_field]`` matches an
    existing entry, that entry is replaced; otherwise the record is appended.

    The caller is responsible for writing the dict back to disk in whatever
    format they prefer (this helper is format-agnostic; pass a parsed YAML
    dict in, get a parsed YAML dict out)."""
    records = existing.get(section_name)
    if not isinstance(records, list):
        records = []
    replaced = False
    for idx, prev in enumerate(records):
        if isinstance(prev, dict) and prev.get(key_field) == record.get(key_field):
            records[idx] = record
            replaced = True
            break
    if not replaced:
        records.append(record)
    existing[section_name] = records
    return existing

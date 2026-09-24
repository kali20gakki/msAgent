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
"""Thin YAML helpers used by the practice-dit skill.

Practice YAMLs are *small* (one ``spec`` block, one ``process`` list, one
``exclude`` list) — we deliberately avoid pulling in msmodelslim's full
yaml_validation_helpers stack so the skill has no transitive dependency on
the quant framework being installed. ``PyYAML`` is the only requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


def _yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except ImportError as exc:  # pragma: no cover — env-mismatch guard
        raise RuntimeError(
            "PyYAML is required by the quantization experience library. "
            "Install it via `pip install pyyaml`."
        ) from exc


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a Python dict. Always returns ``{}`` for empty files."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"YAML not found: {p}")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    parsed = _yaml().safe_load(text)
    return parsed or {}


def dump_yaml(obj: Dict[str, Any], path: str | Path, *, sort_keys: bool = False) -> None:
    """Dump a Python dict to YAML with a stable style (block, indent=2)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = _yaml().safe_dump(
        obj,
        sort_keys=sort_keys,
        allow_unicode=True,
        default_flow_style=False,
        indent=2,
        width=120,
    )
    p.write_text(text, encoding="utf-8")


def find_exclude_target(process: list) -> int | None:
    """Return index of the first ``process`` step whose ``exclude`` is a list, or ``None``.

    Shared by ``merge_exclude`` (read-only) and ``apply_rollback`` (needs the same
    index to write back the merged list without re-scanning).
    """
    for idx, step in enumerate(process):
        if isinstance(step, dict) and isinstance(step.get("exclude"), list):
            return idx
    return None


def merge_exclude(
    base_practice: Dict[str, Any],
    extras: list[str],
) -> tuple[list[str], list[str]]:
    """Append ``extras`` to the first ``spec.process[*].exclude`` list.

    Returns ``(merged, original)`` so callers that need to compute the
    ``appended`` delta (e.g. ``apply_rollback``) don't have to re-locate the
    target step. Honors the existing ``exclude`` list — duplicates are dropped,
    and the *original order* of the base list is preserved so YAML diffs stay
    readable. Does **not** mutate the input dict.
    """
    spec = base_practice.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("base_practice has no `spec` mapping")
    process = spec.get("process")
    if not isinstance(process, list) or not process:
        raise ValueError("base_practice.spec.process must be a non-empty list")

    target_idx = find_exclude_target(process)
    if target_idx is None:
        raise ValueError(
            "No process step with `exclude` list found in base_practice; "
            "expert_rollback requires an existing exclude list to append to."
        )

    base_exclude = list(process[target_idx]["exclude"])
    seen = set(base_exclude)
    merged: list[str] = list(base_exclude)
    for entry in extras:
        if entry and entry not in seen:
            merged.append(entry)
            seen.add(entry)

    return merged, base_exclude

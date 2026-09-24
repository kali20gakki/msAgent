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
"""Precheck: enumerate candidate VBench cache locations BEFORE scoring.

Mandatory precheck gate. The orchestrator MUST run this script before
``scripts/score.py`` and surface the returned ``candidates`` to the user for
explicit confirmation. This prevents an agent from silently running
``download_vbench_cache.sh`` and duplicating a 1.8GB cache that already
exists at a non-default path.

Contract
--------
* **Never** auto-selects a path.
* **Never** downloads anything.
* **Never** writes outside stdout.

The returned ``instruction`` field is a hard directive to the orchestrator:
surface the candidates to the user and wait for explicit confirmation.
Reading it once and proceeding silently is a contract violation.

Usage::

    python scripts/check_vbench_cache.py

Exit 0 on either branch; the JSON ``ok`` flag is the source of truth.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# Required sub-model directories the scoring pipeline reads from. If any is
# missing, the cache is incomplete and the user must authorize a re-download
# or repopulation before score.py can succeed.
REQUIRED_SUBDIRS: tuple = ("clip_model", "umt_model", "amt_model", "raft_model")


_INSTRUCTION_FOUND = (
    "Show 'candidates' to the user. Do NOT auto-select. Wait for explicit "
    "user confirmation of which path to use as --vbench-cache-dir. Only "
    "after the user confirms may the orchestrator invoke score.py."
)

_INSTRUCTION_MISSING = (
    "No VBench cache found on this host. Ask the user: (1) provide the path "
    "to an existing cache directory, or (2) explicitly authorize a download "
    "via ais_bench/configs/vbench_examples/download_vbench_cache.sh. Do NOT "
    "download without explicit user authorization."
)


def _inspect(path: Path) -> Dict[str, Any]:
    """Inspect one candidate path; return a JSON-friendly dict."""
    if not path.is_dir():
        return {"path": str(path), "exists": False}
    subdirs = sorted(d.name for d in path.iterdir() if d.is_dir())
    has_required = [s for s in REQUIRED_SUBDIRS if (path / s).is_dir()]
    missing = [s for s in REQUIRED_SUBDIRS if not (path / s).is_dir()]
    return {
        "path": str(path),
        "exists": True,
        "subdirs": subdirs,
        "has_required": has_required,
        "missing_subdirs": missing,
        "complete": not missing,
    }


def _candidate_paths() -> List[Path]:
    """Well-known locations in priority order (highest first).

    Order matters — the FIRST existing candidate is the most likely correct
    path — but this script never picks automatically; it only suggests
    priority for the user.

    Cases:
      1. ``$VBENCH_CACHE_DIR`` env var (user-explicit; top priority).
      2. ``$HOME/.cache/vbench`` — uses the real ``$HOME``, NOT
         ``Path("~").expanduser()`` which can resolve to ``/root`` when the
         agent runs as root inside a container while the real user is
         ``/home/<user>`` (the bug that produced the 1.8GB duplicate).
      3. ``~/.cache/vbench`` (Python expanduser fallback).
      4. ``/opt/ais_bench/cache/vbench`` (system install).
      5. ``/home/*/.cache/vbench`` (multi-user host scan).
      6. ``/root/.cache/vbench`` (last-resort root home).
    """
    candidates: List[Path] = []

    env = os.environ.get("VBENCH_CACHE_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())

    home = os.environ.get("HOME", "").strip()
    if home:
        candidates.append(Path(home) / ".cache" / "vbench")

    try:
        candidates.append(Path("~/.cache/vbench").expanduser())
    except RuntimeError:
        pass  # $HOME unset/empty; already covered above.

    candidates.append(Path("/opt/ais_bench/cache/vbench"))

    home_root = Path("/home")
    if home_root.is_dir():
        for user_dir in sorted(home_root.iterdir()):
            if user_dir.is_dir():
                candidates.append(user_dir / ".cache" / "vbench")

    candidates.append(Path("/root/.cache/vbench"))

    # De-dup while preserving order.
    seen: set = set()
    unique: List[Path] = []
    for c in candidates:
        s = str(c)
        if s and s not in seen:
            seen.add(s)
            unique.append(c)
    return unique


def main() -> int:
    inspected = [_inspect(c) for c in _candidate_paths()]
    existing = [c for c in inspected if c.get("exists")]

    if existing:
        payload: Dict[str, Any] = {"ok": True, "candidates": existing, "instruction": _INSTRUCTION_FOUND}
    else:
        payload = {"ok": False, "candidates": [], "instruction": _INSTRUCTION_MISSING}

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
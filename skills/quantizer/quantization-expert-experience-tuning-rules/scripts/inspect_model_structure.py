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
"""DEPRECATED — kept for legacy debugging only.

The DiT practice skill (v3) no longer scans ``inference_repo`` to decide
rollback rules. Block-container name is auto-detected from the existing
exclude list (``blocks`` / ``transformer_blocks`` /
``single_transformer_blocks``); structural patterns (``*ffn.down*`` /
``*attn.out_proj*`` / modulation) live in the experience library
L2 §7 and are not user-facing input here.

This script remains in the tree for users who still want to inspect the
modeling files manually (e.g. when the block container name is unusual
and the auto-detect fallback does not kick in). It is **not** invoked by
``apply_rollback.py`` and is **not** part of the documented workflow.

Original docstring preserved below.
---

Static scan of an inference repo's modeling files to surface rollback candidates.

Per the user's decision (3) the scan is *limited* to modeling-related paths:
``modeling_*.py``, ``<repo>/wan/modules/*.py``, ``<repo>/wan/dit/*.py``,
``<repo>/modules/*.py`` — never ``utils`` / ``scripts`` / ``examples`` / ``tests``.

Output: a JSON dict the orchestrator (or human) reads to verify which
rollback rules are sensible for the given model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCAN_ROOTS = ["wan/modules", "wan/dit", "modules", "modeling"]
SCAN_FILE_PATTERNS = ["modeling_*.py", "*.py"]

# FFN down / out_proj / wo / etc. — tokens that frequently appear in
# parameter-sensitive layers and benefit from expert-precision rollback.
LINEAR_TOKENS = ["down_proj", "out_proj", "fc2", "w2", "wo"]
# Block-container class / module names that hint at sequential block indices.
BLOCK_TOKENS = ["blocks", "layers", "transformer_blocks",
                "decoder_layers", "single_transformer_blocks"]

_LINEAR_RE = re.compile(r"([\w.]+)\.(" + "|".join(LINEAR_TOKENS) + r")\b")
_BLOCK_RE = re.compile(
    r"\b(" + "|".join(BLOCK_TOKENS) + r")\s*[:=]?\s*"
    r"(?:nn\.ModuleList\s*\(\s*\[?\s*\[?(\d+)?|nn\.ModuleList|nn\.Sequential|List\()"
)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _resolve_scan_files(repo: Path) -> list[Path]:
    """Return a sorted, deduplicated list of modeling-related Python files."""
    seen: set[Path] = set()
    out: list[Path] = []
    for root_rel in SCAN_ROOTS:
        root = repo / root_rel
        if not root.is_dir():
            continue
        for pattern in SCAN_FILE_PATTERNS:
            for p in root.glob(pattern):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    out.append(p)
    for p in repo.glob("modeling_*.py"):
        if p.is_file() and p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def inspect_model_structure(
    inference_repo: str | Path,
    *,
    model_type: Optional[str] = None,
) -> dict:
    """Scan modeling files; return JSON-serialisable dict with candidates + notes.

    Shape::

        {
          "model_family": str,
          "scan_roots": list[str],
          "files_scanned": int,
          "block_containers": [{"file", "line", "container_name", "inferred_block_count"}],
          "sensitive_candidates": [{"file", "line", "module_path", "matched_token"}],
          "notes": list[str],
        }
    """
    repo = Path(inference_repo).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"inference_repo not found: {repo}")

    result: dict = {
        "model_family": model_type or "dit",
        "scan_roots": list(SCAN_ROOTS),
        "files_scanned": 0,
        "block_containers": [],
        "sensitive_candidates": [],
        "notes": [],
    }

    files = _resolve_scan_files(repo)
    result["files_scanned"] = len(files)
    if not files:
        result["notes"].append(
            f"No modeling files found under any of {SCAN_ROOTS}; "
            "rollback patterns will rely on user-declared patterns only."
        )

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(repo))
        for lineno, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.lstrip()
            if stripped.startswith(("#", '"""', "'''")):
                continue
            for m in _LINEAR_RE.finditer(raw):
                module, token = m.group(1), m.group(2)
                result["sensitive_candidates"].append({
                    "file": rel, "line": lineno,
                    "module_path": f"{module}.{token}",
                    "matched_token": token,
                })
            for m in _BLOCK_RE.finditer(raw):
                container, count = m.group(1), m.group(2)
                result["block_containers"].append({
                    "file": rel, "line": lineno,
                    "container_name": container,
                    "inferred_block_count": int(count) if count else None,
                })

    if not result["block_containers"]:
        result["notes"].append(
            "No block container detected; user must declare rollback_rules.by_block "
            "explicitly."
        )
    if not result["sensitive_candidates"]:
        result["notes"].append(
            "No sensitive linear candidates detected; user must declare "
            "rollback_rules.by_pattern explicitly."
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an inference repo's modeling files for rollback candidates.",
    )
    parser.add_argument("--inference-repo", required=True)
    parser.add_argument("--model-type", default=None)
    args = parser.parse_args()
    result = inspect_model_structure(
        inference_repo=args.inference_repo,
        model_type=args.model_type,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
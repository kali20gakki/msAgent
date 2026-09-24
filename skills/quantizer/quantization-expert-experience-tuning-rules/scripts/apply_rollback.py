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
"""Append block-rollback entries to a DiT Practice YAML's ``exclude`` list.

The DiT tuning experience (see
``../structure-family-pitfalls.md`` §7) reduces the agent's job to **one
number** — how many leading DiT blocks should be excluded from
quantization. Cross-block structural patterns (``*ffn.down*`` /
``*attn.out_proj*`` / modulation exclusion) live in the experience library
and are not user-facing input here.

This script:
    1. reads a base Practice YAML produced by ``quant-tuning-quantize-dit``;
    2. expands ``first_n_blocks`` (or explicit ``block_indices``) into
       ``*<container>.{i}.*`` glob entries;
    3. appends them to the first ``spec.process[*].exclude`` list;
    4. writes the modified YAML, returning md5 + appended entries.

The output is a copy of the input practice YAML with extra entries appended
to ``spec.process[0].exclude`` — nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# Bootstrap: common helpers + sibling yaml_utils.
_HERE = Path(__file__).resolve().parent  # .../quantization-expert-experience-tuning-rules/scripts/
_REPO_SKILLS = _HERE.resolve().parents[1]  # .../quantizer/
_COMMON = _REPO_SKILLS / "msmodelslim-tools-common" / "scripts"
for _p in (_COMMON, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from script_utils import file_md5  # noqa: E402
from yaml_utils import (  # noqa: E402
    dump_yaml,
    find_exclude_target,
    load_yaml,
    merge_exclude,
)


# Container-name candidates tried in order when --block-container is not given.
# Wan2.2 / FLUX (single_transformer) / SD3 / HunyuanDiT all use ``blocks``;
# HunyuanVideo uses ``transformer_blocks``.
_DEFAULT_CONTAINER_CANDIDATES = ("blocks", "transformer_blocks",
                                 "single_transformer_blocks")


# ---------------------------------------------------------------------------
# Input shape (validated, but not via Pydantic to keep dependencies minimal)
# ---------------------------------------------------------------------------


@dataclass
class RollbackRules:
    """User-supplied rollback rule.

    DiT v3 simplification: only one input dimension is needed — the number
    of leading blocks to roll back (or an explicit index list).
    """
    first_n_blocks: Optional[int] = None
    block_indices: Optional[List[int]] = None


def _parse_rules(raw: object) -> RollbackRules:
    """Validate the rollback_rules payload and return a RollbackRules dataclass.

    Accepted shapes (all optional except that at least one must be set)::

        {"first_n_blocks": 5}
        {"block_indices": [0, 1, 5]}
        {"first_n_blocks": 5, "block_indices": [0, 1, 5]}  # merged
    """
    if not isinstance(raw, dict):
        raise ValueError("rollback_rules must decode to a JSON object")

    unknown = set(raw.keys()) - {"first_n_blocks", "block_indices"}
    if unknown:
        raise ValueError(
            f"rollback_rules contains unsupported keys: {sorted(unknown)}; "
            "DiT v3 only accepts 'first_n_blocks' and/or 'block_indices'."
        )

    first_n = raw.get("first_n_blocks")
    indices = raw.get("block_indices")

    if first_n is None and indices is None:
        raise ValueError(
            "rollback_rules must set at least one of "
            "'first_n_blocks' or 'block_indices'."
        )

    if first_n is not None:
        first_n = int(first_n)
        if first_n < 0:
            raise ValueError("first_n_blocks must be non-negative")

    if indices is not None:
        if not isinstance(indices, list) or not all(
            isinstance(x, int) for x in indices
        ):
            raise ValueError("block_indices must be a list[int]")
        indices = [int(x) for x in indices]

    return RollbackRules(
        first_n_blocks=first_n,
        block_indices=indices,
    )


# ---------------------------------------------------------------------------
# Translation: RollbackRules → exclude list
# ---------------------------------------------------------------------------


def _expand(rules: RollbackRules, container_name: str) -> List[str]:
    """Expand RollbackRules into a flat list of ``*<container>.{i}.*`` entries."""
    excludes: List[str] = []
    if rules.first_n_blocks:
        for i in range(rules.first_n_blocks):
            excludes.append(f"*{container_name}.{i}.*")
    if rules.block_indices:
        for i in rules.block_indices:
            excludes.append(f"*{container_name}.{i}.*")
    # De-dup while preserving order.
    seen: set[str] = set()
    return [e for e in excludes if not (e in seen or seen.add(e))]


def _resolve_container(
    base: dict,
    *,
    explicit: Optional[str],
    candidates: tuple[str, ...] = _DEFAULT_CONTAINER_CANDIDATES,
) -> str:
    """Pick the block container name.

    Priority:
      1. ``explicit`` (CLI flag), if given.
      2. First candidate whose name already appears in the existing exclude
         list (covers Wan2.2 / HunyuanVideo / FLUX / SD3 transparently).
      3. ``blocks`` as final fallback (covers most DiT variants).
    """
    if explicit:
        return explicit
    existing: List[str] = []
    process = base.get("spec", {}).get("process") or []
    for step in process:
        excl = step.get("exclude") if isinstance(step, dict) else None
        if isinstance(excl, list):
            existing.extend(excl)
    blob = " ".join(existing)
    for name in candidates:
        if name in blob:
            return name
    return "blocks"


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    ok: bool
    base_practice: str
    output_practice: str
    appended: List[str]
    md5: str
    block_container: str
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_rollback(
    base_practice_path: str | Path,
    rollback_rules: object,
    output_practice_path: str | Path,
    *,
    block_container: Optional[str] = None,
) -> ApplyResult:
    """Read ``base_practice_path``, append block-rollback excludes, write YAML."""
    rules = _parse_rules(rollback_rules)
    base = load_yaml(base_practice_path)

    container = _resolve_container(base, explicit=block_container)
    extras = _expand(rules, container)
    if not extras:
        raise ValueError(
            "rollback_rules produced an empty exclude list; "
            "check first_n_blocks / block_indices values."
        )
    merged, original_exclude = merge_exclude(base, extras)

    process = base["spec"]["process"]
    target_idx = find_exclude_target(process)
    process[target_idx]["exclude"] = merged

    dump_yaml(base, output_practice_path)
    md5 = file_md5(output_practice_path)

    notes: List[str] = []
    if block_container is None and container != "blocks":
        notes.append(
            f"block_container auto-detected as '{container}' "
            "from existing exclude list."
        )

    return ApplyResult(
        ok=True,
        base_practice=str(Path(base_practice_path).expanduser().resolve()),
        output_practice=str(Path(output_practice_path).expanduser().resolve()),
        appended=[e for e in merged if e not in original_exclude],
        md5=md5,
        block_container=container,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append DiT block rollback entries to a Practice YAML's "
            "exclude list (DiT v3, single-dimension input)."
        ),
    )
    parser.add_argument("--base-practice", required=True)
    parser.add_argument(
        "--first-n-blocks",
        type=int,
        default=None,
        help="Roll back the first N DiT blocks (整层回退). Mutually composable with --block-indices.",
    )
    parser.add_argument(
        "--block-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicitly roll back DiT block indices, e.g. --block-indices 0 1 5.",
    )
    parser.add_argument("--output-practice", required=True)
    parser.add_argument(
        "--block-container",
        default=None,
        help=(
            "Block container name (default: auto-detected from existing exclude list; "
            "e.g. transformer_blocks for HunyuanVideo)."
        ),
    )
    return parser


def main() -> int:
    args = _build_cli_parser().parse_args()
    raw_rules: dict = {}
    if args.first_n_blocks is not None:
        raw_rules["first_n_blocks"] = args.first_n_blocks
    if args.block_indices is not None:
        raw_rules["block_indices"] = list(args.block_indices)
    result = apply_rollback(
        base_practice_path=args.base_practice,
        rollback_rules=raw_rules,
        output_practice_path=args.output_practice,
        block_container=args.block_container,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
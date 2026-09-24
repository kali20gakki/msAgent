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
"""Append a single DiT-tuning round to ``{workdir}/history/history.yaml``.

Mirrors the LLM-path ``accuracy_append.py`` shape, but writes DiT-specific
fields (``inference_outputs``, ``fp_baseline_outputs``, and the scoring
fields ``scores`` / ``overall_score`` / ``loss_vs_baseline`` /
``is_satisfied`` populated by ``quant-tuning-score-dit``) as
documented in
``msagent/skills/quantization-accuracy-tuning-orchestrator/references/output_format.md §3``.

Behaviour:

* Idempotent on identical ``practice_id``: re-running with the same id
  overwrites the previous entry instead of duplicating it.
* Auto-creates ``history/`` directory if missing.
* Writes UTF-8 YAML with stable ordering so human diffs stay readable.
* Does not touch LLM/VLM ``records`` — appends a separate DiT section when
  the file already contains LLM data (see ``--append-as`` flag).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap: common helpers + sibling yaml_utils.
_HERE = Path(__file__).resolve().parent
_REPO_SKILLS = _HERE.resolve().parents[1]  # .../quantizer/
_COMMON = _REPO_SKILLS / "msmodelslim-tools-common" / "scripts"
# DiT v3: yaml_utils now lives under the experience library scripts/ folder.
_PRACTICE = _REPO_SKILLS / "quantization-expert-experience-tuning-rules" / "scripts"
for _p in (_COMMON, _PRACTICE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from script_utils import file_md5, now_iso, upsert_yaml_section  # noqa: E402
from yaml_utils import dump_yaml, load_yaml  # noqa: E402


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def append_dit_record(
    history_path: str | Path,
    *,
    practice_id: str,
    practice_path: str | Path,
    inference_outputs: List[str],
    fp_baseline_outputs: Optional[List[str]] = None,
    scores: Optional[Dict[str, float]] = None,
    overall_score: Optional[float] = None,
    loss_vs_baseline: Optional[float] = None,
    is_satisfied: Optional[bool] = None,
    append_as: str = "dit_records",
) -> Dict[str, Any]:
    """Append (or upsert) one DiT record into ``history.yaml``.

    Args:
        history_path: ``{workdir}/history/history.yaml``.
        practice_id: Unique ID for this round (``dit-round-2``).
        practice_path: ``{workdir}/round_{N}/practice.yaml`` (md5 auto-computed).
        inference_outputs: List of video / image paths produced this round.
        fp_baseline_outputs: List of FP baseline paths (when ``quant-tuning-evaluate`` DiT workflow ran in FP baseline mode with --ckpt_dir → FP weights).
        scores: Per-dimension score dict from ``quant-tuning-score-dit``.
        overall_score: Aggregated VBench overall_score (Quality/Semantic/Total).
        loss_vs_baseline: Quant vs FP overall loss; only populated when baseline-outputs is enabled.
        is_satisfied: ``True`` iff ``loss_vs_baseline >= -tolerance``; orchestrator uses this as
            the loop exit signal.
        append_as: Section name to use in the YAML. Defaults to ``dit_records``.

    Returns:
        The full record dict that was written, including ``time`` and ``quant_config_md5``.
    """
    history_p = Path(history_path).expanduser().resolve()
    history_p.parent.mkdir(parents=True, exist_ok=True)

    practice_p = Path(practice_path).expanduser().resolve()
    if not practice_p.is_file():
        raise FileNotFoundError(f"practice_path not found: {practice_p}")

    record: Dict[str, Any] = {
        "practice_id": practice_id,
        "quant_config_md5": file_md5(practice_p),
        "time": now_iso(),
        "practice_path": str(practice_p),
        "inference_outputs": list(inference_outputs),
        "fp_baseline_outputs": list(fp_baseline_outputs) if fp_baseline_outputs else None,
        "scores": scores,
        "overall_score": overall_score,
        "loss_vs_baseline": loss_vs_baseline,
        "is_satisfied": is_satisfied,
    }

    existing = load_yaml(history_p) if history_p.is_file() else {}
    if not isinstance(existing, dict):
        existing = {}
    upsert_yaml_section(existing, append_as, record, key_field="practice_id")
    dump_yaml(existing, history_p, sort_keys=True)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a DiT-tuning round to history.yaml.",
    )
    parser.add_argument("--history-path", required=True,
                        help="Path to history.yaml (will be created if missing).")
    parser.add_argument("--practice-id", required=True,
                        help="Unique ID, e.g. 'dit-round-2'.")
    parser.add_argument("--practice-path", required=True,
                        help="Path to the round's practice.yaml (md5 computed).")
    parser.add_argument("--inference-outputs", required=True,
                        help="Comma-separated list of video/image paths.")
    parser.add_argument("--fp-baseline-outputs", default=None,
                        help="Comma-separated list of FP baseline paths (optional).")
    parser.add_argument("--scores-json", default=None,
                        help="JSON object of per-dimension scores (optional).")
    parser.add_argument("--overall-score", type=float, default=None)
    parser.add_argument("--loss-vs-baseline", type=float, default=None)
    parser.add_argument("--is-satisfied", choices=["true", "false"], default=None)
    parser.add_argument("--append-as", default="dit_records",
                        help="YAML section name (default: dit_records).")
    return parser


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    args = _build_cli_parser().parse_args()
    scores = None
    if args.scores_json:
        scores = json.loads(args.scores_json)
    is_satisfied = None
    if args.is_satisfied is not None:
        is_satisfied = args.is_satisfied == "true"

    record = append_dit_record(
        history_path=args.history_path,
        practice_id=args.practice_id,
        practice_path=args.practice_path,
        inference_outputs=_parse_csv(args.inference_outputs) or [],
        fp_baseline_outputs=_parse_csv(args.fp_baseline_outputs),
        scores=scores,
        overall_score=args.overall_score,
        loss_vs_baseline=args.loss_vs_baseline,
        is_satisfied=is_satisfied,
        append_as=args.append_as,
    )
    print(json.dumps({"ok": True, "record": record}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

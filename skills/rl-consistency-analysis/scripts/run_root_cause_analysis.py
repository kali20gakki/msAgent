#!/usr/bin/env python3
"""Generate structured root-cause evidence from train/rollout dumps."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SKILLS_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = SKILLS_ROOT / "rl-consistency-analysis" / "scripts" / "generate_module_mapping.py"

ACTIVATION_TERMS = ("act_fn", "silu", "swiglu", "gelu", "mul", "activation")
STRUCTURAL_PATTERNS = [
    ("linear_q_down_proj", "fused_qkv_a_proj"),
    ("linear_kv_down_proj", "fused_qkv_a_proj"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_module_keys(dump_data: dict[str, Any]) -> list[str]:
    data = dump_data.get("data", {})
    if not isinstance(data, dict):
        return []
    return [key for key in data if isinstance(key, str) and key.startswith("Module.")]


def parse_layer_idx(key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def infer_block(key: str) -> str:
    lowered = key.lower()
    if ".mlp." in lowered:
        return "mlp"
    if "self_attn" in lowered or "self_attention" in lowered or "attention" in lowered:
        return "attn"
    if "norm" in lowered:
        return "norm"
    if "embed" in lowered:
        return "embed"
    return "other"


def structural_false_positive(train_key: str, rollout_key: str) -> str:
    lowered_train = train_key.lower()
    lowered_rollout = rollout_key.lower()
    for left, right in STRUCTURAL_PATTERNS:
        if left in lowered_train and right in lowered_rollout:
            return f"structural fusion pattern: {left} -> {right}"
    return ""


def is_activation_like(key: str) -> bool:
    lowered = key.lower()
    return any(term in lowered for term in ACTIVATION_TERMS)


def find_previous_aligned(records: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    current = records[index]
    current_layer = parse_layer_idx(current["train_key"])
    current_block = infer_block(current["train_key"])
    for probe in range(index - 1, -1, -1):
        candidate = records[probe]
        if parse_layer_idx(candidate["train_key"]) != current_layer:
            continue
        if infer_block(candidate["train_key"]) != current_block:
            continue
        if candidate.get("output_consistency") == "aligned":
            return candidate
    return None


def find_between(keys: list[str], start_key: str, end_key: str) -> list[str]:
    try:
        start = keys.index(start_key)
        end = keys.index(end_key)
    except ValueError:
        return []
    if start >= end:
        return []
    return keys[start + 1 : end]


def build_matched_sets(mapping_rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    train_keys = set()
    rollout_keys = set()
    for row in mapping_rows:
        peers = row.get("rollout_keys") or []
        if peers:
            train_keys.add(row["train_key"])
            rollout_keys.update(peers)
    return train_keys, rollout_keys


def classify_candidate(module_row: dict[str, Any], unmatched_train: list[str], unmatched_rollout: list[str]) -> tuple[str, str]:
    if unmatched_rollout and any(is_activation_like(key) for key in unmatched_rollout):
        return (
            "missing_or_extra_op_between_aligned_boundary",
            "rollout has unmatched activation-like intermediate modules between the last aligned boundary and this suspect",
        )
    if unmatched_train and any(is_activation_like(key) for key in unmatched_train):
        return (
            "missing_or_extra_op_between_aligned_boundary",
            "train has unmatched activation-like intermediate modules between the last aligned boundary and this suspect",
        )
    bucket = module_row.get("module_bucket", "")
    if bucket == "parameter_sync_mismatch":
        return ("parameter_or_checkpoint_issue", "parameters diverged while inputs remained aligned")
    if bucket == "module_impl_mismatch":
        return ("in_module_impl_difference", "inputs and parameters aligned, but outputs diverged inside the module")
    if bucket in {"upstream_propagation_or_amplification", "systemic_misalignment"}:
        return ("upstream_propagation", "inputs were already divergent before this module")
    return ("needs_manual_review", "no stronger automatic explanation was found")


def run_compare(train_path: Path, rollout_path: Path, out_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--train",
            str(train_path),
            "--rollout",
            str(rollout_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )


def build_report(train_path: Path, rollout_path: Path, out_dir: Path) -> dict[str, Any]:
    train_dump = load_json(train_path)
    rollout_dump = load_json(rollout_path)
    train_keys = get_module_keys(train_dump)
    rollout_keys = get_module_keys(rollout_dump)

    mapping_rows = load_json(out_dir / "output_1_key_mapping.json")["rows"]
    module_rows = load_json(out_dir / "output_4_module_analysis.json")["records"]
    matched_train, matched_rollout = build_matched_sets(mapping_rows)

    suspects: list[dict[str, Any]] = []
    structural_skips: list[dict[str, Any]] = []
    for index, row in enumerate(module_rows):
        rank = row.get("module_priority_rank")
        if rank not in (1, 2):
            continue

        reason = structural_false_positive(row["train_key"], row["rollout_key"])
        if reason:
            structural_skips.append(
                {
                    "train_key": row["train_key"],
                    "rollout_key": row["rollout_key"],
                    "module_bucket": row.get("module_bucket", ""),
                    "skip_reason": reason,
                }
            )
            continue

        previous_aligned = find_previous_aligned(module_rows, index)
        unmatched_train: list[str] = []
        unmatched_rollout: list[str] = []
        if previous_aligned:
            train_segment = find_between(train_keys, previous_aligned["train_key"], row["train_key"])
            rollout_segment = find_between(rollout_keys, previous_aligned["rollout_key"], row["rollout_key"])
            unmatched_train = [key for key in train_segment if key not in matched_train]
            unmatched_rollout = [key for key in rollout_segment if key not in matched_rollout]

        classification, rationale = classify_candidate(row, unmatched_train, unmatched_rollout)
        suspects.append(
            {
                "train_key": row["train_key"],
                "rollout_key": row["rollout_key"],
                "module_priority_rank": rank,
                "module_bucket": row.get("module_bucket", ""),
                "classification": classification,
                "rationale": rationale,
                "previous_aligned_boundary": previous_aligned,
                "unmatched_train_between_boundary": unmatched_train,
                "unmatched_rollout_between_boundary": unmatched_rollout,
                "likely_missing_train_activation": [key for key in unmatched_rollout if is_activation_like(key)],
                "likely_missing_rollout_activation": [key for key in unmatched_train if is_activation_like(key)],
            }
        )

    suspects.sort(
        key=lambda item: (
            0 if item["classification"] == "missing_or_extra_op_between_aligned_boundary" else 1,
            item["module_priority_rank"],
        )
    )

    top_hypothesis = ""
    top_evidence: list[str] = []
    if suspects:
        first = suspects[0]
        if first["classification"] == "missing_or_extra_op_between_aligned_boundary" and first["likely_missing_train_activation"]:
            top_hypothesis = "train may be missing an activation-like operation that exists on rollout between the last aligned boundary and the first downstream mismatch"
            top_evidence = first["likely_missing_train_activation"]
        elif first["classification"] == "parameter_or_checkpoint_issue":
            top_hypothesis = "parameter sync or checkpoint alignment issue"
        elif first["classification"] == "in_module_impl_difference":
            top_hypothesis = "in-module implementation or numeric path mismatch"
        elif first["classification"] == "upstream_propagation":
            top_hypothesis = "upstream mismatch is propagating into this module"

    return {
        "train_path": str(train_path),
        "rollout_path": str(rollout_path),
        "out_dir": str(out_dir),
        "structural_false_positives": structural_skips,
        "root_cause_suspects": suspects,
        "top_hypothesis": top_hypothesis,
        "top_evidence": top_evidence,
        "next_checks": [
            "verify implementation/config of any activation-like unmatched module",
            "confirm whether the missing or extra op is expected to be fused, hidden, or disabled by config",
            "if activation is expected on both sides, compare the actual implementation and bridge/config flags next",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end root cause analysis for train/rollout dump mismatch")
    parser.add_argument("--train", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train_path = Path(args.train)
    rollout_path = Path(args.rollout)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_compare(train_path, rollout_path, out_dir)
    report = build_report(train_path, rollout_path, out_dir)

    (out_dir / "output_5_root_cause_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Root Cause Evidence Summary:")
    print(f"- output_dir: {out_dir}")
    print(f"- top_hypothesis: {report.get('top_hypothesis') or '(none)'}")
    print(f"- structural_false_positives: {len(report['structural_false_positives'])}")
    print(f"- root_cause_suspects: {len(report['root_cause_suspects'])}")
    print(f"- report_json: {out_dir / 'output_5_root_cause_report.json'}")
    print("- report_md: (agent should generate from references/report_template.md)")


if __name__ == "__main__":
    main()

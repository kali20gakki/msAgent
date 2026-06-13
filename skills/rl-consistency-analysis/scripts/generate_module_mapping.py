#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from common import load_json, get_module_items, parse_layer_idx, infer_block, EPS, THRESHOLDS_OK, THRESHOLDS_WARN, \
    is_fused_qkv_pair


def build_rollout_index(rollout_items: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for k, _ in rollout_items:
        idx[k].append(k)
    return idx


def pick_compare_node(entry: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    if target == "output.0":
        out = entry.get("output", [])
        return out[0] if isinstance(out, list) and out else None
    if target == "input_args.0":
        args = entry.get("input_args", [])
        return args[0] if isinstance(args, list) and args else None
    if target == "parameters.weight":
        params = entry.get("parameters", {})
        if isinstance(params, dict):
            p = params.get("weight")
            return p if isinstance(p, dict) else None
    return None


def rel_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), EPS)


def status_from_rel(err: Optional[float]) -> str:
    if err is None:
        return "missing"
    if err <= THRESHOLDS_OK:
        return "ok"
    if err <= THRESHOLDS_WARN:
        return "warn"
    return "alert"


def normalize_shape(shape: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(shape, list):
        return None
    normalized: List[int] = []
    for item in shape:
        if not isinstance(item, int):
            return None
        normalized.append(item)
    return tuple(normalized)


def shape_numel(shape: Optional[Tuple[int, ...]]) -> Optional[int]:
    if shape is None:
        return None
    if not shape:
        return 1
    return reduce(lambda a, b: a * b, shape, 1)


def strip_singletons(shape: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    if shape is None:
        return None
    stripped = tuple(dim for dim in shape if dim != 1)
    return stripped or (1,)


def classify_shape_relation(shape_l: Any, shape_r: Any) -> Tuple[str, str, Optional[int], Optional[int]]:
    left = normalize_shape(shape_l)
    right = normalize_shape(shape_r)
    numel_left = shape_numel(left)
    numel_right = shape_numel(right)
    if left is None or right is None:
        return "unknown", "missing", numel_left, numel_right
    if left == right:
        return "exact_match", "match", numel_left, numel_right
    if numel_left == numel_right:
        if strip_singletons(left) == strip_singletons(right):
            return "singleton_compatible_numel_match", "mismatch", numel_left, numel_right
        return "numel_match", "mismatch", numel_left, numel_right
    return "numel_mismatch", "mismatch", numel_left, numel_right


def value_priority_bucket(numeric_mismatch: bool, shape_relation: str) -> Tuple[int, str, str]:
    if numeric_mismatch and shape_relation == "exact_match":
        return 1, "value_mismatch_shape_match", "alert"
    if numeric_mismatch and shape_relation in {"singleton_compatible_numel_match", "numel_match"}:
        return 1, "value_mismatch_numel_match", "alert"
    if numeric_mismatch and shape_relation == "numel_mismatch":
        return 2, "value_mismatch_shape_mismatch", "warn"
    if (not numeric_mismatch) and shape_relation in {"singleton_compatible_numel_match", "numel_match",
                                                     "numel_mismatch"}:
        return 3, "value_match_shape_mismatch", "warn"
    return 4, "value_match_shape_match", "ok"


def apply_target_context(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_row = next((row for row in rows if row.get("compare_target") == "output.0"), None)
    output_numeric_match = bool(output_row and output_row.get("numeric_consistency") == "match")

    adjusted: List[Dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["focus_level"] = "normal"
        updated["focus_reason"] = ""

        if (
                updated.get("compare_target") == "parameters.weight"
                and updated.get("numeric_consistency") == "mismatch"
                and output_numeric_match
        ):
            updated["priority_rank"] = 5
            updated["priority_bucket"] = "parameter_mismatch_output_aligned"
            updated["value_status"] = "info"
            updated["focus_level"] = "deprioritized"
            updated["focus_reason"] = "Parameter statistics differ, but output.0 remains numerically aligned."

        adjusted.append(updated)
    return adjusted


def compare_single_node(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    dtype_l = left.get("dtype")
    dtype_r = right.get("dtype")
    shape_l = left.get("shape")
    shape_r = right.get("shape")
    dtype_status = "match" if dtype_l == dtype_r else "mismatch"
    shape_relation, shape_status, numel_train, numel_rollout = classify_shape_relation(shape_l, shape_r)

    metrics = {}
    worst_metric = "ok"
    for m in ("Max", "Min", "Mean", "Norm"):
        a = left.get(m) if isinstance(left.get(m), (int, float)) else None
        b = right.get(m) if isinstance(right.get(m), (int, float)) else None
        e = rel_err(float(a) if a is not None else None, float(b) if b is not None else None)
        s = status_from_rel(e)
        metrics[m] = {
            "train": a,
            "rollout": b,
            "rel_err": e,
            "diff_pct": (e * 100.0) if e is not None else None,
            "status": s,
        }
        if s == "alert":
            worst_metric = "alert"
        elif s == "warn" and worst_metric != "alert":
            worst_metric = "warn"

    numeric_mismatch = worst_metric in {"warn", "alert"}
    priority_rank, priority_bucket, value_status = value_priority_bucket(numeric_mismatch, shape_relation)
    return {
        "dtype_train": dtype_l,
        "dtype_rollout": dtype_r,
        "dtype_status": dtype_status,
        "shape_train": shape_l,
        "shape_rollout": shape_r,
        "shape_status": shape_status,
        "shape_relation": shape_relation,
        "numel_train": numel_train,
        "numel_rollout": numel_rollout,
        "metrics": metrics,
        "diff_Max%": metrics["Max"]["diff_pct"],
        "diff_Min%": metrics["Min"]["diff_pct"],
        "diff_Mean%": metrics["Mean"]["diff_pct"],
        "diff_Norm%": metrics["Norm"]["diff_pct"],
        "numeric_consistency": "mismatch" if numeric_mismatch else "match",
        "priority_rank": priority_rank,
        "priority_bucket": priority_bucket,
        "metric_status": worst_metric,
        "value_status": value_status,
    }


def summarize_row_alignment(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "missing"
    if row.get("numeric_consistency") == "mismatch":
        return "mismatch"
    if row.get("numeric_consistency") == "match":
        return "aligned"
    return "missing"


def compare_input_kwargs(train_entry: Dict[str, Any], rollout_entry: Dict[str, Any]) -> Dict[str, Any]:
    train_kwargs = train_entry.get("input_kwargs", {})
    rollout_kwargs = rollout_entry.get("input_kwargs", {})
    if not isinstance(train_kwargs, dict):
        train_kwargs = {}
    if not isinstance(rollout_kwargs, dict):
        rollout_kwargs = {}

    train_keys = {key for key, value in train_kwargs.items() if isinstance(value, dict)}
    rollout_keys = {key for key, value in rollout_kwargs.items() if isinstance(value, dict)}
    shared_keys = sorted(train_keys & rollout_keys)

    if not train_keys and not rollout_keys:
        return {
            "status": "not_applicable",
            "numeric_consistency": "not_applicable",
            "shape_relation": "unknown",
            "compared_keys": [],
            "missing_train_keys": [],
            "missing_rollout_keys": [],
            "worst_metric": "not_applicable",
            "notes": "no comparable input_kwargs",
        }

    compared: List[Dict[str, Any]] = []
    worst_rank = 4
    worst_metric = "ok"
    for key in shared_keys:
        detail = compare_single_node(train_kwargs[key], rollout_kwargs[key])
        compared.append({"name": key, **detail})
        rank = detail.get("priority_rank") or 4
        if rank < worst_rank:
            worst_rank = rank
        metric = detail.get("metric_status", "ok")
        if metric == "alert":
            worst_metric = "alert"
        elif metric == "warn" and worst_metric != "alert":
            worst_metric = "warn"

    missing_train_keys = sorted(rollout_keys - train_keys)
    missing_rollout_keys = sorted(train_keys - rollout_keys)
    numeric_consistency = "match"
    for item in compared:
        if item.get("numeric_consistency") == "mismatch":
            numeric_consistency = "mismatch"
            break
    if not compared and (missing_train_keys or missing_rollout_keys):
        numeric_consistency = "missing"

    shape_relation = "exact_match"
    relation_rank = {
        "exact_match": 0,
        "singleton_compatible_numel_match": 1,
        "numel_match": 2,
        "numel_mismatch": 3,
        "unknown": 4,
    }
    if compared:
        shape_relation = max(
            (item.get("shape_relation", "unknown") for item in compared),
            key=lambda name: relation_rank.get(name, 99),
        )
    elif missing_train_keys or missing_rollout_keys:
        shape_relation = "unknown"

    status = "aligned"
    if numeric_consistency == "mismatch":
        status = "mismatch"
    elif numeric_consistency == "missing":
        status = "missing"
    elif missing_train_keys or missing_rollout_keys:
        status = "partial"

    return {
        "status": status,
        "numeric_consistency": numeric_consistency,
        "shape_relation": shape_relation,
        "compared_keys": shared_keys,
        "missing_train_keys": missing_train_keys,
        "missing_rollout_keys": missing_rollout_keys,
        "worst_metric": worst_metric if compared else "missing",
        "details": compared,
        "notes": "" if compared else "input_kwargs keys did not overlap",
    }


def classify_module_behavior(
        train_key: str,
        rollout_key: str,
        output_row: Optional[Dict[str, Any]],
        parameter_row: Optional[Dict[str, Any]],
        input_arg_row: Optional[Dict[str, Any]],
        input_kwargs_summary: Dict[str, Any],
) -> Dict[str, Any]:
    output_status = summarize_row_alignment(output_row)
    parameter_status = summarize_row_alignment(parameter_row)
    input_arg_status = summarize_row_alignment(input_arg_row)
    input_kwargs_status = input_kwargs_summary.get("status", "missing")

    input_statuses = [status for status in (input_arg_status, input_kwargs_status) if
                      status not in {"missing", "not_applicable"}]
    if any(status == "mismatch" for status in input_statuses):
        combined_input_status = "mismatch"
    elif input_statuses:
        combined_input_status = "aligned"
    else:
        combined_input_status = "missing"

    premise_flags: List[str] = []
    if parameter_status == "missing" or input_kwargs_status in {"missing", "partial"}:
        premise_flags.append("comparison_premise_incomplete")

    if output_status == "mismatch" and combined_input_status == "aligned" and is_fused_qkv_pair(train_key, rollout_key):
        return {
            "module_priority_rank": 5,
            "module_bucket": "fused_qkv_structural_mismatch",
            "module_status": "tolerable_structural_mismatch",
            "focus_level": "deprioritized",
            "reasoning": "Inputs are aligned, but this train-side projection is being compared against a rollout fused_qkv projection that contains multiple sub-projections, so direct output mismatch is structurally tolerable and should be deprioritized.",
            "premise_flags": premise_flags,
        }

    if output_status == "aligned":
        if parameter_status == "aligned" and combined_input_status == "aligned":
            return {
                "module_priority_rank": 4,
                "module_bucket": "normal_aligned",
                "module_status": "normal",
                "focus_level": "normal",
                "reasoning": "output is aligned, and internal inputs/parameters are aligned as well.",
                "premise_flags": premise_flags,
            }
        bucket = "internal_difference_but_output_aligned"
        if premise_flags:
            bucket = "comparison_premise_abnormal_but_output_aligned"
        return {
            "module_priority_rank": 5,
            "module_bucket": bucket,
            "module_status": "closed_aligned",
            "focus_level": "deprioritized",
            "reasoning": "output is aligned, so internal input/parameter differences are non-blocking unless they break the comparison premise.",
            "premise_flags": premise_flags,
        }

    if output_status != "mismatch":
        return {
            "module_priority_rank": 3,
            "module_bucket": "insufficient_output_evidence",
            "module_status": "needs_review",
            "focus_level": "review",
            "reasoning": "output could not be compared cleanly, so the module needs manual inspection.",
            "premise_flags": premise_flags,
        }

    if parameter_status == "aligned" and combined_input_status == "aligned":
        return {
            "module_priority_rank": 1,
            "module_bucket": "module_impl_mismatch",
            "module_status": "output_mismatch",
            "focus_level": "high",
            "reasoning": "same effective inputs and parameters produced mismatched output, which strongly suggests an in-module implementation or numeric-path difference.",
            "premise_flags": premise_flags,
        }
    if parameter_status != "aligned" and combined_input_status == "aligned":
        return {
            "module_priority_rank": 1,
            "module_bucket": "parameter_sync_mismatch",
            "module_status": "output_mismatch",
            "focus_level": "high",
            "reasoning": "output mismatched while inputs aligned but parameters differed, which points to parameter sync or checkpoint alignment issues.",
            "premise_flags": premise_flags,
        }
    if parameter_status == "aligned" and combined_input_status != "aligned":
        return {
            "module_priority_rank": 2,
            "module_bucket": "upstream_propagation_or_amplification",
            "module_status": "output_mismatch",
            "focus_level": "review",
            "reasoning": "output mismatched with aligned parameters but mismatched inputs, so the issue is likely propagated from upstream or amplified here.",
            "premise_flags": premise_flags,
        }
    return {
        "module_priority_rank": 2,
        "module_bucket": "systemic_misalignment",
        "module_status": "output_mismatch",
        "focus_level": "review",
        "reasoning": "inputs and parameters both differed before the output mismatch, which points to broader system-level misalignment rather than a single-module root cause.",
        "premise_flags": premise_flags,
    }


def compare_values(train_entry: Dict[str, Any], rollout_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = ["output.0", "input_args.0", "parameters.weight"]
    rows: List[Dict[str, Any]] = []
    for t in targets:
        left = pick_compare_node(train_entry, t)
        right = pick_compare_node(rollout_entry, t)
        if left is None and right is None:
            continue
        if left is None:
            rows.append(
                {
                    "compare_target": t,
                    "numeric_consistency": "missing",
                    "priority_rank": None,
                    "priority_bucket": "missing_compare_target",
                    "metric_status": "missing",
                    "value_status": "missing_on_train",
                }
            )
            continue
        if right is None:
            rows.append(
                {
                    "compare_target": t,
                    "numeric_consistency": "missing",
                    "priority_rank": None,
                    "priority_bucket": "missing_compare_target",
                    "metric_status": "missing",
                    "value_status": "missing_on_rollout",
                }
            )
            continue
        rows.append({"compare_target": t, **compare_single_node(left, right)})
    return apply_target_context(rows)


def build_module_summary(
        *,
        train_key: str,
        rollout_key: str,
        compare_rows: List[Dict[str, Any]],
        train_entry: Dict[str, Any],
        rollout_entry: Dict[str, Any],
) -> Dict[str, Any]:
    output_row = next((row for row in compare_rows if row.get("compare_target") == "output.0"), None)
    input_arg_row = next((row for row in compare_rows if row.get("compare_target") == "input_args.0"), None)
    parameter_row = next((row for row in compare_rows if row.get("compare_target") == "parameters.weight"), None)
    input_kwargs_summary = compare_input_kwargs(train_entry, rollout_entry)
    module_classification = classify_module_behavior(
        train_key,
        rollout_key,
        output_row,
        parameter_row,
        input_arg_row,
        input_kwargs_summary,
    )
    return {
        "train_key": train_key,
        "rollout_key": rollout_key,
        "output_consistency": summarize_row_alignment(output_row),
        "output_priority_bucket": output_row.get("priority_bucket",
                                                 "missing_compare_target") if output_row else "missing_compare_target",
        "parameter_consistency": summarize_row_alignment(parameter_row),
        "parameter_priority_bucket": parameter_row.get("priority_bucket",
                                                       "missing_compare_target") if parameter_row else "missing_compare_target",
        "input_args_consistency": summarize_row_alignment(input_arg_row),
        "input_args_priority_bucket": input_arg_row.get("priority_bucket",
                                                        "missing_compare_target") if input_arg_row else "missing_compare_target",
        "input_kwargs_consistency": input_kwargs_summary.get("status", "missing"),
        "input_kwargs_numeric_consistency": input_kwargs_summary.get("numeric_consistency", "missing"),
        "input_kwargs_shape_relation": input_kwargs_summary.get("shape_relation", "unknown"),
        "input_kwargs_compared_keys": input_kwargs_summary.get("compared_keys", []),
        "input_kwargs_missing_train_keys": input_kwargs_summary.get("missing_train_keys", []),
        "input_kwargs_missing_rollout_keys": input_kwargs_summary.get("missing_rollout_keys", []),
        "input_kwargs_notes": input_kwargs_summary.get("notes", ""),
        **module_classification,
    }


def build_analysis_summary(
        mapping_rows: List[Dict[str, Any]],
        value_rows: List[Dict[str, Any]],
        module_rows: List[Dict[str, Any]],
        out_dir: Path,
) -> List[str]:
    priority_counts: Dict[str, int] = defaultdict(int)
    missing_compare_target = 0
    for row in value_rows:
        bucket = row.get("priority_bucket")
        if bucket == "missing_compare_target":
            missing_compare_target += 1
        elif bucket:
            priority_counts[bucket] += 1

    module_bucket_counts: Dict[str, int] = defaultdict(int)
    for row in module_rows:
        bucket = row.get("module_bucket")
        if bucket:
            module_bucket_counts[bucket] += 1

    unmatched = [row["train_key"] for row in mapping_rows if not row["rollout_keys"]]
    lines = [
        "Analysis Summary:",
        f"- output_dir: {out_dir}",
        f"- mapping_rows: {len(mapping_rows)}",
        f"- matched_rows: {sum(1 for row in mapping_rows if row['rollout_keys'])}",
        f"- unmatched_rows: {sum(1 for row in mapping_rows if not row['rollout_keys'])}",
        "- priority_counts:",
        f"  - P1 value_mismatch_shape_match: {priority_counts.get('value_mismatch_shape_match', 0)}",
        f"  - P1 value_mismatch_numel_match: {priority_counts.get('value_mismatch_numel_match', 0)}",
        f"  - P2 value_mismatch_shape_mismatch: {priority_counts.get('value_mismatch_shape_mismatch', 0)}",
        f"  - P3 value_match_shape_mismatch: {priority_counts.get('value_match_shape_mismatch', 0)}",
        f"  - P4 value_match_shape_match: {priority_counts.get('value_match_shape_match', 0)}",
        f"  - P5 parameter_mismatch_output_aligned: {priority_counts.get('parameter_mismatch_output_aligned', 0)}",
        f"- missing_compare_target: {missing_compare_target}",
        "- module_bucket_counts:",
        f"  - module_impl_mismatch: {module_bucket_counts.get('module_impl_mismatch', 0)}",
        f"  - parameter_sync_mismatch: {module_bucket_counts.get('parameter_sync_mismatch', 0)}",
        f"  - upstream_propagation_or_amplification: {module_bucket_counts.get('upstream_propagation_or_amplification', 0)}",
        f"  - systemic_misalignment: {module_bucket_counts.get('systemic_misalignment', 0)}",
        f"  - fused_qkv_structural_mismatch: {module_bucket_counts.get('fused_qkv_structural_mismatch', 0)}",
        f"  - internal_difference_but_output_aligned: {module_bucket_counts.get('internal_difference_but_output_aligned', 0)}",
        f"  - comparison_premise_abnormal_but_output_aligned: {module_bucket_counts.get('comparison_premise_abnormal_but_output_aligned', 0)}",
        f"  - normal_aligned: {module_bucket_counts.get('normal_aligned', 0)}",
    ]
    if unmatched:
        lines.append("- first_unmatched_train_keys:")
        for item in unmatched[:8]:
            lines.append(f"  - {item}")
    lines.append("- output_files:")
    for name in (
            "output_0_key_mapping.json",
            "output_1_key_mapping.json",
            "output_1_key_mapping.csv",
            "output_2_mapping.json",
            "output_2_mapping.csv",
            "output_3_value_compare.json",
            "output_3_value_compare.csv",
            "output_4_module_analysis.json",
            "output_4_module_analysis.csv",
    ):
        lines.append(f"  - {out_dir / name}")
    return lines


def compact_metric_summary(row: Dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    if not metrics:
        return row.get("metric_status", "missing")
    pieces = []
    for name in ("Max", "Min", "Mean", "Norm"):
        item = metrics.get(name) or {}
        status = item.get("status")
        rel = item.get("rel_err")
        if status is None:
            continue
        if rel is None:
            pieces.append(f"{name}:missing")
        else:
            pieces.append(f"{name}:{status}@{rel:.3g}")
    return "; ".join(pieces) if pieces else row.get("metric_status", "missing")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate module mapping and value comparison from two dump.json files.")
    parser.add_argument("--train", required=True, help="train dump.json path")
    parser.add_argument("--rollout", required=True, help="rollout dump.json path")
    parser.add_argument("--mapping_key", required=True, help="mapping_key.json path")
    parser.add_argument("--out-dir", default=".", help="output directory")
    args = parser.parse_args()

    train_path = Path(args.train)
    rollout_path = Path(args.rollout)
    mapping_key = load_json(Path(args.mapping_key))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dump = load_json(train_path)
    rollout_dump = load_json(rollout_path)
    train_items = get_module_items(train_dump)
    rollout_items = get_module_items(rollout_dump)
    rollout_idx = build_rollout_index(rollout_items)
    rollout_data = dict(rollout_items)

    mapping_rows: List[Dict[str, Any]] = []
    # output_2: train_key -> rollout_key
    output2_map: Dict[str, Any] = {}
    value_rows: List[Dict[str, Any]] = []
    module_rows: List[Dict[str, Any]] = []

    # Keep train traversal order strictly.
    for train_key, train_entry in train_items:
        layer = parse_layer_idx(train_key)
        block = infer_block(train_key)
        mapping_rollout_key = mapping_key.get(train_key)
        peers = []
        if mapping_rollout_key and isinstance(mapping_rollout_key, list):
            for key in mapping_rollout_key:
                peers.append(rollout_idx.get(key, []))
        elif mapping_rollout_key and isinstance(mapping_rollout_key, str):
            peers = rollout_idx.get(mapping_rollout_key, [])

        cardinality = "1:1" if len(peers) == 1 else ("1:N" if len(peers) > 1 else "1:0")
        mapping_rows.append(
            {
                "train_key": train_key,
                "rollout_keys": peers,
                "layer_index": layer if layer is not None else "-",
                "block": block,
                "cardinality": cardinality,
                "confidence": "high" if peers else "low",
                "notes": "" if peers else "no matched rollout key",
            }
        )

        if peers:
            output2_map[train_key] = peers

        for rk in peers:
            cmp_rows = compare_values(train_entry, rollout_data.get(rk, {}))
            for cr in cmp_rows:
                value_rows.append({"train_key": train_key, "rollout_key": rk, **cr})
            module_rows.append(
                build_module_summary(
                    train_key=train_key,
                    rollout_key=rk,
                    compare_rows=cmp_rows,
                    train_entry=train_entry,
                    rollout_entry=rollout_data.get(rk, {}),
                )
            )

    # output_1
    out1_json = {
        "meta": {
            "train_path": str(train_path),
            "rollout_path": str(rollout_path),
            "order_basis": "train_data_insertion_order",
            "left_key": "train_key",
        },
        "rows": mapping_rows,
    }
    (out_dir / "output_1_key_mapping.json").write_text(json.dumps(out1_json, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")

    with (out_dir / "output_1_key_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_key", "rollout_key", "cardinality", "layer_index", "block", "confidence", "notes"])
        for r in mapping_rows:
            peers = r["rollout_keys"] or [""]
            for p in peers:
                w.writerow(
                    [r["train_key"], p, r["cardinality"], r["layer_index"], r["block"], r["confidence"], r["notes"]])

    # output_2
    (out_dir / "output_2_mapping.json").write_text(json.dumps(output2_map, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")

    with (out_dir / "output_2_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["mapping_type", "train_key", "rollout_key", "layer_index", "block", "confidence", "notes"])
        for r in mapping_rows:
            peers = r["rollout_keys"]
            if not peers:
                continue
            mt = "1:1" if len(peers) == 1 else "1:N"
            for p in peers:
                w.writerow([mt, r["train_key"], p, r["layer_index"], r["block"], r["confidence"], r["notes"]])

    # output_3
    priority_counts: Dict[str, int] = defaultdict(int)
    missing_compare_target = 0
    for row in value_rows:
        bucket = row.get("priority_bucket")
        if bucket == "missing_compare_target":
            missing_compare_target += 1
        elif bucket:
            priority_counts[bucket] += 1

    out3_json = {
        "meta": {
            "eps": EPS,
            "thresholds": {"ok": THRESHOLDS_OK, "warn": THRESHOLDS_WARN},
            "order_basis": "train_data_insertion_order",
            "left_key": "train_key",
            "priority_policy": [
                "P1: value mismatch + shape match",
                "P1: value mismatch + numel match",
                "P2: value mismatch + numel mismatch",
                "P3: value match + shape mismatch",
                "P4: value match + shape match",
                "P5: parameter mismatch + output aligned",
            ],
            "priority_counts": dict(priority_counts),
            "missing_compare_target": missing_compare_target,
        },
        "records": value_rows,
    }
    (out_dir / "output_3_value_compare.json").write_text(json.dumps(out3_json, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")

    with (out_dir / "output_3_value_compare.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "train_key",
                "rollout_key",
                "compare_target",
                "dtype_status",
                "shape_status",
                "shape_relation",
                "numeric_consistency",
                "priority_rank",
                "priority_bucket",
                "metric_status",
                "value_status",
                "metric_summary",
            ]
        )
        for r in value_rows:
            w.writerow(
                [
                    r.get("train_key", ""),
                    r.get("rollout_key", ""),
                    r.get("compare_target", ""),
                    r.get("dtype_status", ""),
                    r.get("shape_status", ""),
                    r.get("shape_relation", ""),
                    r.get("numeric_consistency", ""),
                    r.get("priority_rank", ""),
                    r.get("priority_bucket", ""),
                    r.get("metric_status", ""),
                    r.get("value_status", ""),
                    compact_metric_summary(r),
                ]
            )

    out4_json = {
        "meta": {
            "order_basis": "train_data_insertion_order",
            "module_policy": [
                "Step1: inspect output first",
                "If output aligned, internal differences are downgraded unless comparison premise is broken",
                "If output mismatched, inspect parameters before input_args and input_kwargs",
                "Module-level buckets: module_impl_mismatch, parameter_sync_mismatch, upstream_propagation_or_amplification, systemic_misalignment",
            ],
        },
        "records": module_rows,
    }
    (out_dir / "output_4_module_analysis.json").write_text(json.dumps(out4_json, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")

    with (out_dir / "output_4_module_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "train_key",
                "rollout_key",
                "module_priority_rank",
                "module_status",
                "reasoning",
            ]
        )
        for r in module_rows:
            w.writerow(
                [
                    r.get("train_key", ""),
                    r.get("rollout_key", ""),
                    r.get("module_priority_rank", ""),
                    r.get("module_status", ""),
                    r.get("reasoning", ""),
                ]
            )

    analysis_lines = build_analysis_summary(mapping_rows, value_rows, module_rows, out_dir)
    print("\n".join(analysis_lines))


if __name__ == "__main__":
    main()

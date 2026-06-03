#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from utils import (
    TIME_TABLE_CANDIDATES,
    connect_readonly,
    emit_payload,
    first_existing_table,
    json_dump,
    op_type_from_name,
    safe_float,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect cross-rank timing evidence for selected Ascend communication ops")
    parser.add_argument("--db", required=True)
    parser.add_argument("--op-name", action="append", required=True, help="Selected communication op name. Repeat --op-name to pass multiple selected ops in one call.")
    parser.add_argument("--step-id", action="append", help="Optional selected step id. Repeat --step-id to allow multiple selected steps.")
    parser.add_argument("--output-csv", help="Optional CSV output path for selected-op timing evidence rows")
    parser.add_argument("--format", choices=["json", "md"], default="json")
    return parser


def wait_pattern_label(start_span_ms: float, end_span_ms: float, duration_skew_ms: float, earliest_rank: str, longest_rank: str) -> tuple[str, str]:
    if start_span_ms > 0 and end_span_ms <= max(1.0, start_span_ms * 0.1) and duration_skew_ms > 0 and earliest_rank == longest_rank:
        return "strong_wait_like", "large start skew, small end skew, and earliest-start rank is also the longest-duration rank"
    if start_span_ms > 0 and end_span_ms < start_span_ms and duration_skew_ms > 0:
        return "weak_wait_like", "start skew and duration skew are visible while end skew stays relatively smaller"
    return "not_obvious", "timing alignment alone does not show a strong wait-like pattern"


def collect_wait_evidence(args: argparse.Namespace) -> dict:
    selected_op_names = set(args.op_name or [])
    selected_steps = set(str(x) for x in (args.step_id or []))
    if not selected_op_names:
        raise ValueError("At least one non-empty --op-name must be provided")
    if args.step_id and not selected_steps:
        raise ValueError("When --step-id is provided, at least one non-empty step id is required")

    with connect_readonly(args.db) as conn:
        time_table = first_existing_table(conn, TIME_TABLE_CANDIDATES)
        if not time_table:
            raise RuntimeError(f"Missing tables: {TIME_TABLE_CANDIDATES}")

        filters = ["hccl_op_name IN ({})".format(",".join("?" for _ in selected_op_names))]
        params = list(selected_op_names)
        if selected_steps:
            filters.append("CAST(step AS TEXT) IN ({})".format(",".join("?" for _ in selected_steps)))
            params.extend(selected_steps)
        rows = conn.execute(
            f"""
            SELECT
              step,
              hccl_op_name,
              COALESCE(group_name, '') AS communication_group,
              rank_id,
              start_timestamp,
              elapsed_time,
              wait_time,
              transit_time
            FROM {time_table}
            WHERE {" AND ".join(filters)}
            ORDER BY step, group_name, hccl_op_name, start_timestamp
            """,
            params,
        ).fetchall()
        end_scale = 1000.0
        start_scale = 1000.0

    groups: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        key = (str(row[0]), str(row[1]), str(row[2]))
        start = safe_float(row[4])
        elapse = safe_float(row[5])
        groups[key].append(
            {
                "step_id": row[0],
                "op_name": row[1],
                "communication_group": row[2],
                "rank_id": str(row[3]),
                "start_time": start,
                "end_time": start + elapse * end_scale,
                "elapse_ms": elapse,
                "wait_ms": safe_float(row[6]),
                "transit_ms": safe_float(row[7]),
            }
        )

    timing_evidence = []
    for key, items in groups.items():
        starts = [x["start_time"] for x in items if x["start_time"]]
        ends = [x["end_time"] for x in items if x["end_time"]]
        elapses = [x["elapse_ms"] for x in items]
        waits = [x["wait_ms"] for x in items]
        if not starts or not elapses:
            continue

        earliest = min(items, key=lambda x: x["start_time"] or float("inf"))
        latest = max(items, key=lambda x: x["start_time"] or float("-inf"))
        longest = max(items, key=lambda x: x["elapse_ms"])
        shortest = min(items, key=lambda x: x["elapse_ms"])
        start_span_ms = (max(starts) - min(starts)) / start_scale
        end_span_ms = (max(ends) - min(ends)) / start_scale if ends else 0.0
        avg_elapse = sum(elapses) / len(elapses)
        avg_wait = sum(waits) / len(waits) if waits else 0.0
        wait_ratio = sum(waits) / sum(elapses) if sum(elapses) else 0.0
        max_elapse = max(elapses)
        min_elapse = min(elapses)
        max_wait = max(waits) if waits else 0.0
        min_wait = min(waits) if waits else 0.0
        duration_skew_ms = max(elapses) - min(elapses)
        duration_ratio = max(elapses) / max(min(elapses), 1e-9)
        pattern, pattern_evidence = wait_pattern_label(
            start_span_ms,
            end_span_ms,
            duration_skew_ms,
            str(earliest["rank_id"]),
            str(longest["rank_id"]),
        )

        timing_evidence.append(
            {
                "step_id": key[0],
                "op_type": op_type_from_name(key[1]),
                "op_name": key[1],
                "communication_group": key[2],
                "rank_count": len(items),
                "wait_ratio": wait_ratio,
                "avg_wait_ms": avg_wait,
                "avg_elapse_ms": avg_elapse,
                "max_wait_ms": max_wait,
                "min_wait_ms": min_wait,
                "max_elapse_ms": max_elapse,
                "min_elapse_ms": min_elapse,
                "duration_skew_ms": duration_skew_ms,
                "duration_ratio": duration_ratio,
                "start_span_ms": start_span_ms,
                "end_span_ms": end_span_ms,
                "earliest_start_rank": earliest["rank_id"],
                "latest_start_rank": latest["rank_id"],
                "longest_rank": longest["rank_id"],
                "shortest_rank": shortest["rank_id"],
                "key_rank_summary": f"longest={longest['rank_id']}; shortest={shortest['rank_id']}; earliest={earliest['rank_id']}; latest={latest['rank_id']}",
                "wait_pattern": pattern,
                "pattern_evidence": pattern_evidence,
            }
        )

    timing_evidence.sort(
        key=lambda x: (
            -safe_float(x["wait_ratio"]),
            -safe_float(x["duration_skew_ms"]),
            -safe_float(x["start_span_ms"]),
        )
    )
    payload = {
        "title": "Ascend Selected Communication Op Timing Evidence",
        "summary": {
            "db": args.db,
            "selected_op_names": sorted(selected_op_names),
            "selected_step_ids": sorted(selected_steps),
            "note": "This script only collects timing evidence. It does not classify wait or straggler causes.",
        },
        "timing_evidence": timing_evidence,
    }
    if getattr(args, "output_csv", None):
        payload["summary"]["output_csv"] = write_csv(payload["timing_evidence"], args.output_csv)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        emit_payload(collect_wait_evidence(args), args.format)
        return 0
    except Exception as exc:
        json_dump({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())

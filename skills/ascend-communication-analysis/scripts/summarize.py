#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from utils import (
    GROUP_TABLE_CANDIDATES,
    MATRIX_TABLE_CANDIDATES,
    TIME_TABLE_CANDIDATES,
    connect_readonly,
    emit_payload,
    first_existing_table,
    json_dump,
    normalize_path,
    op_type_from_name,
    safe_float,
    safe_int,
    write_csv,
)


COMMUNICATION_STATS_SQL = """
SELECT
  t.hccl_op_name,
  COALESCE(t.group_name, '') AS communication_group_hash,
  COALESCE(g.group_id, '') AS communication_group_id,
  COALESCE(g.pg_name, '') AS communication_pg_name,
  COALESCE(g.type, '') AS communication_group_type,
  COALESCE(g.rank_set, '') AS communication_rank_set,
  SUM(t.elapsed_time) AS total_elapse_ms,
  AVG(t.elapsed_time) AS avg_elapse_ms,
  MAX(t.elapsed_time) AS max_elapse_ms
FROM {time_table} t
LEFT JOIN {group_table} g
  ON COALESCE(t.group_name, '') = g.group_name
WHERE (:include_total = 1 OR t.hccl_op_name != 'Total Op Info')
GROUP BY
  t.hccl_op_name,
  t.group_name,
  g.group_id,
  g.pg_name,
  g.type,
  g.rank_set
ORDER BY total_elapse_ms DESC;
"""


GROUP_LINK_OVERVIEW_SQL = """
SELECT
  m.group_name AS communication_group_hash,
  COALESCE(g.group_id, '') AS communication_group_id,
  COALESCE(g.pg_name, '') AS communication_pg_name,
  COALESCE(g.type, '') AS communication_group_type,
  COALESCE(g.rank_set, '') AS communication_rank_set,
  m.transport_type,
  COUNT(*) AS row_count,
  COUNT(DISTINCT m.hccl_op_name) AS op_count,
  COUNT(DISTINCT CAST(m.src_rank AS TEXT) || '->' || CAST(m.dst_rank AS TEXT)) AS rank_pair_count,
  SUM(m.transit_size) AS total_transit_size,
  SUM(m.transit_time) AS total_transit_time,
  AVG(m.bandwidth) AS avg_bandwidth,
  MIN(m.bandwidth) AS min_bandwidth,
  MAX(m.bandwidth) AS max_bandwidth,
  SUM(CASE WHEN m.bandwidth < :low_bw_threshold THEN 1 ELSE 0 END) AS low_bandwidth_row_count
FROM {matrix_table} m
LEFT JOIN {group_table} g
  ON m.group_name = g.group_name
WHERE (:include_total = 1 OR m.hccl_op_name != 'Total Op Info')
GROUP BY
  m.group_name,
  g.group_id,
  g.pg_name,
  g.type,
  g.rank_set,
  m.transport_type
ORDER BY total_transit_time DESC
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Ascend communication ops and group-level links from cluster DB")
    parser.add_argument("--db", required=True)
    parser.add_argument("--include-total", action="store_true", help="Include aggregate 'Total Op Info' rows")
    parser.add_argument("--low-bw-threshold", type=float, default=10.0, help="Bandwidth threshold for low-bandwidth row count")
    parser.add_argument("--output-stats-csv", help="Optional CSV output path for communication op statistics")
    parser.add_argument("--output-links-csv", help="Optional CSV output path for group-link overview")
    parser.add_argument("--format", choices=["json", "md"], default="json")
    return parser


def fetch_dicts(conn, sql: str, params: dict) -> list[dict]:
    cursor = conn.execute(sql, params)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def build_communication_stats(op_rows: list[dict]) -> list[dict]:
    stats_by_group_type: dict[tuple[str, str, str, str, str, str], dict] = {}
    grand_total_elapse = sum(safe_float(row.get("total_elapse_ms")) for row in op_rows)

    for row in op_rows:
        key = (
            str(row.get("communication_group_hash") or ""),
            str(row.get("communication_group_id") or ""),
            str(row.get("communication_pg_name") or ""),
            str(row.get("communication_group_type") or ""),
            str(row.get("communication_rank_set") or ""),
            op_type_from_name(str(row.get("hccl_op_name") or "")),
        )
        if key not in stats_by_group_type:
            stats_by_group_type[key] = {
                "communication_group_hash": key[0],
                "communication_group_id": key[1],
                "communication_pg_name": key[2],
                "communication_group_type": key[3],
                "communication_rank_set": key[4],
                "op_type": key[5],
                "op_count": 0,
                "total_elapse_ms": 0.0,
                "max_elapse_ms": 0.0,
            }

        stats = stats_by_group_type[key]
        stats["op_count"] += 1
        stats["total_elapse_ms"] += safe_float(row.get("total_elapse_ms"))
        stats["max_elapse_ms"] = max(safe_float(stats["max_elapse_ms"]), safe_float(row.get("max_elapse_ms")))

    communication_stats = []
    for stats in stats_by_group_type.values():
        total_elapse = safe_float(stats["total_elapse_ms"])
        op_count = safe_int(stats["op_count"])
        communication_stats.append(
            {
                "communication_group_hash": stats["communication_group_hash"],
                "communication_group_id": stats["communication_group_id"],
                "communication_pg_name": stats["communication_pg_name"],
                "communication_group_type": stats["communication_group_type"],
                "communication_rank_set": stats["communication_rank_set"],
                "op_type": stats["op_type"],
                "op_count": op_count,
                "total_elapse_ms": total_elapse,
                "elapsed_time_share": total_elapse / grand_total_elapse if grand_total_elapse else 0.0,
                "avg_elapse_per_op_ms": total_elapse / op_count if op_count else 0.0,
                "max_elapse_ms": safe_float(stats["max_elapse_ms"]),
            }
        )

    communication_stats.sort(
        key=lambda item: (
            -safe_float(item["total_elapse_ms"]),
            item["communication_pg_name"],
            item["communication_group_hash"],
            item["op_type"],
        )
    )
    return communication_stats


def summarize(args: argparse.Namespace) -> dict:
    with connect_readonly(args.db) as conn:
        time_table = first_existing_table(conn, TIME_TABLE_CANDIDATES)
        if not time_table:
            raise RuntimeError(f"Missing tables: {TIME_TABLE_CANDIDATES}")

        group_table = first_existing_table(conn, GROUP_TABLE_CANDIDATES)
        if not group_table:
            raise RuntimeError(f"Missing tables: {GROUP_TABLE_CANDIDATES}")

        matrix_table = first_existing_table(conn, MATRIX_TABLE_CANDIDATES)
        if not matrix_table:
            raise RuntimeError(f"Missing tables: {MATRIX_TABLE_CANDIDATES}")

        params = {
            "include_total": 1 if args.include_total else 0,
            "low_bw_threshold": args.low_bw_threshold,
        }
        op_rows = fetch_dicts(
            conn,
            COMMUNICATION_STATS_SQL.format(time_table=time_table, group_table=group_table),
            params,
        )
        communication_stats = build_communication_stats(op_rows)
        group_link_overview = fetch_dicts(
            conn,
            GROUP_LINK_OVERVIEW_SQL.format(matrix_table=matrix_table, group_table=group_table),
            params,
        )

    payload = {
        "title": "Ascend Communication Summary",
        "summary": {
            "db": str(normalize_path(args.db)),
            "time_table": time_table,
            "group_table": group_table,
            "matrix_table": matrix_table,
            "include_total": args.include_total,
            "low_bw_threshold": args.low_bw_threshold,
        },
        "communication_stats": communication_stats,
        "group_link_overview": group_link_overview,
    }
    if args.output_stats_csv:
        payload["summary"]["output_stats_csv"] = write_csv(communication_stats, args.output_stats_csv)
    if args.output_links_csv:
        payload["summary"]["output_links_csv"] = write_csv(group_link_overview, args.output_links_csv)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        emit_payload(summarize(args), args.format)
        return 0
    except Exception as exc:
        json_dump({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())

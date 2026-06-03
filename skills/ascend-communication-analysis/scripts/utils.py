from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any


TIME_TABLE_CANDIDATES = ["ClusterCommunicationTime"]
BANDWIDTH_TABLE_CANDIDATES = ["ClusterCommunicationBandwidth"]
MATRIX_TABLE_CANDIDATES = ["ClusterCommunicationMatrix"]
GROUP_TABLE_CANDIDATES = ["CommunicationGroupMapping"]


def normalize_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.exists():
        return raw
    text = str(path)
    match = re.match(r"^([A-Za-z]):\\(.*)$", text)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        candidate = Path(f"/mnt/{drive}/{rest}")
        if candidate.exists():
            return candidate
    return raw


def connect_readonly(db_path: str) -> sqlite3.Connection:
    db_file = normalize_path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return sqlite3.connect(str(db_file))


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def first_existing_table(conn: sqlite3.Connection, candidates: list[str]) -> str | None:
    for table in candidates:
        if table_exists(conn, table):
            return table
    return None


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    if not table_exists(conn, table):
        return []
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and not value.strip():
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_integer_part_from_float_num(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def op_type_from_name(op_name: str) -> str:
    value = str(op_name or "")
    value = value.split("@", 1)[0]
    value = value.split("__", 1)[0]
    if value.startswith("hcom_"):
        return value[len("hcom_") :]
    return value.split("_", 1)[0] if value else "unknown"


def split_op_and_suffix(op_key: str) -> tuple[str, str]:
    if "@" in op_key:
        left, right = op_key.split("@", 1)
        return left, right
    return op_key, ""


def json_dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def ensure_parent_dir(path: str | Path) -> Path:
    target = normalize_path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> str:
    target = ensure_parent_dir(output_csv)
    if not rows:
        fieldnames = ["message"]
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"message": "no rows"})
        return str(target)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, list):
                    value = json.dumps(value, ensure_ascii=False)
                normalized[key] = value
            writer.writerow(normalized)
    return str(target)


def render_md_table(rows: list[dict[str, Any]], columns_to_show: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns_to_show) + " |",
        "| " + " | ".join("---" for _ in columns_to_show) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns_to_show:
            value = row.get(col, "")
            if isinstance(value, float):
                value = round(value, 4)
            if isinstance(value, list):
                value = ", ".join(str(x) for x in value)
            cells.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def emit_payload(payload: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        json_dump(payload)
        return

    print(f"# {payload.get('title', 'Communication Analysis')}")
    if "summary" in payload:
        print("\n## Summary")
        for key, value in payload["summary"].items():
            print(f"- **{key}**: {value}")
    for table_name in ("communication_stats", "group_link_overview", "timing_evidence"):
        rows = payload.get(table_name)
        if rows is None:
            continue
        print(f"\n## {table_name}")
        if rows:
            print(render_md_table(rows, list(rows[0].keys())))
        else:
            print("_No rows._")

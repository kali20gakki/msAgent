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

"""Collect detailed Linux CPU topology from lscpu and sysfs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.cpulist import parse_cpu_list


def collect_cpu_topology(
    lscpu_e_text: str,
    smt_siblings_by_cpu: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    lines = [line for line in lscpu_e_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return {"cpus": [], "physical_cores": []}

    header = lines[0].split()
    cpus: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts, strict=False))
        cpu_id = _safe_int(row.get("CPU"))
        if cpu_id is None:
            continue
        cpus.append(
            {
                "cpu": cpu_id,
                "core_id": _safe_int(row.get("CORE")),
                "socket_id": _safe_int(row.get("SOCKET")),
                "numa_node": _safe_int(row.get("NODE")),
                "online": str(row.get("ONLINE", "")).lower() == "yes",
                "max_mhz": _safe_float(row.get("MAXMHZ")),
                "current_mhz": _safe_float(row.get("MHZ")),
                "smt_siblings": list(smt_siblings_by_cpu.get(cpu_id, [cpu_id])) if smt_siblings_by_cpu else [cpu_id],
            }
        )

    return {"cpus": cpus, "physical_cores": _aggregate_physical_cores(cpus)}


def collect_smt_siblings(sys_root: Path = Path("/sys")) -> dict[int, list[int]]:
    cpu_root = sys_root / "devices" / "system" / "cpu"
    online = _read_text(cpu_root / "online")
    if online:
        cpu_ids = sorted(parse_cpu_list(online.strip()))
    else:
        cpu_ids = sorted(_cpu_ids_from_cpu_dirs(cpu_root))

    result: dict[int, list[int]] = {}
    for cpu_id in cpu_ids:
        sibling_text = _read_text(cpu_root / f"cpu{cpu_id}" / "topology" / "thread_siblings_list")
        if sibling_text:
            try:
                result[cpu_id] = sorted(parse_cpu_list(sibling_text.strip()))
                continue
            except ValueError:
                pass
        result[cpu_id] = [cpu_id]
    return result


def collect_distance_matrix(sys_root: Path = Path("/sys")) -> list[list[int]]:
    node_root = sys_root / "devices" / "system" / "node"
    rows: list[list[int]] = []
    try:
        entries = sorted(node_root.iterdir(), key=lambda entry: _node_sort_key(entry.name))
    except (FileNotFoundError, PermissionError, OSError):
        return rows

    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith("node"):
            continue
        text = _read_text(entry / "distance")
        if not text:
            continue
        try:
            rows.append([int(item) for item in text.split()])
        except ValueError:
            continue
    return rows


def _aggregate_physical_cores(cpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cpu in cpus:
        socket_id = cpu.get("socket_id")
        core_id = cpu.get("core_id")
        if socket_id is None or core_id is None:
            continue
        groups.setdefault((int(socket_id), int(core_id)), []).append(cpu)

    cores: list[dict[str, Any]] = []
    for (socket_id, core_id), entries in sorted(groups.items()):
        logical_cpus = sorted(int(entry["cpu"]) for entry in entries)
        numa_node = next(
            (entry.get("numa_node") for entry in entries if entry.get("numa_node") is not None),
            None,
        )
        cores.append(
            {
                "core_key": f"socket{socket_id}-core{core_id}",
                "socket_id": socket_id,
                "core_id": core_id,
                "numa_node": numa_node,
                "logical_cpus": logical_cpus,
            }
        )
    return cores


def _cpu_ids_from_cpu_dirs(cpu_root: Path) -> set[int]:
    cpu_ids: set[int] = set()
    try:
        entries = list(cpu_root.iterdir())
    except (FileNotFoundError, PermissionError, OSError):
        return cpu_ids
    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith("cpu"):
            continue
        value = _safe_int(entry.name[3:])
        if value is not None:
            cpu_ids.add(value)
    return cpu_ids


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _node_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("node"):
        value = _safe_int(name[4:])
        if value is not None:
            return value, name
    return 10**9, name


def _safe_int(  # pylint: disable=duplicate-code
    value: str | None,
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

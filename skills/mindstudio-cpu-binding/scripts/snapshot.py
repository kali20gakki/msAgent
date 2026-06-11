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

"""Snapshot loading and lightweight access helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Snapshot = dict[str, Any]


def load_snapshot(path: str | Path) -> Snapshot:
    snapshot_path = Path(path)
    try:
        with snapshot_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid snapshot JSON {snapshot_path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("snapshot root must be a JSON object")
    data.setdefault("availability", {})
    data["availability"].setdefault("missing", [])
    data["availability"].setdefault("partial", [])
    data["availability"].setdefault("errors", [])
    return data


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def get_cgroup_for_pid(snapshot: Snapshot, pid: int) -> dict[str, Any] | None:
    for group in snapshot.get("cgroup", {}).get("process_groups", []):
        if int(group.get("pid", -1)) == int(pid):
            return group
    return None


def get_rank_mapping_for_pid(snapshot: Snapshot, pid: int) -> dict[str, Any] | None:
    for mapping in snapshot.get("workload", {}).get("rank_mapping", []):
        if int(mapping.get("pid", -1)) == int(pid):
            return mapping
    return None


def get_npu_for_device(snapshot: Snapshot, device_id: str | int | None) -> dict[str, Any] | None:
    if device_id is None:
        return None
    normalized = str(device_id).replace("npu", "")
    for device in snapshot.get("npu_topology", {}).get("devices", []):
        candidates = {str(device.get("device_id")), str(device.get("logical_id"))}
        if normalized in {value.replace("npu", "") for value in candidates}:
            return device
    return None


def effective_cpu_list(snapshot: Snapshot, process: dict[str, Any]) -> str | None:
    group = get_cgroup_for_pid(snapshot, int(process.get("pid", -1)))
    if group and group.get("cpuset_cpus_effective"):
        return group.get("cpuset_cpus_effective")
    return process.get("cpus_allowed_list")

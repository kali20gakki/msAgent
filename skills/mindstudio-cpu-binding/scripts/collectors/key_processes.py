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

"""Summarize key processes and threads for snapshot reporting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

KEY_CLASS_BUCKETS = {
    "communication": "communication_threads",
    "dataloader": "dataloader_threads",
    "npu_fixed": "npu_fixed_threads",
    "sq_task": "sq_task_threads",
}


def collect_key_processes(
    processes: list[dict[str, Any]],
    npu_topology: dict[str, Any],
    target_pids: list[int],
    extra_keywords: list[str] | None = None,
    proc_root: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "discovery_sources": [],
        "main_scheduler_pids": [],
        "sq_task_threads": [],
        "npu_fixed_threads": [],
        "dataloader_threads": [],
        "communication_threads": [],
        "top_threads": [],
    }

    target_pid_set = {int(pid) for pid in target_pids}
    target_threads = _target_key_threads(processes, target_pid_set)
    if target_threads:
        result["discovery_sources"].append("target_pids")
    _add_target_threads(result, target_threads)

    npu_smi_pids = _npu_smi_host_pids(npu_topology)
    if npu_smi_pids:
        result["discovery_sources"].append("npu_smi_info")
        result["npu_smi_host_pids"] = npu_smi_pids

    if proc_root is not None:
        sq_threads = scan_sq_task_threads(proc_root, target_pids=target_pid_set)
        if sq_threads:
            result["discovery_sources"].append("sq_pattern")
            result["sq_task_threads"] = _dedupe_threads(result["sq_task_threads"] + sq_threads)

    extra_matches = _extra_keyword_matches(processes, extra_keywords or [])
    if extra_matches:
        result["discovery_sources"].append("user_extra")
        result["user_extra_matches"] = extra_matches

    result["main_scheduler_pids"] = sorted(set(result["main_scheduler_pids"]))
    result["top_threads"] = sorted(target_threads, key=_top_thread_sort_key)
    return result


def scan_sq_task_threads(proc_root: Path = Path("/proc"), target_pids: set[int] | None = None) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    try:
        pid_dirs = sorted(proc_root.iterdir(), key=lambda entry: _safe_int(entry.name) or -1)
    except (FileNotFoundError, PermissionError, OSError):
        return matches

    for pid_dir in pid_dirs:
        pid = _safe_int(pid_dir.name)
        if pid is None or not pid_dir.is_dir():
            continue
        if target_pids and pid not in target_pids:
            continue
        task_dir = pid_dir / "task"
        try:
            task_dirs = sorted(task_dir.iterdir(), key=lambda entry: _safe_int(entry.name) or -1)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for task in task_dirs:
            tid = _safe_int(task.name)
            if tid is None or not task.is_dir():
                continue
            name = _read_text(task / "comm").strip()
            npu_id = _sq_npu_id(name)
            if npu_id is None:
                continue
            matches.append({"npu_id": npu_id, "pid": pid, "tid": tid, "name": name})
    return matches


def _target_key_threads(processes: list[dict[str, Any]], target_pids: set[int]) -> list[dict[str, Any]]:
    threads: list[dict[str, Any]] = []
    for process in processes:
        pid = int(process["pid"])
        if target_pids and pid not in target_pids:
            continue
        for thread in process.get("threads", []):
            key_class = thread.get("key_class") or "unknown"
            if key_class == "unknown":
                continue
            threads.append(_thread_summary(process, thread))
    return threads


def _top_thread_sort_key(thread: dict[str, Any]) -> tuple[float, int, int]:
    cpu_percent = thread.get("cpu_percent")
    if cpu_percent is not None:
        return (
            -float(cpu_percent),
            -int(thread.get("key_score") or 0),
            int(thread["tid"]),
        )
    return (0.0, -int(thread.get("key_score") or 0), int(thread["tid"]))


def _add_target_threads(result: dict[str, Any], threads: list[dict[str, Any]]) -> None:
    for thread in threads:
        key_class = thread.get("key_class")
        if key_class == "main_scheduler":
            result["main_scheduler_pids"].append(int(thread["pid"]))
            continue
        bucket = KEY_CLASS_BUCKETS.get(str(key_class))
        if bucket is not None:
            result[bucket].append(thread)


def _thread_summary(process: dict[str, Any], thread: dict[str, Any]) -> dict[str, Any]:
    return {
        "pid": int(process["pid"]),
        "tid": int(thread["tid"]),
        "name": thread.get("name") or "",
        "key_class": thread.get("key_class") or "unknown",
        "key_score": int(thread.get("key_score") or 0),
        "current_cpu": thread.get("current_cpu"),
        "numa_node": thread.get("numa_node"),
        "cpu_percent": thread.get("cpu_percent"),
        "cpus_allowed_list": thread.get("cpus_allowed_list") or "",
        "npu_device": thread.get("npu_device") or process.get("npu_device"),
    }


def _npu_smi_host_pids(npu_topology: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for npu_id, pids in sorted(
        npu_topology.get("processes_by_device", {}).items(),
        key=lambda item: str(item[0]),
    ):
        for pid in pids:
            result.append({"npu_id": str(npu_id), "pid": int(pid)})
    return result


def _extra_keyword_matches(processes: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    normalized = [keyword.lower() for keyword in keywords if keyword]
    if not normalized:
        return matches

    for process in processes:
        pid = int(process["pid"])
        command = str(process.get("command") or process.get("comm") or "")
        for keyword in normalized:
            if keyword in command.lower():
                matches.append({"keyword": keyword, "pid": pid, "command": command})
        for thread in process.get("threads", []):
            name = str(thread.get("name") or "")
            for keyword in normalized:
                if keyword in name.lower():
                    matches.append(
                        {
                            "keyword": keyword,
                            "pid": pid,
                            "tid": int(thread["tid"]),
                            "name": name,
                        }
                    )
    return matches


def _dedupe_threads(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    result: list[dict[str, Any]] = []
    for thread in threads:
        key = (int(thread["pid"]), int(thread["tid"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(thread)
    return result


def _sq_npu_id(name: str) -> int | None:
    match = re.fullmatch(r"dev(\d+)_sq(?:_task)?", name)
    if not match:
        return None
    return int(match.group(1))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _safe_int(  # pylint: disable=duplicate-code
    value: Any,
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

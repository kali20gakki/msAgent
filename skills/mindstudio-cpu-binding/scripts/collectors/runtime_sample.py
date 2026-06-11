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

"""Sample runtime CPU usage from /proc task stat snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.collectors.availability import Availability


class ProcSampleReader:
    def __init__(self, root: Path = Path("/proc")) -> None:
        self._root = root

    def snapshot(self, pids: list[int]) -> dict[int, dict[int, dict[str, int | str | None]]]:
        result: dict[int, dict[int, dict[str, int | str | None]]] = {}
        for pid in pids:
            tasks = self._snapshot_pid(pid)
            if tasks:
                result[pid] = tasks
        return result

    def read_loadavg(self) -> str:
        try:
            return (self._root / "loadavg").read_text(encoding="utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    def _snapshot_pid(self, pid: int) -> dict[int, dict[str, int | str | None]]:
        task_dir = self._root / str(pid) / "task"
        try:
            entries = sorted(task_dir.iterdir(), key=lambda entry: _safe_int(entry.name) or -1)
        except (FileNotFoundError, PermissionError, OSError):
            return {}

        tasks: dict[int, dict[str, int | str | None]] = {}
        for entry in entries:
            tid = _safe_int(entry.name)
            if tid is None or not entry.is_dir():
                continue
            stat = _read_text(entry / "stat")
            parsed = _parse_task_stat(stat or "")
            if parsed is not None:
                tasks[tid] = parsed
        return tasks


def collect_proc_runtime_sample(
    before: dict[int, dict[int, dict[str, int | str | None]]],
    after: dict[int, dict[int, dict[str, int | str | None]]],
    sample_seconds: float,
    clock_ticks: int,
    top_threads: int,
    cpu_to_numa: dict[int, int],
    loadavg_text: str,
) -> tuple[dict[str, Any], Availability]:
    availability = Availability()
    thread_samples: list[dict[str, Any]] = []
    process_totals: dict[int, float] = {}
    cpu_migration_observed = False
    usage_by_numa: dict[int, float] = {}

    for pid, before_threads in before.items():
        after_threads = after.get(pid, {})
        for tid, start in before_threads.items():
            end = after_threads.get(tid)
            if end is None:
                availability.add_partial(f"runtime_sample.threads[{tid}]")
                continue
            cpu_percent = _cpu_percent(start, end, sample_seconds, clock_ticks)
            start_processor = start.get("processor")
            end_processor = end.get("processor")
            if start_processor != end_processor:
                cpu_migration_observed = True
            sample = {
                "pid": pid,
                "tid": tid,
                "name": str(end.get("name") or start.get("name") or ""),
                "cpu_percent": cpu_percent,
                "current_cpu": end_processor,
                "numa_node": cpu_to_numa.get(end_processor) if end_processor is not None else None,
            }
            thread_samples.append(sample)
            process_totals[pid] = process_totals.get(pid, 0.0) + cpu_percent
            numa_node = sample["numa_node"]
            if numa_node is not None:
                usage_by_numa[int(numa_node)] = usage_by_numa.get(int(numa_node), 0.0) + cpu_percent

    if not thread_samples:
        availability.add_missing("runtime_sample.top_threads")

    sorted_threads = sorted(thread_samples, key=lambda item: (-float(item["cpu_percent"]), int(item["tid"])))
    top = sorted_threads[: max(top_threads, 0)]
    return (
        {
            "sample_seconds": sample_seconds,
            "process_cpu_percent_total": round(sum(process_totals.values()), 2),
            "process_cpu_percent_by_pid": {pid: round(cpu_percent, 2) for pid, cpu_percent in process_totals.items()},
            "system_loadavg": _parse_loadavg(loadavg_text),
            "cpu_usage_by_numa": [
                {"numa_node": node, "cpu_percent": round(cpu_percent, 2)}
                for node, cpu_percent in sorted(usage_by_numa.items())
            ],
            "threads": sorted_threads,
            "top_threads": top,
            "cpu_migration_observed": cpu_migration_observed,
        },
        availability,
    )


def merge_runtime_sample(processes: list[dict[str, Any]], sample: dict[str, Any]) -> None:
    cpu_by_thread = {
        (int(item["pid"]), int(item["tid"])): float(item["cpu_percent"]) for item in sample.get("threads", [])
    }
    cpu_by_pid = {int(pid): float(value) for pid, value in sample.get("process_cpu_percent_by_pid", {}).items()}
    for process in processes:
        pid = int(process["pid"])
        for thread in process.get("threads", []):
            key = (pid, int(thread["tid"]))
            if key in cpu_by_thread:
                thread["cpu_percent"] = cpu_by_thread[key]
        process["cpu_percent"] = round(cpu_by_pid.get(pid, 0.0), 2)


def _parse_task_stat(stat_text: str) -> dict[str, int | str | None] | None:
    lpar = stat_text.find("(")
    rpar = stat_text.rfind(")")
    if lpar == -1 or rpar == -1 or rpar <= lpar:
        return None
    name = stat_text[lpar + 1 : rpar]
    fields = stat_text[rpar + 1 :].split()
    if len(fields) <= 36:
        return None
    utime = _safe_int(fields[11])
    stime = _safe_int(fields[12])
    processor = _safe_int(fields[36])
    if utime is None or stime is None:
        return None
    return {"name": name, "utime": utime, "stime": stime, "processor": processor}


def _cpu_percent(
    start: dict[str, int | str | None],
    end: dict[str, int | str | None],
    sample_seconds: float,
    clock_ticks: int,
) -> float:
    start_ticks = int(start.get("utime") or 0) + int(start.get("stime") or 0)
    end_ticks = int(end.get("utime") or 0) + int(end.get("stime") or 0)
    if sample_seconds <= 0 or clock_ticks <= 0:
        return 0.0
    delta_ticks = max(0, end_ticks - start_ticks)
    return round((delta_ticks / clock_ticks) / sample_seconds * 100, 2)


def _parse_loadavg(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split()[:3]:
        try:
            values.append(float(token))
        except ValueError:
            break
    return values


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _safe_int(  # pylint: disable=duplicate-code
    value: Any,
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

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

"""Collect process and thread state by reading /proc only.

The collector is split so it can be tested without root or a real /proc:
- `ProcReader` abstracts the root path (production: /proc; tests: tmp dir).
- `classify_thread` is pure and pattern-driven; see KEY_CLASS_PATTERNS.
- `collect_process` reads status / stat / cmdline for the process and
  for each task/<tid>/, returning a dict shaped for the Snapshot schema.

What this collector does NOT do
-------------------------------
- It does not resolve a thread's `numa_node` from its `current_cpu`.
  That cross-references the NUMA topology and is done by the top-level
  collect.py flow once numa_topology is populated.
- It does not write `rank` / `npu_device`. Those are injected from the
  user-supplied --rank-map.
- It does not sample CPU% — runtime_sample backfills those fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from scripts.collectors.availability import Availability
from scripts.collectors.sanitize import sanitize_cmdline

# Ordered list: first match wins. Lower-priority entries (e.g. engine_worker
# whose pattern is broad) come last so dataloader / tokenizer get classified
# correctly first.
KEY_CLASS_PATTERNS: list[tuple[str, str]] = [
    ("sq_task", r"^dev(\d+)_sq(?:_task)?$"),
    (
        "npu_fixed",
        r"^(release_thread|acl_thread|pt_data_pin|pt_autograd_\d+|data_pin)$",
    ),
    ("communication", r"hccl|nccl|comm(?:unication)?"),
    ("blas_worker", r"blas|mkl|openblas"),
    ("openmp_worker", r"omp|openmp"),
    ("dataloader", r"dataloader|data_worker|pt_data"),
    ("api_server", r"api[_-]?server|openai"),
    ("scheduler", r"scheduler|manager"),
    ("tokenizer", r"tokenizer"),
    ("engine_worker", r"worker|engine|local[_-]?rank"),
]

KEY_SCORE: dict[str, int] = {
    "main_scheduler": 100,
    "sq_task": 95,
    "communication": 85,
    "npu_fixed": 80,
    "dataloader": 70,
    "engine_worker": 65,
    "tokenizer": 60,
    "scheduler": 60,
    "api_server": 55,
    "blas_worker": 50,
    "openmp_worker": 50,
    "unknown": 0,
}


class ProcReader:
    def __init__(self, root: Path = Path("/proc")) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def read(self, pid: int, relpath: str) -> str | None:
        try:
            return (self._root / str(pid) / relpath).read_text(encoding="utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError, OSError):
            return None

    def exists(self, pid: int) -> bool:
        return (self._root / str(pid)).is_dir()


def classify_thread(tid: int, pid: int, name: str) -> str:
    for cls_name, pattern in KEY_CLASS_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return cls_name
    if tid == pid:
        return "main_scheduler"
    return "unknown"


def key_score_for(key_class: str) -> int:
    return KEY_SCORE.get(key_class, 0)


def collect_process(reader: ProcReader, pid: int) -> tuple[dict[str, Any] | None, Availability]:
    avail = Availability()
    if not reader.exists(pid):
        avail.add_error(f"linux_proc[{pid}]", f"PID {pid} not found under {reader.root}")
        return None, avail

    status = reader.read(pid, "status") or ""
    stat = reader.read(pid, "stat") or ""
    cmdline = reader.read(pid, "cmdline") or ""

    process: dict[str, Any] = {
        "pid": pid,
        "ppid": _safe_int(_parse_status_field(status, "PPid")),
        "comm": _parse_status_field(status, "Name") or "",
        "state": _parse_status_field(status, "State") or "",
        "command": sanitize_cmdline(cmdline),
        "rank": None,
        "npu_device": None,
        "cpus_allowed_list": _parse_status_field(status, "Cpus_allowed_list") or "",
        "mems_allowed_list": _parse_status_field(status, "Mems_allowed_list") or "",
        "current_cpu": _parse_stat_processor(stat),
        "num_threads": _safe_int(_parse_status_field(status, "Threads")),
        "voluntary_ctxt_switches": _safe_int(_parse_status_field(status, "voluntary_ctxt_switches")),
        "nonvoluntary_ctxt_switches": _safe_int(_parse_status_field(status, "nonvoluntary_ctxt_switches")),
        "cpu_percent": None,
        "nspid_chain": _parse_nspid(status),
        "threads": _collect_threads(reader, pid),
    }
    return process, avail


def _collect_threads(reader: ProcReader, pid: int) -> list[dict[str, Any]]:
    task_dir = reader.root / str(pid) / "task"
    threads: list[dict[str, Any]] = []
    try:
        entries = sorted(task_dir.iterdir(), key=lambda entry: _safe_tid(entry.name))
    except (FileNotFoundError, PermissionError, OSError):
        return threads

    for entry in entries:
        try:
            tid = int(entry.name)
        except ValueError:
            continue
        if not entry.is_dir():
            continue

        tstatus = reader.read(pid, f"task/{tid}/status") or ""
        tstat = reader.read(pid, f"task/{tid}/stat") or ""
        tcomm = reader.read(pid, f"task/{tid}/comm") or ""
        tname = tcomm.strip()
        if not tname:
            tname = _parse_status_field(tstatus, "Name") or ""
        kcls = classify_thread(tid, pid, tname)
        threads.append(
            {
                "tid": tid,
                "name": tname,
                "state": _parse_status_field(tstatus, "State") or "",
                "cpus_allowed_list": _parse_status_field(tstatus, "Cpus_allowed_list") or "",
                "mems_allowed_list": _parse_status_field(tstatus, "Mems_allowed_list") or "",
                "current_cpu": _parse_stat_processor(tstat),
                "numa_node": None,
                "cpu_percent": None,
                "voluntary_ctxt_switches": _safe_int(_parse_status_field(tstatus, "voluntary_ctxt_switches")),
                "nonvoluntary_ctxt_switches": _safe_int(_parse_status_field(tstatus, "nonvoluntary_ctxt_switches")),
                "role_hint": kcls,
                "key_class": kcls,
                "key_score": key_score_for(kcls),
            }
        )
    return threads


# ---------- parsers ----------

# /proc/<pid>/stat fields after the closing ')' of (comm). Index in that
# post-comm slice. Field 39 (1-based) per man 7 proc → index 36 (0-based).
_PROCESSOR_FIELD_INDEX = 36


def _parse_status_field(text: str, field: str) -> str | None:
    prefix = field + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _parse_stat_processor(stat_text: str) -> int | None:
    """Extract the `processor` field from /proc/<pid>/stat.

    The comm field can contain spaces and parentheses, so we anchor on the
    last ')' before splitting the remaining fields.
    """
    rpar = stat_text.rfind(")")
    if rpar == -1:
        return None
    fields = stat_text[rpar + 1 :].split()
    if len(fields) <= _PROCESSOR_FIELD_INDEX:
        return None
    try:
        return int(fields[_PROCESSOR_FIELD_INDEX])
    except ValueError:
        return None


def _parse_nspid(status_text: str) -> list[int]:
    line = _parse_status_field(status_text, "NSpid")
    if not line:
        return []
    chain: list[int] = []
    for token in line.split():
        try:
            chain.append(int(token))
        except ValueError:
            continue
    return chain


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _safe_tid(name: str) -> int:
    try:
        return int(name)
    except ValueError:
        return -1

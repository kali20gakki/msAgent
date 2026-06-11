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

"""Collect cgroup cpuset/quota/throttling state for a PID.

Supports cgroup v2 unified hierarchy and cgroup v1 controller-specific
hierarchies. This module only reads /proc and /sys/fs/cgroup; it never writes
controller files and never changes limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.collectors.availability import Availability


class CgroupReader:
    def __init__(
        self,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        proc_root: Path = Path("/proc"),
    ) -> None:
        self._cgroup_root = cgroup_root
        self._proc_root = proc_root

    def read_sys(self, relpath: str) -> str | None:
        try:
            return (self._cgroup_root / relpath.lstrip("/")).read_text(encoding="utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError, OSError):
            return None

    def read_proc(self, pid: int, relpath: str) -> str | None:
        try:
            return (self._proc_root / str(pid) / relpath).read_text(encoding="utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError, OSError):
            return None


def collect_cgroup_for_pid(reader: CgroupReader, pid: int) -> tuple[dict[str, Any] | None, Availability]:
    avail = Availability()
    cgroup_text = reader.read_proc(pid, "cgroup")
    if cgroup_text is None:
        avail.add_error(f"linux_cgroup[{pid}]", f"Cannot read /proc/{pid}/cgroup")
        return None, avail

    if _is_v2(cgroup_text):
        return _collect_v2(reader, pid, cgroup_text, avail)
    return _collect_v1(reader, pid, cgroup_text, avail)


def _is_v2(cgroup_text: str) -> bool:
    return any(line.startswith("0::") for line in cgroup_text.splitlines())


def _collect_v2(
    reader: CgroupReader, pid: int, cgroup_text: str, avail: Availability
) -> tuple[dict[str, Any], Availability]:
    path = _v2_path(cgroup_text)
    rel = path.lstrip("/")

    cpuset_cpus, cpuset_cpus_fallback = _read_first_available(reader, rel, "cpuset.cpus.effective")
    cpuset_mems, cpuset_mems_fallback = _read_first_available(reader, rel, "cpuset.mems.effective")
    cpu_max = _strip_or_none(reader.read_sys(_join_cgroup_path(rel, "cpu.max")))
    cpu_weight = _safe_int(_strip_or_none(reader.read_sys(_join_cgroup_path(rel, "cpu.weight"))))
    cpu_stat = reader.read_sys(_join_cgroup_path(rel, "cpu.stat"))

    field = f"cgroup.process_groups[{pid}].cpuset_cpus_effective"
    if cpuset_cpus is None:
        avail.add_missing(field)
    elif cpuset_cpus_fallback:
        avail.add_partial(field)
    if cpuset_mems_fallback:
        avail.add_partial(f"cgroup.process_groups[{pid}].cpuset_mems_effective")

    quota_us, period_us = _parse_cpu_max(cpu_max)
    group = {
        "pid": pid,
        "path": path,
        "cpuset_cpus_effective": cpuset_cpus,
        "cpuset_mems_effective": cpuset_mems,
        "cpu_max": cpu_max,
        "cpu_quota_us": quota_us,
        "cpu_period_us": period_us,
        "cpu_weight": cpu_weight,
        "nr_periods": _parse_cpu_stat_field(cpu_stat, "nr_periods"),
        "nr_throttled": _parse_cpu_stat_field(cpu_stat, "nr_throttled"),
        "throttled_usec": _parse_cpu_stat_field(cpu_stat, "throttled_usec"),
    }
    return {"version": "v2", "process_groups": [group]}, avail


def _collect_v1(
    reader: CgroupReader, pid: int, cgroup_text: str, avail: Availability
) -> tuple[dict[str, Any], Availability]:
    controller_paths = _parse_v1_controller_paths(cgroup_text)
    cpuset = controller_paths.get("cpuset")
    cpu = controller_paths.get("cpu") or controller_paths.get("cpuacct")

    cpuset_cpus, cpuset_cpus_fallback = _read_v1_cpuset_value(reader, cpuset, "cpuset.effective_cpus", "cpuset.cpus")
    cpuset_mems, cpuset_mems_fallback = _read_v1_cpuset_value(reader, cpuset, "cpuset.effective_mems", "cpuset.mems")

    cpuset_field = f"cgroup.process_groups[{pid}].cpuset_cpus_effective"
    if cpuset_cpus is None:
        avail.add_missing(cpuset_field)
    elif cpuset_cpus_fallback:
        avail.add_partial(cpuset_field)
    if cpuset_mems_fallback:
        avail.add_partial(f"cgroup.process_groups[{pid}].cpuset_mems_effective")

    quota_us = _safe_int(_strip_or_none(_read_v1_controller_file(reader, cpu, "cpu.cfs_quota_us")))
    period_us = _safe_int(_strip_or_none(_read_v1_controller_file(reader, cpu, "cpu.cfs_period_us")))
    cpu_stat = _read_v1_controller_file(reader, cpu, "cpu.stat")

    group = {
        "pid": pid,
        "path": (cpuset or cpu or {}).get("path", ""),
        "cpuset_cpus_effective": cpuset_cpus,
        "cpuset_mems_effective": cpuset_mems,
        "cpu_max": None,
        "cpu_quota_us": quota_us,
        "cpu_period_us": period_us,
        "cpu_weight": None,
        "nr_periods": _parse_cpu_stat_field(cpu_stat, "nr_periods"),
        "nr_throttled": _parse_cpu_stat_field(cpu_stat, "nr_throttled"),
        "throttled_usec": _parse_cpu_stat_field(cpu_stat, "throttled_time"),
    }
    return {"version": "v1", "process_groups": [group]}, avail


def _v2_path(cgroup_text: str) -> str:
    for line in cgroup_text.splitlines():
        if line.startswith("0::"):
            return line.split(":", 2)[2] or "/"
    return "/"


def _read_first_available(reader: CgroupReader, rel: str, filename: str) -> tuple[str | None, bool]:
    """Read filename from rel or nearest parent.

    Some v2 leaf cgroups do not expose cpuset.* when the controller is only
    enabled higher in the tree. The effective value still inherits from an
    ancestor, so walking parents gives the operational limit while marking the
    field partial to preserve provenance.
    """
    current = Path(rel) if rel else Path(".")
    tried_parent = False
    while True:
        target = filename if str(current) == "." else str(current / filename)
        value = _strip_or_none(reader.read_sys(target))
        if value is not None:
            return value, tried_parent
        if str(current) == ".":
            return None, tried_parent
        current = current.parent
        tried_parent = True


def _join_cgroup_path(rel: str, filename: str) -> str:
    return str(Path(rel) / filename) if rel else filename


def _parse_v1_controller_paths(cgroup_text: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for line in cgroup_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        mount = parts[1]
        path = parts[2] or "/"
        for controller in mount.split(","):
            if controller:
                result[controller] = {"mount": mount, "path": path}
    return result


def _read_v1_cpuset_value(
    reader: CgroupReader, controller: dict[str, str] | None, primary: str, fallback: str
) -> tuple[str | None, bool]:
    if controller is None:
        return None, False
    rel = Path(controller["mount"]) / controller["path"].lstrip("/")
    primary_value = _strip_or_none(reader.read_sys(str(rel / primary)))
    if primary_value is not None:
        return primary_value, False
    return _strip_or_none(reader.read_sys(str(rel / fallback))), True


def _read_v1_controller_file(reader: CgroupReader, controller: dict[str, str] | None, filename: str) -> str | None:
    if controller is None:
        return None
    path = controller["path"].lstrip("/")
    for mount in _v1_mount_candidates(controller["mount"]):
        value = reader.read_sys(str(Path(mount) / path / filename))
        if value is not None:
            return value
    return None


def _v1_mount_candidates(mount: str) -> list[str]:
    if mount == "cpu,cpuacct":
        return [mount, "cpu"]
    if mount == "cpu":
        return [mount, "cpu,cpuacct"]
    return [mount]


def _parse_cpu_max(cpu_max: str | None) -> tuple[int | None, int | None]:
    if not cpu_max:
        return None, None
    parts = cpu_max.split()
    if len(parts) < 2:
        return None, None
    quota = None if parts[0] == "max" else _safe_int(parts[0])
    return quota, _safe_int(parts[1])


def _parse_cpu_stat_field(stat_text: str | None, field: str) -> int | None:
    if not stat_text:
        return None
    for line in stat_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == field:
            return _safe_int(parts[1])
    return None


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _safe_int(  # pylint: disable=duplicate-code
    value: str | None,
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

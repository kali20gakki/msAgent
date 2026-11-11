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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""procfs.py — /proc、/sys 读取子集（配置项/亲和/环境变量）。

只读、不抛异常降级返回 None/空结构，由调用方标注 view_limited。
安全约束：只读脚本内部拼出的固定路径（/proc、/sys 与 /proc/<pid>/*），
不接受外部传入的任意路径。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


def _read_text(path: str, max_bytes: int = 4 * 1024 * 1024) -> Optional[str]:
    """安全读取文本文件（限制大小，失败返回 None）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_bytes)
    except (OSError, ValueError):
        return None


def read_config(key: str) -> Optional[str]:
    """读取 /proc/sys 单配置项，去尾换行；未知键返回 None。"""
    mapping = {
        "numa_balancing": "/proc/sys/kernel/numa_balancing",
        "zone_reclaim_mode": "/proc/sys/vm/zone_reclaim_mode",
        "swappiness": "/proc/sys/vm/swappiness",
        "overcommit_memory": "/proc/sys/vm/overcommit_memory",
        "overcommit_ratio": "/proc/sys/vm/overcommit_ratio",
    }
    path = mapping.get(key)
    if path is None:
        return None
    text = _read_text(path, 4096)
    if text is None:
        return None
    return text.strip()


def read_meminfo() -> Dict[str, int]:
    """读取 /proc/meminfo 关键项（KB）。"""
    result: Dict[str, int] = {}
    text = _read_text("/proc/meminfo") or ""
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1].strip().split()[0])
            except (IndexError, ValueError):
                continue
    return result


def read_swaps() -> List[Dict[str, str]]:
    """读取活跃 swap 设备列表。"""
    text = _read_text("/proc/swaps") or ""
    devices = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5 and fields[1] == "partition":
            devices.append({"filename": fields[0], "size_kb": fields[2]})
    return devices


def read_thp_setting(name: str) -> Optional[str]:
    """读取 THP 配置（enabled/defrag），方括号内为生效值。"""
    path = f"/sys/kernel/mm/transparent_hugepage/{name}"
    text = _read_text(path, 4096)
    if text is None:
        return None
    match = re.search(r"\[(\w+)\]", text)
    return match.group(1) if match else text.strip()


def read_scaling_governor(cpu: int) -> Optional[str]:
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    text = _read_text(path, 4096)
    return text.strip() if text is not None else None


def read_cpu_freq(cpu: int, which: str) -> Optional[int]:
    """读取 cpufreq 值（scaling_cur_freq / cpuinfo_max_freq）。"""
    if which not in ("scaling_cur_freq", "cpuinfo_max_freq", "scaling_min_freq"):
        return None
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{which}"
    text = _read_text(path, 4096)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def read_epp(cpu: int) -> Optional[str]:
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/energy_performance_preference"
    text = _read_text(path, 4096)
    if text is None:
        return None
    return text.strip()


def read_cpuidle_states(cpu: int) -> List[str]:
    """读取 cpu 的 C-state 名单。"""
    states = []
    base = f"/sys/devices/system/cpu/cpu{cpu}/cpuidle"
    if not os.path.isdir(base):
        return states
    for name in sorted(os.listdir(base)):
        if not name.startswith("state"):
            continue
        text = _read_text(os.path.join(base, name, "name"), 4096)
        if text is not None:
            states.append(text.strip())
    return states


def read_interrupts() -> Dict[str, Dict[str, Any]]:
    """读取 /proc/interrupts：{irq_label: {"counts": [per-cpu], "desc": str}}。"""
    text = _read_text("/proc/interrupts")
    if not text:
        return {}
    lines = text.splitlines()
    if not lines:
        return {}
    try:
        ncpu = len(lines[0].split())
    except ValueError:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for line in lines[1:]:
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        label = parts[0].strip()
        rest = parts[1].split()
        counts: List[int] = []
        idx = 0
        while idx < len(rest) and idx < ncpu:
            try:
                counts.append(int(rest[idx]))
            except ValueError:
                break
            idx += 1
        desc = " ".join(rest[idx:]) if idx < len(rest) else ""
        if counts:
            result[label] = {"counts": counts, "desc": desc}
    return result


def read_softirqs() -> Dict[str, List[int]]:
    """读取 /proc/softirqs：{name: [per-cpu 计数]}。"""
    text = _read_text("/proc/softirqs")
    if not text:
        return {}
    lines = text.splitlines()
    if not lines:
        return {}
    try:
        ncpu = len(lines[0].split())
    except ValueError:
        return {}
    result: Dict[str, List[int]] = {}
    for line in lines[1:]:
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        values = parts[1].split()
        counts: List[int] = []
        for value in values[:ncpu]:
            try:
                counts.append(int(value))
            except ValueError:
                counts = []
                break
        if counts:
            result[name] = counts
    return result


def read_pid_status(pid: int) -> Dict[str, str]:
    """读取 /proc/<pid>/status 关键字段。"""
    result: Dict[str, str] = {}
    text = _read_text(f"/proc/{pid}/status", 256 * 1024)
    if text is None:
        return result
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1].strip()
    return result


def read_pid_environ(pid: int) -> Dict[str, str]:
    """读取 /proc/<pid>/environ（NUL 分隔）。权限不足返回空。"""
    env: Dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            raw = fh.read(1024 * 1024)
    except OSError:
        return env
    for item in raw.split(b"\x00"):
        try:
            entry = item.decode("utf-8", errors="replace")
        except ValueError:
            continue
        if "=" in entry:
            key, _, value = entry.partition("=")
            env[key] = value
    return env


def read_pid_cmdline(pid: int) -> str:
    """读取 /proc/<pid>/cmdline（NUL → 空格）。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read(4096)
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def read_pid_affinity(pid: int) -> Optional[str]:
    """读取 Cpus_allowed_list（优先）或 Cpus_allowed 掩码。"""
    status = read_pid_status(pid)
    if "Cpus_allowed_list" in status:
        return status["Cpus_allowed_list"]
    return status.get("Cpus_allowed")


def read_pid_numa_maps_summary(pid: int) -> Dict[str, int]:
    """读取 numa_maps 页数聚合（total_pages 只按 node 归属累计）。

    numa_maps 的页数 token 形如 ``N0=123``（node 0 占 123 页）；``N=123`` 表示
    未绑定到具体 node 的页，不计入 total_pages，避免节点分布口径失真。
    """
    summary = {"total_pages": 0}
    try:
        with open(f"/proc/{pid}/numa_maps", "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(4 * 1024 * 1024)
    except OSError:
        return summary
    for line in text.splitlines():
        for token in line.split()[1:]:
            match = re.fullmatch(r"N(\d+)=(\d+)", token)
            if match:
                summary["total_pages"] += int(match.group(2))
    return summary


def read_irq_affinity(irq: str) -> Optional[str]:
    """读取 /proc/irq/<irq>/smp_affinity_list。irq 仅允许数字或短横线字母。"""
    if not re.fullmatch(r"[0-9A-Za-z\-]+", irq):
        return None
    return _read_text(f"/proc/irq/{irq}/smp_affinity_list", 4096)


def read_cmdline_boot_params() -> str:
    return _read_text("/proc/cmdline", 4096) or ""

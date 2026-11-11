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
"""discover.py — 进程发现（NPU 锚点 + comm/cmdline + cgroup）。

NPU 锚点消费 npu_query 卡↔进程映射（本模块不直接调用 npu-smi）。
只读 /proc，pid 数值严格校验，无任意路径拼接。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import npu_query
import procfs

# comm/cmdline 匹配模式（只读白名单正则，非任意通配）
_TRAIN_PATTERNS = re.compile(r"(torchrun|msrun|deepspeed|torchelast|accelerate)", re.IGNORECASE)
_INFER_PATTERNS = re.compile(r"(vllm|mindie|sglang|tritonserver)", re.IGNORECASE)
_PY_MAIN = re.compile(r"^python[0-9.]*$")
_PID_RE = re.compile(r"^\d+$")


def _list_pids(limit: int = 4096) -> List[int]:
    """列出 /proc 下全部 pid（限流防大机扫描开销）。"""
    pids: List[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return pids
    for entry in entries:
        if _PID_RE.match(entry):
            pids.append(int(entry))
        if len(pids) >= limit:
            break
    return sorted(pids)


def _classify_process(comm: str, cmdline: str, has_npu: bool) -> str:
    """按 process-roles.md 规则分类：推理关键词优先，裸 python 才走训练兜底。"""
    if not has_npu:
        return "other"
    blob = f"{comm} {cmdline}"
    if _INFER_PATTERNS.search(blob):
        return "infer_main"
    if _TRAIN_PATTERNS.search(blob) or _PY_MAIN.match(comm):
        return "train_main"
    return "other"


def discover_npu_anchored(npu_processes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """以 npu-smi 卡↔进程映射为锚点发现目标进程（同 pid 多卡合并为一条）。"""
    targets: Dict[int, Dict[str, Any]] = {}
    for entry in npu_processes:
        pid = entry.get("pid")
        if not isinstance(pid, int):
            continue
        npu = entry.get("npu")
        target = targets.get(pid)
        if target is None:
            comm = _safe_comm(pid)
            cmdline = procfs.read_pid_cmdline(pid)
            affinity = procfs.read_pid_affinity(pid)
            target = {
                "pid": pid,
                "comm": comm,
                "cmdline": cmdline[:256],
                "npus": [],
                "affinity": affinity or "",
                "role": _classify_process(comm, cmdline, has_npu=True),
                "source": "npu_occupancy",
            }
            targets[pid] = target
        if npu is not None and npu not in target["npus"]:
            target["npus"].append(npu)
    return list(targets.values())


def discover_by_pattern(pattern: str, npu_pids: set) -> List[Dict[str, Any]]:
    """按 comm/cmdline 正则匹配发现（pattern 由内部白名单调用方传入，非用户直传）。"""
    if not re.fullmatch(r"[A-Za-z0-9_\-|]+", pattern or ""):
        return []
    regex = re.compile(pattern, re.IGNORECASE)
    targets: List[Dict[str, Any]] = []
    for pid in _list_pids():
        if pid in npu_pids:
            continue
        cmdline = procfs.read_pid_cmdline(pid)
        comm = _safe_comm(pid)
        if regex.search(cmdline) or regex.search(comm):
            affinity = procfs.read_pid_affinity(pid)
            targets.append({
                "pid": pid,
                "comm": comm,
                "cmdline": cmdline[:256],
                "npus": [],
                "affinity": affinity or "",
                "role": _classify_process(comm, cmdline, has_npu=False),
                "source": "pattern_match",
            })
    return targets


def discover_cgroup_local() -> List[Dict[str, Any]]:
    """容器内：本 cgroup 全部进程（/proc/1/cgroup 路径不可用时返回空）。"""
    targets: List[Dict[str, Any]] = []
    own_cgroup: Optional[str] = None
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh.read(4096).splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3:
                    own_cgroup = parts[2].strip("/")
                    break
    except OSError:
        return targets
    if not own_cgroup:
        return targets
    base = f"/sys/fs/cgroup/{own_cgroup}/cgroup.procs"
    try:
        with open(base, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read(1024 * 1024)
    except OSError:
        return targets
    for line in raw.splitlines():
        if not _PID_RE.match(line):
            continue
        pid = int(line)
        comm = _safe_comm(pid)
        affinity = procfs.read_pid_affinity(pid)
        targets.append({
            "pid": pid,
            "comm": comm,
            "cmdline": procfs.read_pid_cmdline(pid)[:256],
            "npus": [],
            "affinity": affinity or "",
            "role": _classify_process(comm, "", has_npu=False),
            "source": "cgroup_local",
        })
    return targets


def _safe_comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(256).strip()
    except OSError:
        return ""


def discover(framework: str = "", npu_cards: str = "") -> Dict[str, Any]:
    """发现入口：NPU 锚点优先 → 框架/卡数模式 → 容器本 cgroup。

    返回 {"targets": [...], "npu_processes": [...], "mode": "..."}。
    """
    npu_processes = npu_query.query_processes()
    targets = discover_npu_anchored(npu_processes)
    mode = "npu_anchored"
    npu_pids = {t["pid"] for t in targets}
    if not targets and framework:
        patterns = {
            "vllm": "vllm", "mindie": "mindie", "megatron": "torchrun",
            "msrun": "msrun", "torchrun": "torchrun",
        }.get(framework.lower())
        if patterns:
            targets = discover_by_pattern(patterns, npu_pids)
            mode = "pattern_match"
    return {"targets": targets, "npu_processes": npu_processes, "mode": mode}

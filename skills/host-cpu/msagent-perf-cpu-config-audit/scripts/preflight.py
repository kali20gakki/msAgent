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
"""preflight.py — 权限/容器/工具探测（决定配置侧/现状侧可判性与 view_limited）。"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict

# 依赖工具可用性由调用方降级（check-items 中声明的等价 /proc、/sys 路径）
_OPTIONAL_TOOLS = ("numactl", "cpupower", "irqbalance", "taskset", "ethtool", "sensors")


def _is_container() -> bool:
    """容器判定：dockerenv/containerenv 标记 + cgroup v1/v2 路径标记。

    cgroup v2 下 /proc/1/cgroup 常退化为 "0::/"，故同时检查 /proc/self/cgroup，
    并补 podman/libpod 标记，避免 v2 容器漏判导致 view_limited 误报 False。
    """
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    markers = ("docker", "kubepods", "containerd", "lxc", "libpod", "podman")
    for path in ("/proc/1/cgroup", "/proc/self/cgroup"):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(65536)
        except OSError:
            continue
        if any(marker in content for marker in markers):
            return True
    return False


def _cgroup_version() -> str:
    """判定 cgroup v1/v2（v2 以 /sys/fs/cgroup/cgroup.controllers 存在为准）。"""
    if os.path.isfile("/sys/fs/cgroup/cgroup.controllers"):
        return "v2"
    return "v1"


def _cgroup_writable() -> bool:
    return os.access("/sys/fs/cgroup", os.W_OK)


def _topology_visible() -> bool:
    return os.path.isdir("/sys/devices/system/node")


def _capabilities() -> list:
    """尽力读取 CapEff（/proc/self/status）。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh.read(65536).splitlines():
                if line.startswith("CapEff:"):
                    return [line.split(":", 1)[1].strip()]
    except OSError:
        pass
    return []


def _optional_tools() -> Dict[str, bool]:
    return {tool: shutil.which(tool) is not None for tool in _OPTIONAL_TOOLS}


def run() -> Dict[str, Any]:
    """preflight 探测结果（写入报告元信息）。"""
    in_container = _is_container()
    topology_visible = _topology_visible()
    return {
        "in_container": in_container,
        "cgroup_ver": _cgroup_version(),
        "cgroup_writable": _cgroup_writable(),
        "topology_visible": topology_visible,
        "caps": _capabilities(),
        "optional_tools": _optional_tools(),
        "view_limited": in_container or not topology_visible,
    }

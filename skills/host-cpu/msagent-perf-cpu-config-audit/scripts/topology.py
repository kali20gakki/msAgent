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
"""topology.py — 完整拓扑采集（NUMA/SNC/SMT/L3/NPU/NIC/存储→node/HCCS）。

只读 /sys 固定路径，失败降级返回空结构。config-audit / bind-tune 各持一份，
口径与字段命名保持一致。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


def _read_text(path: str, max_bytes: int = 65536) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_bytes)
    except (OSError, ValueError):
        return None


def _read_int(path: str) -> Optional[int]:
    text = _read_text(path, 4096)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def parse_cpu_list(text: str) -> List[int]:
    """解析 '0-3,8,10-11' 形式的 CPU 列表为升序 int 列表。"""
    cpus: List[int] = []
    if not text or text.strip().lower() in ("", "n/a"):
        return cpus
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                cpus.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus: List[int]) -> str:
    """int 列表 → '0-3,8' 形式。"""
    if not cpus:
        return ""
    cpus = sorted(set(cpus))
    ranges: List[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def online_cpus() -> List[int]:
    return parse_cpu_list(_read_text("/sys/devices/system/cpu/online") or "")


def node_ids() -> List[int]:
    """实际 NUMA node id 列表（online 优先，不可读时回落 possible）。

    node id 不保证从 0 连续（部分机型/固件为 0-1,4），因此不能用 range(num_nodes())
    反推 id，否则会漏采真实 node 并访问不存在的 node。
    """
    for path in ("/sys/devices/system/node/online",
                 "/sys/devices/system/node/possible"):
        ids = parse_cpu_list(_read_text(path) or "")
        if ids:
            return ids
    return [0]


def num_nodes() -> int:
    return len(node_ids())


def node_cpus(node: int) -> List[int]:
    return parse_cpu_list(_read_text(f"/sys/devices/system/node/node{node}/cpulist") or "")


def node_mem_total_kb(node: int) -> Optional[int]:
    """从 node meminfo 读 MemTotal（KB）。"""
    text = _read_text(f"/sys/devices/system/node/node{node}/meminfo")
    if not text:
        return None
    for line in text.splitlines():
        if " MemTotal:" in line or line.startswith("MemTotal:"):
            try:
                return int(line.split("kB")[0].split()[-1])
            except (IndexError, ValueError):
                return None
    return None


def node_mem_free_kb(node: int) -> Optional[int]:
    text = _read_text(f"/sys/devices/system/node/node{node}/meminfo")
    if not text:
        return None
    for line in text.splitlines():
        if " MemFree:" in line or line.startswith("MemFree:"):
            try:
                return int(line.split("kB")[0].split()[-1])
            except (IndexError, ValueError):
                return None
    return None


def node_distance(node: int) -> List[int]:
    text = _read_text(f"/sys/devices/system/node/node{node}/distance")
    if not text:
        return []
    return [int(x) for x in text.split() if x.strip().isdigit()]


def thread_siblings(cpu: int) -> List[int]:
    return parse_cpu_list(
        _read_text(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") or ""
    )


def l3_domain(cpu: int) -> List[int]:
    """CPU 所属 L3 共享核集：cpuN/cache/index*/shared_cpu_list 中 level=3 那组。

    kernel 的 L3 domain 不在 topology/ 下（topology/shared_cpu_list 并非所有
    机型都存在），cache/indexN 才是按 level 暴露共享核集的路径。
    """
    base = f"/sys/devices/system/cpu/cpu{cpu}/cache"
    if not os.path.isdir(base):
        return []
    for name in sorted(os.listdir(base)):
        if _read_int(os.path.join(base, name, "level")) != 3:
            continue
        return parse_cpu_list(_read_text(os.path.join(base, name, "shared_cpu_list")) or "")
    return []


def detect_snc() -> Dict[str, Any]:
    """SNC 检测：node 数 > 物理封装数时视为 SNC 开启（简化判定）。"""
    n_nodes = num_nodes()
    # 无 package 信息时保守判定
    packages: set = set()
    for cpu in online_cpus()[:64]:
        pkg_id = _read_int(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id")
        if pkg_id is not None:
            packages.add(pkg_id)
    if packages and n_nodes > len(packages):
        return {"enabled": True, "sub_clusters": n_nodes}
    return {"enabled": False, "sub_clusters": 1}


def numa_topology() -> Dict[str, Any]:
    """汇总 NUMA 拓扑（node/cpus/mem/distance），按实际 node id 遍历。"""
    nodes = []
    for node in node_ids():
        nodes.append({
            "node": node,
            "cpus": format_cpu_list(node_cpus(node)),
            "mem_total_kb": node_mem_total_kb(node),
            "mem_free_kb": node_mem_free_kb(node),
            "distance": node_distance(node),
        })
    return {
        "num_nodes": len(nodes),
        "nodes": nodes,
        "snc": detect_snc(),
        "online_cpus": format_cpu_list(online_cpus()),
    }


def smt_siblings() -> List[List[int]]:
    """全机 SMT 兄弟对（去重，仅保留 >1 线程的物理核）。"""
    seen: set = set()
    pairs: List[List[int]] = []
    for cpu in online_cpus():
        siblings = thread_siblings(cpu)
        key = tuple(siblings)
        if len(siblings) > 1 and key not in seen:
            seen.add(key)
            pairs.append(siblings)
    return pairs


def l3_domains() -> List[Dict[str, Any]]:
    """全机 L3 domain 分组（去重）。"""
    seen: set = set()
    domains: List[Dict[str, Any]] = []
    for cpu in online_cpus():
        domain = l3_domain(cpu)
        key = tuple(domain)
        if domain and key not in seen:
            seen.add(key)
            domains.append({"cpus": format_cpu_list(domain)})
    return domains


def nic_numa_map() -> List[Dict[str, Any]]:
    """网卡 → node 映射（仅 /sys/class/net 下带 device 的实体网卡）。

    虚拟网卡（lo/veth/bridge/docker0 等）没有 device 链接，numa_node 读不到，
    保留它们会产出 node=None 的伪网卡事实。
    """
    result: List[Dict[str, Any]] = []
    base = "/sys/class/net"
    if not os.path.isdir(base):
        return result
    for name in sorted(os.listdir(base)):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            continue
        if not os.path.exists(f"{base}/{name}/device"):
            continue
        node = _read_int(f"{base}/{name}/device/numa_node")
        if node is None or node < 0:
            continue
        result.append({"dev": name, "node": node})
    return result


def block_numa_map() -> List[Dict[str, Any]]:
    """块设备 → node 映射。"""
    result: List[Dict[str, Any]] = []
    base = "/sys/block"
    if not os.path.isdir(base):
        return result
    for name in sorted(os.listdir(base)):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            continue
        node = _read_int(f"{base}/{name}/device/numa_node")
        if node is not None:
            result.append({"dev": name, "node": node})
    return result


def net_queues(dev: str) -> int:
    """网卡合并队列数（rx 队列计数）。"""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", dev):
        return 0
    base = f"/sys/class/net/{dev}/queues"
    if not os.path.isdir(base):
        return 0
    return sum(1 for name in os.listdir(base) if name.startswith("rx-"))


def rps_cpus(dev: str, queue: str) -> Optional[str]:
    """读取网卡队列 rps_cpus 掩码（hex）。"""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", dev) or not re.fullmatch(r"rx-\d+", queue):
        return None
    return _read_text(f"/sys/class/net/{dev}/queues/{queue}/rps_cpus", 4096)


def collect() -> Dict[str, Any]:
    """完整拓扑汇总（static-snapshot.json 的 topology 节）。"""
    return {
        "numa": numa_topology(),
        "smt": {"siblings": [format_cpu_list(pair) for pair in smt_siblings()]},
        "l3_domains": l3_domains(),
        "nic": nic_numa_map(),
        "storage": block_numa_map(),
    }

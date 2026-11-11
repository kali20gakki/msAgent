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
"""static_collect.py — 静态采集聚合：topology/procfs/npu-smi 汇总 → facts[]。

单点快照、只读不改（无差分/时序采样；需要持续确认的项经 suspect_items
由 os-observe 承接）。采集到的 npu-smi 数据消费 npu_query 唯一封装点。
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List

import npu_query
import preflight
import procfs
import topology


def _fact(item: str, value: Any, source: str) -> Dict[str, Any]:
    """构造 facts[] 条目（只含 item/value/source，判定归 checker）。"""
    return {"item": item, "value": value, "source": source}


def _collect_numa_facts(facts: List[Dict[str, Any]]) -> None:
    topo = topology.numa_topology()
    for node_info in topo["nodes"]:
        node = node_info["node"]
        facts.append(_fact(f"node{node}_cpus", node_info["cpus"],
                            f"/sys/devices/system/node/node{node}/cpulist"))
        if node_info["mem_total_kb"] is not None:
            facts.append(_fact(f"node{node}_mem_total_kb", node_info["mem_total_kb"],
                               f"/sys/devices/system/node/node{node}/meminfo"))
        if node_info["mem_free_kb"] is not None:
            facts.append(_fact(f"node{node}_mem_free_kb", node_info["mem_free_kb"],
                               f"/sys/devices/system/node/node{node}/meminfo"))
        if node_info["distance"]:
            facts.append(_fact(f"node{node}_distance", node_info["distance"],
                               f"/sys/devices/system/node/node{node}/distance"))
    facts.append(_fact("snc", topo["snc"], "/sys/devices/system/node/*"))
    facts.append(_fact("smt_siblings",
                       [",".join(map(str, pair)) for pair in topology.smt_siblings()],
                       "cpu*/topology/thread_siblings_list"))
    facts.append(_fact("online_cpus", topo["online_cpus"],
                       "/sys/devices/system/cpu/online"))
    for nic in topology.nic_numa_map():
        facts.append(_fact(f"nic_{nic['dev']}_node", nic["node"],
                           f"/sys/class/net/{nic['dev']}/device/numa_node"))
        facts.append(_fact(f"nic_{nic['dev']}_queues", topology.net_queues(nic["dev"]),
                           f"/sys/class/net/{nic['dev']}/queues"))
    for blk in topology.block_numa_map():
        facts.append(_fact(f"block_{blk['dev']}_node", blk["node"],
                           f"/sys/block/{blk['dev']}/device/numa_node"))


def _collect_memory_facts(facts: List[Dict[str, Any]]) -> None:
    meminfo = procfs.read_meminfo()
    if "HugePages_Total" in meminfo:
        facts.append(_fact("hugepages_total", meminfo["HugePages_Total"], "/proc/meminfo"))
        facts.append(_fact("hugepages_free", meminfo["HugePages_Free"], "/proc/meminfo"))
    for name in ("enabled", "defrag"):
        value = procfs.read_thp_setting(name)
        if value is not None:
            facts.append(_fact(f"thp_{name}", value,
                               f"/sys/kernel/mm/transparent_hugepage/{name}"))
    for key in ("swappiness", "overcommit_memory", "overcommit_ratio",
                "numa_balancing", "zone_reclaim_mode"):
        value = procfs.read_config(key)
        if value is not None:
            facts.append(_fact(key, value, f"/proc/sys/vm|kernel/{key}"))
    swaps = procfs.read_swaps()
    facts.append(_fact("active_swap_devices", swaps, "/proc/swaps"))


def _collect_freq_facts(facts: List[Dict[str, Any]]) -> None:
    online = topology.online_cpus()
    sample_cpus = online[:16]  # 采样前 16 核（单点快照足够判定）
    governors: Dict[str, int] = {}
    cur_max_ratio: List[float] = []
    epps: Dict[str, int] = {}
    for cpu in sample_cpus:
        gov = procfs.read_scaling_governor(cpu)
        if gov:
            governors[gov] = governors.get(gov, 0) + 1
        cur = procfs.read_cpu_freq(cpu, "scaling_cur_freq")
        maxf = procfs.read_cpu_freq(cpu, "cpuinfo_max_freq")
        if cur and maxf and maxf > 0:
            cur_max_ratio.append(cur / maxf)
        epp = procfs.read_epp(cpu)
        if epp:
            epps[epp] = epps.get(epp, 0) + 1
    if governors:
        facts.append(_fact("scaling_governor", governors,
                           "cpu*/cpufreq/scaling_governor"))
    if cur_max_ratio:
        facts.append(_fact("freq_cur_over_max",
                           round(sum(cur_max_ratio) / len(cur_max_ratio), 3),
                           "cpu*/cpufreq/scaling_cur_freq"))
    if epps:
        facts.append(_fact("epp", epps, "cpu*/cpufreq/energy_performance_preference"))
    # C-state（采样前 8 核）
    cstates: set = set()
    for cpu in online[:8]:
        cstates.update(procfs.read_cpuidle_states(cpu))
    if cstates:
        facts.append(_fact("cstates", sorted(cstates), "cpu*/cpuidle/state*/name"))
    facts.append(_fact("cmdline_boot_params", procfs.read_cmdline_boot_params(),
                       "/proc/cmdline"))


def _collect_irq_facts(facts: List[Dict[str, Any]]) -> None:
    interrupts = procfs.read_interrupts()
    if interrupts:
        total = sum(sum(v["counts"]) for v in interrupts.values())
        facts.append(_fact("interrupts_total", total, "/proc/interrupts"))
        facts.append(_fact("interrupts_labels", len(interrupts), "/proc/interrupts"))
        # 单核最高占比
        ncpu = max((len(v["counts"]) for v in interrupts.values()), default=0)
        if ncpu and total > 0:
            per_cpu: List[int] = [0] * ncpu
            for entry in interrupts.values():
                for idx, count in enumerate(entry["counts"]):
                    per_cpu[idx] += count
            busiest = max(per_cpu)
            facts.append(_fact("irq_busiest_cpu_pct",
                               round(100.0 * busiest / total, 1), "/proc/interrupts"))
            if per_cpu:
                cpu0_pct = round(100.0 * per_cpu[0] / total, 1)
                facts.append(_fact("irq_cpu0_pct", cpu0_pct, "/proc/interrupts"))
    softirqs = procfs.read_softirqs()
    if softirqs:
        per_name = {name: sum(counts) for name, counts in softirqs.items()}
        facts.append(_fact("softirq_totals", per_name, "/proc/softirqs"))


def _collect_npu_facts(facts: List[Dict[str, Any]]) -> None:
    # 一次 npu-smi info 同时拿总览与卡↔进程（重复执行会让采集耗时翻倍且快照不一致）
    info, procs = npu_query.query_snapshot()
    facts.append(_fact("npu_available", info.get("available", False), "npu-smi info"))
    if info.get("available"):
        # 封装级权威计数：info -l 优先（其 NPU ID 语义为物理封装）；
        # -l 缺失时 info 总览兜底（双 die 机型每封装两行 NPU 行，按 NPU ID 去重）
        info_l = npu_query.query_info_l()
        if info_l and info_l.get("total_count"):
            facts.append(_fact("npu_count", info_l["total_count"], "npu-smi info -l"))
            names = sorted({n.get("name", "") for n in info_l.get("npus", [])
                            if n.get("name")})
            if names:
                facts.append(_fact("npu_names", names, "npu-smi info -l"))
            else:
                facts.append(_fact("npu_names", _dedupe_npu_names(info),
                                   "npu-smi info"))
        else:
            names, npu_ids = _dedupe_npu_names(info, with_ids=True)
            facts.append(_fact("npu_count", len(npu_ids), "npu-smi info"))
            facts.append(_fact("npu_names", sorted(set(names)), "npu-smi info"))
        # die 级映射（双 die 机型每组 2 个 die；单 die 机型每组 1 个）
        die_map = npu_query.query_die_map()
        if die_map and die_map.get("logic_ids"):
            facts.append(_fact("npu_die_count", len(die_map["logic_ids"]),
                               "npu-smi info -m"))
            facts.append(_fact("npu_die_groups", die_map["groups"], "npu-smi info -m"))
            facts.append(_fact("npu_logic_ids", die_map["logic_ids"], "npu-smi info -m"))
        davinci = _count_davinci_devices()
        if davinci is not None:
            facts.append(_fact("davinci_devices", davinci, "/dev"))
    # topo 双形态：form（npu 封装级 / die 级 Phy-ID 矩阵）+ fabric（关系符号判定）
    # fabric 与 CPU Affinity 是两条独立信号；affinity_available 显式记录可得性，
    # 下游不再把「空数组」误读为「没有」
    topo = npu_query.query_topo()
    facts.append(_fact("npu_topo_form", topo.get("form", "unknown"),
                       "npu-smi info -t topo"))
    facts.append(_fact("npu_topo_fabric", topo.get("fabric", "unknown"),
                       "npu-smi info -t topo"))
    facts.append(_fact("npu_topo_rows", topo.get("rows", 0), "npu-smi info -t topo"))
    if topo.get("sio_pairs"):
        facts.append(_fact("npu_sio_pairs", topo["sio_pairs"], "npu-smi info -t topo"))
    affinity = topo.get("affinity", [])
    facts.append(_fact("npu_cpu_affinity", affinity, "npu-smi info -t topo"))
    facts.append(_fact("npu_affinity_available", bool(affinity), "npu-smi info -t topo"))
    if not affinity:
        reason = ("ub_equidistant（UB/HCCS_SW 等距互联，NPU 与所有 CPU 等距，"
                  "无 NUMA 亲和，合法形态）"
                  if topo.get("fabric") in ("ub", "hccs_sw")
                  else "no_affinity_column（topo 无 CPU Affinity 列且 fabric 未知）")
        facts.append(_fact("npu_affinity_unavailable_reason", reason,
                           "npu-smi info -t topo"))
    facts.append(_fact("npu_process_count", len(procs), "npu-smi info"))


def _dedupe_npu_names(info: Dict[str, Any],
                      with_ids: bool = False) -> Any:
    """info 总览 NPU 行去重（双 die 机型每封装两行），名称为纯数字的行丢弃。"""
    names: List[str] = []
    ids: set = set()
    for entry in info.get("npus", []):
        npu_id = entry.get("npu")
        if npu_id in ids:
            continue
        ids.add(npu_id)
        name = str(entry.get("name", ""))
        if name and not name.isdigit():
            names.append(name)
    return (names, ids) if with_ids else names


def _count_davinci_devices() -> Any:
    """/dev/davinciN 软件设备枚举（die 级，与 info -m 的 Logic ID 交叉验证）。"""
    try:
        return len([name for name in os.listdir("/dev")
                    if re.fullmatch(r"davinci\d+", name)])
    except OSError:
        return None


def _collect_cgroup_facts(facts: List[Dict[str, Any]]) -> None:
    ver = preflight._cgroup_version()
    facts.append(_fact("cgroup_version", ver, "/sys/fs/cgroup"))
    if ver == "v2":
        paths = {
            "cpu_max": "/sys/fs/cgroup/cpu.max",
            "cpu_weight": "/sys/fs/cgroup/cpu.weight",
            "memory_max": "/sys/fs/cgroup/memory.max",
            "cpuset_cpus_effective": "/sys/fs/cgroup/cpuset.cpus.effective",
            "cpuset_mems_effective": "/sys/fs/cgroup/cpuset.mems.effective",
        }
    else:
        paths = {
            "cfs_quota_us": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "cpuset_cpus_effective": "/sys/fs/cgroup/cpuset/cpuset.cpus.effective",
            "cpuset_mems_effective": "/sys/fs/cgroup/cpuset/cpuset.mems.effective",
        }
    for key, path in paths.items():
        text = topology._read_text(path, 4096)
        if text is not None:
            facts.append(_fact(key, text.strip(), path))
    # cpu.stat throttled
    stat_path = "/sys/fs/cgroup/cpu.stat" if ver == "v2" else "/sys/fs/cgroup/cpu/cpu.stat"
    stat_text = topology._read_text(stat_path, 65536)
    if stat_text:
        throttled: Dict[str, str] = {}
        for line in stat_text.splitlines():
            key, _, value = line.partition(" ")
            if "throttled" in key:
                throttled[key] = value.strip()
        if throttled:
            facts.append(_fact("cpu_stat_throttled", throttled, stat_path))


def collect(pid: int = 0) -> Dict[str, Any]:
    """静态采集入口。pid > 0 时附加现状侧（affinity/environ/numa_maps）。"""
    facts: List[Dict[str, Any]] = []
    _collect_numa_facts(facts)
    _collect_memory_facts(facts)
    _collect_freq_facts(facts)
    _collect_irq_facts(facts)
    _collect_npu_facts(facts)
    _collect_cgroup_facts(facts)
    if pid > 0:
        affinity = procfs.read_pid_affinity(pid)
        facts.append(_fact("pid_affinity", affinity or "", f"/proc/{pid}/status"))
        environ = procfs.read_pid_environ(pid)
        watch_keys = (
            "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
            "ASCEND_RT_VISIBLE_DEVICES", "HCCL_WHITELIST_DISABLE",
            "MALLOC_ARENA_MAX", "ASCEND_GLOBAL_LOG_LEVEL",
        )
        picked = {k: environ[k] for k in watch_keys if k in environ}
        facts.append(_fact("pid_environ_picked", picked, f"/proc/{pid}/environ"))
        numa_maps = procfs.read_pid_numa_maps_summary(pid)
        facts.append(_fact("pid_numa_maps_summary", numa_maps, f"/proc/{pid}/numa_maps"))
    return {
        "session": {
            "ts": time.time(),
            "host": _hostname(),
            "mode": "in-process" if pid > 0 else "pre-start",
        },
        "facts": facts,
    }


def _hostname() -> str:
    import socket
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"

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

"""完整拓扑采集：NUMA / CPU / SMT / L3 / NPU / NIC / 存储 -> node。

只读 /sys、/proc 与 npu-smi，不依赖任何额外安装的工具。
"""

import glob
import logging
import os
import platform
import re
import socket
import time

from cpu_list import parse_cpulist

_LOG = logging.getLogger(__name__)

# locality_mode legacy 别名唯一映射表：model / emit_plan / report 统一引用
LOCALITY_ALIAS = {"npu_local": "topo_affinity", "balanced_partition": "global_slice"}


def _read(path, default=None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError) as exc:
        _LOG.debug("read %s failed: %s", path, type(exc).__name__)
        return default
    except Exception as exc:  # noqa: BLE001 - sysfs 缺失/竞态统一降级
        _LOG.debug("read %s failed: %s: %s", path, type(exc).__name__, exc)
        return default


def cpus_to_str(cpus):
    """[0,1,2,5] -> '0-2,5'。"""
    cpus = sorted(set(int(c) for c in cpus))
    if not cpus:
        return ""
    ranges = []
    start = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append(str(start) if start == prev else "%d-%d" % (start, prev))
        start = prev = c
    ranges.append(str(start) if start == prev else "%d-%d" % (start, prev))
    return ",".join(ranges)


def _node_id_from_path(path):
    m = re.search(r"node(\d+)$", path)
    return int(m.group(1)) if m else None


def _pci_numa(addr):
    """从 /sys/bus/pci/devices/<addr>/numa_node 读取设备本地 node。"""
    if not addr:
        return None
    value = _read("/sys/bus/pci/devices/%s/numa_node" % addr)
    if value is None:
        return None
    try:
        node = int(value)
    except ValueError:
        return None
    return node if node >= 0 else None


def _pci_local_cpus(addr):
    """PCI 设备 local_cpulist（比 numa_node 更精确的 CPU 亲和）。"""
    if not addr:
        return []
    value = _read("/sys/bus/pci/devices/%s/local_cpulist" % addr)
    return parse_cpulist(value) if value else []


def affinity_summary(npus):
    """亲和粒度：每卡独立（per_npu）还是多卡共享同一核区间（group）。

    例：A5 上 NPU0/1/6/7 → 288-383、NPU2/3/4/5 → 96-191，属 group 粒度——
    亲和描述的是 NPU 主机口就近的 CPU 组，不是单卡私有核集。
    """
    groups = {}
    for npu in npus or []:
        key = npu.get("affinity")
        if key:
            groups.setdefault(key, []).append(npu.get("id"))
    if not groups:
        return "none", {}
    granularity = "per_npu" if all(len(v) == 1 for v in groups.values()) else "group"
    return granularity, {k: sorted(v) for k, v in sorted(groups.items())}


def sanitize_affinity(npus, online_cpus):
    """亲和区间与本机可见 CPU 无交集时丢弃亲和，避免产出空核集。

    返回 (has_affinity, degraded)：degraded 表示亲和来自 PCI 回退或被判定无效。
    """
    online = set(online_cpus or [])
    for npu in npus or []:
        ac = npu.get("affinity_cpus") or []
        if ac and online and not (set(ac) & online):
            npu["affinity_cpus"] = []
            npu["affinity"] = None
            npu["affinity_source"] = "invalid"
            npu["affinity_note"] = "affinity_invalid"
    has_affinity = any((n.get("affinity_cpus") for n in npus or []))
    degraded = any(n.get("affinity_source") in ("pci_local_cpulist", "pci_numa", "invalid")
                   for n in npus or [])
    return has_affinity, degraded


def _ub_present():
    """UB（灵衢）总线是否可见：超节点形态下存在 /sys/bus/ub/devices。"""
    try:
        return bool(os.listdir("/sys/bus/ub/devices"))
    except Exception as exc:  # noqa: BLE001 - 超节点探测失败按不存在处理
        _LOG.debug("probe /sys/bus/ub/devices failed: %s", type(exc).__name__)
        return False


def _collect_numa_nodes():
    """采集 NUMA node 与 CPU→node 映射，node sysfs 缺失时退化为单 node。"""
    nodes = []
    cpu_to_node = {}

    for path in sorted(glob.glob("/sys/devices/system/node/node*"), key=_node_id_from_path):
        nid = _node_id_from_path(path)
        if nid is None:
            continue
        cpulist = _read(os.path.join(path, "cpulist"), "") or ""
        cpus = parse_cpulist(cpulist)
        distance = _read(os.path.join(path, "distance"), "") or ""
        meminfo = _read(os.path.join(path, "meminfo"), "") or ""
        m = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
        mem_gb = round(int(m.group(1)) / 1024.0 / 1024.0, 1) if m else None
        node = {
            "node": nid,
            "cpus": cpus,
            "cpulist": cpulist,
            "distance": [int(x) for x in distance.split() if x.isdigit()],
            "mem_gb": mem_gb,
        }
        nodes.append(node)
        for c in cpus:
            cpu_to_node[c] = nid

    numa_source = "sysfs"
    if not nodes:
        # 非 NUMA 系统 / node sysfs 不可见：退化为单 node，不阻断后续规划
        online_cpus = parse_cpulist(_read("/sys/devices/system/cpu/online", "") or "")
        if online_cpus:
            nodes = [{
                "node": 0,
                "cpus": online_cpus,
                "cpulist": cpus_to_str(online_cpus),
                "distance": [10],
                "mem_gb": None,
            }]
            for c in online_cpus:
                cpu_to_node[c] = 0
            numa_source = "single_node_fallback"
    return nodes, cpu_to_node, numa_source


def _collect_cpu_topology(nodes, cpu_to_node):
    """采集在线 CPU 拓扑（core/package/siblings/L3）与 L3 共享域。"""
    online = _read("/sys/devices/system/cpu/online", "") or ""
    cpus = []
    l3_map = {}
    smt_enabled = False
    # 同域 CPU 的 siblings / L3 值相同：首次读到后按域内成员缓存，后续 CPU 直接复用
    siblings_cache = {}
    l3_cache = {}
    for cpu in parse_cpulist(online):
        base = "/sys/devices/system/cpu/cpu%d" % cpu
        topo = os.path.join(base, "topology")
        core_id = _read(os.path.join(topo, "core_id"))
        package_id = _read(os.path.join(topo, "physical_package_id"))
        siblings = siblings_cache.get(cpu)
        if siblings is None:
            siblings = _read(os.path.join(topo, "thread_siblings_list"), "")
            for member in parse_cpulist(siblings):
                siblings_cache[member] = siblings
        siblings_cpus = parse_cpulist(siblings)
        if len(siblings_cpus) > 1:
            smt_enabled = True
        l3 = l3_cache.get(cpu)
        if l3 is None:
            l3 = _read(os.path.join(base, "cache/index3/shared_cpu_list"), "")
            if not l3:
                l3 = _read(os.path.join(base, "cache/index2/shared_cpu_list"), "")
            if l3:
                for member in parse_cpulist(l3):
                    l3_cache[member] = l3
        cpus.append({
            "cpu": cpu,
            "node": cpu_to_node.get(cpu),
            "core_id": int(core_id) if core_id and core_id.isdigit() else None,
            "package_id": int(package_id) if package_id and package_id.isdigit() else None,
            "thread_siblings": siblings_cpus,
            "l3_shared": l3,
        })
        if l3:
            l3_map.setdefault(l3, []).append(cpu)

    l3_domains = []
    for idx, (shared, l3_cpus) in enumerate(sorted(l3_map.items())):
        l3_domains.append({
            "id": idx,
            "shared_cpu_list": shared,
            "cpus": l3_cpus,
            "node": node_of_cpus(l3_cpus, nodes),
        })
    return cpus, l3_domains, smt_enabled


def _collect_npu_topology(npu_info, nodes):
    """采集 NPU 拓扑：优先 npu-smi topo 亲和，缺失时回退 PCI local_cpulist / numa_node。"""
    npus = []
    devices = {d.get("id"): d for d in (npu_info.get("devices") or [])}
    topo_items = npu_info.get("topo") or {}
    for nid in sorted(set(devices) | set(topo_items)):
        dev = devices.get(nid, {})
        item = topo_items.get(nid, {})
        affinity_cpus = list(item.get("affinity_cpus") or [])
        affinity_source = "npu_smi_topo" if affinity_cpus else None
        if not affinity_cpus and dev.get("bus_id"):
            # 无 topo 时的降级：PCI local_cpulist > PCI numa_node
            local = _pci_local_cpus(dev["bus_id"])
            if local:
                affinity_cpus = local
                affinity_source = "pci_local_cpulist"
            else:
                pci_node = _pci_numa(dev["bus_id"])
                if pci_node is not None:
                    affinity_cpus = next(
                        (n["cpus"] for n in nodes if n["node"] == pci_node), []
                    )
                    affinity_source = "pci_numa"
        pci_node = _pci_numa(dev.get("bus_id"))
        npus.append({
            "id": nid,
            "name": dev.get("name"),
            "health": dev.get("health"),
            "bus_id": dev.get("bus_id"),
            "hbm_used_mb": dev.get("hbm_used_mb"),
            "hbm_total_mb": dev.get("hbm_total_mb"),
            "power_w": dev.get("power_w"),
            "temp_c": dev.get("temp_c"),
            "affinity": item.get("affinity") or (cpus_to_str(affinity_cpus) if affinity_cpus else None),
            "affinity_cpus": affinity_cpus,
            "affinity_source": affinity_source or "none",
            "numa_node": node_of_cpus(affinity_cpus, nodes) if affinity_cpus else pci_node,
            "hccs": item.get("hccs") or {},
        })
    return npus


def _annotate_npu_nodes(npus, cpu_to_node):
    """为每张 NPU 标注亲和 CPU 跨越的 node 集合。"""
    for npu in npus:
        npu_cpus = npu.get("affinity_cpus") or []
        spanned = sorted({cpu_to_node.get(c) for c in npu_cpus if cpu_to_node.get(c) is not None})
        npu["affinity_nodes"] = spanned
        if len(spanned) > 1:
            npu["affinity_note"] = "affinity_spans_nodes"


def _collect_nics():
    """采集物理网卡及其 NUMA 归属（跳过虚拟网卡）。"""
    nics = []
    virtual_prefixes = ("lo", "veth", "docker", "br-", "virbr", "bond", "tun", "tap")
    for iface in sorted(glob.glob("/sys/class/net/*")):
        name = os.path.basename(iface)
        if name.startswith(virtual_prefixes):
            continue
        dev_numa = os.path.join(iface, "device", "numa_node")
        dev_link = os.path.join(iface, "device")
        if not os.path.exists(dev_link):
            continue
        pci = os.path.basename(os.path.realpath(dev_link))
        numa = _read(dev_numa)
        if numa and numa.lstrip("-").isdigit() and int(numa) >= 0:
            numa_val = int(numa)
            nic_src = "device_numa_node"
        else:
            numa_val = _pci_numa(pci)
            nic_src = "pci_numa" if numa_val is not None else "unknown"
        nics.append({
            "name": name,
            "numa_node": numa_val,
            "numa_source": nic_src,
            "operstate": _read(os.path.join(iface, "operstate")),
            "speed": _read(os.path.join(iface, "speed")),
            "mac": _read(os.path.join(iface, "address")),
            "pci": pci,
        })
    return nics


def _collect_storage():
    """采集 NVMe/SATA 块设备及其 NUMA 归属与容量。"""
    storage = []
    for dev in sorted(glob.glob("/sys/block/*")):
        name = os.path.basename(dev)
        if name.startswith(("loop", "ram", "sr", "dm-", "md")):
            continue
        if not (name.startswith("nvme") or name.startswith("sd")):
            continue
        dev_link = os.path.join(dev, "device")
        pci = os.path.basename(os.path.realpath(dev_link)) if os.path.exists(dev_link) else None
        numa = _read(os.path.join(dev, "device", "numa_node"))
        if numa and numa.lstrip("-").isdigit() and int(numa) >= 0:
            numa_val = int(numa)
            st_src = "device_numa_node"
        else:
            numa_val = _pci_numa(pci)
            st_src = "pci_numa" if numa_val is not None else "unknown"
        size = _read(os.path.join(dev, "size"))
        try:
            size_gb = round(int(size) * 512 / 1000.0 / 1000.0 / 1000.0, 1) if size else None
        except ValueError:
            size_gb = None
        storage.append({
            "name": name,
            "numa_node": numa_val,
            "numa_source": st_src,
            "size_gb": size_gb,
            "model": _read(os.path.join(dev, "device", "model")),
            "rotational": _read(os.path.join(dev, "queue", "rotational")),
            "pci": pci,
        })
    return storage


def collect(npu_info=None):
    npu_info = npu_info or {}
    nodes, cpu_to_node, numa_source = _collect_numa_nodes()
    cpus, l3_domains, smt_enabled = _collect_cpu_topology(nodes, cpu_to_node)
    npus = _collect_npu_topology(npu_info, nodes)

    topo_meta = npu_info.get("topo_meta") or {}
    fabric = "ub" if _ub_present() else (topo_meta.get("fabric_hint") or "unknown")
    # 双 die 机型：同封装 die 分组（npu-smi info -m）→ global_slice 不可跨 node 拆分的单位
    die_map = npu_info.get("die_map") or {}
    die_groups = die_map.get("groups") or []
    # die 信息缺失与单 die 机型必须区分：缺失时 die_count 留空并标 die_available=false，
    # 不能回落成 NPU 数（否则与同一份 meta 里的空 die_groups 自相矛盾）。
    die_available = bool(die_map.get("logic_ids"))
    die_count = len(die_map["logic_ids"]) if die_available else None

    online_cpu_set = {c["cpu"] for c in cpus}
    has_cpu_affinity, degraded = sanitize_affinity(npus, online_cpu_set)
    locality_mode = "unknown"
    if npus:
        locality_mode = "topo_affinity" if has_cpu_affinity else "global_slice"

    affinity_granularity, affinity_groups = affinity_summary(npus)
    _annotate_npu_nodes(npus, cpu_to_node)

    nics = _collect_nics()
    storage = _collect_storage()

    return {
        "meta": {
            "host": socket.gethostname(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "ts_ns": time.time_ns(),
            "numa_source": numa_source,
            "fabric": fabric,
            "locality_mode": locality_mode,
            "topo_format": topo_meta.get("format") or "unknown",
            "has_cpu_affinity": has_cpu_affinity,
            "affinity_granularity": affinity_granularity,
            "affinity_groups": affinity_groups,
            "degraded": degraded,
            "has_nic_columns": bool(topo_meta.get("has_nic_columns")),
            # 双 die 机型：每封装 die 分组（如 [[0,1],[2,3],...]），单 die 机型为每组 1 个
            "die_groups": die_groups,
            "die_count": die_count,
            "die_available": die_available,
            "die_source": "npu_smi_info_m" if die_available else "unavailable",
            "package_count": npu_info.get("package_count") or len(npus),
        },
        "nodes": nodes,
        "cpus": cpus,
        "smt": {"enabled": smt_enabled},
        "l3_domains": l3_domains,
        "npu": npus,
        "nic": nics,
        "storage": storage,
    }


def node_of_cpus(cpus, nodes):
    """CPU 集合归属 node：完全包含于某 node 直接返回；否则取重叠最多者。

    平票取先遍历到的高重叠 node（nodes 按 node id 升序，结果确定）。
    """
    if not cpus:
        return None
    cpus = set(cpus)
    best = None
    best_count = -1
    for node in nodes:
        overlap = len(cpus & set(node["cpus"]))
        if overlap == len(cpus):
            return node["node"]
        if overlap > best_count:
            best = node["node"]
            best_count = overlap
    return best if best_count > 0 else None


def distance(topo, a, b):
    for node in topo.get("nodes", []):
        if node["node"] == a:
            dist = node.get("distance") or []
            if 0 <= b < len(dist):
                return dist[b]
    return None


def npu_of_node(topo, node):
    return [n for n in topo.get("npu", []) if n.get("numa_node") == node]


def nics_of_node(topo, node):
    return [n for n in topo.get("nic", []) if n.get("numa_node") == node]


def storage_of_node(topo, node):
    return [s for s in topo.get("storage", []) if s.get("numa_node") == node]


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), ensure_ascii=False, indent=2))

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

"""拓扑建模：把 unit 映射到 NPU node / L3 domain，并补充全局约束。

不重新采集，只消费 topology + discover 的结果。
目标 node 规则：实质占用卡集（过滤控制内存）所在 node 取交集；
全部同 node 直接用；跨 node 时取卡数最多的 node，并标注 multi_npu_cross_node。
"""


LEGACY = {"npu_local": "topo_affinity", "balanced_partition": "global_slice"}


def _l3_of_cpus(cpus, topo):
    cpus = set(cpus)
    for dom in topo.get("l3_domains") or []:
        if cpus and cpus <= set(dom.get("cpus") or []):
            return dom["id"]
    return None


def _target_node(unit, npu_by_id):
    cards = unit.get("npus_significant") or unit.get("npus") or []
    nodes = []
    for card in cards:
        node = (npu_by_id.get(card) or {}).get("numa_node")
        if node is not None:
            nodes.append(node)
    if not nodes:
        return unit.get("npu_node"), None
    counts = {}
    for node in nodes:
        counts[node] = counts.get(node, 0) + 1
    # 平票时取 node id 较小者（确定性 tie-break，保证同输入同方案）
    best = max(counts.items(), key=lambda x: (x[1], -x[0]))[0]
    if len(set(nodes)) > 1:
        return best, {
            "kind": "multi_npu_cross_node",
            "severity": "medium",
            "handling": "observe",
            "evidence": "实质占用卡 %s 跨 node %s；按多数卡 node%s 求解" % (
                cards, sorted(set(nodes)), best
            ),
        }
    return best, None


def _assign_global_slice(topo, units):
    """无 NPU→CPU 亲和（超节点/UB 等距）时按封装分组、再按 node 均分。

    分组单位 = 封装（primary_npu）：双 die 机型同一封装的两个 die 上的 unit
    必须落在同一 node（同封装 die 经 SIO 片内直连，跨 node 会让协同走远端）；
    单 die 机型每封装 1 个 unit，退化为按 unit 均分。
    每 group 落到一个 node，unit 间核集由 zone_alloc 保证不重叠。
    """
    nodes = sorted(n["node"] for n in (topo.get("nodes") or []))
    ordered = sorted(
        units,
        key=lambda u: (
            u.get("primary_npu") if u.get("primary_npu") is not None else 10 ** 6,
            u.get("unit_id") or 0,
        ),
    )
    if not nodes:
        for unit in ordered:
            unit["target_node"] = None
            unit["target_node_source"] = "none"
            unit["target_cpus"] = []
        return
    # 按封装聚组：双 die 机型 primary_npu 是 die 级编号（0-15，见 npu-smi info -m），
    # 需先经 die_groups 归到封装；单 die 机型 die_groups 每组 1 个 die，等价于按 primary_npu 分组
    die_groups = (topo.get("meta") or {}).get("die_groups") or []
    die_to_package = {}
    for package_idx, group in enumerate(die_groups):
        for die in group:
            die_to_package[die] = package_idx

    groups = []
    index = {}
    for unit in ordered:
        npu = unit.get("primary_npu")
        if npu is None:
            key = ("unit", unit.get("unit_id") or 0)
        else:
            key = die_to_package.get(npu, npu)
        if key not in index:
            index[key] = len(groups)
            groups.append((key, []))
        groups[index[key]][1].append(unit)
    for gi, (key, members) in enumerate(groups):
        node = nodes[gi % len(nodes)]
        for unit in members:
            unit["target_node"] = node
            unit["target_node_source"] = "global_slice"
            unit["target_cpus"] = []
            unit["package_group"] = key if not isinstance(key, tuple) else None
            if len(members) > 1:
                # 同封装多 unit（双 die / 多 rank）：同 node 内再按 unit 均分
                unit["package_shared_with"] = [
                    u.get("unit_id") for u in members if u is not unit
                ]


def _index_npus(topo):
    """建立 NPU id 与 NUMA node 的索引。"""
    npu_by_node = {}
    npu_by_id = {}
    for npu in topo.get("npu") or []:
        npu_by_id[npu["id"]] = npu
        node = npu.get("numa_node")
        if node is not None:
            npu_by_node.setdefault(node, []).append(npu)
    return npu_by_node, npu_by_id


def _locality_mode(topo):
    """解析求解策略，兼容 legacy 别名。"""
    raw_mode = (topo.get("meta") or {}).get("locality_mode")
    return LEGACY.get(raw_mode, raw_mode) or "topo_affinity"


def _apply_global_slice(topo, units):
    """global_slice 形态：按封装分组后派生 L3 字段。"""
    _assign_global_slice(topo, units)
    for unit in units:
        unit["target_l3"] = None
        unit["current_l3"] = _l3_of_cpus(unit.get("current_affinity_cpus") or [], topo)


def _apply_topo_affinity(topo, units, npu_by_id, npu_by_node):
    """按 NPU 亲和派生 target node / L3 / 目标 CPU。"""
    for unit in units:
        target_node, deviation = _target_node(unit, npu_by_id)
        unit["target_node"] = target_node
        unit["target_node_source"] = "topo_affinity"
        if deviation:
            unit.setdefault("deviations", []).append(deviation)
        unit["target_l3"] = _l3_of_cpus(unit.get("npu_affinity_cpus") or [], topo)
        unit["current_l3"] = _l3_of_cpus(unit.get("current_affinity_cpus") or [], topo)
        if target_node is not None:
            unit["target_cpus"] = sorted({
                c for npu in npu_by_node.get(target_node, [])
                for c in (npu.get("affinity_cpus") or [])
            })
        else:
            unit["target_cpus"] = []


def _shared_node_findings(npu_by_node):
    """多个 NPU 共享同一 node CPU 池的提示。"""
    findings = []
    for node, npus in sorted(npu_by_node.items()):
        if len(npus) > 1:
            findings.append({
                "kind": "shared_npu_node",
                "severity": "info",
                "node": node,
                "evidence": "node%s 上有 NPU %s，共享同一 CPU 池；每 unit 核数需按 node 均分" % (
                    node, [n["id"] for n in npus]
                ),
            })
    return findings


def _global_slice_findings(topo):
    """global_slice 形态与双 die 同封装约束的提示。"""
    findings = []
    findings.append({
        "kind": "no_npu_cpu_affinity",
        "severity": "info",
        "handling": "observe",
        "evidence": "未观测到 NPU→CPU 亲和（%s 形态）；按封装分组把 CPU 池均分到各 node，NPU 与 CPU 不再一一对应" % (
            (topo.get("meta") or {}).get("fabric") or "超节点/UB"
        ),
    })
    meta = topo.get("meta") or {}
    pairs = [g for g in (meta.get("die_groups") or []) if len(g) > 1]
    if pairs:
        findings.append({
            "kind": "die_pair_same_node",
            "severity": "info",
            "handling": "applied",
            "evidence": "双 die 机型（%d 封装 / %d die）：同封装 die 对 %s 的 unit 已分到同一 node，"
                        "不跨 node 拆分（同封装 die 经 SIO 片内直连）" % (
                            meta.get("package_count") or len(pairs),
                            meta.get("die_count") or sum(len(g) for g in pairs),
                            pairs,
                        ),
        })
    return findings


def _unknown_node_findings(units):
    """NPU 未解析到 NUMA node 的提示。"""
    findings = []
    for unit in units:
        if unit.get("target_node") is None:
            npu = unit.get("primary_npu")
            findings.append({
                "kind": "npu_node_unknown",
                "severity": "medium",
                "unit_id": unit.get("unit_id"),
                "evidence": "NPU%s 未解析到 NUMA node（topo 可能缺失）" % (
                    npu if npu is not None else "-"),
            })
    return findings


def build(topo, units):
    npu_by_node, npu_by_id = _index_npus(topo)

    locality_mode = _locality_mode(topo)
    if locality_mode == "global_slice":
        _apply_global_slice(topo, units)
    else:
        _apply_topo_affinity(topo, units, npu_by_id, npu_by_node)

    findings = _shared_node_findings(npu_by_node)
    if locality_mode == "global_slice":
        findings.extend(_global_slice_findings(topo))
    findings.extend(_unknown_node_findings(units))

    return {
        "units": units,
        "npu_by_node": {str(k): [n["id"] for n in v] for k, v in sorted(npu_by_node.items())},
        "locality_mode": locality_mode,
        "findings": findings,
    }

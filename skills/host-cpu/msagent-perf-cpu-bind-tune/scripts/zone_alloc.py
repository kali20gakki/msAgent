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

"""划区与核预算：每个 node 预留系统中断区/干扰隔离区，其余为业务区。

A2 实机：24 CPU/node、无 SMT；默认 system 2 + interference 1 = 3，
业务区 21；同一 node 上多个 unit 均分连续核集。
"""

import topology


DEFAULT_POLICY = {
    "smt": "exclusive",
    "irq_reserve_per_node": 2,
    "interference_reserve_per_node": 1,
}


def allocate(topo, units, policy=None):
    policy = dict(DEFAULT_POLICY, **(policy or {}))
    by_node = {}
    for unit in units:
        node = unit.get("target_node")
        if node is None:
            continue
        by_node.setdefault(node, []).append(unit)

    zones = {}
    alloc = {}
    for node in topo.get("nodes") or []:
        nid = node["node"]
        node_units = by_node.get(nid, [])
        if node_units:
            cpus = sorted({
                c for u in node_units for c in (u.get("target_cpus") or [])
            })
            if not cpus:
                cpus = list(node["cpus"])
        else:
            cpus = list(node["cpus"])

        irq_n = int(policy.get("irq_reserve_per_node") or 0)
        int_n = int(policy.get("interference_reserve_per_node") or 0)
        # 预留区从 CPU 编号高端截取（尾部），业务区保持低端连续：
        # 1) unit 核集（compute/comm/io/main）始终连续，便于 OMP_PLACES=cores；
        # 2) 内核 softirq/irqbalance 默认倾向低端 CPU，高端核天然受打扰概率更低；
        # 3) 预留大于 node 半数时放弃预留（小 node 全给业务），由 C1 报告预算问题。
        if irq_n + int_n >= len(cpus):
            irq_n, int_n = 0, 0
        interference = cpus[-int_n:] if int_n else []
        system = cpus[-(int_n + irq_n):(-int_n if int_n else None)] if irq_n else []
        business = cpus[:-(int_n + irq_n)] if (int_n + irq_n) else cpus

        # 只保留 cpulist 一份核集描述，避免 cpus/cpulist 双字段口径漂移
        zones[str(nid)] = {
            "node": nid,
            "cpulist": topology.cpus_to_str(cpus),
            "system_irq": topology.cpus_to_str(system),
            "interference": topology.cpus_to_str(interference),
            "business": topology.cpus_to_str(business),
            "business_cpus": business,
            "unit_ids": [u["unit_id"] for u in node_units],
        }

        if node_units:
            for unit, cpuset in zip(
                sorted(node_units, key=lambda x: x["unit_id"]),
                _split_contiguous(business, len(node_units))
            ):
                alloc[unit["unit_id"]] = cpuset

    return {
        "policy": policy,
        "zones": zones,
        "alloc": alloc,
    }


def _split_contiguous(cpus, n):
    if n <= 0:
        return []
    if n == 1:
        return [list(cpus)]
    base = len(cpus) // n
    rem = len(cpus) % n
    out = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append(list(cpus[idx:idx + size]))
        idx += size
    return out

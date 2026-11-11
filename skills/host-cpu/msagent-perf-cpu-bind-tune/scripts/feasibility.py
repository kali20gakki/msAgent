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

"""规划可行性检查 C1-C5。

C1 核预算不足 / C2 L3 跨域 / C3 unit 核集冲突 / C4 SMT 启用 / C5 容量回退。
"""

# C1 最小核数 = 角色最小需求（comm 1 + io 1 + compute 1），与 unit_alloc
# 「compute 至少 1 核」的收缩下限对应；调整任一侧时需同步。
MIN_UNIT_CPUS = 3


def _smt_enabled(topo):
    """SMT 判定：优先用拓扑里的确认值，缺失时按线程兄弟核推断。

    部分拓扑来源不写 topo["smt"]，只看该字段会让 C4 永不触发。
    """
    smt = topo.get("smt") or {}
    if "enabled" in smt:
        return bool(smt.get("enabled"))
    return any(len(set(c.get("thread_siblings") or [])) > 1
               for c in topo.get("cpus") or [])


def check(topo, units, view_limited=False):
    """C1-C5 检查。

    view_limited 由采集侧（preflight / discover）传入，确保 flags 与 plan.summary
    在容器、/proc/1 不可读、numa_maps 不可读等受限场景如实反映证据受限。
    """
    unresolved = []
    warnings = []
    flags = {
        "smt_shared": False,
        "oversubscribed": False,
        "cross_numa": False,
        "l3_split": False,
        "view_limited": bool(view_limited),
    }

    for unit in units:
        uid = unit.get("unit_id")
        cpus = unit.get("cpuset_cpus") or []
        if len(cpus) < MIN_UNIT_CPUS:
            flags["oversubscribed"] = True
            unresolved.append({
                "check": "C1",
                "unit_id": uid,
                "reason": "可用核 %d < 最小需求 %d（comm 1 + io 1 + compute 1）" % (
                    len(cpus), MIN_UNIT_CPUS),
            })

        l3 = _l3_split(cpus, topo)
        if l3:
            flags["l3_split"] = True
            warnings.append({
                "check": "C2",
                "unit_id": uid,
                "reason": "unit 核集跨 L3 domain",
            })

        # 先清历史值：复用已加载 unit 时 gap<=0 不能残留上一次的差值
        unit.pop("capacity_gap", None)
        current = len(unit.get("current_affinity_cpus") or [])
        planned = len(cpus)
        gap = current - planned
        if gap > 0:
            unit["capacity_gap"] = gap
            mode = (topo.get("meta") or {}).get("locality_mode")
            tail = ("按 node 均分后少于全机亲和" if mode in ("global_slice", "balanced_partition")
                    else "local-first 以 NUMA 正确性优先")
            warnings.append({
                "check": "C5",
                "unit_id": uid,
                "reason": "plan 核数 %d < 当前亲和 %d，差 %d；%s" % (
                    planned, current, gap, tail
                ),
            })

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a = set(units[i].get("cpuset_cpus") or [])
            b = set(units[j].get("cpuset_cpus") or [])
            if a & b:
                unresolved.append({
                    "check": "C3",
                    "unit_id": [units[i].get("unit_id"), units[j].get("unit_id")],
                    "reason": "核集相交：%s" % sorted(a & b),
                })

    # 计划把 unit 从当前 node 迁到其它 node 时标记跨 node 求解
    flags["cross_numa"] = any(
        u.get("current_node") is not None and u.get("target_node") is not None
        and u.get("current_node") != u.get("target_node")
        for u in units
    )

    if _smt_enabled(topo):
        flags["smt_shared"] = True
        warnings.append({
            "check": "C4",
            "reason": "检测到 SMT；当前策略为 exclusive，未主动启用兄弟线程",
        })
    else:
        flags["smt_shared"] = False

    return {
        "unresolved": unresolved,
        "warnings": warnings,
        "flags": flags,
    }


def _l3_split(cpus, topo):
    cpus = set(cpus)
    if not cpus:
        return False
    domains = topo.get("l3_domains") or []
    if not domains:
        return False
    for dom in domains:
        if cpus <= set(dom.get("cpus") or []):
            return False
    return True

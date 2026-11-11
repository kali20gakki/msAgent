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

"""进程发现与单元建模：npu-smi 卡↔进程为锚点，补齐进程树/affinity/线程角色。

输出 units[]（训练 rank / 推理实例），以及 NUMA 亲和偏差。
"""

import os

import npu_query
import procfs
import roles
import topology


FRAMEWORK_PATTERNS = [
    ("sglang", ("sglang",)),
    ("vllm", ("vllm", "vllmworker")),
    ("verl", ("verl", "ray::", "raylet")),
    ("mindspeed", ("mindspeed",)),
    ("megatron", ("megatron", "torchrun", "msrun")),
]

# 实质占用卡判定阈值：低于该内存（控制/共享内存，如 A2 的 117MB 条目）或
# 低于主卡 10% 的条目不计入 node 交集，避免多卡控制内存污染求解。
# 两个值均来自 A2 实测（48GB 主卡 vs 117MB 控制内存），换机型在 references/ 实测记录中复核。
CONTROL_MEM_FLOOR_MB = 1024
CONTROL_MEM_RATIO = 0.1

# 远端页占比告警阈值（对应 references/process-roles.md 实测记录）：
# > 20% 说明存在明显跨 node 内存访问，> 50% 升级为高严重度。
REMOTE_RATIO_WARN = 0.2
REMOTE_RATIO_HIGH = 0.5

# unit 明细默认只保留每个角色前 N 条：数千线程时全量明细会撑大产物并挤占上下文，
# 计数与漂移统计仍按全量计算。
THREAD_DETAIL_LIMIT = 20


def classify_framework(cmdline, comm):
    text = ("%s %s" % (cmdline or "", comm or "")).lower()
    for name, patterns in FRAMEWORK_PATTERNS:
        for p in patterns:
            if p in text:
                return name
    return "unknown"


def classify_thread(comm):
    return roles.classify(comm)


def _cpus(text):
    return topology.parse_cpulist(text)


def _all_cpu_set(topo):
    """全机 CPU 集合：topo 生命周期内不变，由调用方预计算后复用。"""
    return set(c["cpu"] for c in topo.get("cpus") or [])


def _is_global_affinity(affinity_cpus, topo, all_cpus=None):
    """亲和覆盖全机 CPU（未绑核）时为真：此时无法归属单一 node。"""
    if not affinity_cpus:
        return False
    all_cpus = _all_cpu_set(topo) if all_cpus is None else all_cpus
    return bool(all_cpus) and set(affinity_cpus) >= all_cpus


def _node_of_affinity(affinity_cpus, topo, all_cpus=None):
    if not affinity_cpus:
        return None
    if _is_global_affinity(affinity_cpus, topo, all_cpus):
        return None
    return topology.node_of_cpus(affinity_cpus, topo.get("nodes") or [])


def _proc_node(affinity_cpus, cgroup_cpus, topo, all_cpus=None):
    """进程 node 归属：全机亲和视为未绑核，不退回 cgroup 推断以免伪报 node 偏差。"""
    all_cpus = _all_cpu_set(topo) if all_cpus is None else all_cpus
    if _is_global_affinity(affinity_cpus, topo, all_cpus):
        return None
    node = _node_of_affinity(affinity_cpus, topo, all_cpus)
    if node is None and cgroup_cpus and not _is_global_affinity(cgroup_cpus, topo, all_cpus):
        node = topology.node_of_cpus(cgroup_cpus, topo.get("nodes") or [])
    return node


def _npu_map(topo):
    return {n["id"]: n for n in topo.get("npu") or []}


def _node_of_cpu(cpu, topo):
    return topology.node_of_cpus([cpu], topo.get("nodes") or []) if cpu is not None else None


def _significant_npus(proc):
    """实质占用卡集：过滤控制内存条目，避免污染多卡 node 交集。"""
    primary_mem = proc.get("primary_mem_mb") or 0
    threshold = max(CONTROL_MEM_FLOOR_MB, primary_mem * CONTROL_MEM_RATIO)
    cards = sorted({
        e["npu"] for e in (proc.get("entries") or [])
        if (e.get("mem_mb") or 0) >= threshold
    })
    return cards or ([proc.get("primary_npu")] if proc.get("primary_npu") is not None else [])


def _collect_proc_basics(proc, topo, current_uid, npu_by_id, all_cpus):
    """采集进程身份、亲和、cgroup 与 NPU 归属等基础字段。"""
    pid = proc["pid"]
    status = procfs.read_status(pid)
    if status is None:
        return None
    affinity = status.get("Cpus_allowed_list", "")
    affinity_cpus = _cpus(affinity)
    cgroup = procfs.read_cgroup(pid)
    cgroup_cpuset = procfs.read_cgroup_cpuset(pid, cgroup)
    cgroup_cpus = _cpus(cgroup_cpuset) if cgroup_cpuset else []
    cgroup_mems = procfs.read_cgroup_mems(pid, cgroup)
    target_uid = procfs.read_uid(pid)
    is_own_process = target_uid is not None and current_uid is not None and target_uid == current_uid
    last_cpu = procfs.read_last_cpu(pid)
    last_cpu_node = _node_of_cpu(last_cpu, topo)
    cmdline = procfs.read_cmdline(pid)
    comm = status.get("Name", "")
    framework = classify_framework(cmdline, comm)
    npu = npu_by_id.get(proc["primary_npu"]) or {}
    npu_node = npu.get("numa_node")
    npu_affinity = npu.get("affinity_cpus") or []

    proc_node = _proc_node(affinity_cpus, cgroup_cpus, topo, all_cpus)

    return {
        "proc": proc,
        "pid": pid,
        "status": status,
        "affinity": affinity,
        "affinity_cpus": affinity_cpus,
        "cgroup": cgroup,
        "cgroup_cpuset": cgroup_cpuset,
        "cgroup_mems": cgroup_mems,
        "target_uid": target_uid,
        "current_uid": current_uid,
        "is_own_process": is_own_process,
        "last_cpu": last_cpu,
        "last_cpu_node": last_cpu_node,
        "cmdline": cmdline,
        "comm": comm,
        "framework": framework,
        "npu": npu,
        "npu_node": npu_node,
        "npu_affinity": npu_affinity,
        "proc_node": proc_node,
    }


def _collect_numa_pages(pid, npu_node):
    """解析 numa_maps 并计算远端页占比；npu_node 未知或不可读时标为不可观测。"""
    numa = procfs.read_numa_maps(pid)
    remote_ratio = None
    limited = False
    if not numa.get("ok") or not numa.get("total"):
        limited = True
    elif npu_node is not None:
        remote = sum(v for k, v in numa["pages"].items() if k != npu_node)
        remote_ratio = round(remote / float(numa["total"]), 4)
    return numa, remote_ratio, limited


def _collect_threads(pid, topo, npu_node, all_cpus):
    """采集进程线程并按角色归类，统计 node 漂移数与各 node 线程分布。"""
    threads = procfs.read_threads(pid)
    # 不与模块 roles 同名，避免遮蔽 roles.classify
    by_role = {"compute": [], "comm": [], "io": [], "main": []}
    thread_drift = 0
    thread_last_cpu_nodes = {}
    for t in threads:
        role = classify_thread(t.get("comm"))
        t["role"] = role
        t["last_cpu_node"] = _node_of_cpu(t.get("last_cpu"), topo)
        if t["last_cpu_node"] is not None:
            thread_last_cpu_nodes[t["last_cpu_node"]] = thread_last_cpu_nodes.get(t["last_cpu_node"], 0) + 1
        by_role[role].append(t)
        if t.get("affinity"):
            tnode = _node_of_affinity(_cpus(t["affinity"]), topo, all_cpus)
            if npu_node is not None and tnode is not None and tnode != npu_node:
                thread_drift += 1
    return threads, by_role, thread_drift, thread_last_cpu_nodes


def _evidence_level(numa, last_cpu_node, affinity_cpus):
    """按 numa_maps / last_cpu / 亲和 的可得性判定证据级别。"""
    if numa.get("ok"):
        return "full"
    if last_cpu_node is not None:
        return "affinity+last_cpu"
    if affinity_cpus:
        return "affinity_only"
    return "npu_smi_only"


def _collect_proc_metrics(ctx, topo, all_cpus):
    """采集 numa_maps、线程角色与证据级别，返回 unit 装配所需指标。"""
    numa, remote_ratio, limited = _collect_numa_pages(ctx["pid"], ctx["npu_node"])
    threads, by_role, thread_drift, thread_last_cpu_nodes = _collect_threads(
        ctx["pid"], topo, ctx["npu_node"], all_cpus
    )
    evidence_level = _evidence_level(numa, ctx["last_cpu_node"], ctx["affinity_cpus"])
    return {
        "numa": numa,
        "remote_ratio": remote_ratio,
        "limited": limited,
        "threads": threads,
        "by_role": by_role,
        "thread_drift": thread_drift,
        "thread_last_cpu_nodes": thread_last_cpu_nodes,
        "evidence_level": evidence_level,
    }


def _process_deviations(ctx, metrics):
    """进程 node 偏移、亲和越界与远端页占比相关偏差。"""
    deviations = []
    proc_node = ctx["proc_node"]
    npu_node = ctx["npu_node"]
    if proc_node is not None and npu_node is not None and proc_node != npu_node:
        deviations.append({
            "kind": "process_node_vs_npu_node",
            "severity": "high",
            "handling": "solve",
            "evidence": "进程 node%s != NPU%d node%s" % (proc_node, ctx["proc"]["primary_npu"], npu_node),
        })
    if ctx["npu_affinity"] and ctx["affinity_cpus"]:
        outside = sorted(set(ctx["affinity_cpus"]) - set(ctx["npu_affinity"]))
        if outside:
            deviations.append({
                "kind": "affinity_outside_npu_range",
                "severity": "high",
                "handling": "solve",
                "evidence": "当前亲和 %s，超出 NPU%d 推荐范围 %s；越界 CPU %s" % (
                    ctx["affinity"], ctx["proc"]["primary_npu"], ctx["npu"].get("affinity"),
                    topology.cpus_to_str(outside)
                ),
            })
    remote_ratio = metrics["remote_ratio"]
    if remote_ratio is not None and remote_ratio > REMOTE_RATIO_WARN:
        deviations.append({
            "kind": "remote_page_ratio",
            "severity": "high" if remote_ratio > REMOTE_RATIO_HIGH else "medium",
            "handling": "solve",
            "evidence": "远端页占比 %.1f%%（npu-smi 主卡 node%s）" % (remote_ratio * 100, npu_node),
        })
    elif not metrics["numa"].get("ok"):
        deviations.append({
            "kind": "numa_maps_unavailable",
            "severity": "info",
            "handling": "view_limited",
            "evidence": "numa_maps 不可读：%s（远端页占比标 not_observable；证据级别 %s）" % (
                metrics["numa"].get("reason"), metrics["evidence_level"]
            ),
        })
    return deviations


def _thread_deviations(ctx, thread_drift):
    """线程 node 漂移、最近 CPU 偏移与跨用户控制相关偏差。"""
    deviations = []
    npu_node = ctx["npu_node"]
    if thread_drift:
        deviations.append({
            "kind": "thread_node_drift",
            "severity": "medium",
            "handling": "solve",
            "evidence": "%d 个线程所在 node 与 NPU node 不一致" % thread_drift,
        })
    if (ctx["last_cpu"] is not None and ctx["last_cpu_node"] is not None
            and npu_node is not None and ctx["last_cpu_node"] != npu_node):
        deviations.append({
            "kind": "last_cpu_node_mismatch",
            "severity": "low",
            "handling": "observe",
            "evidence": "最近运行 CPU %s 在 node%s，NPU%d 在 node%s（弱证据，权限受限时的替代观测）" % (
                ctx["last_cpu"], ctx["last_cpu_node"], ctx["proc"]["primary_npu"], npu_node
            ),
        })
    if ctx["target_uid"] is not None and ctx["current_uid"] is not None and ctx["target_uid"] != ctx["current_uid"]:
        deviations.append({
            "kind": "cross_user_control",
            "severity": "medium",
            "handling": "handoff",
            "evidence": "目标进程 UID %s，当前用户 UID %s；taskset/cgroup 变更需 sudo 或同用户执行" % (
                ctx["target_uid"], ctx["current_uid"]
            ),
        })
    return deviations


def _io_deviations(npu_node, topo):
    """NPU node 无本地网卡 / 本地 NVMe 的附属偏差。"""
    deviations = []
    nics = topology.nics_of_node(topo, npu_node) if npu_node is not None else []
    if npu_node is not None and not nics:
        up_nics = [n for n in topo.get("nic") or [] if n.get("operstate") == "up"]
        deviations.append({
            "kind": "nic_node_mismatch",
            "severity": "medium",
            "handling": "handoff",
            "evidence": "NPU node%s 无物理网卡；UP 网卡位于 node %s（转 config-audit IRQ 亲和）" % (
                npu_node, sorted({n.get("numa_node") for n in up_nics if n.get("numa_node") is not None})
            ),
        })
    storage = topology.storage_of_node(topo, npu_node) if npu_node is not None else []
    if npu_node is not None and not storage:
        deviations.append({
            "kind": "storage_node_mismatch",
            "severity": "low",
            "handling": "note",
            "evidence": "NPU node%s 无本地 NVMe；模型/数据加载跨 node" % npu_node,
        })
    return deviations


def _build_unit(unit_id, ctx, metrics, deviations):
    """组装单个 unit（训练 rank / 推理实例）输出条目。"""
    threads = metrics["threads"]
    by_role = metrics["by_role"]
    unit = {
        "unit_id": unit_id,
        "unit_type": "instance" if ctx["framework"] in ("vllm", "sglang") else "rank",
        "framework": ctx["framework"],
        "pid": ctx["pid"],
        "pids": [ctx["pid"]],
        "ppid": int(ctx["status"].get("PPid")) if ctx["status"].get("PPid", "").isdigit() else None,
        "process_name": ctx["comm"],
        "cmdline": ctx["cmdline"][:400],
        "primary_npu": ctx["proc"]["primary_npu"],
        "npus": ctx["proc"]["npus"],
        "npus_significant": _significant_npus(ctx["proc"]),
        "npu_node": ctx["npu_node"],
        "npu_affinity": ctx["npu"].get("affinity"),
        "npu_affinity_cpus": ctx["npu_affinity"],
        "current_node": ctx["proc_node"],
        "current_affinity": ctx["affinity"],
        "current_affinity_cpus": ctx["affinity_cpus"],
        "cgroup_cpuset": ctx["cgroup_cpuset"],
        "cgroup_mems": ctx["cgroup_mems"],
        "cgroup_path": ctx["cgroup"].get("path"),
        "cgroup_version": ctx["cgroup"].get("version"),
        "target_uid": ctx["target_uid"],
        "is_own_process": ctx["is_own_process"],
        "mems_allowed": ctx["status"].get("Mems_allowed_list"),
        "last_cpu": ctx["last_cpu"],
        "last_cpu_node": ctx["last_cpu_node"],
        "thread_last_cpu_nodes": metrics["thread_last_cpu_nodes"],
        "evidence_level": metrics["evidence_level"],
        "thread_count": len(threads),
        "thread_roles": {k: len(v) for k, v in by_role.items()},
        "threads_by_role": {
            k: [{"tid": t["tid"], "comm": t["comm"], "affinity": t.get("affinity"),
                 "last_cpu": t.get("last_cpu"), "last_cpu_node": t.get("last_cpu_node")}
                for t in v[:THREAD_DETAIL_LIMIT]]
            for k, v in by_role.items()
        },
        "threads_truncated": {k: len(v) - THREAD_DETAIL_LIMIT for k, v in by_role.items()
                              if len(v) > THREAD_DETAIL_LIMIT},
        "numa_maps": metrics["numa"],
        "remote_page_ratio": metrics["remote_ratio"],
        "deviations": deviations,
        "before": {
            "Cpus_allowed_list": ctx["affinity"],
            "cgroup_cpuset": ctx["cgroup_cpuset"],
            "source": "/proc/%d/status" % ctx["pid"],
        },
    }
    return unit


def collect_units(npu_info, topo):
    primaries = npu_query.primary_processes(npu_info.get("processes") or [])
    npu_by_id = _npu_map(topo)
    units = []
    view_limited = False
    current_uid = os.getuid() if hasattr(os, "getuid") else None
    all_cpus = _all_cpu_set(topo)

    for proc in primaries:
        ctx = _collect_proc_basics(proc, topo, current_uid, npu_by_id, all_cpus)
        if ctx is None:
            continue
        metrics = _collect_proc_metrics(ctx, topo, all_cpus)
        if metrics["limited"]:
            view_limited = True

        deviations = []
        deviations.extend(_process_deviations(ctx, metrics))
        deviations.extend(_thread_deviations(ctx, metrics["thread_drift"]))
        deviations.extend(_io_deviations(ctx["npu_node"], topo))

        # unit_id 用输出侧序号而非输入下标：跳过无法采集的进程后仍连续
        units.append(_build_unit(len(units), ctx, metrics, deviations))

    return {
        "units": units,
        "view_limited": view_limited,
        "unknown_hbm_npus": npu_query.unknown_hbm_npus(npu_info),
    }


def unknown_hbm(npu_info):
    """兼容入口：判定唯一实现在 npu_query.unknown_hbm_npus。"""
    return npu_query.unknown_hbm_npus(npu_info)


if __name__ == "__main__":
    import json
    q = npu_query.query()
    topo = topology.collect(q)
    print(json.dumps(collect_units(q, topo), ensure_ascii=False, indent=2))

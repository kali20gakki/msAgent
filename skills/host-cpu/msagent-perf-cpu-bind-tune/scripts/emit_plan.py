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

"""plan 组装：命令、回滚、验证脚本、收益预估。

只产出动作契约，不执行 taskset；apply 由主会话确认后执行。
"""

import hashlib
import json
import logging
import os
import shlex
import socket
import sys
import tempfile
import time

_LOG = logging.getLogger(__name__)


def _ts():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _hash(obj):
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _script(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _apply_command(plan_path, extra):
    return "%s %s %s%s" % (
        shlex.quote(sys.executable or "python3"),
        shlex.quote(_script("apply_plan.py")),
        shlex.quote(plan_path),
        extra,
    )


def _unit_commands(unit, plan_path, form):
    """执行命令：执行时枚举线程，不落 tid 快照。"""
    if form != "in_run" or not unit.get("pid"):
        return [], []
    return (
        [_apply_command(plan_path, " --roles comm,io")],
        [_apply_command(plan_path, " --rollback")],
    )


def _build_unit_output(units, plan_path, form):
    """组装 units 输出字段，并收集各 unit 原始 cgroup cpuset。"""
    unit_out = []
    all_before = []
    for unit in units:
        commands, rollback = _unit_commands(unit, plan_path, form)
        if form == "pre_run":
            commands, rollback = [], []
        all_before.append((unit.get("before") or {}).get("cgroup_cpuset"))
        unit_out.append({
            "unit_id": unit.get("unit_id"),
            "unit_type": unit.get("unit_type"),
            "framework": unit.get("framework"),
            "pid": unit.get("pid"),
            "pids": unit.get("pids"),
            "process_name": unit.get("process_name"),
            "primary_npu": unit.get("primary_npu"),
            "npus": unit.get("npus"),
            "npus_significant": unit.get("npus_significant"),
            "numa_node": unit.get("target_node"),
            "target_node_source": unit.get("target_node_source"),
            "npu_affinity": unit.get("npu_affinity"),
            "current_node": unit.get("current_node"),
            "current_affinity": unit.get("current_affinity"),
            "cpuset": unit.get("cpuset"),
            "main_cpuset": unit.get("main_cpuset"),
            "compute_cpuset": unit.get("compute_cpuset"),
            "work_cpuset": unit.get("work_cpuset"),
            "thread_binding": unit.get("thread_binding"),
            "roles": unit.get("roles"),
            "env": unit.get("env"),
            "mechanism": "sched_setaffinity" if form == "in_run" else "launch_wrapper",
            "command_source": "runtime_task_scan" if form == "in_run" else "env_only",
            "commands": commands,
            "before": unit.get("before"),
            "rollback": rollback,
            "target_uid": unit.get("target_uid"),
            "is_own_process": unit.get("is_own_process"),
            "requires_privilege": bool(unit.get("pid")) and not bool(unit.get("is_own_process")),
            "evidence_level": unit.get("evidence_level"),
            "last_cpu": unit.get("last_cpu"),
            "last_cpu_node": unit.get("last_cpu_node"),
            "mems_allowed": unit.get("mems_allowed"),
            "cgroup_mems": unit.get("cgroup_mems"),
            "deviations": unit.get("deviations"),
            "capacity_gap": unit.get("capacity_gap", 0),
            "thread_roles": unit.get("thread_roles"),
        })
    return unit_out, all_before


def _common_container_cpuset(all_before):
    """所有 unit 的 cgroup cpuset 一致时视为容器级 cpuset。"""
    unique = {x for x in all_before if x}
    if len(unique) == 1:
        return unique.pop()
    return None


def _privilege_profile(form, units, preflight_info):
    """推导执行主体、权限需求与 numa_maps 可读性。"""
    target_uids = sorted({u.get("target_uid") for u in units if u.get("target_uid") is not None})
    has_pid = any(u.get("pid") for u in units)
    own_processes = bool(has_pid) and all(u.get("is_own_process") for u in units if u.get("pid"))
    is_root = bool(preflight_info.get("is_root"))
    can_execute = bool(preflight_info.get("has_taskset")) and (own_processes or is_root)
    if form == "pre_run":
        execution_mode = "plan_only"
    elif can_execute:
        execution_mode = "self"
    else:
        execution_mode = "delegated"
    # root 具备 CAP_SYS_NICE，跨用户目标进程也不额外需要提权
    requires_privilege = form == "in_run" and not own_processes and not is_root
    can_read_numa_maps = (
        any((u.get("numa_maps") or {}).get("ok") for u in units)
        if units else bool(preflight_info.get("numa_maps_self_ok"))
    )
    return {
        "target_uids": target_uids,
        "own_processes": own_processes,
        "execution_mode": execution_mode,
        "requires_privilege": requires_privilege,
        "can_read_numa_maps": can_read_numa_maps,
    }


def _collect_unit_deviations(units):
    """汇总 unit 偏差，并生成容量回退与证据提示。"""
    deviations = []
    notes = []
    for u in units:
        deviations.extend(u.get("deviations") or [])
        if u.get("capacity_gap", 0) > 0:
            notes.append("unit %s plan 核数 %d < 当前亲和 %d（差 %d）；local-first 优先 NUMA 正确性" % (
                u.get("unit_id"), len(u.get("cpuset_cpus") or []),
                len(u.get("current_affinity_cpus") or []), u.get("capacity_gap")
            ))
        for d in u.get("deviations") or []:
            if d.get("kind") == "numa_maps_unavailable":
                notes.append("unit %s numa_maps 不可读：远端页占比标 not_observable" % u.get("unit_id"))
            if d.get("kind") == "nic_node_mismatch":
                notes.append("unit %s：%s" % (u.get("unit_id"), d.get("evidence")))
    return deviations, notes


def _hbm_invisible_notes(unknown):
    """对有 HBM 占用但无可见进程的 NPU 追加确认提示。"""
    notes = []
    for item in unknown:
        notes.append("NPU%d 有 HBM 占用 %.0f MB 但无可见进程，可能存在其他容器/残留，绑核前需确认" % (
            item["npu"], item["hbm_used_mb"]
        ))
    return notes


def _environment_notes(preflight_info, form, locality_mode, requires_privilege, can_read_numa_maps,
                       fabric, die_groups):
    """按权限、可观测性与求解策略补充环境提示。"""
    notes = []
    if requires_privilege:
        notes.append("目标进程属于其他用户/root；taskset/cgroup 变更需 sudo 或同用户执行，plan 不自动执行")
    if form == "in_run" and not can_read_numa_maps:
        notes.append("numa_maps 不可读；远端页占比 not_observable，已用 Cpus_allowed_list + 最近运行 CPU 作为弱证据")
    if form == "in_run" and not preflight_info.get("has_taskset"):
        notes.append("taskset 不可用；只能输出方案，由管理员选择 cgroup/启动包装执行")
    if locality_mode == "global_slice":
        notes.append(
            "求解策略 global_slice：未观测到 NPU→CPU 亲和（%s 形态），按封装分组把 CPU 池均分到各 node，"
            "保证 unit 间核集不重叠且切片对齐 NUMA 边界；NIC/存储/IRQ 仍按各自 node 就近"
            % (fabric or "超节点矩阵")
        )
        if any(len(g) > 1 for g in (die_groups or [])):
            notes.append(
                "双 die 机型：同一封装的两个 die 上的 unit 已锁定同一 node（同封装 die 经 SIO 片内直连，"
                "跨 node 拆分会让 die 间协同走远端）"
            )
    return notes


def _expected_effect(deviations, locality_mode):
    """评估收益上限等级，并给出证据依据。"""
    level = "low"
    if any(d.get("kind") in ("process_node_vs_npu_node", "affinity_outside_npu_range") for d in deviations):
        level = "high"
    elif any(d.get("kind") in ("remote_page_ratio", "thread_node_drift", "nic_node_mismatch") for d in deviations):
        level = "medium"

    basis = []
    for d in deviations:
        if d.get("kind") in ("process_node_vs_npu_node", "affinity_outside_npu_range",
                             "remote_page_ratio", "thread_node_drift"):
            evidence = d.get("evidence")
            if evidence:
                basis.append(evidence)
    if not basis:
        if locality_mode == "global_slice":
            basis.append("未观测到 NPU→CPU 亲和（超节点矩阵），按封装分组均分 CPU 池（global_slice）")
        else:
            basis.append("未发现跨 node 偏差；按拓扑生成预规划")
    return level, basis


def _verify_spec():
    """验证方案：配置生效判据、运行时观测与收益指标。"""
    return {
        "config_effective": {
            "commands": [
                "grep -m1 '^Cpus_allowed_list:' /proc/<pid>/status",
                "grep -m1 '^Cpus_allowed_list:' /proc/<pid>/task/<tid>/status",
            ],
        },
        "runtime": {
            "consumer": "tracing --plan <plan.json>",
            "check": "units[].roles[].cpus 落核/迁移/越界",
        },
        "effect": {
            "metrics": ["sched_delay_p99_us", "remote_page_ratio", "migration_per_s"],
            "threshold": "改善 ≥ 收益上限的 30%（§3.1.5）",
            "window": "同参数复采",
        },
    }


def _preflight_block(preflight_info):
    """preflight 快照块。"""
    return {
        "npu_smi": preflight_info.get("npu_smi"),
        "in_container": preflight_info.get("in_container"),
        "cgroup_version": preflight_info.get("cgroup_version"),
        "has_taskset": preflight_info.get("has_taskset"),
        "user": preflight_info.get("user"),
        "uid": preflight_info.get("uid"),
        "is_root": preflight_info.get("is_root"),
        "proc_hidden": preflight_info.get("proc_hidden"),
    }


def _privilege_block(preflight_info, priv):
    """权限块：账号、目标 uid、可读性与执行模式。"""
    return {
        "current_user": preflight_info.get("user"),
        "current_uid": preflight_info.get("uid"),
        "is_root": preflight_info.get("is_root"),
        "target_uids": priv["target_uids"],
        "own_processes": priv["own_processes"],
        "can_read_numa_maps": priv["can_read_numa_maps"],
        # root 可跨用户执行；非 root 仅当目标进程全部属于当前用户时可直接绑核
        "can_taskset": bool(preflight_info.get("has_taskset"))
        and (priv["own_processes"] or bool(preflight_info.get("is_root"))),
        "requires_privilege": priv["requires_privilege"],
        "execution_mode": priv["execution_mode"],
        "notes": [
            n for n in [
                "目标进程属于其他用户/root；需 sudo 或同用户执行" if priv["requires_privilege"] else None,
                "numa_maps 不可读；使用 affinity + last_cpu 弱证据" if not priv["can_read_numa_maps"] else None,
                "taskset 不可用；只能输出方案" if not preflight_info.get("has_taskset") else None,
            ] if n
        ],
    }


def _apply_block(plan_path, bind_compute):
    """apply 动作契约块：bind_compute 与 unit 级 compute 绑定语义保持一致。"""
    return {
        "script": _script("apply_plan.py"),
        "command": _apply_command(plan_path, " --roles comm,io"),
        "rollback": _apply_command(plan_path, " --rollback"),
        "roles": "comm,io",
        "bind_compute": bool(bind_compute),
        "dynamic_task_scan": True,
        "cgroup_clip": True,
    }


def _plan_flags(feasibility, preflight_info):
    """plan flags：feasibility 为权威来源，采集侧 view_limited 兜底覆盖。"""
    flags = dict(feasibility.get("flags") or {})
    flags["view_limited"] = bool(flags.get("view_limited") or preflight_info.get("view_limited"))
    return flags


def _meta_value(topo, preflight_info, key):
    """拓扑 meta 优先，preflight_info 仅作 legacy 别名兜底。"""
    value = (topo.get("meta") or {}).get(key)
    if value not in (None, "", []):
        return value
    return preflight_info.get(key)


def _summary_block(ctx, flags):
    """summary 汇总块。"""
    return {
        "unit_count": len(ctx["units"]),
        "deviation_count": len(ctx["deviations"]),
        "finding_count": len(ctx["findings"]),
        "unknown_hbm_npus": ctx["unknown"],
        "view_limited": bool(flags.get("view_limited")),
        "execution_mode": ctx["priv"]["execution_mode"],
        "requires_privilege": ctx["priv"]["requires_privilege"],
    }


def _assemble_plan(ctx):
    """装填 plan JSON 输出字段（字段顺序即对外契约）。"""
    preflight_info = ctx["preflight_info"]
    feasibility = ctx["feasibility"]
    flags = ctx["flags"]
    plan = {
        "schema_version": "hostcpu-plan/1",
        "plan_id": ctx["plan_id"],
        "host": ctx["host"],
        "ts_ns": int(time.time() * 1e9),
        "mode": "container" if preflight_info.get("in_container") else "host",
        "form": ctx["form"],
        "fabric": ctx["fabric"],
        "locality_mode": ctx["locality_mode"],
        "affinity_granularity": ctx["affinity_granularity"],
        "preflight": _preflight_block(preflight_info),
        "privilege": _privilege_block(preflight_info, ctx["priv"]),
        "topology_ref": {
            "source": "self.topology",
            "schema": "static-snapshot/topology",
            "hash": _hash(ctx["topo"]),
        },
        "executable": ctx["priv"]["execution_mode"] == "self",
        "plan_file": os.path.basename(ctx["plan_path"]),
        "apply": _apply_block(ctx["plan_path"], ctx["bind_compute"]),
        "before": {
            "container_cpuset": ctx["container_cpuset"],
            "source": "/proc/<pid>/cgroup",
        },
        "policy": ctx["zones_alloc"].get("policy"),
        "zones": ctx["zones_alloc"].get("zones"),
        "units": ctx["units"],
        "baseline": {
            "metrics": {
                "sched_delay_p99_us": None,
                "migration_per_s": None,
                "remote_page_ratio": None,
                "step_time_ms": None,
            },
            "source": "none",
            "window": None,
        },
        "verify": ctx["verify"],
        "expected_effect": {
            "level": ctx["level"],
            "basis": ctx["basis"][:5],
            "upper_bound": "消除跨 node 亲和与越界核；容量回退见 notes",
        },
        "verify_script": os.path.basename(ctx["verify_path"]),
        "verify_script_path": ctx["verify_path"],
        "report": os.path.basename(ctx["report_path"]),
        "unresolved": feasibility.get("unresolved") or [],
        "warnings": feasibility.get("warnings") or [],
        "findings": ctx["findings"],
        "flags": flags,
        "notes": ctx["notes"],
        "summary": _summary_block(ctx, flags),
    }
    return plan


def build_plan(topo, units, zones_alloc, feasibility, preflight_info, form, out_dir):
    ts = _ts()
    host = socket.gethostname()
    # pid 后缀避免同一秒内并发调用生成相同 plan_id
    plan_id = "hostcpu-plan-%s-%s-%d" % (host, ts, os.getpid())
    base = "hostcpu_plan_%s_%s" % (host, ts)
    plan_path = os.path.join(out_dir, base + ".json")
    verify_path = os.path.join(out_dir, "hostcpu_verify_%s_%s.sh" % (host, ts))
    report_path = os.path.join(out_dir, "hostcpu_report_%s_%s.html" % (host, ts))

    unit_out, all_before = _build_unit_output(units, plan_path, form)

    container_cpuset = _common_container_cpuset(all_before)
    priv = _privilege_profile(form, units, preflight_info)

    # 求解策略与拓扑事实以 topo["meta"] 为唯一权威（与 model / feasibility 一致），
    # preflight_info 仅保留 legacy 别名兜底，避免无亲和机器被误标 topo_affinity。
    locality_mode = _meta_value(topo, preflight_info, "locality_mode") or "topo_affinity"
    locality_mode = {"npu_local": "topo_affinity", "balanced_partition": "global_slice"}.get(
        locality_mode, locality_mode)
    fabric = _meta_value(topo, preflight_info, "fabric")
    affinity_granularity = _meta_value(topo, preflight_info, "affinity_granularity")
    die_groups = _meta_value(topo, preflight_info, "die_groups") or []
    deviations, notes = _collect_unit_deviations(units)

    unknown = (preflight_info.get("unknown_hbm_npus") or [])
    notes.extend(_hbm_invisible_notes(unknown))
    notes.extend(_environment_notes(preflight_info, form, locality_mode,
                                    priv["requires_privilege"], priv["can_read_numa_maps"],
                                    fabric, die_groups))

    level, basis = _expected_effect(deviations, locality_mode)
    flags = _plan_flags(feasibility, preflight_info)
    findings = list(preflight_info.get("findings") or [])
    # 存在 compute 核集且 apply=exec 时，执行块应声明 bind_compute=true
    bind_compute = any(
        b.get("role") == "compute" and b.get("apply") == "exec" and bool(b.get("cpuset"))
        for u in units for b in (u.get("thread_binding") or [])
    )

    verify = _verify_spec()

    ctx = {
        "plan_id": plan_id,
        "host": host,
        "form": form,
        "plan_path": plan_path,
        "verify_path": verify_path,
        "report_path": report_path,
        "topo": topo,
        "preflight_info": preflight_info,
        "zones_alloc": zones_alloc,
        "units": unit_out,
        "container_cpuset": container_cpuset,
        "priv": priv,
        "verify": verify,
        "locality_mode": locality_mode,
        "fabric": fabric,
        "affinity_granularity": affinity_granularity,
        "level": level,
        "basis": basis,
        "deviations": deviations,
        "unknown": unknown,
        "notes": notes,
        "feasibility": feasibility,
        "flags": flags,
        "findings": findings,
        "bind_compute": bind_compute,
    }
    plan = _assemble_plan(ctx)
    return plan, plan_path, verify_path, report_path


def write_plan(plan, path):
    """原子写入：同目录临时文件 + os.replace，避免并发/中断产生半截 plan。"""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".hostcpu-plan-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_verify_script(plan, path):
    verify_py = _script("verify_config.py")
    plan_file = plan.get("plan_file") or ""
    lines = [
        "#!/bin/sh",
        "# generated by msagent-perf-cpu-bind-tune",
        "# plan: %s" % plan.get("plan_id"),
        'DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
        "PY=python3",
        'command -v "$PY" >/dev/null 2>&1 || PY=/usr/bin/python3',
        'exec "$PY" %s "$DIR"/%s --threads' % (shlex.quote(verify_py), shlex.quote(plan_file)),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o755)
    except OSError as exc:
        _LOG.warning("verify 脚本 chmod +x 失败（%s）：请手动执行 chmod +x %s", exc, path)

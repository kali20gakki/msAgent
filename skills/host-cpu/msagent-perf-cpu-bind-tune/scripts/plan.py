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

"""msagent-perf-cpu-bind-tune 主入口。

默认 plan-only：采集 -> 建模 -> 划区 -> 分配 -> 可行性 -> plan/HTML/验证脚本。
不执行 taskset；execute 仅生成待确认执行预案。

用法：
  python3 plan.py --out ./out
  python3 plan.py --out ./out --dry-run --print-plan
  python3 plan.py --out ./out --irq-reserve 2 --interference-reserve 1
"""

import argparse
import json
import os
import sys

import discover
import emit_plan
import feasibility
import model
import npu_query
import preflight
import report
import topology
import unit_alloc
import workspace
import zone_alloc


def _non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("必须为非负整数：%s" % text)
    return value


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Host CPU bind-tune plan-only solver")
    p.add_argument("--out", default=None, help="output directory (default: <workspace>/workspaces/<host>/<timestamp>)")
    p.add_argument("--execute", action="store_true", help="在摘要中输出 apply/rollback 执行命令（仍需主会话确认）")
    p.add_argument("--no-html", action="store_true", help="skip HTML report")
    p.add_argument("--dry-run", action="store_true", help="do not write any output files")
    p.add_argument("--print-plan", action="store_true", help="print full plan JSON (with summary)")
    p.add_argument("--irq-reserve", type=_non_negative_int, default=2, help="system/IRQ reserve CPUs per node")
    p.add_argument("--interference-reserve", type=_non_negative_int, default=1,
                   help="interference reserve CPUs per node")
    return p.parse_args(argv)


def _pre_run_units(topo):
    units = []
    for idx, npu in enumerate(topo.get("npu") or []):
        units.append({
            "unit_id": idx,
            "unit_type": "rank",
            "framework": "unknown",
            "pid": None,
            "pids": [],
            "process_name": "pre_run",
            "primary_npu": npu.get("id"),
            "npus": [npu.get("id")],
            "npu_node": npu.get("numa_node"),
            "npu_affinity": npu.get("affinity"),
            "npu_affinity_cpus": npu.get("affinity_cpus") or [],
            "current_node": None,
            "current_affinity": None,
            "current_affinity_cpus": [],
            "cgroup_cpuset": None,
            "thread_count": 0,
            "thread_roles": {},
            "threads_by_role": {},
            "numa_maps": {"ok": False, "reason": "pre_run", "pages": {}, "total": 0},
            "remote_page_ratio": None,
            "deviations": [],
            "before": None,
        })
    return units


def _prepare_out_dir(args):
    """解析输出目录，dry-run 时不创建目录。"""
    if args.out:
        out_dir = os.path.abspath(os.path.expanduser(args.out))
    else:
        out_dir = workspace.session_dir(create=not args.dry_run)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _collect_units(form, npu_info, topo):
    """按执行形态发现运行中 unit 或构造预运行 unit。"""
    if form == "in_run":
        discovered = discover.collect_units(npu_info, topo)
        units = discovered.get("units") or []
    else:
        discovered = {"units": [], "unknown_hbm_npus": npu_query.unknown_hbm_npus(npu_info), "view_limited": False}
        units = _pre_run_units(topo)
    return discovered, units


def _allocate_zones(args, topo, units):
    """划区并把 CPU 池分配到各 unit。"""
    policy = {
        "smt": "exclusive",
        "irq_reserve_per_node": args.irq_reserve,
        "interference_reserve_per_node": args.interference_reserve,
    }
    zones_alloc = zone_alloc.allocate(topo, units, policy)

    for unit in units:
        cpus = zones_alloc.get("alloc", {}).get(unit.get("unit_id")) or []
        if cpus:
            unit_alloc.allocate(unit, cpus)
    return zones_alloc


def _check_feasibility(topo, units, discovered):
    """执行可行性检查并补充视图受限标记。"""
    checks = feasibility.check(topo, units)
    checks["flags"]["view_limited"] = bool(discovered.get("view_limited"))
    return checks


def _preflight_context(pre, topo, modeled, discovered):
    """汇合 preflight 采集与建模结论，供 plan 组装使用。"""
    preflight_info = dict(pre)
    preflight_info["unknown_hbm_npus"] = discovered.get("unknown_hbm_npus") or []
    preflight_info["locality_mode"] = modeled.get("locality_mode")
    preflight_info["fabric"] = (topo.get("meta") or {}).get("fabric")
    preflight_info["affinity_granularity"] = (topo.get("meta") or {}).get("affinity_granularity")
    preflight_info["die_groups"] = (topo.get("meta") or {}).get("die_groups") or []
    preflight_info["findings"] = modeled.get("findings") or []
    return preflight_info


def _write_outputs(args, plan, topo, paths):
    """按 dry-run / no-html 开关落盘 plan、验证脚本与报告。"""
    if not args.dry_run:
        emit_plan.write_plan(plan, paths["plan_path"])
        emit_plan.write_verify_script(plan, paths["verify_path"])
        if not args.no_html:
            report.write(plan, topo, paths["report_path"])


def _build_summary(args, form, plan, units, paths):
    """组装 stdout 摘要 JSON。"""
    summary = {
        "status": "ok",
        "form": form,
        "dry_run": args.dry_run,
        "executable": plan.get("executable"),
        "out_dir": paths["out_dir"],
        "plan": None if args.dry_run else paths["plan_path"],
        "report": None if (args.dry_run or args.no_html) else paths["report_path"],
        "verify_script": None if args.dry_run else paths["verify_path"],
        "unit_count": len(units),
        "deviation_count": sum(len(u.get("deviations") or []) for u in units),
        "expected_effect": (plan.get("expected_effect") or {}).get("level"),
        "execution_mode": (plan.get("privilege") or {}).get("execution_mode"),
        "requires_privilege": (plan.get("privilege") or {}).get("requires_privilege"),
        "unresolved": plan.get("unresolved") or [],
        "warnings": plan.get("warnings") or [],
        "units": [
            {
                "unit_id": u.get("unit_id"),
                "framework": u.get("framework"),
                "pid": u.get("pid"),
                "primary_npu": u.get("primary_npu"),
                "current_node": u.get("current_node"),
                "current_affinity": u.get("current_affinity"),
                "target_node": u.get("target_node"),
                "cpuset": u.get("cpuset"),
                "main_cpuset": u.get("main_cpuset"),
                "work_cpuset": u.get("work_cpuset"),
                "target_uid": u.get("target_uid"),
                "is_own_process": u.get("is_own_process"),
                "evidence_level": u.get("evidence_level"),
                "roles": {k: v.get("cpus") for k, v in (u.get("roles") or {}).items()},
                "capacity_gap": u.get("capacity_gap", 0),
                "deviations": [d.get("kind") for d in (u.get("deviations") or [])],
            }
            for u in units
        ],
    }
    if args.execute:
        # 显式请求时把 apply/rollback 命令带进摘要，供主会话确认后直接执行
        apply_cfg = plan.get("apply") or {}
        summary["execute"] = {
            "command": apply_cfg.get("command"),
            "rollback": apply_cfg.get("rollback"),
            "roles": apply_cfg.get("roles"),
            "units": [
                {"unit_id": u.get("unit_id"),
                 "commands": u.get("commands") or [],
                 "rollback": u.get("rollback") or []}
                for u in plan.get("units") or []
            ],
        }
    return summary


def run(args):
    # dry-run 语义 = 不产生任何文件系统副作用：不建目录、不写产物
    out_dir = _prepare_out_dir(args)

    pre = preflight.collect()
    npu_info = npu_query.query()
    form, blocked = preflight.detect_form(npu_info, pre)
    if blocked:
        blocked["preflight"] = pre
        return blocked, None, None

    topo = topology.collect(npu_info)
    max_node_cpus = max((len(n.get("cpus") or []) for n in topo.get("nodes") or []), default=0)
    reserve = args.irq_reserve + args.interference_reserve
    if max_node_cpus and reserve >= max_node_cpus:
        return {"status": "error",
                "error": "预留核数 %d（irq %d + interference %d）不小于单 node 核数 %d，请调小参数" % (
                    reserve, args.irq_reserve, args.interference_reserve, max_node_cpus)}, None, None
    discovered, units = _collect_units(form, npu_info, topo)

    modeled = model.build(topo, units)
    units = modeled.get("units") or []

    zones_alloc = _allocate_zones(args, topo, units)
    checks = _check_feasibility(topo, units, discovered)
    preflight_info = _preflight_context(pre, topo, modeled, discovered)

    plan, plan_path, verify_path, report_path = emit_plan.build_plan(
        topo, units, zones_alloc, checks, preflight_info, form, out_dir
    )
    paths = {
        "out_dir": out_dir,
        "plan_path": plan_path,
        "verify_path": verify_path,
        "report_path": report_path,
    }
    _write_outputs(args, plan, topo, paths)
    summary = _build_summary(args, form, plan, units, paths)
    return summary, plan, topo


def main(argv):
    args = _parse_args(argv)
    try:
        summary, plan, topo = run(args)
    except Exception as exc:  # noqa: BLE001 - 调用方是 Agent 子进程，错误必须可结构化解析
        print(json.dumps(
            {"status": "error", "error": "%s: %s" % (type(exc).__name__, exc)},
            ensure_ascii=False, indent=2,
        ))
        return 2
    if summary.get("status") in ("blocked", "error"):
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    if args.print_plan:
        print(json.dumps({"summary": summary, "plan": plan}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

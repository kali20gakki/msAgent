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

"""按 plan 执行线程级绑核（主会话确认后运行）。

Linux 只有线程级亲和：本脚本在执行时枚举 /proc/<pid>/task，按 comm 归类后
逐个设置亲和，避免 plan 生成时的 tid 快照过期，也避免漏绑 compute 线程。

默认绑主线程 + comm/io 线程，并按 plan 的 apply 语义绑定 compute（新计划为 exec，
与 OMP_NUM_THREADS 核数一致）；--bind-compute 仅用于强制绑定标 inherit 的旧计划。
目标核先与 cgroup cpuset.effective 取交集，越界部分不会生效。

安全边界（对应 RFC §3.1.5 / references/permissions.md）：
- 只接受参数列表式调用，不拼接 shell；
- 执行前校验 pid 存在 + 进程属主：目标非当前用户且当前非 root 时拒绝执行
  （delegated 场景应sudo / 以目标用户身份运行本脚本），防止误绑他人进程；
- --dry-run 不产生任何系统调用副作用，用于主会话展示动作清单。

用法：
  python3 apply_plan.py <plan.json> --roles comm,io [--bind-compute] [--dry-run]
  python3 apply_plan.py <plan.json> --rollback
"""

import argparse
import json
import os
import sys

import procfs
import roles
import topology


def _cpus(text):
    return set(topology.parse_cpulist(text))


def _clip(target, cgroup_text):
    target = set(target)
    cgroup = _cpus(cgroup_text) if cgroup_text else set()
    if not cgroup:
        return target, False
    clipped = target & cgroup
    return clipped, clipped != target


def _set_affinity(tid, cpus):
    os.sched_setaffinity(int(tid), sorted(int(c) for c in cpus))


def _binding(unit):
    return {b.get("role"): b for b in (unit.get("thread_binding") or [])}


def _owner_refusal(pid, dry_run):
    """属主校验：目标进程非当前用户且当前非 root 时拒绝实际执行。

    dry-run 放行（无副作用）；root 可操作任意进程（delegated 经 sudo 走此路径）。
    返回 None 表示允许，否则返回结构化拒绝原因。
    """
    if dry_run or not hasattr(os, "getuid"):
        return None
    current = os.getuid()
    if current == 0:
        return None
    target = procfs.read_uid(pid)
    if target is not None and target != current:
        return ("目标进程 uid=%s 与当前用户 uid=%d 不一致；"
                "请用 sudo 或以目标进程所属用户执行（execution_mode=delegated）" % (target, current))
    return None


def _plan_rollback(unit, dry_run):
    pid = unit.get("pid")
    before = _cpus((unit.get("before") or {}).get("Cpus_allowed_list") or "")
    if not before:
        return {"unit_id": unit.get("unit_id"), "pid": pid, "ok": False,
                "error": "before.Cpus_allowed_list 缺失，无法回滚"}
    tids = [t for t in procfs.list_tids(pid) if t != pid]
    if not dry_run:
        _set_affinity(pid, before)
        for tid in tids:
            _set_affinity(tid, before)
    return {
        "unit_id": unit.get("unit_id"), "pid": pid, "ok": True,
        "cpuset": topology.cpus_to_str(before), "threads": len(tids),
    }


def _plan_unit(unit, wanted, bind_compute, dry_run):
    pid = unit.get("pid")
    binding = _binding(unit)
    cgroup = procfs.read_cgroup_cpuset(pid)
    result = {
        "unit_id": unit.get("unit_id"), "pid": pid, "ok": True,
        "main": None, "roles": {}, "warnings": [],
    }

    main_binding = binding.get("main") or {}
    # 旧计划 compute=inherit：compute 线程继承主线程亲和，若主线程只绑 1 核，
    # OMP_NUM_THREADS 声明的 N 核实际会挤在 1 核上。此时主线程改绑 work_cpuset
    # （compute∪main），保证继承路径可用核集与 OMP 核数一致。
    inherit_compute = ((binding.get("compute") or {}).get("apply") == "inherit"
                       and not bind_compute)
    if inherit_compute and unit.get("work_cpuset"):
        main_cpus = unit["work_cpuset"]
    else:
        main_cpus = main_binding.get("cpuset") or unit.get("main_cpuset") or unit.get("cpuset")
    main_target, main_clipped = _clip(_cpus(main_cpus), cgroup)
    if not main_target:
        result["ok"] = False
        result["warnings"].append("主线程目标核与 cgroup cpuset 无交集")
    else:
        if not dry_run:
            _set_affinity(pid, main_target)
        result["main"] = {
            "cpuset": topology.cpus_to_str(main_target),
            "clipped": main_clipped,
        }
        if main_clipped:
            result["warnings"].append("主线程目标核超出 cgroup cpuset，已裁剪")

    # 执行时枚举线程，按 comm 归类；跳过主线程（已按 main 绑定）
    tasks = [t for t in procfs.list_tids(pid) if t != pid]
    for role, item in binding.items():
        if role == "main":
            continue
        if role == "compute":
            # compute 以 plan 的 apply 语义为准（默认 exec）；--bind-compute 可强制绑定 inherit 计划
            if item.get("apply") == "inherit" and not bind_compute:
                result["roles"]["compute"] = {
                    "apply": "inherit", "threads": 0,
                    "cpuset": item.get("cpuset"),
                }
                continue
        elif role not in wanted:
            continue
        if item.get("apply") == "skip" or not item.get("cpuset"):
            result["roles"][role] = {"apply": "skip", "threads": 0}
            continue

        target, clipped = _clip(_cpus(item.get("cpuset")), cgroup)
        matched = [tid for tid in tasks if roles.classify(procfs.read_comm(pid, tid)) == role]
        # dry-run 不落亲和，但仍报告待绑核线程数，便于执行前评估影响面
        would_bind = len(matched) if target else 0
        if not dry_run and target:
            applied = []
            failed_tid = None
            try:
                for tid in matched:
                    failed_tid = tid
                    before = _cpus((procfs.read_status(tid) or {}).get("Cpus_allowed_list") or "")
                    _set_affinity(tid, target)
                    applied.append((tid, before))
            except OSError as exc:
                rollback_failed = []
                for done_tid, before in reversed(applied):
                    if not before:
                        continue
                    try:
                        _set_affinity(done_tid, before)
                    except OSError:
                        rollback_failed.append(done_tid)
                raise OSError(
                    "%s；tid=%s 绑定失败，已回滚 %d 个线程%s" % (
                        exc, failed_tid, len(applied) - len(rollback_failed),
                        "，回滚失败 tid=%s" % rollback_failed if rollback_failed else "",
                    )
                ) from exc
        result["roles"][role] = {
            "apply": "exec",
            "threads": would_bind,
            "cpuset": topology.cpus_to_str(target),
            "clipped": clipped,
        }
        if clipped:
            result["warnings"].append("%s 目标核超出 cgroup cpuset，已裁剪" % role)
    return result


def main(argv):
    parser = argparse.ArgumentParser(description="按 plan 执行线程级绑核（需主会话确认）")
    parser.add_argument("plan", help="hostcpu_plan_*.json")
    parser.add_argument("--roles", default="comm,io", help="要执行的线程角色（默认 comm,io）")
    parser.add_argument("--bind-compute", action="store_true",
                        help="强制绑定 plan 标记 inherit 的 compute 计划")
    parser.add_argument("--rollback", action="store_true", help="按 before 恢复原始亲和")
    parser.add_argument("--dry-run", action="store_true", help="只打印动作，不修改亲和")
    args = parser.parse_args(argv)

    if not hasattr(os, "sched_setaffinity"):
        print(json.dumps({"ok": False, "error": "当前平台不支持 sched_setaffinity（需 Linux）"},
                         ensure_ascii=False, indent=2))
        return 2

    try:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except OSError as exc:
        print(json.dumps({"ok": False, "error": "计划文件不可读：%s" % exc},
                         ensure_ascii=False, indent=2))
        return 2
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": "计划文件 JSON 解析失败：%s" % exc},
                         ensure_ascii=False, indent=2))
        return 2
    if not isinstance(plan, dict) or not isinstance(plan.get("units") or [], list):
        print(json.dumps({"ok": False, "error": "计划文件格式非法：需为含 units 数组的 JSON 对象"},
                         ensure_ascii=False, indent=2))
        return 2

    wanted = {x.strip() for x in args.roles.split(",") if x.strip()}
    results = []
    ok = True
    for unit in plan.get("units") or []:
        if not isinstance(unit, dict):
            results.append({"unit_id": None, "pid": None, "ok": False,
                            "error": "plan 中 unit 非对象，已跳过"})
            ok = False
            continue
        missing = [k for k in ("unit_id", "cpuset", "thread_binding", "roles") if k not in unit]
        if missing:
            results.append({"unit_id": unit.get("unit_id"), "pid": unit.get("pid"), "ok": False,
                            "error": "unit 缺少必需字段：%s" % ", ".join(missing)})
            ok = False
            continue
        pid = unit.get("pid")
        if not pid:
            # 无 pid 的 unit 不能在线绑核：必须显式失败，避免整体 ok=true 的假成功
            results.append({"unit_id": unit.get("unit_id"), "pid": None, "ok": False,
                            "error": "unit 无 pid（pre_run 计划或进程信息缺失），无法在线绑核"})
            ok = False
            continue
        if not procfs.pid_exists(pid):
            results.append({"unit_id": unit.get("unit_id"), "pid": pid, "ok": False,
                            "error": "pid 不存在（可能已退出或 pid 复用）"})
            ok = False
            continue
        refusal = _owner_refusal(pid, args.dry_run)
        if refusal:
            results.append({"unit_id": unit.get("unit_id"), "pid": pid, "ok": False,
                            "error": refusal, "requires_privilege": True})
            ok = False
            continue
        try:
            if args.rollback:
                item = _plan_rollback(unit, args.dry_run)
            else:
                item = _plan_unit(unit, wanted, args.bind_compute, args.dry_run)
        except PermissionError:
            item = {"unit_id": unit.get("unit_id"), "pid": pid, "ok": False,
                    "error": "权限不足；需 sudo 或由目标进程所属用户执行（delegated）"}
        except OSError as exc:
            item = {"unit_id": unit.get("unit_id"), "pid": pid, "ok": False,
                    "error": str(exc)}
        if not item.get("ok"):
            ok = False
        results.append(item)

    print(json.dumps({"ok": ok, "dry_run": args.dry_run, "rollback": args.rollback,
                      "results": results}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

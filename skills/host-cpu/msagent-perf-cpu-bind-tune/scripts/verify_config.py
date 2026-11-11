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

"""配置生效校验 / 现状 gap 分析。

用法：
  python3 verify_config.py <plan.json> [--threads] [--bind-compute]

plan-only 时输出"现状 vs 目标"的 gap（预期有 FAIL）；
apply 后复跑，全部 PASS 才算落地。

线程校验在执行时枚举 /proc/<pid>/task 并按 comm 归类，不依赖 plan 生成时的 tid。
目标核先与 cgroup cpuset.effective 取交集，避免"假 FAIL / 假 PASS"。

本脚本只读（不修改亲和），无特权要求；未知参数经 argparse 直接报错，
避免静默忽略导致的"校验口径与预期不符"。
"""

import argparse
import json
import sys

import procfs
import roles
import topology


def _cpus(text):
    return set(topology.parse_cpulist(text))


def _clip(target, cgroup_text):
    cgroup = _cpus(cgroup_text) if cgroup_text else set()
    if not cgroup:
        return set(target), False
    clipped = set(target) & cgroup
    return clipped, clipped != set(target)


def _actual(pid):
    status = procfs.read_status(pid) or {}
    return _cpus(status.get("Cpus_allowed_list") or "")


def _parse_args(argv):
    p = argparse.ArgumentParser(description="bind-tune 配置生效校验（只读）")
    p.add_argument("plan", help="hostcpu_plan_*.json")
    p.add_argument("--threads", action="store_true", help="逐线程校验（按 comm 归类）")
    p.add_argument("--bind-compute", action="store_true",
                   help="强制校验 plan 标 inherit 的 compute 绑定")
    return p.parse_args(argv)


def main(argv):
    args = _parse_args(argv)
    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    results = []
    ok = True
    for unit in plan.get("units") or []:
        pid = unit.get("pid")
        if not pid:
            # 无 pid 无法校验，必须显式记失败，避免「零校验即 PASS」
            ok = False
            results.append({
                "unit_id": unit.get("unit_id"),
                "pid": None,
                "role": "unit",
                "expect": unit.get("cpuset") or "",
                "actual": "",
                "match": False,
                "clipped": False,
                "threads": [],
                "error": "unit 无 pid（未启动或进程信息缺失），无法校验",
            })
            continue
        binding = {b.get("role"): b for b in (unit.get("thread_binding") or [])}
        cgroup = procfs.read_cgroup_cpuset(pid)

        main_expect, main_clipped = _clip(
            _cpus((binding.get("main") or {}).get("cpuset")
                  or unit.get("main_cpuset") or unit.get("cpuset")),
            cgroup,
        )
        main_actual = _actual(pid)
        main_match = main_actual == main_expect
        if not main_match:
            ok = False
        results.append({
            "unit_id": unit.get("unit_id"),
            "pid": pid,
            "role": "main",
            "expect": topology.cpus_to_str(main_expect),
            "actual": topology.cpus_to_str(main_actual),
            "match": main_match,
            "clipped": main_clipped,
            "threads": [],
        })

        if not args.threads:
            continue
        tasks = [t for t in procfs.list_tids(pid) if t != pid]
        for role in ("comm", "io", "compute"):
            item = binding.get(role) or {}
            enabled = item.get("apply") != "inherit" or args.bind_compute
            if not enabled or item.get("apply") == "skip" or not item.get("cpuset"):
                continue
            expect, clipped = _clip(_cpus(item.get("cpuset")), cgroup)
            mismatches = []
            matched = 0
            total = 0
            for tid in tasks:
                if roles.classify(procfs.read_comm(pid, tid)) != role:
                    continue
                total += 1
                actual = _actual(tid)
                if actual == expect:
                    matched += 1
                elif len(mismatches) < 10:
                    mismatches.append({
                        "tid": tid,
                        "comm": procfs.read_comm(pid, tid),
                        "actual": topology.cpus_to_str(actual),
                    })
            role_match = total == 0 or not mismatches
            if not role_match:
                ok = False
            results[-1]["threads"].append({
                "role": role,
                "expect": topology.cpus_to_str(expect),
                "matched": matched,
                "total": total,
                "match": role_match,
                "clipped": clipped,
                "mismatches": mismatches,
            })

    if not results:
        # 空计划不能按「零失败」判为通过：没有任何单元被校验过
        print(json.dumps({"ok": False, "error": "计划中没有 unit，未执行任何校验", "results": []},
                         ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"ok": ok, "results": results}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

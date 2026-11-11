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
"""audit.py — config-audit 主入口：参数解析 + 五步骤编排。

五步骤：目标解析（发现链）→ 现状快照（单点）→ 分析判定 → 动作清单 → 归档。
只读、不执行任何修改动作（执行由主会话确认交互承载，见 RFC §3.1.5）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

# 同目录脚本（skill 自包含，不跨目录 import）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline as baseline_mod
import checker as checker_mod
import classify as classify_mod
import discover as discover_mod
import preflight as preflight_mod
import report as report_mod
import rules_locality as rules_mod
import static_collect as collect_mod
import topology as topology_mod
import workspace


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audit.py",
        description="msagent-perf-cpu-config-audit：主机配置与环境变量只读预检（8 域）",
    )
    target = parser.add_argument_group("审计目标")
    target.add_argument("--framework", default="",
                        help="框架名（vllm/sglang/verl/megatron/mindspeed/torchrun/msrun）")
    target.add_argument("--npu-cards", default="",
                        help="卡集（如 0-7），用于推导期望卡数")
    target.add_argument("--pid", type=int, default=0,
                        help="目标进程 pid（事中模式，启用现状侧检查项）")
    scope = parser.add_argument_group("检查范围")
    scope.add_argument("--quick", action="store_true", default=True,
                       help="快检（约 18 项，默认）")
    scope.add_argument("--full", action="store_true",
                       help="深检（全量 + 现状侧）")
    scope.add_argument("--focus", default="",
                        help="聚焦域：numa,bind,hugepage,freq,irq,cgroup,env,npu")
    scope.add_argument("--baseline", default="",
                        help="上一轮 hostcpu-baseline.json（对比输入）")
    scope.add_argument("--expected", default="",
                        help="用户期望值 JSON 文件（阈值覆盖）")
    out = parser.add_argument_group("输出")
    out.add_argument("--output", default="",
                     help="输出目录（默认 <cwd>/hostcpu_audit_<host>_<时间戳>/）")
    out.add_argument("--dump-thresholds", default="", metavar="PATH",
                     help="导出当前生效阈值模板（JSON，含说明）后退出；改数值后经 --expected 传回")
    out.add_argument("--single-file", action="store_true",
                     help="HTML 报告内联资源（便于转发归档）")
    out.add_argument("--stamp", default="",
                     help="产物文件名时间戳（默认取输出目录名，供编排层固定命名）")
    return parser.parse_args(argv)


def _output_dir(args: argparse.Namespace, host: str) -> str:
    if args.output:
        out = os.path.abspath(os.path.expanduser(args.output))
    else:
        out = workspace.session_dir(host=host)
    os.makedirs(out, exist_ok=True)
    return out


def _expected_cards(npu_cards: str) -> int:
    """从卡集参数推导期望卡数（'0-7' → 8）。"""
    total = 0
    for part in npu_cards.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                total += int(hi) - int(lo) + 1
        elif part.isdigit():
            total += 1
    return total


def _focus_filter(results: List[Dict[str, Any]], focus: str) -> List[Dict[str, Any]]:
    if not focus:
        return results
    wanted = {f.strip() for f in focus.split(",") if f.strip()}
    return [r for r in results if r["domain"] in wanted]


def _resolve_targets(args: argparse.Namespace):
    """步骤1：预检 + 发现链 + 目标分类，返回 (pre, targets, pid)。"""
    pre = preflight_mod.run()
    discovery = discover_mod.discover(framework=args.framework,
                                      npu_cards=args.npu_cards)
    targets = classify_mod.classify_targets(discovery["targets"])
    # pid 显式指定时以 pid 为准（事中模式）
    pid = args.pid or (targets[0]["pid"] if targets else 0)
    if pid and not _pid_exists(pid):
        print(f"warn: pid {pid} 不存在，降级为事前模式", file=sys.stderr)
        pid = 0
    print(f"[1/5] 目标解析：mode={discovery['mode']} targets={len(targets)} "
          f"pid={pid or '-'} 事中={'是' if pid else '否'}")
    return pre, targets, pid


def _take_snapshot(pre: Dict[str, Any], targets: List[Dict[str, Any]], pid: int):
    """步骤2：单点快照采集，返回 (snapshot, online_cpus, num_cpus)。"""
    snapshot = collect_mod.collect(pid=pid)
    snapshot["preflight"] = pre
    snapshot["targets"] = targets
    # 一次读取 online 核集，格式化与计数共用同一快照（避免重复 sysfs 读取与口径分叉）
    cpus = topology_mod.online_cpus()
    online_cpus = topology_mod.format_cpu_list(cpus) or "0"
    num_cpus = len(cpus)
    print(f"[2/5] 现状快照：{len(snapshot['facts'])} 条事实（单点快照）")
    return snapshot, online_cpus, num_cpus


def _export_thresholds(args: argparse.Namespace,
                       overrides: Dict[str, Any]) -> bool:
    """导出阈值模板；已导出返回 True（调用方据此提前退出）。"""
    if not args.dump_thresholds:
        return False
    merged = dict(checker_mod.DEFAULT_THRESHOLDS)
    merged.update(overrides)
    template = checker_mod.dump_thresholds_template(
        merged, args.dump_thresholds)
    print(f"阈值模板已导出：{template}")
    print("修改数值后通过 --expected <该文件> 传入即可覆盖判定阈值。")
    return True


def _build_judgement(args: argparse.Namespace, snapshot: Dict[str, Any],
                     quick: bool, online_cpus: str, num_cpus: int,
                     targets: List[Dict[str, Any]]):
    """步骤3：机型 profile → 阈值覆盖 → 判定 + R 规则 + 嫌疑项。

    导出阈值模板时返回 None（调用方提前退出）。
    """
    profile_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "references", "profiles")
    npu_names = _fact_lookup(snapshot["facts"], "npu_names")
    npu_names = [n for n in (npu_names or []) if not str(n).isdigit()]
    profile_path = _pick_profile(profile_dir, npu_names)
    if profile_path:
        print(f"      机型 profile：{os.path.basename(profile_path)}")
    overrides = checker_mod.load_threshold_overrides(
        profile_path, args.expected)
    if _export_thresholds(args, overrides):
        return None
    profile_extra = checker_mod.load_profile_extra(profile_path)
    expected_affinity = ""
    requested_cpus = 0
    if args.expected:
        expected_affinity = _read_expected_affinity(args.expected)
        # 申请的 CPU 核数由期望核集推导；npu 卡数是设备数，不能当 CPU 申请量
        requested_cpus = len(checker_mod._parse_cpu_set(expected_affinity))
    check = checker_mod.Checker(thresholds=overrides,
                                profile_extra=profile_extra)
    results = check.run(
        snapshot["facts"], quick=quick,
        online_cpus=online_cpus, num_cpus=num_cpus,
        expected_affinity=expected_affinity,
        requested_cpus=requested_cpus,
        expected_cards=_expected_cards(args.npu_cards),
    )
    results = _focus_filter(results, args.focus)
    rules = rules_mod.LocalityRules(results, snapshot["facts"], targets)
    findings = rules.run(online_cpus)
    suspects = check.extract_suspects()
    print(f"[3/5] 判定：{len(results)} 项（CRIT/WARN/FACT/OK），"
          f"R 规则命中 {len(findings)} 条")
    return results, findings, suspects


def _diff_baseline(args: argparse.Namespace, snapshot: Dict[str, Any]):
    """基线对比：返回 (current_baseline, baseline_diff)。"""
    current_baseline = baseline_mod.build_baseline(snapshot["facts"])
    baseline_diff: List[Dict[str, Any]] = []
    if args.baseline:
        previous = baseline_mod.load_baseline(args.baseline)
        if previous:
            baseline_diff = baseline_mod.diff_baseline(current_baseline, previous)
    return current_baseline, baseline_diff


def _save_artifacts(out_dir: str, snapshot: Dict[str, Any],
                    findings: List[Dict[str, Any]],
                    actions: List[Dict[str, Any]],
                    current_baseline: Dict[str, Any],
                    suspects: List[Dict[str, Any]]) -> str:
    """步骤5：落盘快照/发现/基线/嫌疑项，返回基线文件路径。"""
    _json_dump(os.path.join(out_dir, "static-snapshot.json"), snapshot)
    report_mod.save_findings(findings, actions,
                             os.path.join(out_dir, "findings.json"))
    baseline_path = baseline_mod.save_baseline(
        current_baseline, os.path.join(out_dir, "hostcpu-baseline.json"))
    report_mod.suspect_items_json(suspects,
                                  os.path.join(out_dir, "suspect_items.json"))
    return baseline_path


def _render_report(args: argparse.Namespace, host: str, out_dir: str,
                   results: List[Dict[str, Any]],
                   findings: List[Dict[str, Any]],
                   suspects: List[Dict[str, Any]], meta: Dict[str, Any],
                   baseline_diff: List[Dict[str, Any]],
                   actions: List[Dict[str, Any]]) -> str:
    """步骤5：模板存在时渲染 HTML，返回报告路径（无模板返回空串）。"""
    template = _template_path()
    if not os.path.isfile(template):
        return ""
    payload = report_mod.build_payload(results, findings, suspects,
                                       meta, baseline_diff,
                                       actions=actions)
    # 文件名带 host + stamp，避免固定名 report.html 被二次运行覆盖，
    # 也便于编排层（perf-triage）按阶段目录归档后仍能识别归属
    stamp = workspace.safe_name(args.stamp or os.path.basename(os.path.normpath(out_dir)))
    safe_host = workspace.safe_name(host)
    return report_mod.render_html(
        payload, template,
        os.path.join(out_dir,
                     "hostcpu_config_report_%s_%s.html" % (safe_host, stamp)),
        single_file=args.single_file)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
    quick = not args.full
    host = _hostname()

    # ---- 步骤1 目标解析（发现链）----
    pre, targets, pid = _resolve_targets(args)

    # ---- 步骤2 现状快照（单点，全部自包含）----
    snapshot, online_cpus, num_cpus = _take_snapshot(pre, targets, pid)

    # ---- 步骤3 分析判定 ----
    judgement = _build_judgement(args, snapshot, quick, online_cpus,
                                 num_cpus, targets)
    if judgement is None:
        return 0
    results, findings, suspects = judgement

    # ---- baseline diff ----
    current_baseline, baseline_diff = _diff_baseline(args, snapshot)

    # ---- 步骤4 动作清单（Finding 补全 + 待改排序）----
    actions = _build_actions(results)
    print(f"[4/5] 待改清单：{len(actions)} 项（命令 + 回滚 + 验证，主会话确认后执行）")

    # ---- 步骤5 归档 ----
    out_dir = _output_dir(args, host)
    meta = {
        "host": host, "mode": "in-process" if pid else "pre-start",
        "quick": quick, "pid": pid, "framework": args.framework,
        "view_limited": pre.get("view_limited", False),
        "in_container": pre.get("in_container", False),
    }
    baseline_path = _save_artifacts(out_dir, snapshot, findings, actions,
                                    current_baseline, suspects)
    html_path = _render_report(args, host, out_dir, results, findings,
                               suspects, meta, baseline_diff, actions)
    print(f"[5/5] 归档完成 → {out_dir}")
    if html_path:
        print(f"      报告：{html_path}")
    print(f"      基线：{baseline_path}")

    print()
    print(report_mod.terminal_summary(results, findings, meta))
    return 0


# ---- 工具函数 ----

def _build_actions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 CRIT/WARN 结果生成待改清单（命令 + 回滚 + 验证 + 通俗解释，不自动执行）。"""
    actions: List[Dict[str, Any]] = []
    templates = {
        "D1": {"type": "CPUFREQ_GOVERNOR",
               "explain": "CPU 频率策略是 powersave，CPU 没有跑满频率，Host 侧算力被压低。",
               "commands": ["cpupower frequency-set -g performance"],
               "rollback": ["cpupower frequency-set -g powersave"],
               "verify": "复检 D1：scaling_governor = performance",
               "risk": "low", "level": "L1"},
        "C2": {"type": "THP_MODE",
               "explain": "透明大页未启用（never），大内存访问的地址翻译开销大。",
               "commands": ["echo madvise > /sys/kernel/mm/transparent_hugepage/enabled"],
               "rollback": ["echo never > /sys/kernel/mm/transparent_hugepage/enabled"],
               "verify": "复检 C2：thp_enabled = madvise",
               "risk": "low", "level": "L1"},
        "A8": {"type": "SYSCTL",
               "explain": "numa_balancing=1 时内核自动迁移内存页，会抵消绑核并引发 TLB 抖动。",
               "commands": ["sysctl -w kernel.numa_balancing=0"],
               "rollback": ["sysctl -w kernel.numa_balancing=1"],
               "verify": "复检 A8：numa_balancing = 0",
               "risk": "low", "level": "L1"},
        "A9": {"type": "SYSCTL",
               "explain": "zone_reclaim_mode=1 时本组内存紧张会同步回收，造成停顿。",
               "commands": ["sysctl -w vm.zone_reclaim_mode=0"],
               "rollback": ["sysctl -w vm.zone_reclaim_mode=1"],
               "verify": "复检 A9：zone_reclaim_mode = 0",
               "risk": "medium", "level": "L1"},
    }
    for result in results:
        if result["status"] not in ("CRIT", "WARN"):
            continue
        template = templates.get(result["id"])
        if template:
            actions.append({"action_id": f"A-{result['id']}", **template,
                            "target": {"scope": "system_runtime"},
                            "expected_effect": result.get("evidence", "")})
    return actions


# aarch64 CPU implementer/part → 常见机型名（可按需扩展）
_ARM_PART_NAMES = {
    ("0x48", "0xd01"): "Kunpeng-920",
}


def _norm_id(text: Any) -> str:
    """硬件标识归一化：小写、去分隔符、去 ascend 前缀（'Ascend910B'→'910b'）。"""
    text = re.sub(r"[-_\s]+", "", str(text)).lower()
    if text.startswith("ascend"):
        text = text[6:]
    return text


def _hw_match(pattern: str, value: str) -> bool:
    """前缀语义匹配：pattern 为空=通配；否则 pattern（归一化后）须为 value 的前缀。

    即 profile 声明的家族（如 'Ascend910B'）涵盖具体型号（如 npu-smi 显示的 '910B2'）。
    """
    p, v = _norm_id(pattern), _norm_id(value)
    if not p:
        return True
    return bool(v) and v.startswith(p)


def _cpu_identity() -> str:
    """本机 CPU 标识：x86 model name / aarch64 part 映射 + DMI 产品名（白名单只读）。"""
    parts: List[str] = []
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(65536)
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if m:
            parts.append(m.group(1).strip())
        impl = re.search(r"^CPU implementer\s*:\s*(0x[0-9a-fA-F]+)$", text, re.M)
        part = re.search(r"^CPU part\s*:\s*(0x[0-9a-fA-F]+)$", text, re.M)
        if part:
            key = ((impl.group(1).lower() if impl else ""), part.group(1).lower())
            parts.append(_ARM_PART_NAMES.get(key, part.group(1)))
    except OSError:
        pass
    try:
        with open("/sys/class/dmi/id/product_name", "r",
                  encoding="utf-8", errors="replace") as fh:
            product = fh.read().strip()
        if product:
            parts.append(product)
    except OSError:
        pass
    return " ".join(parts)


def _pick_profile(profile_dir: str, npu_names: List[str] = None) -> str:
    """选取机型 profile：match.cpu/npu 与本机身份匹配（前缀语义），最长匹配优先。

    未命中返回空串（走默认阈值表）。cpu/npu 均 CPU+NPU 双条件命中才算；
    多个 profile 同时命中时取（cpu 长度 + npu 长度）最大者，保证精确机型优先。
    """
    if not os.path.isdir(profile_dir):
        return ""
    cpu_id = _cpu_identity()
    npu_values = [str(v).strip() for v in (npu_names or []) if str(v).strip()]
    best_path, best_score = "", -1
    for name in sorted(os.listdir(profile_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(profile_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        match = data.get("match") or {}
        if not isinstance(match, dict):
            continue
        if not _hw_match(match.get("cpu", ""), cpu_id):
            continue
        npu_pattern = str(match.get("npu", ""))
        if npu_values:
            if not any(_hw_match(npu_pattern, v) for v in npu_values):
                continue
        elif _norm_id(npu_pattern):
            continue  # 本机无 NPU 型号信息时，带 NPU 条件的 profile 不命中
        score = len(_norm_id(match.get("cpu", ""))) + len(_norm_id(npu_pattern))
        if score > best_score:
            best_path, best_score = path, score
    return best_path


def _read_expected_affinity(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return str(data.get("expected_affinity", "")) if isinstance(data, dict) else ""
    except (OSError, ValueError):
        return ""


def _json_dump(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _fact_lookup(facts: List[Dict[str, Any]], item: str) -> Any:
    """从 facts 中取指定项的值（不存在返回 None）。"""
    for fact in facts:
        if fact.get("item") == item:
            return fact.get("value")
    return None


def _pid_exists(pid: int) -> bool:
    return pid > 0 and os.path.isdir(f"/proc/{pid}")


def _template_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "resources", "report_template.html")


def _hostname() -> str:
    import socket
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())

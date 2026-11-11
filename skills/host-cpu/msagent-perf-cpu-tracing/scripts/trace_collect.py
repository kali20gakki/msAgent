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

"""collect 入口：tracefs 直采（唯一引擎，不使用 trace-cmd）。

流程：preflight -> 预算 -> 逐卡采集（instance 隔离）-> 原始文本存档
      -> 自动 parse/analyze/report（--no-post 可跳过）。

全卡必须显式 --cards all（或给出卡列表），逐卡串行，每卡独立 instance 与 DB 段。
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import budget
import preflight
import workspace
from convert import engine_detect, tracefs_ctl

BUFFER_KB_MIN = 256
BUFFER_KB_MAX = 8192
DEVICE_PATTERN = re.compile(r"^/dev/davinci(\d+)$")


def parse_list(text):
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def resolve_events(raw):
    if not raw:
        return "min", budget.events_for_tier("min")
    raw = raw.strip()
    if raw in budget.TIERS:
        return raw, budget.events_for_tier(raw)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if len(names) > budget.MAX_EVENT_TYPES:
        raise SystemExit("事件类型超过上限 %d" % budget.MAX_EVENT_TYPES)
    return "custom", names


def plan_cpus(plan_path, card):
    if not plan_path or not os.path.isfile(plan_path):
        return []
    with open(plan_path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    cpus = []
    for unit in plan.get("units") or []:
        if unit.get("primary_npu") is not None and card is not None and int(unit["primary_npu"]) != card:
            continue
        for role in ("compute", "comm", "io", "main"):
            cpus.extend(parse_list((unit.get("roles") or {}).get(role, {}).get("cpus", "")))
    return sorted(set(cpus))


def _parse_args(argv):
    """构造并解析采集命令行参数。"""
    parser = argparse.ArgumentParser(description="tracefs ftrace 采集（单卡/逐卡）")
    parser.add_argument("--card", type=int, default=None)
    parser.add_argument("--cards", default="", help="all（按 plan/设备枚举展开）或卡列表，如 0 或 0-3")
    parser.add_argument("--cpus", default="", help="目标核集（缺省取 plan 中该卡 unit 核集）")
    parser.add_argument("--plan", help="bind-tune plan.json")
    parser.add_argument("--duration", type=float, default=0, help="单段时长（秒，0=默认 30s）")
    parser.add_argument("--events", default="min", help="min|std|ext 或事件列表")
    parser.add_argument("--target-comm", default="", help="comm 过滤（识别抢占方时不要加）")
    parser.add_argument("--buffer-kb", type=int, default=2048,
                        help="每核 ring buffer（%d~%d KB）" % (BUFFER_KB_MIN, BUFFER_KB_MAX))
    parser.add_argument("--rate", type=float, default=budget.REF_RATE_PER_CORE)
    parser.add_argument("--verify", action="store_true", help="验证模式：预算不做减核收窄")
    parser.add_argument("--out", help="输出目录")
    parser.add_argument("--host", default=None)
    parser.add_argument("--tracefs", help="tracefs 根（默认自动探测）")
    parser.add_argument("--no-post", action="store_true", help="只采集，不做转换/分析")
    return parser.parse_args(argv)


def _all_cards(plan_path):
    """--cards all 展开：优先 plan 的 primary_npu 集合，其次 /dev/davinci* 枚举，最后退回 0-7。"""
    if plan_path and os.path.isfile(plan_path):
        with open(plan_path, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
        npus = sorted({int(unit["primary_npu"]) for unit in plan.get("units") or []
                       if unit.get("primary_npu") is not None})
        if npus:
            return npus
    npus = sorted(int(match.group(1)) for match in
                  (DEVICE_PATTERN.match(path) for path in glob.glob("/dev/davinci*")) if match)
    if npus:
        return npus
    print("提示：未从 plan 或 /dev/davinci* 推导出卡列表，按 0-7 尝试；请用 --cards 显式指定",
          file=sys.stderr)
    return list(range(8))


def _resolve_cards(args):
    """按 --cards/--card 确定待采集卡列表（空列表由调用方拒绝）。"""
    if args.cards.strip():
        cards = _all_cards(args.plan) if args.cards.strip() == "all" else parse_list(args.cards)
    elif args.card is not None:
        cards = [args.card]
    else:
        cards = [0]
    return cards


def _plan_targets(args, cards, tier, events):
    """逐卡规划核集与预算，返回 (per_card, 退出码)。

    budget.plan 的收窄动作（事件降档 / 时长减半 / 减核）必须回灌到实际采集参数，
    否则「预算说收窄、采集仍按原参数跑」会突破模块自身的硬约束。
    自定义事件列表（tier=custom）不受降档影响，按用户给定的事件集采集。
    """
    per_card = []
    for card in cards:
        cpus = parse_list(args.cpus) or plan_cpus(args.plan, card)
        if not cpus:
            print(json.dumps({"status": "rejected", "message": "缺少目标核集：给 --cpus 或 --plan",
                              "card": card}, ensure_ascii=False, indent=2))
            return None, 3
        duration = args.duration or 30
        ok, estimate, adjustments = budget.plan(
            cpus=len(cpus), tier=tier, duration_s=duration,
            cards=len(cards), rate_per_core=args.rate, verify=args.verify)
        applied_tier = estimate.get("tier") or tier
        applied_cpus = cpus
        if estimate.get("cpus") and int(estimate["cpus"]) < len(cpus):
            # 预算减核后按排序前缀截断，保持核集连续可用
            applied_cpus = sorted(cpus)[: int(estimate["cpus"])]
        per_card.append({
            "card": card,
            "cpus": applied_cpus,
            "tier": applied_tier,
            "events": list(events) if tier not in budget.TIERS else budget.events_for_tier(applied_tier),
            "duration": float(estimate.get("duration_s") or duration),
            "budget": estimate,
            "ok": ok,
            "adjustments": adjustments,
        })
        if not ok:
            print(json.dumps({"status": "rejected", "message": "预算超限，已拒绝采集",
                              "cards": per_card}, ensure_ascii=False, indent=2))
            return None, 3
    return per_card, 0


def _collect_cards(info, out_dir, stamp, per_card, args):
    """逐卡串行采集原始 trace，并按需触发后处理。"""
    comms = [item.strip() for item in args.target_comm.split(",") if item.strip()] or None

    results = []
    for item in per_card:
        card = item["card"]
        raw_path = os.path.join(out_dir, "trace_card%d_%s.txt" % (card, stamp))
        started = time.time()
        try:
            stats = tracefs_ctl.capture_window(
                info["path"], item["cpus"], item["events"], item["duration"], raw_path,
                comms=comms, buffer_kb=args.buffer_kb)
        except tracefs_ctl.TracefsError as exc:
            results.append({
                "card": card, "cpus": item["cpus"], "raw": raw_path,
                "tier": item["tier"], "events": item["events"], "duration_s": item["duration"],
                "status": "capture_failed", "error": str(exc),
                "elapsed_s": round(time.time() - started, 2),
                "budget": item["budget"], "adjustments": item["adjustments"],
            })
            continue
        entry = {
            "card": card,
            "cpus": item["cpus"],
            "tier": item["tier"],
            "events": item["events"],
            "duration_s": item["duration"],
            "raw": raw_path,
            "bytes": stats["bytes"],
            "overrun": stats["overrun"],
            "filters": stats.get("filters"),
            "elapsed_s": round(time.time() - started, 2),
            "budget": item["budget"],
            "adjustments": item["adjustments"],
            "status": "ok" if stats["bytes"] else "empty",
        }
        results.append(entry)
        if not args.no_post and entry["status"] == "ok":
            entry["post"] = _post(out_dir, raw_path, item, args, card)
            if entry["post"].get("status") != "ok":
                entry["status"] = "post_failed"
    return results


def main(argv=None):
    args = _parse_args(argv)

    if not BUFFER_KB_MIN <= args.buffer_kb <= BUFFER_KB_MAX:
        print(json.dumps({"status": "rejected",
                          "message": "--buffer-kb 需在 %d~%d 之间" % (BUFFER_KB_MIN, BUFFER_KB_MAX),
                          "buffer_kb": args.buffer_kb}, ensure_ascii=False, indent=2))
        return 3

    info = preflight.run(args.tracefs)
    engine = engine_detect.detect(info)
    if not engine["available"]:
        print(json.dumps({"status": "unavailable", "engine": engine, "message": "tracefs 不可写，judged 不可采集",
                          "hint": info.get("hint")}, ensure_ascii=False, indent=2))
        return 2

    cards = _resolve_cards(args)
    if not cards:
        print(json.dumps({"status": "rejected",
                          "message": "--cards 解析为空：请给出卡号（如 0 或 0-3）或 all"},
                         ensure_ascii=False, indent=2))
        return 3

    tier, events = resolve_events(args.events)
    per_card, code = _plan_targets(args, cards, tier, events)
    if per_card is None:
        return code

    out_dir = args.out or workspace.session_dir(host=args.host)
    os.makedirs(out_dir, exist_ok=True)
    stamp = os.path.basename(os.path.normpath(out_dir))

    results = _collect_cards(info, out_dir, stamp, per_card, args)

    failed = [item for item in results if item.get("status") != "ok"]
    applied_events = {tuple(item["events"]) for item in per_card}
    applied_durations = {item["duration"] for item in per_card}
    manifest = {
        "status": "partial" if failed else "collected",
        "engine": engine["engine"],
        "host": args.host or workspace.host_name(),
        "tier": tier,
        "events": per_card[0]["events"] if len(applied_events) == 1 else "mixed",
        "verify": bool(args.verify),
        "buffer_kb": args.buffer_kb,
        "duration_s": per_card[0]["duration"] if len(applied_durations) == 1 else "mixed",
        "cards": results,
    }
    with open(os.path.join(out_dir, "collect_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 4 if failed else 0


def _post(out_dir, raw_path, item, args, card):
    commands = [
        [sys.executable, os.path.join(HERE, "trace_parse.py"),
         "--input", raw_path, "--out", out_dir, "--card", str(card),
         "--cpus", _cpulist(item["cpus"]), "--expect-cpus", _cpulist(item["cpus"]),
         "--tier", item["tier"], "--host", args.host or workspace.host_name(),
         "--target-comm", args.target_comm or ""],
    ]
    if args.plan:
        commands[0].extend(["--plan", args.plan])
    try:
        parsed = subprocess.run(commands[0], capture_output=True, text=True, encoding="utf-8")
        if parsed.returncode != 0:
            return {"status": "parse_failed", "stderr": parsed.stderr[-800:]}
        db = (json.loads(parsed.stdout) or {}).get("db")
        if not db:
            return {"status": "parse_failed", "stderr": parsed.stderr[-800:],
                    "error": "trace_parse 输出缺少 db 字段"}
        analyzed = subprocess.run(
            [sys.executable, os.path.join(HERE, "analyze_trace.py"), "--db", db, "--out", out_dir,
             "--card", str(card), "--target-comm", args.target_comm or ""]
            + (["--plan", args.plan] if args.plan else []),
            capture_output=True, text=True, encoding="utf-8")
        if analyzed.returncode != 0:
            return {"status": "analyze_failed", "stderr": analyzed.stderr[-800:], "db": db}
        payload = os.path.join(out_dir, "report_payload.json")
        reported = subprocess.run(
            [sys.executable, os.path.join(HERE, "report.py"), "--payload", payload, "--out", out_dir],
            capture_output=True, text=True, encoding="utf-8")
        return {"status": "ok" if reported.returncode == 0 else "report_failed",
                "db": db, "report": reported.stdout.strip()[-800:]}
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}


def _cpulist(cpus):
    cpus = sorted(set(cpus))
    if not cpus:
        return ""
    spans, start, prev = [], cpus[0], cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        spans.append((start, prev))
        start = prev = cpu
    spans.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in spans)


if __name__ == "__main__":
    raise SystemExit(main())

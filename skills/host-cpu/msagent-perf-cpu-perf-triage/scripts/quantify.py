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
"""quantify.py — 归因量化（两级口径确定性实现，不含 LLM）。

定性级：配置判型（沿用 config-audit 结论，不硬折算数值）
定量级：互斥时间分解——打扰% + 等待% + 效率（第一期 not_observable）
残余  ：100% - 已归因；> 30% 标「未收敛」

数值一律取上游产物（config_triage / phase_verdict_1 中的 accounting），
本脚本只做占比换算与截断，不重算事件域事实。
"""
from __future__ import annotations

import argparse
import json
import math
import os

import workspace

RESIDUAL_CONVERGED_MAX = workspace.RESIDUAL_CONVERGED_MAX
WAIT_CLASS_NAMES = {
    "io": "IO 等待", "lock": "锁/同步", "net": "网络",
    "upstream": "上游数据", "timer": "定时器", "block": "长阻塞",
}


def _ratio(value):
    """上游占比安全转换：缺失按 0，非数值/非有限值返回 None，结果截断到 [0, 1]。"""
    if value is None:
        return 0.0
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ratio):
        return None
    return min(max(ratio, 0.0), 1.0)


def _int_value(value):
    """上游整数安全转换：非法值按 0，避免窗口字段类型异常中断量化。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="归因量化（两级口径）")
    parser.add_argument("--session", required=True)
    parser.add_argument("--config-triage", default="")
    parser.add_argument("--trace-verdict", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    config_path = args.config_triage or os.path.join(args.session, "00-config", "config_triage.json")
    trace_path = args.trace_verdict or os.path.join(args.session, "01-trace", "phase_verdict_1.json")
    out_path = args.out or os.path.join(args.session, "attribution.json")

    config = workspace.load_json(config_path, default={}) or {}
    trace = workspace.load_json(trace_path, default={}) or {}
    result = build_attribution(config, trace)
    workspace.write_json(out_path, result)
    workspace.manifest_upsert(args.session, "attribution", {
        "converged": result["converged"],
        "disturb_pct": result["disturb_pct"],
        "wait_pct": result["wait_pct"],
        "residual_pct": result["residual_pct"],
        "artifacts": [os.path.basename(out_path)],
    })
    print(json.dumps({k: result[k] for k in
                      ("config_typing", "disturb_pct", "wait_pct", "wait_breakdown",
                       "efficiency", "residual_pct", "converged", "basis")},
                     ensure_ascii=False, indent=2))
    return 0


def build_attribution(config, trace):
    degradation = list(config.get("degradation") or []) + list(trace.get("degradation") or [])
    config_typing = {
        "level": config.get("impact_level") or "无",
        "candidate": bool(config.get("config_primary_candidate")),
        "status": config.get("status") or "UNAVAILABLE",
        "findings_count": config.get("findings_count", 0),
        "action_count": config.get("action_count", 0),
        "top": config.get("top_findings") or [],
        "note": "定性级：判型与影响等级，不硬折算数值占比",
    }
    observed = [c for c in (trace.get("cards") or []) if c.get("status") == "OBSERVED"]
    if not observed:
        degradation.append("阶段1 未取得可观测数据：打扰/等待两维均不可量化")
        return _assemble(config_typing, None, None, {}, degradation, basis="none")

    card, accounting, wait = _worst_card(observed)
    if not accounting:
        degradation.append("运行时产物未产出 accounting（版本不匹配），打扰/等待按不可观测处理")
        return _assemble(config_typing, None, None, {}, degradation, basis="none")

    window_ns = _int_value(accounting.get("window_ns"))
    disturb_ratio = _ratio(accounting.get("disturb_ratio"))
    wait_ratio = _ratio(accounting.get("wait_ratio"))
    invalid = [name for name, value in (("disturb_ratio", disturb_ratio),
                                        ("wait_ratio", wait_ratio)) if value is None]
    if invalid:
        degradation.append("accounting 数值非法（%s），该维按 0 处理" % "、".join(invalid))
    disturb_pct = round((disturb_ratio or 0.0) * 100, 1)
    wait_pct = round((wait_ratio or 0.0) * 100, 1)
    breakdown = _wait_breakdown(wait, window_ns)
    if not wait or not wait.get("interval_count"):
        degradation.append("等待账为空：线程持续 runnable（可能为 busy-wait 自旋，归效率维）或事件缺失")
    basis = "card%s" % card.get("card") if card.get("card") is not None else "card-"
    if len(observed) > 1:
        degradation.append("多卡结论取最差卡 %s，其余卡明细见 phase_verdict_1.json" % basis)
    return _assemble(config_typing, disturb_pct, wait_pct, breakdown, degradation, basis=basis,
                     window_ns=window_ns, card=card)


def _worst_card(observed):
    """取打扰+等待占比最高的卡作为主结论（暴露最严重卡）。"""
    def score(card):
        acc = card.get("accounting") or (card.get("summary") or {}).get("accounting") or {}
        return (_ratio(acc.get("disturb_ratio")) or 0.0) + (_ratio(acc.get("wait_ratio")) or 0.0)

    card = max(observed, key=score)
    accounting = card.get("accounting") or (card.get("summary") or {}).get("accounting") or {}
    wait = (card.get("summary") or {}).get("wait") or {}
    return card, accounting, wait


def _wait_breakdown(wait, window_ns):
    out = {}
    # by_class 是占合计睡眠时长的相对份额，与观测窗时长无关；window_ns 仅保留
    # 在签名中以兼容调用方，不参与分类明细的取舍（缺失时不得静默丢数据）。
    if not wait:
        return out
    for key, item in (wait.get("by_class") or {}).items():
        # by_class 为占「合计睡眠时长」的相对份额（并行睡眠会叠加，不是占窗比）
        share = item.get("share")
        if share is None:
            share = item.get("ratio") or 0
        out[WAIT_CLASS_NAMES.get(key, key)] = round(float(share) * 100, 1)
    return out


def _assemble(config_typing, disturb_pct, wait_pct, breakdown, degradation,
              basis="none", window_ns=0, card=None):
    known = sum(x for x in (disturb_pct, wait_pct) if x is not None)
    if disturb_pct is None and wait_pct is None:
        residual = None
        converged = False
    else:
        residual = round(max(0.0, 100.0 - known), 1)
        converged = residual <= RESIDUAL_CONVERGED_MAX
    return {
        "config_typing": config_typing,
        "disturb_pct": disturb_pct,
        "wait_pct": wait_pct,
        "wait_breakdown": breakdown,
        "efficiency": "not_observable",
        "residual_pct": residual,
        "converged": converged,
        "basis": basis,
        "window_ns": window_ns,
        "verdict": (card or {}).get("verdict"),
        "degradation": degradation,
        "note": ("两期口径：定性级（配置判型）+ 定量级（打扰/等待/效率三段互斥）。"
                 "效率维第一期 not_observable（perf-collect 缺位），其时间计入残余；"
                 "残余 > 30% 标未收敛，不强行归类。"),
    }


if __name__ == "__main__":
    raise SystemExit(main())

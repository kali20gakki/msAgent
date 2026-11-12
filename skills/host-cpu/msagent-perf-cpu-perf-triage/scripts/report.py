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
"""report.py — 会话汇总页 index.html + manifest.json 索引 + 终端摘要。

设计约束：
  汇总页只做导航与结论呈现，**不合并各阶段报告内容**（阶段报告均为 single-file
  self-contained，用相对路径互链即可离线打开）；
  manifest.json 是唯一机读索引，上层只读它、不解析 HTML。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import socket
import time

import workspace

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "resources", "triage-report.template.html")


def main(argv=None):
    parser = argparse.ArgumentParser(description="归因汇总页与 manifest 索引")
    parser.add_argument("--session", required=True)
    parser.add_argument("--attribution", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    attribution = workspace.load_json(
        args.attribution or os.path.join(args.session, "attribution.json"), default={}) or {}
    action_plan = workspace.load_json(
        args.actions or os.path.join(args.session, "actions", "action_plan.json"),
        default={}) or {}
    config = workspace.load_json(
        os.path.join(args.session, "00-config", "config_triage.json"), default={}) or {}
    trace = workspace.load_json(
        os.path.join(args.session, "01-trace", "phase_verdict_1.json"), default={}) or {}

    payload = build_payload(args.session, config, trace, attribution, action_plan)
    out_path = args.out or os.path.join(args.session, "index.html")
    render(payload, out_path)
    manifest = _update_manifest(args.session, payload, out_path)
    print(json.dumps({"index": out_path, "manifest": manifest,
                      "conclusion": payload["conclusion"],
                      "actions": len(payload["actions"])},
                     ensure_ascii=False, indent=2))
    return 0


def build_payload(session, config, trace, attribution, action_plan):
    cards = [c for c in (trace.get("cards") or []) if c.get("status") == "OBSERVED"]
    return {
        "meta": {
            "run_id": os.path.basename(os.path.normpath(session)),
            "host": _resolve_host(session, config, trace),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
        "conclusion": _conclusion(config, trace, attribution),
        "attribution": attribution,
        "cards": cards,
        "actions": action_plan.get("actions") or [],
        "degradation": attribution.get("degradation") or [],
        "phases": [
            {"name": "00 · 配置复核", "status": config.get("status") or "未执行",
             "detail": _config_detail(config), "reports": _reports(session, "00-config/*.html")},
            {"name": "01 · 系统打扰", "status": trace.get("status") or "未执行",
             "detail": _trace_detail(cards),
             "reports": _reports(session, "01-trace/*/*.html") + _reports(session, "01-trace/*.html")},
            {"name": "02 · 代码效率", "status": "not_observable",
             "detail": "第一期缺位（perf-collect 未交付），该维时间计入残余",
             "reports": []},
        ],
    }


RESIDUAL_CONVERGED_MAX = workspace.RESIDUAL_CONVERGED_MAX


def _conclusion(config, trace, attribution):
    if config.get("config_primary_candidate"):
        return "配置主因候选（判型 %s）：先试证配置类动作，成本最低且可回滚" % (
            config.get("impact_level") or "未知")
    if attribution.get("converged"):
        return "已收敛：打扰 + 等待可解释主要异常时间（残余 %.1f%%）" % (
            attribution.get("residual_pct") or 0)
    if attribution.get("disturb_pct") is None and attribution.get("wait_pct") is None:
        if not config.get("findings_count") and not config.get("action_count"):
            return ("无法归因：配置判型与运行时数据均缺失，需补齐依赖"
                    "（config-audit）后重跑")
        return "未收敛：无可观测的运行时数据，仅完成配置判型"
    return "未收敛：残余 %.1f%% > %.0f%%，效率维缺位（第二期 perf-collect 补齐）" % (
        attribution.get("residual_pct") or 0, RESIDUAL_CONVERGED_MAX)


def _resolve_host(session, config, trace):
    """主机名来源优先级：运行时 summary.meta.host → config_triage.host → run_id 前缀 → 本机名。

    run_id 形如 `<host>_<yyyymmdd-hhmmss>`，故用 rpartition 取前缀；
    不能用 session 的父目录名（那是 workspaces/<host>/<ts> 里的 <ts> 上级，会误取到宿主目录名）。
    本机名仅作最后兜底：远程/迁移会话中它并不代表采集主机。
    """
    for card in trace.get("cards") or []:
        host = ((card.get("summary") or {}).get("meta") or {}).get("host")
        if host:
            return host
    if config.get("host"):
        return config["host"]
    run_id = os.path.basename(os.path.normpath(session))
    prefix = run_id.rpartition("_")[0]
    if prefix:
        return prefix
    try:
        name = socket.gethostname()
        if name:
            return name
    except OSError:
        pass
    return "host"


def _config_detail(config):
    if config.get("status") in ("REUSED", "RERUN"):
        detail = "%s · Finding %d 条 / Action %d 条 · 影响面 %s" % (
            config.get("status"), config.get("findings_count") or 0,
            config.get("action_count") or 0, config.get("impact_level") or "无")
        source = config.get("source") or ""
        if config.get("status") == "REUSED" and source:
            # 复用时不生成本会话的配置报告，注明来源会话避免「无报告」困惑
            origin = os.path.basename(os.path.dirname(os.path.dirname(source)))
            if origin:
                detail += " · 复用自 %s" % origin
        return detail
    return "配置维缺失：%s" % (config.get("status") or "未执行")


def _trace_detail(cards):
    if not cards:
        return "无可观测卡（tracefs 不可用或预算超限）"
    parts = []
    for card in cards:
        parts.append("卡%s：%s · 覆盖率 %s · 打扰 %.1f%% · 等待 %.1f%%" % (
            card.get("card"), card.get("verdict") or "UNOBSERVED",
            "%.0f%%" % (_safe_float(card.get("coverage")) * 100),
            _pct(card, "disturb_ratio"), _pct(card, "wait_ratio")))
    return "；".join(parts)


def _safe_float(value, default=0.0):
    """上游数值安全转换：非数值/非有限值退化为默认值，避免整份报告生成失败。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _pct(card, key):
    acc = card.get("accounting") or (card.get("summary") or {}).get("accounting") or {}
    return _safe_float(acc.get(key)) * 100


def _reports(session, pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(session, pattern))):
        rel = os.path.relpath(path, session).replace(os.sep, "/")
        out.append({"name": os.path.basename(path), "href": rel})
    return out


def render(payload, out_path):
    with open(_TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    safe = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(template.replace("__PAYLOAD__", safe))
    return out_path


def _update_manifest(session, payload, out_path=None):
    path = os.path.join(session, "manifest.json")
    manifest = workspace.load_json(path, default={}) or {}
    manifest["run_id"] = payload["meta"]["run_id"]
    manifest["host"] = payload["meta"]["host"]
    manifest["generated_at"] = payload["meta"]["generated_at"]
    manifest["conclusion"] = payload["conclusion"]
    manifest["attribution"] = {
        k: payload["attribution"].get(k) for k in
        ("disturb_pct", "wait_pct", "wait_breakdown", "efficiency",
         "residual_pct", "converged", "basis")
    }
    index_ref = "index.html"
    if out_path:
        try:
            rel = os.path.relpath(out_path, session).replace(os.sep, "/")
        except ValueError:  # Windows 跨盘符无法求相对路径
            rel = ""
        index_ref = os.path.basename(out_path) if not rel or rel.startswith("../") else rel
    manifest["reports"] = {"index": index_ref}
    stages = []
    for phase in payload["phases"]:
        for report in phase["reports"]:
            if report["href"] not in stages:
                stages.append(report["href"])
    if stages:
        manifest["reports"]["stage"] = stages
    manifest["actions"] = len(payload["actions"])
    workspace.write_json(path, manifest)
    return path


if __name__ == "__main__":
    raise SystemExit(main())

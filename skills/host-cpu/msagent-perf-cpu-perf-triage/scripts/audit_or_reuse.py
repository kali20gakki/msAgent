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
"""audit_or_reuse.py — 阶段0：配置复核（复用优先，缺失/过期才重跑）。

职责边界：
  只做「时效校验 + 调用 + 判型汇总」，不重实现 config-audit 的阈值判定逻辑；
  Finding / Action 原样携带，供 action_build.py 复用。

复用规则（默认 30min）：
  候选 findings.json 存在且 mtime 距今 <= --max-age-min → REUSED；
  否则子进程调 config-audit 的 audit.py（快检模式）→ RERUN；
  显式 --skip-audit 时不执行审计 → SKIPPED（判型缺失，进降级）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import workspace

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.dirname(os.path.dirname(HERE))
HOST_CPU_DIR = "host-cpu"
CONFIG_AUDIT_DIR = "msagent-perf-cpu-config-audit"

_RISK_ORDER = {"high": 3, "medium": 2, "low": 1, "": 0}
_IMPACT_BY_RISK = {"high": "高", "medium": "中", "low": "低", "": "无"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="阶段0：配置复核（复用或重跑 config-audit）")
    parser.add_argument("--session", required=True, help="会话根目录")
    parser.add_argument("--findings", default="", help="候选 findings.json（缺省取会话内 00-config/）")
    parser.add_argument("--max-age-min", type=float, default=30.0, help="findings 时效上限（分钟）")
    parser.add_argument("--config-audit-dir", default="", help="config-audit skill 目录（缺省自动探测）")
    parser.add_argument("--skip-audit", action="store_true", help="不执行审计，只用现有 findings")
    args = parser.parse_args(argv)

    phase = workspace.phase_dir(args.session, "00-config")
    candidate = args.findings or os.path.join(phase, "findings.json")
    result = {"phase": "00-config", "candidate": candidate,
              "max_age_min": args.max_age_min, "degradation": []}

    source, status = "", ""
    if os.path.isfile(candidate):
        age_min = (time.time() - os.path.getmtime(candidate)) / 60.0
        result["age_min"] = round(age_min, 1)
        if age_min <= args.max_age_min:
            source, status = candidate, "REUSED"
        else:
            result["degradation"].append(
                "findings 已过期（%.0f min > %.0f min），需重跑" % (age_min, args.max_age_min))
    else:
        result["degradation"].append("会话内无 findings.json，需执行审计")

    if status != "REUSED":
        if args.skip_audit:
            status = "SKIPPED"
            if os.path.isfile(candidate):
                # 保留过期 findings 作为旁证，避免用户以为用了旧结论却拿到空判型
                source = candidate
                result["degradation"].append("显式跳过审计：沿用过期 findings 仅作旁证")
            else:
                result["degradation"].append("显式跳过审计：配置判型缺失，仅作旁证")
        else:
            audit_dir = _resolve_audit_dir(args.config_audit_dir)
            if not audit_dir:
                status = "UNAVAILABLE"
                result["degradation"].append(
                    "未找到 config-audit skill（可用 --config-audit-dir 指定），配置维缺失")
            else:
                code, out, err = _run_audit(audit_dir, phase, args.session)
                if code != 0:
                    status = "FAILED"
                    result["degradation"].append(
                        "config-audit 执行失败（exit=%d）：%s" % (code, (err or out)[-300:]))
                else:
                    produced = os.path.join(phase, "findings.json")
                    if not isinstance(workspace.load_json(produced), dict):
                        status = "FAILED"
                        result["degradation"].append(
                            "config-audit 执行成功但未产出有效 findings.json：%s" % produced)
                    else:
                        source, status = produced, "RERUN"

    payload = _summarize(source) if source and os.path.isfile(source) else {}
    result.update(payload)
    result["status"] = status
    result["source"] = source
    out_path = os.path.join(phase, "config_triage.json")
    workspace.write_json(out_path, result)
    workspace.manifest_upsert(args.session, "00-config", {
        "status": status, "source": source,
        "config_primary_candidate": result.get("config_primary_candidate", False),
        "impact_level": result.get("impact_level", ""),
        "artifacts": [os.path.relpath(out_path, args.session)],
    })
    print(json.dumps({k: result.get(k) for k in
                      ("status", "source", "findings_count", "action_count",
                       "impact_level", "config_primary_candidate", "degradation")},
                     ensure_ascii=False, indent=2))
    return 0 if status in ("REUSED", "RERUN", "SKIPPED") else 1


def _skill_roots():
    """可能的 skills 根：仓库为 <root>/host-cpu/<skill>，也兼容平铺 <root>/<skill>。"""
    roots = []
    env = os.environ.get("HOSTCPU_SKILLS_DIR")
    if env:
        roots.append(env)
    parent = os.path.dirname(SKILLS_DIR)
    if parent:
        roots.append(parent)
    roots.append(SKILLS_DIR)
    return roots


def _resolve_audit_dir(explicit):
    """探测 config-audit skill 的 scripts 目录（只用于定位 CLI，不读其产物）。

    兼容 <root>/host-cpu/<skill> 与 <root>/<skill> 两种布局；
    HOSTCPU_SKILLS_DIR 可指向 skills 根，也可指向 skills/host-cpu。
    """
    candidates = []
    if explicit:
        candidates.append(os.path.join(explicit, "scripts")
                          if os.path.isdir(os.path.join(explicit, "scripts")) else explicit)
    for root in _skill_roots():
        candidates.append(os.path.join(root, HOST_CPU_DIR, CONFIG_AUDIT_DIR, "scripts"))
        candidates.append(os.path.join(root, CONFIG_AUDIT_DIR, "scripts"))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "audit.py")):
            return path
    return ""


def _run_audit(audit_dir, out_dir, session):
    cmd = [sys.executable, os.path.join(audit_dir, "audit.py"),
           "--output", out_dir,
           "--stamp", os.path.basename(os.path.normpath(session))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=600)
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def _summarize(findings_path):
    """从 findings.json 提取判型要素（不重判定，只汇总已产出的结论）。"""
    data = workspace.load_json(findings_path, default={}) or {}
    findings = data.get("findings") or []
    actions = data.get("actions") or []
    top_risk = ""
    for action in actions:
        risk = str(action.get("risk") or "").lower()
        if _RISK_ORDER.get(risk, 0) > _RISK_ORDER.get(top_risk, 0):
            top_risk = risk
    strong_findings = [f for f in findings if float(f.get("confidence") or 0) >= 0.7]
    return {
        "findings_count": len(findings),
        "action_count": len(actions),
        "impact_level": _IMPACT_BY_RISK.get(top_risk, "无"),
        # 判型：存在高置信 R 规则结论，或存在高风险待改项 → 配置主因候选
        "config_primary_candidate": bool(strong_findings) or top_risk == "high",
        "top_findings": [{"finding_id": f.get("finding_id"), "problem": f.get("problem"),
                          "confidence": f.get("confidence"),
                          "suggested_cpuset": f.get("suggested_cpuset", "")}
                         for f in findings[:5]],
        "findings": findings,
        "actions": actions,
    }


if __name__ == "__main__":
    raise SystemExit(main())

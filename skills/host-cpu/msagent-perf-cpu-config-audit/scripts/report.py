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
"""report.py — HTML 分节报告 + 终端摘要 + findings.json 落盘。

HTML 由 resources/report_template.html 模板 + JSON payload 渲染，
不拼接 HTML 片段（模板占位符 __PAYLOAD__ 注入）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# 域中文名
DOMAIN_NAMES = {
    "numa": "A · NUMA 与拓扑", "bind": "B · 绑核与 CPU 隔离",
    "hugepage": "C · 大页与内存", "freq": "D · 频率与电源",
    "irq": "E · 中断与软中断", "cgroup": "F · cgroup 与配额",
    "env": "G · 运行时环境变量", "npu": "H · NPU 适配",
}

_STATUS_ORDER = {"CRIT": 0, "WARN": 1, "FACT": 2, "OK": 3}


def _short(value: Any, limit: int = 40) -> str:
    """终端摘要取值：dict/列表截断展示，避免 HTML 转义（终端非 HTML 上下文）。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def build_payload(results: List[Dict[str, Any]], findings: List[Dict[str, Any]],
                 suspects: List[Dict[str, Any]], meta: Dict[str, Any],
                 baseline_diff: List[Dict[str, Any]],
                 actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造注入模板的 payload。"""
    # 按域分组 + 排序（CRIT → WARN → FACT → OK）
    # OK/FACT 只进折叠表（项/检查项/判定/当前值），去掉仅供解释的字段，控制报告体积
    domains: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        if result["status"] in ("OK", "FACT"):
            result = {k: v for k, v in result.items() if k not in ("explain", "advice")}
        domains.setdefault(result["domain"], []).append(result)
    for domain in domains.values():
        domain.sort(key=lambda r: (_STATUS_ORDER.get(r["status"], 9), r.get("id", "")))
    counts = _status_counts(results)
    top5 = _top5(results)
    return {
        "meta": meta, "counts": counts, "top5": top5,
        "domains": {DOMAIN_NAMES.get(k, k): v for k, v in domains.items()},
        "domain_keys": [DOMAIN_NAMES.get(k, k) for k in domains],
        "findings": findings, "suspects": suspects,
        "baseline_diff": baseline_diff,
        "actions": actions or [],
    }


def _status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"CRIT": 0, "WARN": 0, "FACT": 0, "OK": 0}
    for result in results:
        if result["status"] in counts:
            counts[result["status"]] += 1
    return counts


def _top5(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bad = [r for r in results if r["status"] in ("CRIT", "WARN")]
    bad.sort(key=lambda r: (_STATUS_ORDER[r["status"]],
                            -{"high": 3, "medium": 2, "low": 1, "": 0}.get(r.get("benefit", ""), 0)))
    return bad[:5]


def render_html(payload: Dict[str, Any], template_path: str, output_path: str,
                single_file: bool = False) -> str:
    """模板 + payload → hostcpu_config_report_<host>_<ts>.html。"""
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    safe_payload = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    rendered = template.replace("__PAYLOAD__", safe_payload)
    if single_file:
        rendered = rendered.replace("__PAYLOAD__", safe_payload)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    return output_path


def save_findings(findings: List[Dict[str, Any]], actions: List[Dict[str, Any]],
                   path: str) -> str:
    data = {"findings": findings, "actions": actions}
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def terminal_summary(results: List[Dict[str, Any]], findings: List[Dict[str, Any]],
                     meta: Dict[str, Any]) -> str:
    """终端摘要（对话内直接输出）：得分 + Top 问题 + 复检建议。"""
    counts = _status_counts(results)
    lines: List[str] = []
    score = _score(results)
    lines.append(f"主机配置审计摘要 · {meta.get('host', '?')} · 模式={meta.get('mode', '?')}")
    lines.append(f"环境得分：{score}/100   "
                 f"CRIT {counts['CRIT']}   WARN {counts['WARN']}   "
                 f"OK {counts['OK']}   FACT {counts['FACT']}")
    for item in _top5(results):
        lines.append(f"  [{item['status']}] {item['id']} {item['name']}："
                     f"{_short(item.get('value_text') or item['value'])}"
                     f"（阈值 {_short(item['threshold'])}）")
    for finding in findings:
        lines.append(f"  [规则 {finding.get('rule')}] {finding['problem']}"
                     f"（suggested_cpuset={finding.get('suggested_cpuset') or '-'}）")
    lines.append(f"  OK: {counts['OK']} 项通过")
    lines.append("复检建议：修复后同参数复跑本审计对比 baseline。")
    return "\n".join(lines)


def _score(results: List[Dict[str, Any]]) -> int:
    """环境得分（CRIT/WARN 加权，仅作趋势指标，不替代逐项判定）。"""
    counts = _status_counts(results)
    penalty = counts["CRIT"] * 20 + counts["WARN"] * 5
    return max(0, 100 - penalty)


def suspect_items_json(suspects: List[Dict[str, Any]], path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(suspects, fh, ensure_ascii=False, indent=2)
    return path

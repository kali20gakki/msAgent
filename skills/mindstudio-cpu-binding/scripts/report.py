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

"""HTML report renderer."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from scripts.cpulist import parse_cpu_list
from scripts.topology_view import build_topology_view, render_topology_html


def render_report(
    snapshot: dict[str, Any],
    findings: list[dict[str, Any]],
    plan: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _document(snapshot, findings, plan)
    path.write_text(document, encoding="utf-8")


def _document(snapshot: dict[str, Any], findings: list[dict[str, Any]], plan: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>mindstudio-cpu-binding CPU 绑核优化报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
h1, h2 {{ color: #0f172a; }}
.card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); }}
details.card {{ padding: 0; overflow: hidden; }}
details.card > summary {{ cursor: pointer; list-style: none; padding: 16px; font-weight: 700; color: #0f172a; }}
details.card > summary::-webkit-details-marker {{ display: none; }}
details.card > summary::before {{ content: '▸'; display: inline-block; margin-right: 8px; transition: transform 0.15s ease; }}
details.card[open] > summary::before {{ transform: rotate(90deg); }}
details.card > :not(summary) {{ padding-left: 16px; padding-right: 16px; }}
details.card > :last-child {{ padding-bottom: 16px; }}
.collapsible-section {{ scroll-margin-top: 16px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
.metric {{ background: #eff6ff; border-radius: 10px; padding: 12px; }}
.metric strong {{ display: block; font-size: 22px; color: #1d4ed8; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; background: #e0f2fe; color: #0369a1; }}
.high {{ background: #fee2e2; color: #991b1b; }}
.medium {{ background: #fef3c7; color: #92400e; }}
.low {{ background: #e0f2fe; color: #075985; }}
.cpu-grid {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.cpu {{ min-width: 28px; padding: 5px; border-radius: 6px; text-align: center; background: #e5e7eb; font-size: 12px; }}
.allowed {{ background: #bfdbfe; }}
.target {{ background: #bbf7d0; }}
.overlap {{ background: #99f6e4; }}
.topology-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; margin-top: 12px; }}
.topology-graph {{ margin: 16px 0; border: 1px solid #e5e7eb; border-radius: 12px; background: #fafafa; overflow-x: auto; }}
.topology-svg {{ display: block; width: 100%; min-width: 640px; height: auto; }}
.graph-server {{ fill: #455A64; stroke: #263238; stroke-width: 2; }}
.graph-numa {{ fill: #4CAF50; stroke: #2E7D32; stroke-width: 1.5; }}
.graph-npu {{ fill: #FF5722; stroke: #BF360C; stroke-width: 1.5; }}
.graph-node-label {{ fill: #fff; font-size: 13px; font-weight: 700; text-anchor: middle; dominant-baseline: middle; }}
.graph-node-detail {{ fill: #f8fafc; font-size: 9px; text-anchor: middle; dominant-baseline: middle; }}
.topology-edge-label {{ fill: #475569; font-size: 10px; text-anchor: middle; paint-order: stroke; stroke: #fafafa; stroke-width: 3px; }}
.topology-legend text {{ fill: #334155; font-size: 12px; dominant-baseline: middle; }}
.topology-node {{ border: 1px solid #dbeafe; border-radius: 12px; padding: 12px; background: #f8fafc; }}
.topology-item {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; margin: 8px 0; background: #fff; }}
.topology-item strong, .topology-item span {{ display: block; }}
.process-item.cross-numa {{ border-color: #fca5a5; background: #fef2f2; }}
.process-item.local {{ border-color: #86efac; background: #f0fdf4; }}
pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow: auto; }}
</style>
</head>
<body>
<h1>mindstudio-cpu-binding CPU 绑核优化报告</h1>
{_summary(snapshot, findings, plan)}
{_process_table(snapshot, plan)}
{_topology_view(snapshot, plan)}
{_cpu_grid(snapshot, plan)}
{_key_processes(snapshot)}
{_findings(findings)}
{_plan(plan)}
{_gaps(snapshot)}
</body>
</html>
"""


def _summary(snapshot: dict[str, Any], findings: list[dict[str, Any]], plan: dict[str, Any]) -> str:
    high = sum(1 for finding in findings if finding.get("severity") == "high")
    medium = sum(1 for finding in findings if finding.get("severity") == "medium")
    return f"""
<div class="summary">
  <div class="metric"><span>目标 PID</span><strong>{len(snapshot.get("processes", []))}</strong></div>
  <div class="metric"><span>问题发现</span><strong>{len(findings)}</strong></div>
  <div class="metric"><span>High / Medium</span><strong>{high} / {medium}</strong></div>
  <div class="metric"><span>执行后端</span><strong>{html.escape(plan.get("executor_backend", "dry-run"))}</strong></div>
</div>
<div class="card"><strong>总体结论：</strong>{html.escape(plan.get("summary", ""))}</div>
"""


def _process_table(snapshot: dict[str, Any], plan: dict[str, Any]) -> str:
    action_by_pid = {action["pid"]: action for action in plan.get("apply_actions", [])}
    rows = []
    for process in snapshot.get("processes", []):
        pid = int(process.get("pid"))
        action = action_by_pid.get(pid, {})
        rows.append(
            "<tr>"
            f"<td>{pid}</td>"
            f"<td>{html.escape(str(process.get('rank', '')))}</td>"
            f"<td>{html.escape(str(process.get('npu_device', '')))}</td>"
            f"<td>{html.escape(str(process.get('cpus_allowed_list', '')))}</td>"
            f"<td>{html.escape(str(action.get('effective_cpu_list', '')))}</td>"
            f"<td>{html.escape(str(action.get('target_cpu_list', '无需调整')))}</td>"
            "</tr>"
        )
    return (
        """
<div class="card">
<h2>当前 CPU 绑定状态</h2>
<table>
<thead><tr><th>PID</th><th>Rank</th><th>NPU</th><th>当前 CPU Range</th><th>有效 CPU Range</th><th>推荐 CPU Range</th></tr></thead>
<tbody>
"""
        + "\n".join(rows)
        + "\n</tbody></table></div>"
    )


def _topology_view(snapshot: dict[str, Any], plan: dict[str, Any]) -> str:
    view = build_topology_view(snapshot, plan)
    return render_topology_html(view)


def _cpu_grid(snapshot: dict[str, Any], plan: dict[str, Any]) -> str:
    allowed = set()
    target = set()
    for process in snapshot.get("processes", []):
        allowed |= parse_cpu_list(process.get("cpus_allowed_list"))
    for action in plan.get("apply_actions", []):
        target |= parse_cpu_list(action.get("target_cpu_list"))

    sections = []
    for node in snapshot.get("numa_topology", {}).get("nodes", []):
        cpus = sorted(parse_cpu_list(node.get("cpus")))
        cells = []
        for cpu in cpus:
            classes = ["cpu"]
            if cpu in allowed and cpu in target:
                classes.append("overlap")
            elif cpu in target:
                classes.append("target")
            elif cpu in allowed:
                classes.append("allowed")
            cells.append(f"<span class='{' '.join(classes)}'>{cpu}</span>")
        sections.append(f"<h3>NUMA {node.get('node')}</h3><div class='cpu-grid'>{''.join(cells)}</div>")
    return "<div class='card'><h2>CPU / NUMA 视图</h2>" + "\n".join(sections) + "</div>"


def _key_processes(snapshot: dict[str, Any]) -> str:
    key_processes = snapshot.get("key_processes") or {}
    if not key_processes:
        return "<details class='card collapsible-section' open><summary>关键进程与线程</summary><p>未采集到 key_processes 信息。</p></details>"

    rows = []
    rows.extend(_key_process_rows("主调度进程", key_processes.get("main_scheduler_pids", [])))
    rows.extend(_key_process_rows("SQ 线程", key_processes.get("sq_task_threads", [])))
    rows.extend(_key_process_rows("NPU 固定线程", key_processes.get("npu_fixed_threads", [])))
    rows.extend(_key_process_rows("通信线程", key_processes.get("communication_threads", [])))
    rows.extend(_key_process_rows("DataLoader 线程", key_processes.get("dataloader_threads", [])))
    rows.extend(_key_process_rows("Top CPU 线程", key_processes.get("top_threads", [])))
    if not rows:
        rows.append("<tr><td colspan='6'>未识别到关键进程或线程。</td></tr>")

    return (
        "<details class='card collapsible-section' open><summary>关键进程与线程</summary>"
        "<table><thead><tr>"
        "<th>类别</th><th>PID</th><th>TID</th><th>名称</th><th>NPU</th><th>CPU%</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></details>"
    )


def _key_process_rows(category: str, items: list[Any]) -> list[str]:
    rows = []
    for item in items:
        data = {"pid": item} if isinstance(item, int) else item
        rows.append(
            "<tr>"
            f"<td>{html.escape(category)}</td>"
            f"<td>{html.escape(str(data.get('pid', '')))}</td>"
            f"<td>{html.escape(str(data.get('tid', '')))}</td>"
            f"<td>{html.escape(str(data.get('name', data.get('key_class', ''))))}</td>"
            f"<td>{html.escape(str(data.get('npu_id', data.get('npu_device', ''))))}</td>"
            f"<td>{html.escape(str(data.get('cpu_percent', '')))}</td>"
            "</tr>"
        )
    return rows


def _findings(findings: list[dict[str, Any]]) -> str:
    cards = []
    for finding in findings:
        severity = html.escape(str(finding.get("severity", "info")))
        evidence = "".join(f"<li>{html.escape(str(item))}</li>" for item in finding.get("evidence", []))
        recommendations = "".join(f"<li>{html.escape(str(item))}</li>" for item in finding.get("recommendations", []))
        cards.append(
            f"<div class='card'><span class='badge {severity}'>{severity}</span> "
            f"<strong>{html.escape(str(finding.get('id')))} {html.escape(str(finding.get('title')))}</strong>"
            f"<p>{html.escape(str(finding.get('judgement')))}</p>"
            f"<h4>证据</h4><ul>{evidence}</ul>"
            f"<h4>建议</h4><ul>{recommendations}</ul></div>"
        )
    return "<h2>问题发现</h2>" + "\n".join(cards)


def _plan(plan: dict[str, Any]) -> str:
    commands = "\n".join(action.get("apply_command", "") for action in plan.get("apply_actions", []))
    rollback = "\n".join(action.get("rollback_command", "") for action in plan.get("rollback_actions", []))
    state = json.dumps(plan.get("rollback_state_preview", {}), ensure_ascii=False, indent=2)
    return f"""
<div class="card">
<h2>推荐方案与回滚预览</h2>
<p>执行前需要用户确认；当前实现默认 dry-run，不修改系统状态。</p>
<h3>应用命令</h3><pre>{html.escape(commands)}</pre>
<h3>回滚命令</h3><pre>{html.escape(rollback)}</pre>
<h3>rollback-state 预览</h3><pre>{html.escape(state)}</pre>
</div>
"""


def _gaps(snapshot: dict[str, Any]) -> str:
    missing = snapshot.get("availability", {}).get("missing", [])
    if not missing:
        return "<div class='card'><h2>信息缺口</h2><p>未发现关键缺失字段。</p></div>"
    items = "".join(f"<li>{html.escape(str(item))}</li>" for item in missing)
    return f"<div class='card'><h2>信息缺口</h2><ul>{items}</ul></div>"

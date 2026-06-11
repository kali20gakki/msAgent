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

"""Snapshot-driven CPU/NPU/NUMA topology view model and renderers."""

from __future__ import annotations

import html
from typing import Any

from scripts.cpulist import numa_nodes_for_cpu_list, parse_cpu_list


STATUS_LABELS = {
    "cross-numa": "跨 NUMA",
    "local": "本地 NUMA",
    "unknown": "待确认",
}

LINK_STYLES = {
    "HCCS": {"color": "#FF9E80", "width": 3, "dasharray": ""},
    "HCCS_SW": {"color": "#FFB74D", "width": 3, "dasharray": ""},
    "PIX": {"color": "#64B5F6", "width": 2, "dasharray": ""},
    "PXB": {"color": "#81C784", "width": 2, "dasharray": ""},
    "PHB": {"color": "#BA68C8", "width": 2, "dasharray": ""},
    "SYS": {"color": "#BDBDBD", "width": 1.5, "dasharray": "6 4"},
    "SIO": {"color": "#FFD54F", "width": 2, "dasharray": ""},
    "NA": {"color": "#E0E0E0", "width": 1, "dasharray": "5 5"},
}


def build_topology_view(snapshot: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    numa_nodes = snapshot.get("numa_topology", {}).get("nodes", [])
    npu_devices = snapshot.get("npu_topology", {}).get("devices", [])
    processes = snapshot.get("processes", [])
    action_by_pid = _actions_by_pid(plan)
    missing = _missing_fields(snapshot, numa_nodes, npu_devices)

    process_views = []
    cross_numa_count = 0
    for process in processes:
        action = action_by_pid.get(int(process.get("pid")))
        view_process = _process_view(process, action, numa_nodes)
        process_views.append(view_process)
        if view_process["status"] == "cross-numa":
            cross_numa_count += 1

    nodes = []
    for node in sorted(numa_nodes, key=lambda item: int(item.get("node", -1))):
        node_id = int(node.get("node"))
        node_processes = [process for process in process_views if process["home_numa_node"] == node_id]

        nodes.append(
            {
                "node": node_id,
                "cpus": str(node.get("cpus", "")),
                "logical_cpu_count": node.get("logical_cpu_count"),
                "physical_core_count": node.get("physical_core_count"),
                "memory_total_mb": node.get("memory_total_mb"),
                "local_npus": [_npu_view(device) for device in npu_devices if _device_numa_node(device) == node_id],
                "processes": node_processes,
            }
        )

    known_nodes = {node["node"] for node in nodes}
    orphan_processes = [process for process in process_views if process["home_numa_node"] not in known_nodes]
    topology_graph = _topology_graph(nodes, npu_devices)

    return {
        "summary": {
            "numa_count": len(numa_nodes),
            "logical_cpu_count": snapshot.get("system", {}).get("total_logical_cpus"),
            "npu_count": len(npu_devices),
            "process_count": len(processes),
            "cross_numa_process_count": cross_numa_count,
        },
        "topology_graph": topology_graph,
        "numa_nodes": nodes,
        "orphan_processes": orphan_processes,
        "warnings": _warnings(processes, numa_nodes),
        "missing": missing,
    }


def render_topology_html(view: dict[str, Any]) -> str:
    summary = view.get("summary", {})
    node_cards = "".join(_render_numa_card(node) for node in view.get("numa_nodes", []))
    orphan_card = _render_orphan_processes(view.get("orphan_processes", []))
    warning_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in view.get("warnings", []))
    missing_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in view.get("missing", []))

    graph = _render_topology_svg(view.get("topology_graph", {}))
    warnings = f"<h3>拓扑提示</h3><ul>{warning_items}</ul>" if warning_items else ""
    missing = f"<h3>信息缺口</h3><ul>{missing_items}</ul>" if missing_items else ""

    return f"""
<details class='card collapsible-section topology-view' open>
<summary>CPU / NPU / NUMA 拓扑关系</summary>
<div class="summary topology-summary">
  <div class="metric"><span>NUMA Node</span><strong>{html.escape(str(summary.get("numa_count", 0)))}</strong></div>
  <div class="metric"><span>逻辑 CPU</span><strong>{html.escape(str(summary.get("logical_cpu_count") or "未知"))}</strong></div>
  <div class="metric"><span>NPU</span><strong>{html.escape(str(summary.get("npu_count", 0)))}</strong></div>
  <div class="metric"><span>跨 NUMA 进程</span><strong>{html.escape(str(summary.get("cross_numa_process_count", 0)))}</strong></div>
</div>
{graph}
<div class="topology-grid">{node_cards}{orphan_card}</div>
{warnings}
{missing}
</details>
"""


def render_topology_text(view: dict[str, Any]) -> str:
    lines = ["CPU / NPU / NUMA 拓扑关系"]
    summary = view.get("summary", {})
    lines.append(
        f"NUMA={summary.get('numa_count', 0)}, NPU={summary.get('npu_count', 0)}, "
        f"process={summary.get('process_count', 0)}, cross_numa={summary.get('cross_numa_process_count', 0)}"
    )
    for node in view.get("numa_nodes", []):
        lines.append(f"NUMA {node['node']}: CPUs {node.get('cpus', '')}")
        for npu in node.get("local_npus", []):
            lines.append(f"  NPU {npu.get('device_id')}: {npu.get('pci_bus_id', '')}")
        for process in node.get("processes", []):
            lines.append(
                f"  PID {process.get('pid')}: current={process.get('current_cpu_list', '')}, "
                f"target={process.get('target_cpu_list', '')}, status={process.get('status', '')}"
            )
    for item in view.get("missing", []):
        lines.append(f"缺少 {item}")
    return "\n".join(lines)


def _topology_graph(numa_nodes: list[dict[str, Any]], npu_devices: list[dict[str, Any]]) -> dict[str, Any]:
    total_numa = max(len(numa_nodes), 1)
    max_local_npus = max((len(node.get("local_npus", [])) for node in numa_nodes), default=0)
    npu_half_spread = max((max_local_npus - 1) * 60, 0)
    side_margin = max(180, npu_half_spread + 80)
    width = max(800, side_margin * 2 + 520 * max(total_numa - 1, 0) + 240)
    height = 520 if max_local_npus > 3 or total_numa > 2 else 440
    server_x = width / 2
    server_y = 70
    numa_y = 180
    npu_y = 390
    numa_step = 520 if total_numa > 1 else 0

    nodes = [
        {
            "id": "server",
            "label": "Server",
            "kind": "server",
            "x": server_x,
            "y": server_y,
        }
    ]
    edges = []
    npu_positions = {}

    for index, node in enumerate(numa_nodes):
        x = side_margin + index * numa_step if total_numa > 1 else server_x
        numa_id = f"numa-{node['node']}"
        nodes.append(
            {
                "id": numa_id,
                "label": f"NUMA {node['node']}",
                "kind": "numa",
                "x": x,
                "y": numa_y,
                "detail": node.get("cpus", ""),
            }
        )
        edges.append(
            {
                "source": "server",
                "target": numa_id,
                "type": "NUMA",
                "color": "#B2DFDB",
                "width": 2,
                "dasharray": "",
            }
        )

        local_npus = node.get("local_npus", [])
        npu_step = 120 if len(local_npus) > 1 else 0
        for npu_index, npu in enumerate(local_npus):
            npu_x = x + (npu_index - (len(local_npus) - 1) / 2) * npu_step
            npu_id = f"npu-{npu['device_id']}"
            npu_positions[str(npu["device_id"])] = npu_id
            nodes.append(
                {
                    "id": npu_id,
                    "label": f"NPU {npu['device_id']}",
                    "kind": "npu",
                    "x": npu_x,
                    "y": npu_y,
                    "detail": npu.get("pci_bus_id", ""),
                }
            )
            edges.append(
                {
                    "source": numa_id,
                    "target": npu_id,
                    "type": "locality",
                    "color": "#C5E1A5",
                    "width": 1.5,
                    "dasharray": "5 4",
                }
            )

    edges.extend(_npu_interconnect_edges(npu_devices, npu_positions))
    return {"nodes": nodes, "edges": edges, "width": width, "height": height}


def _npu_interconnect_edges(npu_devices: list[dict[str, Any]], npu_positions: dict[str, str]) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    for device in npu_devices:
        source_device = str(device.get("device_id", ""))
        source = npu_positions.get(source_device)
        if not source:
            continue
        for link in device.get("links", []):
            target_device = str(link.get("target") or link.get("dst") or link.get("device_id") or "")
            target = npu_positions.get(target_device)
            if not target:
                continue
            key = tuple(sorted([source_device, target_device]))
            if key in seen:
                continue
            seen.add(key)
            link_type = str(link.get("type") or link.get("link_type") or "NA")
            style = LINK_STYLES.get(link_type, LINK_STYLES["NA"])
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "type": link_type,
                    "color": style["color"],
                    "width": style["width"],
                    "dasharray": style["dasharray"],
                }
            )
    return edges


def _render_topology_svg(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    if not nodes:
        return ""
    width = int(graph.get("width", 800))
    height = int(graph.get("height", 440))
    node_by_id = {node["id"]: node for node in nodes}
    edges = "".join(_render_graph_edge(edge, node_by_id) for edge in graph.get("edges", []))
    rendered_nodes = "".join(_render_graph_node(node) for node in nodes)
    legend = _render_graph_legend(height)
    return f"""
<div class="topology-graph">
<svg class="topology-svg" viewBox="0 0 {width} {height}" role="img" aria-label="CPU NPU NUMA topology graph">
  {edges}
  {rendered_nodes}
  {legend}
</svg>
</div>
"""


def _render_graph_edge(edge: dict[str, Any], node_by_id: dict[str, dict[str, Any]]) -> str:
    source = node_by_id.get(edge.get("source"))
    target = node_by_id.get(edge.get("target"))
    if not source or not target:
        return ""
    dash = f' stroke-dasharray="{html.escape(str(edge.get("dasharray")))}"' if edge.get("dasharray") else ""
    label_x = (float(source["x"]) + float(target["x"])) / 2
    label_y = (float(source["y"]) + float(target["y"])) / 2 - 8
    return (
        f'<line x1="{source["x"]}" y1="{source["y"]}" x2="{target["x"]}" y2="{target["y"]}" '
        f'stroke="{html.escape(str(edge.get("color", "#CBD5E1")))}" stroke-width="{html.escape(str(edge.get("width", 1)))}"{dash} />'
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="topology-edge-label">{html.escape(str(edge.get("type", "")))}</text>'
    )


def _render_graph_node(node: dict[str, Any]) -> str:
    kind = str(node.get("kind"))
    x = html.escape(str(node.get("x", 0)))
    y = html.escape(str(node.get("y", 0)))
    label = html.escape(str(node.get("label", "")))
    detail = html.escape(_truncate(str(node.get("detail", "")), 18))
    title = html.escape(str(node.get("detail", "")))
    if kind == "server":
        return f'<circle cx="{x}" cy="{y}" r="34" class="graph-server"/><text x="{x}" y="{y}" class="graph-node-label">{label}</text>'
    if kind == "numa":
        return f'<g><title>{title}</title><rect x="{float(node.get("x", 0)) - 54}" y="{float(node.get("y", 0)) - 26}" width="108" height="52" rx="14" class="graph-numa"/><text x="{x}" y="{y}" class="graph-node-label">{label}</text><text x="{x}" y="{float(node.get("y", 0)) + 16}" class="graph-node-detail">{detail}</text></g>'
    return f'<g><title>{title}</title><rect x="{float(node.get("x", 0)) - 46}" y="{float(node.get("y", 0)) - 24}" width="92" height="48" rx="10" class="graph-npu"/><text x="{x}" y="{y}" class="graph-node-label">{label}</text><text x="{x}" y="{float(node.get("y", 0)) + 15}" class="graph-node-detail">{detail}</text></g>'


def _render_graph_legend(height: int) -> str:
    y = height - 50
    text_y = y + 4
    return f"""
  <g class="topology-legend">
    <circle cx="40" cy="{y}" r="8" class="graph-server"/><text x="56" y="{text_y}">Server</text>
    <rect x="130" y="{y - 8}" width="18" height="16" rx="4" class="graph-numa"/><text x="156" y="{text_y}">NUMA</text>
    <rect x="230" y="{y - 8}" width="18" height="16" rx="4" class="graph-npu"/><text x="256" y="{text_y}">NPU</text>
    <line x1="325" y1="{y}" x2="360" y2="{y}" stroke="#FF9E80" stroke-width="3"/><text x="368" y="{text_y}">NPU interconnect</text>
  </g>
"""


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)] + "…"


def _actions_by_pid(plan: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not plan:
        return {}
    return {int(action["pid"]): action for action in plan.get("apply_actions", []) if action.get("pid") is not None}


def _missing_fields(
    snapshot: dict[str, Any],
    numa_nodes: list[dict[str, Any]],
    npu_devices: list[dict[str, Any]],
) -> list[str]:
    missing = []
    if not numa_nodes:
        missing.append("numa_topology.nodes")
    if "npu_topology" not in snapshot or "devices" not in snapshot.get("npu_topology", {}):
        missing.append("npu_topology.devices")
    if npu_devices and any(device.get("numa_node") is None for device in npu_devices):
        missing.append("npu_topology.devices[*].numa_node")
    return missing


def _warnings(processes: list[dict[str, Any]], numa_nodes: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for process in processes:
        pid = process.get("pid")
        current_nodes = numa_nodes_for_cpu_list(process.get("cpus_allowed_list"), numa_nodes)
        if len(current_nodes) > 1:
            warnings.append(f"PID {pid} 当前 allowed CPU 覆盖多个 NUMA 节点：{sorted(current_nodes)}。")
    return warnings


def _process_view(
    process: dict[str, Any],
    action: dict[str, Any] | None,
    numa_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    pid = int(process.get("pid"))
    current_cpu_list = str(process.get("cpus_allowed_list", ""))
    target_cpu_list = str((action or {}).get("target_cpu_list") or "")
    effective_cpu_list = str((action or {}).get("effective_cpu_list") or "")
    npu_device = process.get("npu_device")
    home_numa_node = _first_numa_node(target_cpu_list, numa_nodes)
    if home_numa_node is None:
        home_numa_node = _first_numa_node(effective_cpu_list, numa_nodes)
    if home_numa_node is None:
        home_numa_node = _first_numa_node(current_cpu_list, numa_nodes)

    current_nodes = numa_nodes_for_cpu_list(current_cpu_list, numa_nodes)
    target_nodes = numa_nodes_for_cpu_list(target_cpu_list, numa_nodes)
    status = "unknown"
    if len(current_nodes) > 1:
        status = "cross-numa"
    elif target_nodes and current_nodes and current_nodes <= target_nodes:
        status = "local"

    return {
        "pid": pid,
        "role": _process_role(process),
        "npu_device": str(npu_device) if npu_device is not None else "",
        "current_cpu_list": current_cpu_list,
        "effective_cpu_list": effective_cpu_list,
        "target_cpu_list": target_cpu_list,
        "home_numa_node": home_numa_node,
        "current_numa_nodes": sorted(current_nodes),
        "target_numa_nodes": sorted(target_nodes),
        "status": status,
    }


def _process_role(process: dict[str, Any]) -> str:
    rank = process.get("rank")
    if rank is not None:
        return f"rank{rank}"
    comm = process.get("comm")
    if comm:
        return str(comm)
    return "process"


def _first_numa_node(cpu_list: str | None, numa_nodes: list[dict[str, Any]]) -> int | None:
    nodes = sorted(numa_nodes_for_cpu_list(cpu_list, numa_nodes))
    if not nodes:
        return None
    return nodes[0]


def _npu_view(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_id": str(device.get("device_id", "")),
        "logical_id": str(device.get("logical_id", "")),
        "pci_bus_id": str(device.get("pci_bus_id", "")),
        "numa_node": device.get("numa_node"),
        "local_cpus": str(device.get("local_cpus", "")),
        "health": str(device.get("health", "")),
    }


def _device_numa_node(device: dict[str, Any]) -> int | None:
    node = device.get("numa_node")
    if node is None:
        return None
    return int(node)


def _render_numa_card(node: dict[str, Any]) -> str:
    npus = "".join(_render_npu(npu) for npu in node.get("local_npus", [])) or "<p>未发现本地 NPU。</p>"
    processes = (
        "".join(_render_process(process) for process in node.get("processes", []))
        or "<p>未发现映射到该 NUMA 的目标进程。</p>"
    )
    physical = node.get("physical_core_count") or "未知"
    logical = node.get("logical_cpu_count") or len(parse_cpu_list(node.get("cpus")))
    return f"""
<div class="topology-node">
<h3>NUMA {html.escape(str(node.get("node")))}</h3>
<p><strong>CPU Range:</strong> {html.escape(str(node.get("cpus", "")))}</p>
<p><strong>Core:</strong> physical={html.escape(str(physical))}, logical={html.escape(str(logical))}</p>
<div class="topology-npus"><h4>本地 NPU</h4>{npus}</div>
<div class="topology-processes"><h4>进程 / Rank / Worker</h4>{processes}</div>
</div>
"""


def _render_npu(npu: dict[str, Any]) -> str:
    return f"""
<div class="topology-item npu-item">
<strong>NPU {html.escape(str(npu.get("device_id", "")))}</strong>
<span>PCI {html.escape(str(npu.get("pci_bus_id", "")))}</span>
<span>Local CPU {html.escape(str(npu.get("local_cpus", "")))}</span>
<span>Health {html.escape(str(npu.get("health", "")))}</span>
</div>
"""


def _render_process(process: dict[str, Any]) -> str:
    status = str(process.get("status", "unknown"))
    status_label = STATUS_LABELS.get(status, status)
    return f"""
<div class="topology-item process-item {html.escape(status)}">
<strong>PID {html.escape(str(process.get("pid")))} / {html.escape(str(process.get("role", "")))}</strong>
<span>NPU {html.escape(str(process.get("npu_device", "")))}</span>
<span>当前 CPU {html.escape(str(process.get("current_cpu_list", "")))}</span>
<span>有效 CPU {html.escape(str(process.get("effective_cpu_list", "")))}</span>
<span>推荐 CPU {html.escape(str(process.get("target_cpu_list", "")))}</span>
<span>状态：{html.escape(status_label)}</span>
</div>
"""


def _render_orphan_processes(processes: list[dict[str, Any]]) -> str:
    if not processes:
        return ""
    content = "".join(_render_process(process) for process in processes)
    return f"""
<div class="topology-node">
<h3>NUMA 待确认</h3>
<p>这些进程无法根据 Snapshot 映射到确定 NUMA 节点。</p>
{content}
</div>
"""

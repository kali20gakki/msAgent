#!/usr/bin/env python3
"""
Ascend NPU Memory Snapshot 高层分析工具

支持六种分析模式:
    overview  - 总体概览
    peak      - 峰值分析
    fragment  - 碎片分析
    leak      - 泄漏检测
    oom       - OOM 分析
    compare   - 跨快照对比
    all       - 全模式分析 + HTML 报告

用法:
    python snapshot_analyze.py snapshot.db --mode overview
    python snapshot_analyze.py snapshot.db --mode peak
    python snapshot_analyze.py snapshot.db --mode fragment
    python snapshot_analyze.py snapshot.db --mode leak
    python snapshot_analyze.py snapshot.db --mode oom
    python snapshot_analyze.py snapshot.db --mode compare --ref other.db
    python snapshot_analyze.py snapshot.db --mode all -o report.html
"""

import argparse
import html
import json
import os
import sys
from typing import Any, Dict, List, Optional

import snapshot_queries as q


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _health_status(frag_pct: float, segment_count: int, has_oom: bool) -> Dict[str, str]:
    if has_oom or frag_pct > 15:
        return {"icon": "[ERR]", "level": "严重", "text": "严重 (OOM 或碎片率 >15%)"}
    elif frag_pct > 5 or segment_count > 100:
        return {"icon": "[WARN]", "level": "需关注", "text": "需关注 (碎片率偏高或 Segment 偏多)"}
    else:
        return {"icon": "[OK]", "level": "健康", "text": "健康"}


def _metric_status(value: float, thresholds: tuple) -> str:
    low, high = thresholds
    if value < low:
        return "[OK]"
    elif value <= high:
        return "[WARN]"
    return "[ERR]"


def analyze_overview(db_path: str) -> Dict[str, Any]:
    overview = q.get_device_overview(db_path)
    device_count = q.get_device_count(db_path)
    oom_events = q.get_oom_events(db_path)

    total_reserved = sum(d["reserved_bytes"] for d in overview)
    total_allocated = sum(d["allocated_bytes"] for d in overview)
    total_segments = sum(d["segment_count"] for d in overview)
    frag_pct = round((total_reserved - total_allocated) / total_reserved * 100, 1) if total_reserved > 0 else 0

    health = _health_status(frag_pct, total_segments, len(oom_events) > 0)

    return {
        "mode": "overview",
        "health": health,
        "device_count": device_count,
        "oom_count": len(oom_events),
        "summary": {
            "reserved": total_reserved,
            "reserved_fmt": _format_bytes(total_reserved),
            "allocated": total_allocated,
            "allocated_fmt": _format_bytes(total_allocated),
            "frag_pct": frag_pct,
            "segment_count": total_segments,
        },
        "devices": overview,
    }


def _format_frames(frames_json: Optional[str], max_frames: int = 5) -> str:
    """将 frames_json 格式化为可读的调用路径，跳过 boilerplate 帧并截断超长函数名"""
    if not frames_json:
        return "(无堆栈信息)"
    try:
        frames_list = json.loads(frames_json)
        if not frames_list:
            return "(空堆栈)"

        skip_prefixes = ("torch::unwind::", "torch::CapturedTraceback::", "std::promise", "std::error_code")

        parts = []
        for f in frames_list:
            if isinstance(f, dict):
                name = f.get("name", f.get("filename", "??"))
                if any(name.startswith(p) for p in skip_prefixes):
                    continue
                if len(name) > 80:
                    name = name[:77] + "..."
                line = f.get("line", 0)
                parts.append(f"{name}:{line}" if line else name)
            elif isinstance(f, str):
                if any(f.startswith(p) for p in skip_prefixes):
                    continue
                if len(f) > 80:
                    f = f[:77] + "..."
                parts.append(f)
            else:
                parts.append(str(f))
            if len(parts) >= max_frames:
                break
        return html.escape(" → ".join(parts) if parts else "(无有效堆栈帧)")
    except (json.JSONDecodeError, TypeError):
        return html.escape(str(frames_json)[:120])


def analyze_peak(db_path: str, device_index: Optional[int] = None) -> Dict[str, Any]:
    timeline = q.get_peak_timeline(db_path, device_index)

    if not timeline:
        return {"mode": "peak", "error": "无时序数据"}

    per_device = {}
    current_dev = None
    reserved = 0
    device_start_idx = {}

    for i, event in enumerate(timeline):
        dev = event.get("device_index", 0)
        if current_dev is None:
            current_dev = dev
        if dev != current_dev:
            current_dev = dev
            reserved = 0

        if dev not in device_start_idx:
            device_start_idx[dev] = i

        action = event["action"]
        size = event.get("size", 0)

        if action in ("segment_alloc", "segment_map"):
            reserved += size
        elif action in ("segment_free", "segment_unmap"):
            reserved -= size

        if dev not in per_device:
            per_device[dev] = {
                "peak_reserved": 0,
                "peak_index": i,
                "peak_event": event,
                "baseline_reserved": 0,
                "baseline_index": 0,
            }
        pd = per_device[dev]

        if reserved > pd["peak_reserved"]:
            pd["peak_reserved"] = reserved
            pd["peak_index"] = i
            pd["peak_event"] = event

    for dev, pd in per_device.items():
        dev_timeline = [e for e in timeline if e.get("device_index", 0) == dev]
        cutoff = max(1, len(dev_timeline) // 10)
        if cutoff < len(dev_timeline):
            pd["baseline_index"] = dev_timeline[cutoff].get("trace_index", 0)
            baseline_r = 0
            for j in range(cutoff + 1):
                event = dev_timeline[j]
                action = event["action"]
                size = event.get("size", 0)
                if action in ("segment_alloc", "segment_map"):
                    baseline_r += size
                elif action in ("segment_free", "segment_unmap"):
                    baseline_r -= size
            pd["baseline_reserved"] = baseline_r

    alloc_events = q.get_peak_alloc_events(db_path, device_index, limit=10)
    peak_contributors = []
    for evt in alloc_events:
        peak_contributors.append({
            "trace_index": evt["trace_index"],
            "action": evt["action"],
            "size": evt["size"],
            "size_fmt": _format_bytes(evt["size"]),
            "addr": hex(evt["addr"]) if evt.get("addr") else "N/A",
            "call_path": _format_frames(evt.get("frames_json")),
        })

    top_blocks = q.get_top_blocks_with_stack(db_path, device_index, limit=10)
    peak_blocks = []
    for blk in top_blocks:
        peak_blocks.append({
            "size": blk["size"],
            "size_fmt": _format_bytes(blk["size"]),
            "requested_size": blk["requested_size"],
            "requested_fmt": _format_bytes(blk["requested_size"]),
            "state": blk["state"],
            "segment_addr": hex(blk["segment_address"]) if blk.get("segment_address") else "N/A",
            "call_path": _format_frames(blk.get("frames_json")),
        })

    devices = []
    for dev, pd in sorted(per_device.items()):
        peak_r = pd["peak_reserved"]
        base_r = pd["baseline_reserved"]
        devices.append({
            "device_index": dev,
            "peak_trace_index": pd["peak_event"].get("trace_index"),
            "peak": {
                "reserved": peak_r,
                "reserved_fmt": _format_bytes(peak_r),
            },
            "baseline": {
                "description": "基线 (该设备时间线前 10% 位置)",
                "trace_index": pd["baseline_index"],
                "reserved": base_r,
                "reserved_fmt": _format_bytes(base_r),
            },
            "deltas": {
                "description": "该设备从基线到峰值的增长量",
                "reserved": peak_r - base_r,
                "reserved_fmt": _format_bytes(peak_r - base_r),
                "reserved_pct": round((peak_r - base_r) / base_r * 100, 1) if base_r > 0 else None,
            },
        })

    overall_peak = sum(pd["peak_reserved"] for pd in per_device.values())
    overall_baseline = sum(pd["baseline_reserved"] for pd in per_device.values())

    return {
        "mode": "peak",
        "device": device_index,
        "devices": devices,
        "peak": {
            "reserved": overall_peak,
            "reserved_fmt": _format_bytes(overall_peak),
        },
        "baseline": {
            "description": "基线 (各设备时间线前 10% 位置)",
            "reserved": overall_baseline,
            "reserved_fmt": _format_bytes(overall_baseline),
        },
        "deltas": {
            "description": "从基线到峰值的增长量",
            "reserved": overall_peak - overall_baseline,
            "reserved_fmt": _format_bytes(overall_peak - overall_baseline),
            "reserved_pct": round((overall_peak - overall_baseline) / overall_baseline * 100, 1) if overall_baseline > 0 else None,
        },
        "peak_alloc_events": {
            "description": "峰值归因: 最大的 segment_alloc / segment_map 事件 (按 size 降序)，这些操作的代码路径是峰值的主要贡献者",
            "top": peak_contributors,
        },
        "peak_blocks": {
            "description": "峰值归因: 最大的 active_allocated block (按 size 降序)，展示占用峰值内存的具体 tensor 分配",
            "top": peak_blocks,
        },
    }


def analyze_fragment(db_path: str, device_index: Optional[int] = None) -> Dict[str, Any]:
    overview = q.get_device_overview(db_path)
    frag_detail = q.get_fragmentation_detail(db_path, device_index)
    block_dist = q.get_block_state_dist(db_path, device_index)

    total_reserved = sum(d["reserved_bytes"] for d in overview)
    total_allocated = sum(d["allocated_bytes"] for d in overview)
    frag_pct = round((total_reserved - total_allocated) / total_reserved * 100, 1) if total_reserved > 0 else 0

    free_bytes = total_reserved - total_allocated
    pending_bytes = sum(b["total_size"] for b in block_dist if b["state"] == "active_pending_free")
    pending_pct = round(pending_bytes / free_bytes * 100, 1) if free_bytes > 0 else 0

    is_pseudo = pending_pct > 50

    top_fragmented = sorted(frag_detail, key=lambda x: x["frag_pct"], reverse=True)[:5]

    return {
        "mode": "fragment",
        "device": device_index,
        "overall": {
            "frag_pct": frag_pct,
            "status": _metric_status(frag_pct, (5, 15)),
            "free_bytes": free_bytes,
            "free_fmt": _format_bytes(free_bytes),
            "pending_bytes": pending_bytes,
            "pending_fmt": _format_bytes(pending_bytes),
            "pending_pct": pending_pct,
            "is_pseudo": is_pseudo,
        },
        "top_fragmented": top_fragmented,
    }


def analyze_leak(db_path: str, device_index: Optional[int] = None) -> Dict[str, Any]:
    trend = q.get_net_segment_trend(db_path, device_index)
    long_blocks = q.get_long_lived_blocks(db_path, device_index)
    stack_attr = q.get_stack_attribution(db_path, device_index)

    is_monotonic = False
    if len(trend) >= 2:
        values = [t["net_segments"] for t in trend]
        is_monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1)) and values[-1] > values[0]

    long_lived_count = len(long_blocks)

    if is_monotonic and long_lived_count >= 5:
        risk = {"icon": "[ERR]", "level": "高风险"}
    elif is_monotonic or long_lived_count >= 3:
        risk = {"icon": "[WARN]", "level": "中风险"}
    else:
        risk = {"icon": "[OK]", "level": "低风险"}

    trend_start = trend[0]["net_segments"] if trend else 0
    trend_end = trend[-1]["net_segments"] if trend else 0

    suspects = []
    for attr in stack_attr[:3]:
        key_path = _format_frames(attr.get("frames_json"))

        suspects.append({
            "stack_id": attr["stack_id"],
            "total_size": attr["total_size"],
            "total_size_fmt": _format_bytes(attr["total_size"]),
            "block_count": attr["block_count"],
            "pct": attr["pct"],
            "key_path": key_path,
        })

    return {
        "mode": "leak",
        "device": device_index,
        "risk": risk,
        "monotonic_growth": {
            "detected": is_monotonic,
            "start": trend_start,
            "end": trend_end,
        },
        "long_lived_blocks": long_lived_count,
        "suspects": suspects,
    }


def analyze_oom(db_path: str, device_index: Optional[int] = None) -> Dict[str, Any]:
    oom_events = q.get_oom_events(db_path, device_index)

    if not oom_events:
        return {"mode": "oom", "detected": False, "events": []}

    results = []
    for oom in oom_events:
        dev = oom["device_index"]
        idx = oom["trace_index"]
        device_free = oom.get("device_free", 0)

        start_idx = max(0, idx - 50)
        pre_events = q.get_trace_range(db_path, dev, start_idx, idx)

        pre_allocs = [e for e in pre_events if e["action"] == "alloc"]
        pre_allocs.sort(key=lambda x: x.get("size", 0), reverse=True)
        last_5 = pre_allocs[:5] if len(pre_allocs) >= 5 else pre_allocs

        last_5_total = sum(e.get("size", 0) for e in last_5)

        if len(last_5) >= 3:
            stack_counts: Dict[str, int] = {}
            for e in last_5:
                frames = e.get("frames_json", "")
                key = str(frames)[:60] if frames else "(无堆栈)"
                stack_counts[key] = stack_counts.get(key, 0) + 1
            max_repeat = max(stack_counts.values())
            if max_repeat >= 3:
                root_cause = "同一堆栈重复分配"
            elif last_5_total > device_free * 3:
                root_cause = "一连串大分配"
            else:
                root_cause = "单次过大分配请求"
        else:
            root_cause = "数据不足，无法推断"

        results.append({
            "device": dev,
            "oom_index": idx,
            "device_free": device_free,
            "device_free_fmt": _format_bytes(device_free),
            "last_5_allocations": [
                {
                    "size": e.get("size", 0),
                    "size_fmt": _format_bytes(e.get("size", 0)),
                    "action": e.get("action"),
                    "frames": e.get("frames_json"),
                }
                for e in reversed(last_5)
            ],
            "root_cause": root_cause,
        })

    return {"mode": "oom", "detected": True, "events": results}


def analyze_compare(db_path: str, ref_path: str) -> Dict[str, Any]:
    if not os.path.exists(ref_path):
        return {"mode": "compare", "error": f"参考文件不存在: {ref_path}"}

    a_overview = q.get_device_overview(db_path)
    b_overview = q.get_device_overview(ref_path)

    a_reserved = sum(d["reserved_bytes"] for d in a_overview)
    a_allocated = sum(d["allocated_bytes"] for d in a_overview)
    a_segments = sum(d["segment_count"] for d in a_overview)
    a_frag = round((a_reserved - a_allocated) / a_reserved * 100, 1) if a_reserved > 0 else 0

    b_reserved = sum(d["reserved_bytes"] for d in b_overview)
    b_allocated = sum(d["allocated_bytes"] for d in b_overview)
    b_segments = sum(d["segment_count"] for d in b_overview)
    b_frag = round((b_reserved - b_allocated) / b_reserved * 100, 1) if b_reserved > 0 else 0

    a_oom = len(q.get_oom_events(db_path))
    b_oom = len(q.get_oom_events(ref_path))

    def _diff(a_val, b_val):
        return b_val - a_val

    def _trend(diff):
        if diff > 0:
            return "[+]"
        elif diff < 0:
            return "[-]"
        return "[=]"

    metrics = [
        {
            "name": "Reserved",
            "a": a_reserved, "a_fmt": _format_bytes(a_reserved),
            "b": b_reserved, "b_fmt": _format_bytes(b_reserved),
            "diff": _diff(a_reserved, b_reserved),
            "diff_fmt": _format_bytes(abs(_diff(a_reserved, b_reserved))),
            "trend": _trend(_diff(a_reserved, b_reserved)),
        },
        {
            "name": "Allocated",
            "a": a_allocated, "a_fmt": _format_bytes(a_allocated),
            "b": b_allocated, "b_fmt": _format_bytes(b_allocated),
            "diff": _diff(a_allocated, b_allocated),
            "diff_fmt": _format_bytes(abs(_diff(a_allocated, b_allocated))),
            "trend": _trend(_diff(a_allocated, b_allocated)),
        },
        {
            "name": "碎片率",
            "a": a_frag, "a_fmt": f"{a_frag}%",
            "b": b_frag, "b_fmt": f"{b_frag}%",
            "diff": round(b_frag - a_frag, 1),
            "diff_fmt": f"{abs(round(b_frag - a_frag, 1))}%",
            "trend": _trend(b_frag - a_frag),
        },
        {
            "name": "Segment 数",
            "a": a_segments, "a_fmt": str(a_segments),
            "b": b_segments, "b_fmt": str(b_segments),
            "diff": _diff(a_segments, b_segments),
            "diff_fmt": str(abs(_diff(a_segments, b_segments))),
            "trend": _trend(_diff(a_segments, b_segments)),
        },
        {
            "name": "OOM 事件",
            "a": a_oom, "a_fmt": str(a_oom),
            "b": b_oom, "b_fmt": str(b_oom),
            "diff": _diff(a_oom, b_oom),
            "diff_fmt": str(abs(_diff(a_oom, b_oom))),
            "trend": _trend(_diff(a_oom, b_oom)),
        },
    ]

    conn_a = q._connect(db_path)
    try:
        conn_b = q._connect(ref_path)
        try:
            a_addrs = set(r[0] for r in conn_a.execute("SELECT address FROM segments").fetchall())
            b_addrs = set(r[0] for r in conn_b.execute("SELECT address FROM segments").fetchall())
            new_addrs = b_addrs - a_addrs

            new_segments = []
            for addr in list(new_addrs)[:5]:
                row = conn_b.execute(
                    "SELECT address, total_size, cs.frames_json FROM segments s "
                    "LEFT JOIN call_stacks cs ON s.stack_id = cs.id WHERE s.address = ?",
                    (addr,),
                ).fetchone()
                if row:
                    new_segments.append({
                        "address": hex(row[0]),
                        "size": row[1],
                        "size_fmt": _format_bytes(row[1]),
                        "frames": row[2],
                    })

            grown_segments = []
            for addr in a_addrs & b_addrs:
                a_row = conn_a.execute("SELECT total_size FROM segments WHERE address = ?", (addr,)).fetchone()
                b_row = conn_b.execute("SELECT total_size FROM segments WHERE address = ?", (addr,)).fetchone()
                if a_row and b_row and b_row[0] > a_row[0]:
                    grown_segments.append({
                        "address": hex(addr),
                        "size_a": a_row[0],
                        "size_a_fmt": _format_bytes(a_row[0]),
                        "size_b": b_row[0],
                        "size_b_fmt": _format_bytes(b_row[0]),
                        "growth": b_row[0] - a_row[0],
                        "growth_fmt": _format_bytes(b_row[0] - a_row[0]),
                    })

            grown_segments.sort(key=lambda x: x["growth"], reverse=True)
            grown_segments = grown_segments[:3]
        finally:
            conn_b.close()
    finally:
        conn_a.close()

    one_line = (
        f"Reserved {_trend(_diff(a_reserved, b_reserved))} "
        f"{_format_bytes(abs(_diff(a_reserved, b_reserved)))} "
        f"({round(_diff(a_reserved, b_reserved) / a_reserved * 100, 1) if a_reserved > 0 else 0}%), "
        f"碎片率 {_trend(b_frag - a_frag)} {abs(round(b_frag - a_frag, 1))}%"
    )

    return {
        "mode": "compare",
        "one_line": one_line,
        "metrics": metrics,
        "new_segments": new_segments,
        "new_segment_count": len(new_addrs),
        "grown_segments": grown_segments,
    }


def generate_html_report(results: Dict[str, Any], db_path: str, output_path: str):
    """生成自包含 HTML 报告"""
    overview = results.get("overview", {})
    devices = overview.get("devices", [])
    summary = overview.get("summary", {})
    health = overview.get("health", {})

    peak = results.get("peak", {})
    fragment = results.get("fragment", {})
    leak = results.get("leak", {})
    oom = results.get("oom", {})
    compare = results.get("compare", {})

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _echarts_path = os.path.join(_script_dir, "echarts.min.js")
    if os.path.exists(_echarts_path):
        with open(_echarts_path, "r", encoding="utf-8") as _f:
            _echarts_js = _f.read()
        _echarts_tag = f"<script>{_echarts_js}</script>"
    else:
        _echarts_tag = '<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>'

    devices_json = json.dumps(devices, ensure_ascii=False, default=str).replace("</", "<\\/")
    peak_json = json.dumps(peak, ensure_ascii=False, default=str).replace("</", "<\\/")
    fragment_json = json.dumps(fragment, ensure_ascii=False, default=str).replace("</", "<\\/")
    leak_json = json.dumps(leak, ensure_ascii=False, default=str).replace("</", "<\\/")
    oom_json = json.dumps(oom, ensure_ascii=False, default=str).replace("</", "<\\/")
    compare_json = json.dumps(compare, ensure_ascii=False, default=str).replace("</", "<\\/")

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ascend NPU Memory Snapshot 分析报告</title>
{_echarts_tag}
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; color: #333; }}
.nav {{ position: sticky; top: 0; background: #1a1a2e; padding: 12px 24px; z-index: 100; display: flex; gap: 20px; flex-wrap: wrap; }}
.nav a {{ color: #ccc; text-decoration: none; font-size: 14px; }}
.nav a:hover {{ color: #fff; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
.header {{ background: #fff; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.header h1 {{ font-size: 22px; margin-bottom: 8px; }}
.header .meta {{ color: #888; font-size: 13px; }}
.health-card {{ display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 600; margin: 12px 0; }}
.health-ok {{ background: #e8f5e9; color: #2e7d32; }}
.health-warn {{ background: #fff3e0; color: #e65100; }}
.health-err {{ background: #ffebee; color: #c62828; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
.card .label {{ font-size: 13px; color: #888; margin-bottom: 8px; }}
.card .value {{ font-size: 24px; font-weight: 700; }}
.card .status {{ font-size: 12px; margin-top: 4px; }}
.section {{ background: #fff; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 18px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #eee; }}
.chart {{ width: 100%; height: 450px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }}
th {{ background: #f8f9fa; font-weight: 600; }}
tr:hover {{ background: #f8f9fa; }}
.insight {{ background: #f0f7ff; border-left: 3px solid #1976d2; padding: 12px 16px; margin-top: 16px; border-radius: 0 6px 6px 0; font-size: 14px; }}
.suggestions {{ display: grid; gap: 12px; }}
.suggestion {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }}
.suggestion.high {{ border-left: 4px solid #c62828; }}
.suggestion.medium {{ border-left: 4px solid #e65100; }}
.suggestion .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px; }}
.tag-high {{ background: #ffebee; color: #c62828; }}
.tag-medium {{ background: #fff3e0; color: #e65100; }}
</style>
</head>
<body>

<div class="nav">
    <a href="#overview">总览</a>
    <a href="#devices">设备对比</a>
    <a href="#peak">时序曲线</a>
    <a href="#fragment">碎片分析</a>
    <a href="#leak">泄漏检测</a>
    <a href="#oom">OOM</a>
    <a href="#attribution">堆栈归因</a>
    <a href="#suggestions">建议</a>
</div>

<div class="container">

<div class="header" id="overview">
    <h1>Ascend NPU Memory Snapshot 分析报告</h1>
    <div class="meta">文件: {html.escape(os.path.basename(db_path))} | 分析时间: <span id="analysis-time"></span></div>
    <div class="health-card health-{health.get('level', 'warn').lower().replace('需关注', 'warn').replace('严重', 'err').replace('健康', 'ok')}">
        {health.get('icon', '?')} 健康状态: {health.get('text', '未知')}
    </div>
</div>

<div class="cards" id="cards">
    <div class="card"><div class="label">Reserved</div><div class="value">{summary.get('reserved_fmt', '-')}</div></div>
    <div class="card"><div class="label">Allocated</div><div class="value">{summary.get('allocated_fmt', '-')}</div></div>
    <div class="card"><div class="label">碎片率</div><div class="value">{summary.get('frag_pct', '-')}%</div></div>
    <div class="card"><div class="label">Segment 数</div><div class="value">{summary.get('segment_count', '-')}</div></div>
    <div class="card"><div class="label">OOM 事件</div><div class="value">{overview.get('oom_count', '-')}</div></div>
</div>

<div class="section" id="devices">
    <h2>设备内存概览</h2>
    <div class="chart" id="chart-devices"></div>
</div>

<div class="section" id="peak">
    <h2>内存时序曲线 &amp; 峰值分析</h2>
    <div class="chart" id="chart-peak"></div>
    <div id="peak-content"></div>
</div>

<div class="section" id="fragment">
    <h2>碎片分析</h2>
    <div class="chart" id="chart-fragment"></div>
</div>

<div class="section" id="leak">
    <h2>泄漏检测</h2>
    <div id="leak-content"></div>
</div>

<div class="section" id="oom">
    <h2>OOM 分析</h2>
    <div id="oom-content"></div>
</div>

<div class="section" id="attribution">
    <h2>堆栈归因</h2>
    <div class="chart" id="chart-attribution"></div>
</div>

<div class="section" id="suggestions">
    <h2>优化建议</h2>
    <div class="suggestions" id="suggestions-content"></div>
</div>

</div>

<script>
document.getElementById('analysis-time').textContent = new Date().toLocaleString();

var devicesData = {devices_json};
var peakData = {peak_json};
var fragmentData = {fragment_json};
var leakData = {leak_json};
var oomData = {oom_json};
var compareData = {compare_json};

(function() {{
    var devNames = devicesData.map(function(d) {{ return 'Device ' + d.device_index; }});
    var reserved = devicesData.map(function(d) {{ return (d.reserved_bytes / (1024*1024*1024)).toFixed(2); }});
    var allocated = devicesData.map(function(d) {{ return (d.allocated_bytes / (1024*1024*1024)).toFixed(2); }});
    var active = devicesData.map(function(d) {{ return (d.active_bytes / (1024*1024*1024)).toFixed(2); }});

    var chart = echarts.init(document.getElementById('chart-devices'));
    chart.setOption({{
        tooltip: {{ trigger: 'axis' }},
        legend: {{ data: ['Reserved', 'Allocated', 'Active'] }},
        xAxis: {{ type: 'category', data: devNames }},
        yAxis: {{ type: 'value', name: 'GB' }},
        series: [
            {{ name: 'Reserved', type: 'bar', data: reserved, itemStyle: {{ color: '#5470c6' }} }},
            {{ name: 'Allocated', type: 'bar', data: allocated, itemStyle: {{ color: '#91cc75' }} }},
            {{ name: 'Active', type: 'bar', data: active, itemStyle: {{ color: '#fac858' }} }}
        ]
    }});
}})();

(function() {{
    var peakDevices = peakData.devices || [];
    var devNames = peakDevices.map(function(d) {{ return 'Device ' + d.device_index; }});
    var peakReserved = peakDevices.map(function(d) {{ return (d.peak?.reserved || 0) / (1024*1024*1024); }});
    var baseReserved = peakDevices.map(function(d) {{ return (d.baseline?.reserved || 0) / (1024*1024*1024); }});

    var chart = echarts.init(document.getElementById('chart-peak'));
    chart.setOption({{
        title: {{ text: '内存峰值分析 (按设备)', subtext: '基线: 各设备时间线前 10% 位置' }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ data: ['峰值 Reserved', '基线 Reserved'] }},
        xAxis: {{ type: 'category', data: devNames }},
        yAxis: {{ type: 'value', name: 'GB' }},
        series: [
            {{ name: '峰值 Reserved', type: 'bar', data: peakReserved, itemStyle: {{ color: '#c62828' }} }},
            {{ name: '基线 Reserved', type: 'bar', data: baseReserved, itemStyle: {{ color: '#5470c6' }} }}
        ]
    }});
}})();

(function() {{
    var peakHtml = '';
    var deltas = peakData.deltas || {{}};
    peakHtml += '<div class="insight" style="margin-bottom:16px"><strong>总体增幅</strong>: ';
    peakHtml += '全局峰值 ' + (peakData.peak?.reserved_fmt || 'N/A') + ' ';
    peakHtml += '(<span style="color:#c62828;font-weight:700">+' + (deltas.reserved_fmt || 'N/A') + (deltas.reserved_pct != null ? ', +' + deltas.reserved_pct + '%' : '') + '</span>)</div>';

    var peakDevices = peakData.devices || [];
    if (peakDevices.length > 0) {{
        peakHtml += '<h3>各设备峰值详情</h3>';
        peakHtml += '<table><tr><th>设备</th><th>峰值 Reserved</th><th>基线 Reserved</th><th>增幅</th><th>峰值 trace_index</th></tr>';
        peakDevices.forEach(function(d) {{
            var deltas = d.deltas || {{}};
            peakHtml += '<tr><td>Device ' + d.device_index + '</td><td style="font-weight:600">' + (d.peak?.reserved_fmt || 'N/A') + '</td><td>' + (d.baseline?.reserved_fmt || 'N/A') + '</td><td style="color:#c62828;font-weight:600">+' + (deltas.reserved_fmt || 'N/A') + (deltas.reserved_pct != null ? ' (+' + deltas.reserved_pct + '%)' : '') + '</td><td>' + (d.peak_trace_index || 'N/A') + '</td></tr>';
        }});
        peakHtml += '</table>';
    }}

    if (peakData.peak_alloc_events && peakData.peak_alloc_events.top && peakData.peak_alloc_events.top.length > 0) {{
        peakHtml += '<h3 style="margin-top:16px">峰值归因: 最大的 Segment 分配操作 (segment_alloc / segment_map)</h3>';
        peakHtml += '<p style="color:#888;font-size:13px;margin-bottom:8px">以下操作分配了最大的 segment，是峰值内存的主要贡献者。调用路径指向触发这些分配的代码。</p>';
        peakHtml += '<table><tr><th>#</th><th>大小</th><th>操作</th><th>地址</th><th>调用路径</th></tr>';
        peakData.peak_alloc_events.top.forEach(function(evt, i) {{
            peakHtml += '<tr><td>' + (i+1) + '</td><td style="font-weight:600">' + (evt.size_fmt || 'N/A') + '</td><td>' + (evt.action || '') + '</td><td style="font-family:monospace;font-size:12px">' + (evt.addr || 'N/A') + '</td><td style="font-size:12px;color:#555;max-width:400px;word-break:break-all">' + (evt.call_path || '') + '</td></tr>';
        }});
        peakHtml += '</table>';
    }}

    if (peakData.peak_blocks && peakData.peak_blocks.top && peakData.peak_blocks.top.length > 0) {{
        peakHtml += '<h3 style="margin-top:16px">峰值归因: 最大的活跃 Block (active_allocated)</h3>';
        peakHtml += '<p style="color:#888;font-size:13px;margin-bottom:8px">以下 block 是峰值时刻占用内存最大的 tensor 分配。调用路径指向创建这些 tensor 的代码。</p>';
        peakHtml += '<table><tr><th>#</th><th>大小</th><th>请求大小</th><th>调用路径</th></tr>';
        peakData.peak_blocks.top.forEach(function(blk, i) {{
            peakHtml += '<tr><td>' + (i+1) + '</td><td style="font-weight:600">' + (blk.size_fmt || 'N/A') + '</td><td>' + (blk.requested_fmt || 'N/A') + '</td><td style="font-size:12px;color:#555;max-width:400px;word-break:break-all">' + (blk.call_path || '') + '</td></tr>';
        }});
        peakHtml += '</table>';
    }}

    if (!peakData.peak_alloc_events && !peakData.peak_blocks) {{
        peakHtml += '<p>无峰值归因数据</p>';
    }}
    document.getElementById('peak-content').innerHTML = peakHtml;
}})();

(function() {{
    var topFrag = fragmentData.top_fragmented || [];
    var names = topFrag.map(function(s) {{ return s.address ? '0x' + s.address.toString(16).substring(0, 8) + '...' : 'N/A'; }});
    var used = topFrag.map(function(s) {{ return (100 - (s.frag_pct || 0)).toFixed(1); }});
    var waste = topFrag.map(function(s) {{ return (s.frag_pct || 0).toFixed(1); }});

    var chart = echarts.init(document.getElementById('chart-fragment'));
    chart.setOption({{
        title: {{ text: '碎片化最严重的 Segment TOP 5', subtext: '整体碎片率: ' + (fragmentData.overall?.frag_pct || 'N/A') + '%' }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ data: ['使用率', '碎片率'] }},
        xAxis: {{ type: 'category', data: names }},
        yAxis: {{ type: 'value', name: '%', max: 100 }},
        series: [
            {{ name: '使用率', type: 'bar', stack: 'total', data: used, itemStyle: {{ color: '#91cc75' }} }},
            {{ name: '碎片率', type: 'bar', stack: 'total', data: waste, itemStyle: {{ color: '#ee6666' }} }}
        ]
    }});
}})();

(function() {{
    var leakHtml = '';
    if (leakData.risk) {{
        leakHtml += '<div class="health-card health-' + (leakData.risk.level === '高风险' ? 'err' : leakData.risk.level === '中风险' ? 'warn' : 'ok') + '">';
        leakHtml += leakData.risk.icon + ' 风险等级: ' + leakData.risk.level + '</div>';
        leakHtml += '<p style="margin-top:12px">单调增长: ' + (leakData.monotonic_growth?.detected ? '是' : '否') + '</p>';
        leakHtml += '<p>长生命周期 Block: ' + (leakData.long_lived_blocks || 0) + ' 个</p>';
    }}
    if (leakData.suspects && leakData.suspects.length > 0) {{
        leakHtml += '<h3 style="margin-top:16px">泄漏嫌疑 TOP 3</h3>';
        leakData.suspects.forEach(function(s, i) {{
            leakHtml += '<div class="suggestion high" style="margin-top:8px"><strong>#' + (i+1) + '</strong> ';
            leakHtml += '累计: ' + (s.total_size_fmt || 'N/A') + ' | ' + (s.block_count || 0) + ' 个 block';
            leakHtml += '<br><span style="color:#888;font-size:13px">' + (s.key_path || '') + '</span></div>';
        }});
    }}
    if (!leakData.risk) {{
        leakHtml = '<p>无泄漏检测数据</p>';
    }}
    document.getElementById('leak-content').innerHTML = leakHtml;
}})();

(function() {{
    var oomHtml = '';
    if (oomData.detected && oomData.events) {{
        oomData.events.forEach(function(evt) {{
            oomHtml += '<div class="health-card health-err" style="margin-bottom:12px">OOM 事件: Device ' + evt.device + ', trace_index = ' + evt.oom_index + '</div>';
            oomHtml += '<p>OOM 时可用内存: ' + (evt.device_free_fmt || 'N/A') + '</p>';
            oomHtml += '<p>根因推断: ' + (evt.root_cause || 'N/A') + '</p>';
            if (evt.last_5_allocations && evt.last_5_allocations.length > 0) {{
                oomHtml += '<h4 style="margin-top:12px">OOM 前最后分配:</h4><table><tr><th>大小</th><th>类型</th></tr>';
                evt.last_5_allocations.forEach(function(a) {{
                    oomHtml += '<tr><td>' + (a.size_fmt || 'N/A') + '</td><td>' + (a.action || '') + '</td></tr>';
                }});
                oomHtml += '</table>';
            }}
        }});
    }} else {{
        oomHtml = '<p>未检测到 OOM 事件</p>';
    }}
    document.getElementById('oom-content').innerHTML = oomHtml;
}})();

(function() {{
    var chart = echarts.init(document.getElementById('chart-attribution'));
    chart.setOption({{
        title: {{ text: '堆栈归因 TOP 10' }},
        tooltip: {{ trigger: 'axis' }},
        xAxis: {{ type: 'value', name: 'GB' }},
        yAxis: {{ type: 'category', data: ['数据加载中...'] }},
        series: [{{ type: 'bar', data: [0] }}]
    }});
}})();

(function() {{
    var suggestions = [];
    var fragPct = fragmentData.overall?.frag_pct || 0;
    var leakRisk = leakData.risk?.level || '';
    var hasOom = oomData.detected || false;

    if (hasOom) {{
        suggestions.push({{ level: 'high', tag: 'OOM', text: '减小 batch_size 或启用 gradient checkpointing 以降低峰值内存。', ref: '' }});
    }}
    if (fragPct > 15) {{
        suggestions.push({{ level: 'high', tag: '碎片', text: '碎片率 ' + fragPct + '% 偏高，建议定期调用 torch.npu.empty_cache() 释放空闲 segment。', ref: '' }});
    }}
    if (leakRisk === '高风险') {{
        suggestions.push({{ level: 'high', tag: '泄漏', text: '检测到高风险疑似泄漏，建议检查 DDP/FSDP 的 gradient bucket 配置。', ref: 'https://pytorch.org/docs/stable/ddp.html' }});
    }}
    if (fragPct > 5 && fragPct <= 15) {{
        suggestions.push({{ level: 'medium', tag: '碎片', text: '碎片率 ' + fragPct + '% 略高，可考虑增大 PYTORCH_NPU_ALLOC_CONF 中的 expandable_segments 参数。', ref: '' }});
    }}
    if (leakRisk === '中风险') {{
        suggestions.push({{ level: 'medium', tag: '泄漏', text: '检测到中风险疑似泄漏，建议检查循环中 tensor 是否正确 detach 或使用 torch.no_grad()。', ref: '' }});
    }}
    if (suggestions.length === 0) {{
        suggestions.push({{ level: 'medium', tag: '监控', text: '内存状态正常，建议在训练脚本中增加周期性 snapshot 采集以便后续趋势对比。', ref: '' }});
    }}

    var html = '';
    suggestions.forEach(function(s) {{
        html += '<div class="suggestion ' + s.level + '"><span class="tag tag-' + s.level + '">' + s.tag + '</span>' + s.text;
        if (s.ref) html += ' <a href="' + s.ref + '" target="_blank">参考</a>';
        html += '</div>';
    }});
    document.getElementById('suggestions-content').innerHTML = html;
}})();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Ascend NPU Memory Snapshot 高层分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("db_path", help="SQLite 数据库路径")
    parser.add_argument("--mode", "-m", required=True,
                        choices=["overview", "peak", "fragment", "leak", "oom", "compare", "all"],
                        help="分析模式")
    parser.add_argument("--ref", help="对比参考 DB 路径 (compare 模式)")
    parser.add_argument("--device", "-d", type=int, help="指定分析的 device")
    parser.add_argument("--output", "-o", help="HTML 报告输出路径 (all 模式)")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")

    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"错误: 文件不存在: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    results: Dict[str, Any] = {}

    try:
        if args.mode == "overview":
            results["overview"] = analyze_overview(args.db_path)

        elif args.mode == "peak":
            results["peak"] = analyze_peak(args.db_path, args.device)

        elif args.mode == "fragment":
            results["fragment"] = analyze_fragment(args.db_path, args.device)

        elif args.mode == "leak":
            results["leak"] = analyze_leak(args.db_path, args.device)

        elif args.mode == "oom":
            results["oom"] = analyze_oom(args.db_path, args.device)

        elif args.mode == "compare":
            if not args.ref:
                print("错误: compare 模式需要 --ref 参数", file=sys.stderr)
                sys.exit(1)
            results["compare"] = analyze_compare(args.db_path, args.ref)

        elif args.mode == "all":
            results["overview"] = analyze_overview(args.db_path)
            results["peak"] = analyze_peak(args.db_path, args.device)
            results["fragment"] = analyze_fragment(args.db_path, args.device)
            results["leak"] = analyze_leak(args.db_path, args.device)
            results["oom"] = analyze_oom(args.db_path, args.device)

            output_path = args.output or (os.path.splitext(args.db_path)[0] + "_report.html")
            generate_html_report(results, args.db_path, output_path)
            results["report_path"] = output_path

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        elif args.mode != "all":
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            report_path = results.get("report_path", "")
            print(f"报告已生成: {report_path}")
            overview = results.get("overview", {})
            summary = overview.get("summary", {})
            health = overview.get("health", {})
            print(f"\n{health.get('icon', '?')} 健康状态: {health.get('text', '未知')}")
            print(f"Reserved: {summary.get('reserved_fmt', '-')}")
            print(f"Allocated: {summary.get('allocated_fmt', '-')}")
            print(f"碎片率: {summary.get('frag_pct', '-')}%")
            print(f"Segment 数: {summary.get('segment_count', '-')}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
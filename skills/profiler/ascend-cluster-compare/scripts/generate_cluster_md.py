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
"""
generate_cluster_md.py
基于两个集群提取的 JSON 数据生成 Markdown 集群比对报告（与 HTML 报告同口径）。

数据与结论来源:
1. 集群 JSON 内嵌 advanced_analysis 字段（cluster_data_extractor.py 提取单集群
   JSON 时同步执行 free_analysis / communication_bottleneck，并已写入中文
   「分析与判断」字段）;
2. --advanced 外部 JSON（run_advanced_analysis.py 输出，含双集群时间拆解对比）。
两者自动合并（复用 generate_cluster_report.merge_advanced），无进阶数据时输出引导说明。

设计约束:
- 核心指标计算复用 generate_cluster_report.py 的口径（us_to_ms / safe_div /
  get_eff_bw / classify_bw_change），保证 HTML 与 MD 结论一致;
- 进阶分析「分析与判断」「综合判断」复用 advanced_insights.py，
  与单集群 JSON 内嵌字段、HTML 报告三者同源同口径;
- 全部分析、成因、推断原因、判断文字均为中文输出。

用法:
    python generate_cluster_md.py --data-a a.json --data-b b.json --output report.md \
        [--advanced advanced_analysis.json] [--top-ranks 20]
"""
import argparse
import json
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 复用 HTML 报告的指标计算与进阶分析合并逻辑（口径一致的关键）
import generate_cluster_report as _rep

try:
    import advanced_insights as _insights
except Exception:  # 缺失时仅少「分析与判断」块，其余章节不受影响
    _insights = None


# ==================== 展示常量（中文） ====================

STATUS_TXT = {
    "success": "已完成", "cached": "缓存结果", "failed": "执行失败",
    "not_available": "不可用", "skipped": "已跳过", "pending": "未执行",
}
VERDICT_TXT = {
    "real_severe": "真实劣化", "real_minor": "轻微劣化",
    "statistical": "口径差异", "none": "一致",
}
BW_VERDICT_TEXT = {
    "real_severe": "真实链路劣化：有效吞吐显著下降，需排查硬件",
    "real_minor": "轻微吞吐劣化：有效吞吐小幅下降，建议关注",
    "statistical": "均值带宽下降系流量构成差异，有效吞吐一致，非链路劣化",
    "none": "带宽水平基本一致",
}


# ==================== Markdown 工具 ====================

def _fmt(v):
    """None → '—'，其余转字符串。"""
    return "—" if v is None else str(v)


def _md_cell(v):
    """单元格转义: 竖线与换行会破坏 markdown 表格。"""
    return str(v).replace("|", "\\|").replace("\r\n", " ").replace("\n", " ")


def md_table(headers, rows):
    """headers + rows(list[list]) → markdown 表格文本。"""
    if not rows:
        return ""
    head = "| " + " | ".join(_md_cell(h) for h in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def _avg(rows, key):
    return sum(float(r[key]) for r in rows.values()) / len(rows) if rows else 0


# ==================== 进阶分析 MD 渲染 ====================

def render_feature_md(feat, show_desc=True):
    """单个进阶分析 feature → MD 文本（状态 + 数据表 + 分析与判断）。"""
    name = feat.get("name", "")
    status = feat.get("status", "pending")
    data = feat.get("data") or {}
    lines = []
    lines.append(f"- 状态：{STATUS_TXT.get(status, status)}")
    desc = feat.get("desc", "")
    if show_desc and desc:
        lines.append(f"- 说明：{desc}")
    reason = feat.get("reason", "")
    if reason and status not in ("success", "cached"):
        lines.append(f"- 原因：{reason}")
    try:
        if status in ("success", "cached"):
            if name == _rep.FEAT_NAME_TIME_CMP:
                overall = data.get("overall") or {}
                metrics = data.get("metrics") or []
                rows = [[m, _fmt((overall.get(m) or {}).get("base_avg_ms")),
                         _fmt((overall.get(m) or {}).get("cur_avg_ms")),
                         _fmt((overall.get(m) or {}).get("diff_avg_ms"))]
                        for m in metrics if (overall.get(m) or {})]
                if rows:
                    lines += ["", "总体指标对比（ms，基准 → 对比）：", "",
                              md_table(["指标", "基准均值", "对比均值", "差值"], rows)]
                tops = data.get("top_ranks") or []
                rc = data.get("rank_col") or "Rank"
                trows = []
                for t in tops[:10]:
                    diff = next((t.get(f"{m}_diff_ms") for m in metrics
                                 if t.get(f"{m}_diff_ms") is not None), None)
                    trows.append([_fmt(t.get("rank")), _fmt(t.get("step_count")), _fmt(diff)])
                if trows:
                    lines += ["", f"Top 差异 {rc}（按 Stage 差值降序，ms）：", "",
                              md_table([rc, "Step 数", "差值均值"], trows)]
            elif name == _rep.FEAT_NAME_FREE:
                reasons = data.get("reasons") or []
                rows = [[_md_cell(r0.get("reason", "-")), _fmt(r0.get("count", "-")),
                         _fmt(r0.get("total_ms", "-")), _fmt(r0.get("pct", 0))]
                        for r0 in reasons[:10]]
                if rows:
                    lines += ["", f"空闲时间成因聚合（Top 10，总空闲 {_fmt(data.get('total_ms'))} ms）：", "",
                              md_table(["成因（Reason）", "次数", "总时长 (ms)", "占比 %"], rows)]
                else:
                    lines += ["", "未采集到空闲片段数据。"]
            elif name == _rep.FEAT_NAME_COMM_BN:
                items = data.get("items") or []
                rows = [[_fmt(it.get("op", "-")), _fmt(it.get("slow_rank", "-")),
                         _fmt(it.get("fast_rank", "-")), _fmt(it.get("duration_ms", "-")),
                         _md_cell(it.get("reason", ""))]
                        for it in items[:10]]
                if rows:
                    lines += ["", "通信瓶颈 Top 10（慢/快 Rank 耗时差，ms）：", "",
                              md_table(["通信算子", "慢 Rank", "快 Rank", "耗时差 (ms)", "推断原因"], rows)]
                else:
                    lines += ["", "未识别到显著通信瓶颈。"]
            else:
                lines += ["", "结果数据为空。"]
    except Exception as exc:  # 单 feature 渲染容错
        lines += ["", f"数据渲染失败：{exc}"]
    # 「分析与判断」: 与单集群 JSON 内嵌字段 / HTML 报告同口径
    if _insights is not None and status in ("success", "cached"):
        try:
            items = _insights.feature_insights(feat)
        except Exception:
            items = []
        if items:
            lines += ["", "**分析与判断**", ""]
            lines += [f"- {x}" for x in items]
    return "\n".join(lines)


def render_advanced_md(merged):
    """合并后的进阶分析结构 → MD 章节（含 A/B 综合判断）。"""
    feats = (merged or {}).get("features") or []
    tool = (merged or {}).get("tool") or {}
    clusters = (merged or {}).get("clusters") or {}
    lines = []
    meta = []
    if tool.get("version"):
        meta.append(f"msprof-analyze {tool['version']}")
    for key in ("A", "B"):
        c = clusters.get(key) or {}
        if c.get("data_dir"):
            base = os.path.basename(str(c["data_dir"]).rstrip("\\").rstrip("/"))
            meta.append(f"集群{key} `{base}`")
    if meta:
        lines.append(f"> 数据来源：{' · '.join(meta)}")
        lines.append("")
    if not feats:
        lines.append("本次无进阶分析数据。默认情况下，提取器（cluster_data_extractor.py）生成单集群 JSON 时"
                     "会同步执行 free_analysis / communication_bottleneck 并嵌入 JSON 的 advanced_analysis "
                     "字段；出现此提示说明提取时使用了 `--no-advanced` 或该环境 msprof-analyze 不可用。"
                     "补救方式：去掉 `--no-advanced` 重新提取集群 JSON，或运行 `run_advanced_analysis.py` "
                     "生成进阶分析 JSON 后通过 `--advanced` 传入重新生成本报告。")
        return "\n".join(lines)

    # 按 feature 名称分组（保持首次出现顺序），同名多条记录渲染为 A/B 对照
    groups = []
    for f in feats:
        for g in groups:
            if g[0] == f.get("name", ""):
                g[1].append(f)
                break
        else:
            groups.append((f.get("name", ""), [f]))

    for _, group in groups:
        lines.append(f"#### {group[0].get('label') or group[0].get('name', '')}")
        lines.append("")
        if len(group) == 1:
            feat = group[0]
            tag = {"A": "集群A（基准）", "B": "集群B（对比）"}.get(feat.get("cluster"))
            if tag:
                lines.append(f"**{tag}**")
                lines.append("")
            lines.append(render_feature_md(feat))
        else:
            for i, feat in enumerate(group):
                tag = {"A": "集群A（基准）", "B": "集群B（对比）"}.get(feat.get("cluster"),
                                                                      f"结果 {i + 1}")
                lines.append(f"**{tag}**")
                lines.append("")
                lines.append(render_feature_md(feat, show_desc=(i == 0)))
                lines.append("")
            # A/B 综合判断（双侧均有成功数据时输出）
            if _insights is not None and len(group) >= 2:
                fa = next((f for f in group if f.get("cluster") == "A"), group[0])
                fb = next((f for f in group if f.get("cluster") == "B"), group[-1])
                try:
                    citems = _insights.compare_insights(fa, fb)
                except Exception:
                    citems = []
                if citems:
                    lines.append("**综合判断（基准 vs 对比）**")
                    lines.append("")
                    lines += [f"- {x}" for x in citems]
        lines.append("")
    return "\n".join(lines).rstrip()


# ==================== 比对主文档 ====================

def generate_md(data_a, data_b, advanced_json=None, top_ranks=20):
    sa = data_a.get("step_summary", {})
    sb = data_b.get("step_summary", {})
    steps = sorted(set(list(sa.keys()) + list(sb.keys())), key=lambda x: int(x) if x.isdigit() else x)
    ra = data_a.get("rank_summary", {})
    rb = data_b.get("rank_summary", {})

    avg_a_stage = _avg(sa, "avg_stage")
    avg_b_stage = _avg(sb, "avg_stage")
    avg_a_compute = _avg(sa, "avg_compute")
    avg_b_compute = _avg(sb, "avg_compute")
    avg_a_comm = _avg(sa, "avg_comm")
    avg_b_comm = _avg(sb, "avg_comm")
    avg_a_free = _avg(sa, "avg_free")
    avg_b_free = _avg(sb, "avg_free")
    avg_a_overlap = _avg(sa, "avg_overlap")
    avg_b_overlap = _avg(sb, "avg_overlap")

    delta_stage = avg_b_stage - avg_a_stage
    delta_compute = avg_b_compute - avg_a_compute
    delta_comm = avg_b_comm - avg_a_comm
    delta_free = avg_b_free - avg_a_free
    stage_delta_pct = _rep.safe_div(delta_stage, avg_a_stage) * 100 if avg_a_stage else 0

    # 贡献度: Stage 差异不显著（<1%）时降级为 None
    stage_significant = abs(stage_delta_pct) >= 1.0
    if stage_significant:
        compute_contrib = _rep.safe_div(delta_compute, delta_stage) * 100
        comm_contrib = _rep.safe_div(delta_comm, delta_stage) * 100
        free_contrib = _rep.safe_div(delta_free, delta_stage) * 100
    else:
        compute_contrib = comm_contrib = free_contrib = None

    total_a = avg_a_stage if avg_a_stage > 0 else 1
    total_b = avg_b_stage if avg_b_stage > 0 else 1
    a_comp_pct = round(_rep.safe_div(avg_a_compute, total_a) * 100, 1)
    a_comm_pct = round(_rep.safe_div(avg_a_comm, total_a) * 100, 1)
    a_free_pct = round(_rep.safe_div(avg_a_free, total_a) * 100, 1)
    b_comp_pct = round(_rep.safe_div(avg_b_compute, total_b) * 100, 1)
    b_comm_pct = round(_rep.safe_div(avg_b_comm, total_b) * 100, 1)
    b_free_pct = round(_rep.safe_div(avg_b_free, total_b) * 100, 1)

    # 带宽: 均值带宽（KPI）+ 有效吞吐（跨带宽类型均值）+ 逐类型判定
    bwa = data_a.get("comm_bandwidth", [])
    bwb = data_b.get("comm_bandwidth", [])
    bw_a_val = round(sum(float(b.get("avg_bw", b.get("bandwidth_size", 0)) or 0) for b in bwa) / len(bwa), 2) if bwa else 0
    bw_b_val = round(sum(float(b.get("avg_bw", b.get("bandwidth_size", 0)) or 0) for b in bwb) / len(bwb), 2) if bwb else 0
    bw_delta_pct = _rep.safe_div(bw_b_val - bw_a_val, bw_a_val) * 100 if bw_a_val else 0
    eff_a_val = round(sum(_rep.get_eff_bw(b) for b in bwa) / len(bwa), 2) if bwa else 0
    eff_b_val = round(sum(_rep.get_eff_bw(b) for b in bwb) / len(bwb), 2) if bwb else 0
    eff_delta_pct = _rep.safe_div(eff_b_val - eff_a_val, eff_a_val) * 100 if eff_a_val else 0

    rank_order = {"real_severe": 0, "real_minor": 1, "statistical": 2, "none": 3}
    bw_type_verdicts = []
    for t in sorted(set([b.get("transport_type", b.get("band_type", "?")) for b in bwa + bwb])):
        ea = next((b for b in bwa if b.get("transport_type", b.get("band_type", "?")) == t), None)
        eb = next((b for b in bwb if b.get("transport_type", b.get("band_type", "?")) == t), None)
        avg_drop = abs(_rep.safe_div(float(eb.get("avg_bw", 0) or 0) - float(ea.get("avg_bw", 0) or 0),
                                     float(ea.get("avg_bw", 0) or 0)) * 100) if (ea and eb) else 0
        eff_drop = abs(_rep.safe_div(_rep.get_eff_bw(eb) - _rep.get_eff_bw(ea),
                                     _rep.get_eff_bw(ea)) * 100) if (ea and eb and _rep.get_eff_bw(ea)) else 0
        verdict = _rep.classify_bw_change(avg_drop, eff_drop) if (ea and eb) else "none"
        bw_type_verdicts.append((t, avg_drop, eff_drop, verdict))
    bw_verdict = min((v[3] for v in bw_type_verdicts), key=lambda x: rank_order[x]) if bw_type_verdicts else "none"

    # 整体状态
    if stage_delta_pct > 10:
        status_text = "严重劣化"
    elif stage_delta_pct > 5:
        status_text = "中度劣化"
    elif stage_delta_pct > 0:
        status_text = "轻微劣化"
    else:
        status_text = "性能改善"

    # 诊断摘要
    diagnosis = (f"对比集群 B 的 Stage 总耗时{'增加' if delta_stage > 0 else '减少'} "
                 f"{abs(_rep.us_to_ms(delta_stage)):.1f} ms（{stage_delta_pct:+.1f}%）")
    if not stage_significant:
        diagnosis += "，整体差异不显著（<1%），分项贡献度不具统计意义；波动多为分项间相互转移。"
    elif comm_contrib is not None and comm_contrib > 50:
        diagnosis += f"，通信是主要{'劣化' if delta_comm > 0 else '改善'}来源（贡献度 {comm_contrib:.1f}%）。"
    elif compute_contrib is not None and compute_contrib > 50:
        diagnosis += f"，计算是主要{'劣化' if delta_compute > 0 else '改善'}来源（贡献度 {compute_contrib:.1f}%）。"
    else:
        diagnosis += "，劣化来源分散。"

    # 关键定位点
    key_points = []
    if delta_comm != 0:
        cs = f"（贡献度 {comm_contrib:.1f}%）" if comm_contrib is not None else ""
        v = ("通信为主要劣化来源" if (delta_comm > 0 and comm_contrib is not None and comm_contrib > 50)
             else "通信有改善" if delta_comm < 0 else "通信变化不显著")
        key_points.append(f"通信时间{'增加' if delta_comm > 0 else '减少'} {abs(_rep.us_to_ms(delta_comm)):.1f} ms{cs}——{v}")
    if a_comm_pct != b_comm_pct:
        if b_comm_pct > b_comp_pct and a_comm_pct <= a_comp_pct:
            dominant = "通信反超计算"
        elif b_comm_pct > a_comm_pct:
            dominant = "通信占比上升"
        elif b_comm_pct < a_comm_pct:
            dominant = "通信占比下降"
        else:
            dominant = "通信占比持平"
        key_points.append(f"负载转换：通信占比 {a_comm_pct}% → {b_comm_pct}%——{dominant}")
    if bw_verdict == "real_severe":
        key_points.append(f"有效吞吐下降 {abs(eff_delta_pct):.1f}%（{eff_a_val:.1f} → {eff_b_val:.1f} GB/s）——真实链路劣化，需排查硬件")
    elif bw_verdict == "real_minor":
        key_points.append(f"有效吞吐下降 {abs(eff_delta_pct):.1f}%（{eff_a_val:.1f} → {eff_b_val:.1f} GB/s）——轻微吞吐劣化，建议关注")
    elif bw_verdict == "statistical":
        key_points.append(f"均值带宽 {bw_delta_pct:+.1f}%（{bw_a_val:.1f} → {bw_b_val:.1f} GB/s），有效吞吐 {eff_delta_pct:+.1f}%——均值下降系流量构成差异，非链路劣化")
    if delta_free > 0 and (free_contrib is None or free_contrib > 10):
        fs = f"（贡献度 {free_contrib:.1f}%）" if free_contrib is not None else ""
        key_points.append(f"空闲时间增加 {_rep.us_to_ms(delta_free):.1f} ms{fs}——可能存在 Host 下发瓶颈")

    L = []
    L.append("# 集群性能比对报告")
    L.append("")
    L.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 集群A（基准）：`{data_a.get('data_dir', '?')}`")
    L.append(f"- 集群B（对比）：`{data_b.get('data_dir', '?')}`")
    L.append(f"- 规模：集群A {len(ra)} Rank / {len(sa)} Step，集群B {len(rb)} Rank / {len(sb)} Step")
    L.append("")

    # 一、综合结论
    L.append("## 一、综合结论")
    L.append("")
    L.append(f"**整体状态：{status_text}**")
    L.append("")
    L.append(diagnosis)
    L.append("")
    if key_points:
        L.append("**关键定位点**")
        L.append("")
        L += [f"- {p}" for p in key_points]
    else:
        L.append("- 未发现显著性能差异")
    L.append("")

    # 二、核心指标对比
    contrib_s = lambda c: ("—（Stage 差异不显著）" if c is None else f"{c:.1f}%")
    L.append("## 二、核心指标对比")
    L.append("")
    kpi_rows = [
        ["Stage 总耗时均值 (ms)", _rep.us_to_ms(avg_a_stage), _rep.us_to_ms(avg_b_stage),
         f"{_rep.fmt_signed(_rep.us_to_ms(delta_stage))}（{stage_delta_pct:+.1f}%）", f"整体{status_text}"],
        ["计算均值 (ms)", _rep.us_to_ms(avg_a_compute), _rep.us_to_ms(avg_b_compute),
         f"{_rep.fmt_signed(_rep.us_to_ms(delta_compute))}", f"贡献度 {contrib_s(compute_contrib)}"],
        ["通信均值 (ms)", _rep.us_to_ms(avg_a_comm), _rep.us_to_ms(avg_b_comm),
         f"{_rep.fmt_signed(_rep.us_to_ms(delta_comm))}", f"贡献度 {contrib_s(comm_contrib)}"],
        ["空闲均值 (ms)", _rep.us_to_ms(avg_a_free), _rep.us_to_ms(avg_b_free),
         f"{_rep.fmt_signed(_rep.us_to_ms(delta_free))}", f"贡献度 {contrib_s(free_contrib)}"],
        ["计算/通信/空闲占比 (A→B)", f"计算 {a_comp_pct}% · 通信 {a_comm_pct}% · 空闲 {a_free_pct}%",
         f"计算 {b_comp_pct}% · 通信 {b_comm_pct}% · 空闲 {b_free_pct}%", "—", "负载构成迁移"],
        ["均值带宽 (GB/s)", bw_a_val, bw_b_val, f"{bw_delta_pct:+.1f}%", "对流量构成敏感，仅参考"],
        ["有效吞吐 (GB/s)", eff_a_val, eff_b_val, f"{eff_delta_pct:+.1f}%", BW_VERDICT_TEXT.get(bw_verdict, "")],
    ]
    L.append(md_table(["指标", "集群A（基准）", "集群B（对比）", "变化", "备注"], kpi_rows))
    if bw_type_verdicts:
        L.append("")
        L.append("带宽逐类型判定：")
        L.append("")
        L.append(md_table(["带宽类型", "均值差 %", "有效吞吐差 %", "判定"],
                          [[t, f"{a:.1f}", f"{e:.1f}", VERDICT_TXT.get(v, v)]
                           for t, a, e, v in bw_type_verdicts]))
    L.append("")

    # 三、劣化根因
    L.append("## 三、劣化根因")
    L.append("")
    deg = []
    if stage_delta_pct > 10:
        deg.append(f"**P0** Stage 总耗时增幅 {stage_delta_pct:.1f}%（{_rep.fmt_signed(_rep.us_to_ms(delta_stage))} ms）")
    if comm_contrib is not None and comm_contrib > 50 and delta_comm > 0:
        deg.append(f"**P0** 通信劣化主导（贡献度 {comm_contrib:.1f}%，{_rep.fmt_signed(_rep.us_to_ms(delta_comm))} ms）")
    if bw_verdict == "real_severe":
        deg.append(f"**P0** 有效吞吐暴跌 {abs(eff_delta_pct):.1f}%（真实链路劣化，{eff_a_val:.1f}→{eff_b_val:.1f} GB/s）")
    elif bw_verdict == "real_minor":
        deg.append(f"**P1** 有效吞吐轻微下降 {abs(eff_delta_pct):.1f}%（{eff_a_val:.1f}→{eff_b_val:.1f} GB/s）")
    elif bw_verdict == "statistical":
        deg.append("**OK** 均值带宽下降为统计口径差异（流量构成变化），有效吞吐一致")
    if compute_contrib is not None and compute_contrib > 50 and delta_compute > 0:
        deg.append(f"**P1** 计算劣化（贡献度 {compute_contrib:.1f}%，{_rep.fmt_signed(_rep.us_to_ms(delta_compute))} ms）")
    if free_contrib is not None and free_contrib > 10 and delta_free > 0:
        deg.append(f"**P1** 空闲增加（贡献度 {free_contrib:.1f}%，{_rep.fmt_signed(_rep.us_to_ms(delta_free))} ms）")
    if deg:
        L += [f"- {d}" for d in deg]
    else:
        L.append("- 未发现显著劣化")
    L.append("")

    # 四、行动建议
    L.append("## 四、行动建议")
    L.append("")
    acts = []
    if bw_verdict == "real_severe":
        acts.append(("网络拓扑与硬件检查", "有效吞吐显著下降，重点检查 HCCS 链路状态、交换机拥塞、光模块降级。"))
    elif bw_verdict == "real_minor":
        acts.append(("网络链路跟踪观察", "有效吞吐轻微下降，建议持续跟踪各带宽类型趋势，暂无需硬件排查。"))
    elif bw_verdict == "statistical":
        acts.append(("带宽口径确认", "均值带宽下降源于算子流量构成差异（包大小分布变化），有效吞吐一致，无需排查硬件。"))
    if comm_contrib is not None and comm_contrib > 50:
        acts.append(("集合通信算法调优", "对比通信域切分策略、AllReduce/AllGather 算法参数。"))
    if compute_contrib is not None and compute_contrib > 50:
        acts.append(("算子性能排查", "检查计算算子变化、精度配置、Kernel 编译优化。"))
    if free_contrib is not None and free_contrib > 10:
        acts.append(("Host 下发效率", "检查算子下发、流同步、CPU 负载。"))
    if not acts:
        acts.append(("持续监控", "性能差异在可接受范围，建议持续监控。"))
    L += [f"{i}. **{t}**：{d}" for i, (t, d) in enumerate(acts, 1)]
    L.append("")

    # 五、Step 级耗时对比
    L.append(f"## 五、Step 级耗时对比（ms，共 {len(steps)} 个 Step）")
    L.append("")
    step_rows = []
    for s in steps:
        va = (sa.get(s) or {}).get("avg_stage")
        vb = (sb.get(s) or {}).get("avg_stage")
        ta = _rep.us_to_ms(va) if va is not None else "—"
        tb = _rep.us_to_ms(vb) if vb is not None else "—"
        if va is not None and vb is not None:
            d = _rep.us_to_ms(vb - va)
            p = _rep.safe_div(vb - va, va) * 100 if va else 0
            dt, pt = _rep.fmt_signed(d), f"{_rep.fmt_signed(round(p, 1))}%"
        else:
            dt = pt = "—"
        step_rows.append([s, ta, tb, dt, pt])
    L.append(md_table(["Step", "集群A Stage", "集群B Stage", "差值", "变化"], step_rows))
    L.append("")

    # 六、Rank 级差异
    all_ranks = sorted(set(list(ra.keys()) + list(rb.keys())), key=lambda x: int(x) if x.isdigit() else 999)
    rank_rows = []
    for rid in all_ranks:
        va = (ra.get(rid) or {}).get("avg_stage")
        vb = (rb.get(rid) or {}).get("avg_stage")
        ta = _rep.us_to_ms(va) if va is not None else "—"
        tb = _rep.us_to_ms(vb) if vb is not None else "—"
        if va is None or vb is None:
            rank_rows.append([rid, ta, tb, "—", "—", "单侧数据"])
            continue
        diff = vb - va
        pct = _rep.safe_div(diff, va) * 100 if va else 0
        trend = "劣化" if diff > 0 else "改善" if diff < 0 else "持平"
        rank_rows.append([rid, ta, tb, _rep.fmt_signed(_rep.us_to_ms(diff)),
                          f"{_rep.fmt_signed(round(pct, 1))}%", trend])
    rank_rows.sort(key=lambda r: abs(float(r[3])) if r[3] not in ("—", "") else 0, reverse=True)
    if top_ranks and top_ranks > 0:
        shown = rank_rows[:top_ranks]
        scope = f"按差值绝对值降序 Top {top_ranks}（共 {len(rank_rows)} 个 Rank，可用 --top-ranks 0 展开全部）"
    else:
        shown = rank_rows
        scope = f"全部 {len(rank_rows)} 个 Rank（按差值绝对值降序）"
    L.append(f"## 六、Rank 级差异（{scope}）")
    L.append("")
    L.append(md_table(["Rank", "集群A Stage (ms)", "集群B Stage (ms)", "差值 (ms)", "变化", "趋势"], shown))
    L.append("")

    # 七、通信算子差异
    is_a_text = data_a.get("format") == "text"
    is_b_text = data_b.get("format") == "text"
    ops_a_raw = _rep.filter_total_ops(_rep.filter_total_ops(data_a.get("comm_time_ops", []), "op_name"), "hccl_op_name")
    ops_b_raw = _rep.filter_total_ops(_rep.filter_total_ops(data_b.get("comm_time_ops", []), "op_name"), "hccl_op_name")
    ops_a = {op.get("op_name", op.get("hccl_op_name", "?")): op for op in ops_a_raw}
    ops_b = {op.get("op_name", op.get("hccl_op_name", "?")): op for op in ops_b_raw}
    op_rows = []
    for op in set(list(ops_a.keys()) + list(ops_b.keys())):
        ea_raw = ops_a.get(op, {}).get("avg_elapsed", ops_a.get(op, {}).get("avg_elapsed_time", 0))
        eb_raw = ops_b.get(op, {}).get("avg_elapsed", ops_b.get(op, {}).get("avg_elapsed_time", 0))
        ea = float(ea_raw or 0) if is_a_text else _rep.us_to_ms(float(ea_raw or 0))
        eb = float(eb_raw or 0) if is_b_text else _rep.us_to_ms(float(eb_raw or 0))
        op_rows.append([op, round(ea, 2), round(eb, 2), _rep.fmt_signed(round(eb - ea, 2))])
    op_rows.sort(key=lambda r: float(str(r[3]).replace("+", "")) if r[3] not in ("—", "") else 0, reverse=True)
    L.append("## 七、通信算子差异 Top 10（ms）")
    L.append("")
    if op_rows:
        L.append(md_table(["通信算子", "集群A 均值", "集群B 均值", "差值"], op_rows[:10]))
    else:
        L.append("无通信算子级数据。")
    L.append("")

    # 八、进阶分析
    L.append("## 八、进阶分析（msprof-analyze -m 专项）")
    L.append("")
    merged = _rep.merge_advanced(data_a, data_b, advanced_json)
    L.append(render_advanced_md(merged))
    L.append("")

    L.append("---")
    L.append("*本报告由 generate_cluster_md.py 自动生成；进阶分析「分析与判断」「综合判断」与"
             "单集群 JSON 内嵌字段、HTML 报告同口径（advanced_insights.py），全部结论为规则化中文输出。*")
    return "\n".join(L) + "\n"


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="集群比对 Markdown 报告生成器（仅支持双集群比对）")
    parser.add_argument("--data-a", required=True, help="比对: 集群A（基准/正常）JSON")
    parser.add_argument("--data-b", required=True, help="比对: 集群B（对比/异常）JSON")
    parser.add_argument("--output", required=True, help="输出 Markdown 文件路径")
    parser.add_argument("--advanced", default=None,
                        help="可选: run_advanced_analysis.py 输出的进阶分析 JSON（含双集群时间拆解对比; 内嵌结果缺失时亦作回退来源）")
    parser.add_argument("--top-ranks", type=int, default=20,
                        help="Rank 级差异表展示条数（按差值绝对值降序, 0=全部, 默认 20）")
    args = parser.parse_args()

    with open(args.data_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(args.data_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)
    advanced = None
    if args.advanced:
        with open(args.advanced, "r", encoding="utf-8") as f:
            advanced = json.load(f)

    md = generate_md(data_a, data_b, advanced, top_ranks=args.top_ranks)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown 比对报告已生成: {args.output}")


if __name__ == "__main__":
    main()

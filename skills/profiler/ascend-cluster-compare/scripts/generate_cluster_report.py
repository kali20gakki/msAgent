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
generate_cluster_report.py
基于两个集群提取的 JSON 数据生成 HTML 集群比对报告（深色主题版）。
仅支持双集群比对，不含单集群分析。

进阶分析数据来源（自动合并渲染）:
1. 集群 JSON 内嵌的 advanced_analysis 字段: cluster_data_extractor.py 提取单集群
   JSON 时同步执行 free_analysis / communication_bottleneck（比对前即已获得）;
2. --advanced 外部 JSON: run_advanced_analysis.py 输出, 含双集群时间拆解对比
   （cluster_time_compare_summary）等。合并时成功记录优先, 内嵌结果按 A/B 归属。
"""
import argparse
import json
import os
import sys
from datetime import datetime

# 同目录导入分析判断共享模块（与 cluster_data_extractor.py 中文结论口径一致;
# 缺失时降级为不渲染「分析与判断」块, 报告其余部分不受影响）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
try:
    import advanced_insights as _insights
except Exception:
    _insights = None

US_TO_MS = 1000.0

def safe_div(a, b):
    if abs(b) < 1e-9:
        return 0
    return round(a / b, 4)

def us_to_ms(us_val):
    if us_val is None:
        return 0
    return round(float(us_val) / US_TO_MS, 2)

def fmt_signed(val):
    """带正负号的格式化"""
    if val >= 0:
        return f"+{val}"
    return str(val)


# 进阶分析 feature 名称（与 run_advanced_analysis.py 保持一致）
FEAT_NAME_TIME_CMP = "cluster_time_compare_summary"
FEAT_NAME_FREE = "free_analysis"
FEAT_NAME_COMM_BN = "communication_bottleneck"


def contrib_kpi(contrib, delta):
    """贡献度 KPI 卡展示三元组 (text, color, note)。

    颜色/语义跟随 ΔX 方向: ΔX>0 劣化(橙), ΔX<0 改善(绿)。
    contrib 为 None（ΔStage 极小降级）时展示 "—" 并说明原因。
    """
    if contrib is None:
        return "—", "text-slate-500", "Stage 差异不显著，贡献度无统计意义"
    if delta > 0:
        return f"{contrib:.1f}%", "text-orange-400", "占 Stage 变化比例（各分项可相互抵消）"
    if delta < 0:
        return f"{contrib:.1f}%", "text-emerald-400", "占 Stage 变化比例（改善）"
    return "0.0%", "text-slate-500", "该分项无变化"


# ==================== 带宽劣化智能解读 ====================

def filter_total_ops(rows, name_key):
    """防御性过滤: 剔除算子列表中的 Total 汇总行（旧版提取器可能未过滤）"""
    return [r for r in rows
            if not str(r.get(name_key, "")).strip().lower().startswith("total")]


def get_eff_bw(entry):
    """读取带宽记录的有效吞吐字段（旧 JSON 无此字段时按 avg_size/avg_time 现算）"""
    eff = entry.get("effective_bw")
    if eff is not None:
        try:
            return float(eff)
        except (TypeError, ValueError):
            pass
    size = float(entry.get("avg_size", 0) or 0)
    t = float(entry.get("avg_time", 0) or 0)
    return round(size / t, 4) if t > 1e-9 else 0


def classify_bw_change(avg_drop_pct, eff_drop_pct):
    """带宽变化分级。

    判定原则: 有效吞吐（传输量/传输时间）反映链路真实能力；
    均值带宽对流量构成（包大小分布）敏感，仅作参考。
      - real_severe : 有效吞吐大幅下降，真实链路劣化
      - real_minor  : 有效吞吐小幅下降，需关注
      - statistical : 均值下降但有效吞吐一致 → 统计口径差异（流量构成不同），非劣化
      - none        : 无显著变化
    """
    if eff_drop_pct >= 15:
        return "real_severe"
    if eff_drop_pct >= 5:
        return "real_minor"
    if avg_drop_pct >= 10:
        return "statistical"
    return "none"

# ==================== 进阶分析合并 ====================

def _pick_feature(*feats_lists, name):
    """按来源顺序挑选指定 feature 的最佳记录。

    优先返回 status ∈ {success, cached}（已有数据）的第一条;
    若各来源均无成功记录, 返回来源顺序上的第一条记录（保留失败原因展示）。
    """
    fallback = None
    for feats in feats_lists:
        f = next((x for x in (feats or []) if x.get("name") == name), None)
        if f is None:
            continue
        if f.get("status") in ("success", "cached"):
            return f
        if fallback is None:
            fallback = f
    return fallback


def merge_advanced(data_a, data_b, external_advanced=None):
    """合并三路进阶分析结果为比对报告可渲染的结构。

    来源:
    1. data_a/data_b 内嵌 advanced_analysis（提取单集群 JSON 时同步执行, scope=single,
       仅含 free_analysis / communication_bottleneck, 天然按集群归属）;
    2. --advanced 外部 JSON（run_advanced_analysis.py 输出, 含双集群
       cluster_time_compare_summary 与 clusters 信息）。

    合并策略:
    - time_cmp 仅外部结果提供（内嵌模式不含）;
    - free/comm_bn 对 A、B 各取内嵌版本, 内嵌无成功记录时回退外部对应集群的结果;
    - tool / clusters 信息外部优先, 缺失时由内嵌结果或 data_a/data_b 补全。
    返回 None 表示无任何进阶分析数据（主流程渲染引导卡）。
    """
    ext = external_advanced or {}
    ext_feats = ext.get("features") or []
    # 外部结果按 cluster 归属拆分（旧版输出可能无 cluster 标注, 视为共用来源）
    ext_common = [f for f in ext_feats if f.get("cluster") not in ("A", "B")]
    ext_a = [f for f in ext_feats if f.get("cluster") == "A"] or ext_common
    ext_b = [f for f in ext_feats if f.get("cluster") == "B"] or ext_common
    emb_a = data_a.get("advanced_analysis") or {}
    emb_b = data_b.get("advanced_analysis") or {}
    emb_feats_a = emb_a.get("features") or []
    emb_feats_b = emb_b.get("features") or []

    features = []
    # 1) 双集群时间拆解对比（仅外部结果提供）
    t = _pick_feature(ext_feats, name=FEAT_NAME_TIME_CMP)
    if t:
        features.append(t)
    # 2) 单集群 feature: A/B 各自取内嵌结果, 缺失时回退外部结果（并补 cluster 标注）
    for tag, emb, ext_side in (("A", emb_feats_a, ext_a), ("B", emb_feats_b, ext_b)):
        for name in (FEAT_NAME_FREE, FEAT_NAME_COMM_BN):
            f = _pick_feature(emb, ext_side, name=name)
            if f:
                g = dict(f)
                g["cluster"] = tag
                features.append(g)
    if not features:
        return None

    # tool 信息: 外部可用者优先, 其次内嵌可用者, 最后任一存在者
    tool = None
    for cand in (ext.get("tool"), emb_a.get("tool"), emb_b.get("tool")):
        if (cand or {}).get("available"):
            tool = cand
            break
    if not tool:
        tool = next((c for c in (ext.get("tool"), emb_a.get("tool"), emb_b.get("tool")) if c), {})
    # clusters 信息: 外部优先, 缺失时由 data_a/data_b 的 data_dir 补全
    clusters = dict(ext.get("clusters") or {})
    for tag, d in (("A", data_a), ("B", data_b)):
        c = dict(clusters.get(tag) or {})
        if not c.get("data_dir") and d.get("data_dir"):
            c["data_dir"] = d.get("data_dir")
        if c:
            clusters[tag] = c
    return {
        "generated_at": ext.get("generated_at")
                        or emb_a.get("generated_at") or emb_b.get("generated_at") or "",
        "merged": True,
        "scope": "compare",
        "tool": tool,
        "clusters": clusters,
        "features": features,
    }


# ==================== 进阶分析渲染 ====================

def _adv_table(headers, rows):
    """进阶分析小表：rows 为已拼好的 <tr> 内容片段列表。"""
    if not rows:
        return ""
    th = "".join(f'<th class="px-3 py-2 text-left text-xs font-medium text-slate-500 uppercase tracking-wide">{h}</th>' for h in headers)
    trs = "".join(f'<tr class="border-t border-slate-800 hover:bg-slate-800/40">{r}</tr>' for r in rows)
    return f'<div class="overflow-x-auto rounded-lg border border-slate-800"><table class="w-full text-sm text-slate-300"><thead><tr class="bg-slate-900/60">{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def _insights_block(items, title="分析与判断"):
    """「分析与判断」块：中文结论列表 → 样式化清单（空列表返回空串）。"""
    if not items:
        return ""
    lis = "".join(f'<li class="mb-1 leading-relaxed">{x}</li>' for x in items)
    return (f'<div class="mt-3 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-3">'
            f'<div class="text-xs font-semibold text-indigo-300 mb-1.5">{title}</div>'
            f'<ul class="list-disc list-inside text-sm text-slate-300">{lis}</ul></div>')


def _render_feature_section(feat, badge_map, show_desc=True):
    """渲染单个进阶分析 feature（标题 + 状态徽章 + 数据表格/原因说明）。

    返回 (head, body) HTML 片段; 单 feature 渲染异常时降级为错误提示, 不影响整体报告。
    """
    name = feat.get("name", "")
    label = feat.get("label") or name
    desc = feat.get("desc", "")
    status = feat.get("status", "pending")
    reason = feat.get("reason", "")
    data = feat.get("data") or {}
    badge_cls, badge_txt = badge_map.get(status, badge_map["pending"])
    head = (f'<h4 class="text-sm font-bold text-white mb-1">{label}'
            f'<span class="ml-2 px-2 py-0.5 rounded text-xs font-normal {badge_cls}">{badge_txt}</span></h4>')
    head += f'<p class="text-xs text-slate-500 mb-3">{desc}</p>' if (show_desc and desc) else '<div class="mb-3"></div>'
    body = ""
    try:
        if status in ("success", "cached"):
            if name == "cluster_time_compare_summary":
                # 总体指标表（base/cur/diff 均值，ms）
                overall = data.get("overall") or {}
                metrics = data.get("metrics") or []
                rows = []
                for m in metrics:
                    o = overall.get(m) or {}
                    fmt = lambda v: "—" if v is None else f"{v}"
                    rows.append(f'<td class="px-3 py-2 font-medium text-white">{m}</td>'
                                f'<td class="px-3 py-2 text-right">{fmt(o.get("base_avg_ms"))}</td>'
                                f'<td class="px-3 py-2 text-right">{fmt(o.get("cur_avg_ms"))}</td>'
                                f'<td class="px-3 py-2 text-right font-mono">{fmt(o.get("diff_avg_ms"))}</td>')
                if rows:
                    body += ('<div class="text-xs text-slate-400 mb-1">总体指标对比（ms，基准 → 对比）</div>'
                             + _adv_table(["指标", "基准均值", "对比均值", "差值"], rows))
                # Top 差异 Rank 表
                top = data.get("top_ranks") or []
                rc = data.get("rank_col") or "Rank"
                trows = []
                for t in top[:10]:
                    diff = next((t.get(f"{m}_diff_ms") for m in metrics if t.get(f"{m}_diff_ms") is not None), None)
                    diff_s = "—" if diff is None else f"{diff}"
                    up = isinstance(diff, (int, float)) and diff > 0
                    trows.append(f'<td class="px-3 py-2 font-mono text-white">{t.get("rank", "-")}</td>'
                                 f'<td class="px-3 py-2 text-right">{t.get("step_count", "-")}</td>'
                                 f'<td class="px-3 py-2 text-right font-mono {"text-red-400" if up else ""}">{diff_s}</td>')
                if trows:
                    body += (f'<div class="text-xs text-slate-400 mt-3 mb-1">Top 差异 {rc}（按 Stage 差值降序，ms）</div>'
                             + _adv_table([rc, "Step 数", "差值均值"], trows))
            elif name == "free_analysis":
                reasons = data.get("reasons") or []
                rows = []
                for r0 in reasons[:10]:
                    pct = r0.get("pct", 0)
                    rows.append(f'<td class="px-3 py-2">{r0.get("reason", "-")}</td>'
                                f'<td class="px-3 py-2 text-right">{r0.get("count", "-")}</td>'
                                f'<td class="px-3 py-2 text-right font-mono">{r0.get("total_ms", "-")}</td>'
                                f'<td class="px-3 py-2 text-right font-mono {"text-orange-400" if (pct or 0) > 30 else ""}">{pct}</td>')
                if rows:
                    body += (f'<div class="text-xs text-slate-400 mb-1">空闲时间成因聚合（Top 10，总空闲 {data.get("total_ms", "—")} ms）</div>'
                             + _adv_table(["成因（Reason）", "次数", "总时长 (ms)", "占比 %"], rows))
                else:
                    body = '<p class="text-xs text-slate-500">未采集到空闲片段数据。</p>'
            elif name == "communication_bottleneck":
                items = data.get("items") or []
                rows = []
                for it in items[:10]:
                    rows.append(f'<td class="px-3 py-2 font-mono text-white">{it.get("op", "-")}</td>'
                                f'<td class="px-3 py-2 text-center font-mono">{it.get("slow_rank", "-")}</td>'
                                f'<td class="px-3 py-2 text-center font-mono">{it.get("fast_rank", "-")}</td>'
                                f'<td class="px-3 py-2 text-right font-mono text-orange-400">{it.get("duration_ms", "-")}</td>'
                                f'<td class="px-3 py-2 text-xs">{it.get("reason", "")}</td>')
                if rows:
                    body += ('<div class="text-xs text-slate-400 mb-1">通信瓶颈 Top 10（慢/快 Rank 耗时差，ms）</div>'
                             + _adv_table(["通信算子", "慢 Rank", "快 Rank", "耗时差 (ms)", "推断原因"], rows))
                else:
                    body = '<p class="text-xs text-slate-500">未识别到显著通信瓶颈。</p>'
            if not body:
                body = '<p class="text-xs text-slate-500">结果数据为空。</p>'
        else:
            body = f'<p class="text-xs text-slate-500">{reason or "工具不可用或当前数据格式不支持该特性"}</p>'
    except Exception as exc:  # 渲染容错：单特性异常不影响整体报告
        body = f'<p class="text-xs text-red-400">渲染失败: {exc}</p>'
    # 「分析与判断」：优先使用 feature 内嵌 analysis 字段（提取阶段写入的中文结论），
    # 缺失时由 advanced_insights 按数据规则计算（兼容旧 JSON 与外部进阶分析输出）。
    if _insights is not None and status in ("success", "cached"):
        try:
            body += _insights_block(_insights.feature_insights(feat))
        except Exception:
            pass  # 判断块生成失败不影响数据表格展示
    return head, body


def render_advanced_html(advanced):
    """渲染进阶分析 JSON 为 HTML 区块（支持同 feature 的 A/B 双卡对照）。

    advanced 结构: {tool:{version,available,path}, clusters:{A,B},
                    features:[{name,label,desc,available,reason,status,command,data,cluster}]}
    status 取值: success / cached / failed / not_available / skipped / pending
    同名 feature 多条记录（如集群 A、B 各一条内嵌结果）时合并在同一张卡片内
    对照展示, 卡内以 "集群A · 基准 / 集群B · 对比" 分区; 单条带 cluster 标注的
    记录也会显示归属。
    """
    if not advanced:
        return ""
    feats = advanced.get("features") or []
    if not feats:
        return ""
    tool = advanced.get("tool") or {}
    clusters = advanced.get("clusters") or {}

    badge_map = {
        "success": ("bg-emerald-500/15 text-emerald-400", "已完成"),
        "cached": ("bg-sky-500/15 text-sky-400", "缓存结果"),
        "failed": ("bg-red-500/15 text-red-400", "执行失败"),
        "not_available": ("bg-slate-500/15 text-slate-400", "不可用"),
        "skipped": ("bg-slate-500/15 text-slate-400", "已跳过"),
        "pending": ("bg-slate-500/15 text-slate-400", "未执行"),
    }
    cluster_tag_map = {"A": "集群A · 基准", "B": "集群B · 对比"}

    # 按 feature 名称分组（保持首次出现顺序）, 同名多条记录渲染为 A/B 对照
    groups = []
    for f in feats:
        name = f.get("name", "")
        for g in groups:
            if g[0] == name:
                g[1].append(f)
                break
        else:
            groups.append((name, [f]))

    blocks = ""
    for _, group in groups:
        inner = ""
        if len(group) == 1:
            feat = group[0]
            head, body = _render_feature_section(feat, badge_map)
            tag_txt = cluster_tag_map.get(feat.get("cluster"), "")
            if tag_txt:
                inner += f'<div class="text-xs font-semibold text-indigo-300 mb-2">{tag_txt}</div>'
            inner += f'{head}{body}'
        else:
            for i, feat in enumerate(group):
                head, body = _render_feature_section(feat, badge_map, show_desc=(i == 0))
                ctag = feat.get("cluster")
                tag_txt = cluster_tag_map.get(ctag, f"集群{ctag}" if ctag else f"结果 {i + 1}")
                sep = ' mb-5 pb-5 border-b border-slate-800' if i < len(group) - 1 else ''
                inner += (f'<div class="{sep}">'
                          f'<div class="text-xs font-semibold text-indigo-300 mb-2">{tag_txt}</div>'
                          f'{head}{body}</div>')
            # A/B 对照综合判断：双侧均有成功数据时输出「综合判断（基准 vs 对比）」块
            if _insights is not None and len(group) >= 2:
                try:
                    fa = next((f for f in group if f.get("cluster") == "A"), group[0])
                    fb = next((f for f in group if f.get("cluster") == "B"), group[-1])
                    inner += _insights_block(_insights.compare_insights(fa, fb),
                                             title="综合判断（基准 vs 对比）")
                except Exception:
                    pass  # 综合判断生成失败不影响 A/B 数据展示
        blocks += f'<div class="card p-5 mb-4">{inner}</div>'

    if not blocks:
        return ""
    meta = []
    if tool.get("version"):
        meta.append(f"msprof-analyze {tool['version']}")
    for key in ("A", "B"):
        c = clusters.get(key) or {}
        if c.get("data_dir"):
            meta.append(f"集群{key} {os.path.basename(str(c['data_dir']).rstrip(chr(92)).rstrip('/'))}")
    meta_html = f'<p class="text-xs text-slate-600 mt-3">{" · ".join(meta)}</p>' if meta else ""
    return ('<div class="mb-8"><h3 class="text-xl font-bold text-white mb-1">🔬 进阶分析（msprof-analyze -m 专项）</h3>'
            f'<p class="text-sm text-slate-500 mb-4">基础耗时比对之外的专项深度分析结论（含提取阶段内嵌的单集群结果与双集群时间拆解对比）</p>{blocks}{meta_html}</div>')

# ==================== 比对报告 ====================

def generate_compare_report(data_a, data_b, template_path, output_path, advanced_json=None):
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    sa = data_a.get("step_summary", {})
    sb = data_b.get("step_summary", {})
    steps = sorted(set(list(sa.keys()) + list(sb.keys())), key=lambda x: int(x) if x.isdigit() else x)
    ra = data_a.get("rank_summary", {})
    rb = data_b.get("rank_summary", {})

    avg_a_stage = sum(s["avg_stage"] for s in sa.values()) / len(sa) if sa else 0
    avg_b_stage = sum(s["avg_stage"] for s in sb.values()) / len(sb) if sb else 0
    avg_a_compute = sum(s["avg_compute"] for s in sa.values()) / len(sa) if sa else 0
    avg_b_compute = sum(s["avg_compute"] for s in sb.values()) / len(sb) if sb else 0
    avg_a_comm = sum(s["avg_comm"] for s in sa.values()) / len(sa) if sa else 0
    avg_b_comm = sum(s["avg_comm"] for s in sb.values()) / len(sb) if sb else 0
    avg_a_free = sum(s["avg_free"] for s in sa.values()) / len(sa) if sa else 0
    avg_b_free = sum(s["avg_free"] for s in sb.values()) / len(sb) if sb else 0
    avg_a_overlap = sum(s["avg_overlap"] for s in sa.values()) / len(sa) if sa else 0
    avg_b_overlap = sum(s["avg_overlap"] for s in sb.values()) / len(sb) if sb else 0

    delta_stage = avg_b_stage - avg_a_stage
    delta_compute = avg_b_compute - avg_a_compute
    delta_comm = avg_b_comm - avg_a_comm
    delta_free = avg_b_free - avg_a_free

    stage_delta_pct = safe_div(delta_stage, avg_a_stage) * 100 if avg_a_stage else 0

    # 修复: 贡献度 = ΔX / ΔStage 仅在 Stage 差异显著（|增幅| >= 1%）时才有统计意义。
    # ΔStage 趋近 0 时该比值会爆炸（如 368.5% / -123.4%），此时各分项变化互相抵消，
    # 百分比贡献度不再可解释，降级为 None（报告中展示 "—" 并说明原因）。
    stage_significant = abs(stage_delta_pct) >= 1.0
    if stage_significant:
        compute_contrib = safe_div(delta_compute, delta_stage) * 100
        comm_contrib = safe_div(delta_comm, delta_stage) * 100
        free_contrib = safe_div(delta_free, delta_stage) * 100
    else:
        compute_contrib = None
        comm_contrib = None
        free_contrib = None

    # 占比
    total_a = avg_a_stage if avg_a_stage > 0 else 1
    total_b = avg_b_stage if avg_b_stage > 0 else 1
    a_comp_pct = round(safe_div(avg_a_compute, total_a) * 100, 1)
    a_comm_pct = round(safe_div(avg_a_comm, total_a) * 100, 1)
    a_free_pct = round(safe_div(avg_a_free, total_a) * 100, 1)
    b_comp_pct = round(safe_div(avg_b_compute, total_b) * 100, 1)
    b_comm_pct = round(safe_div(avg_b_comm, total_b) * 100, 1)
    b_free_pct = round(safe_div(avg_b_free, total_b) * 100, 1)

    bwa = data_a.get("comm_bandwidth", [])
    bwb = data_b.get("comm_bandwidth", [])
    # 修复: KPI 用跨带宽类型的均值（原实现取任意第一条记录，代表性差）
    bw_a_val = round(sum(float(b.get("avg_bw", b.get("bandwidth_size", 0)) or 0) for b in bwa) / len(bwa), 2) if bwa else 0
    bw_b_val = round(sum(float(b.get("avg_bw", b.get("bandwidth_size", 0)) or 0) for b in bwb) / len(bwb), 2) if bwb else 0
    bw_delta_pct = safe_div(bw_b_val - bw_a_val, bw_a_val) * 100 if bw_a_val else 0

    # 修复: 有效吞吐（总传输量/总传输时间）对比与逐类型判定
    eff_a_val = round(sum(get_eff_bw(b) for b in bwa) / len(bwa), 2) if bwa else 0
    eff_b_val = round(sum(get_eff_bw(b) for b in bwb) / len(bwb), 2) if bwb else 0
    eff_delta_pct = safe_div(eff_b_val - eff_a_val, eff_a_val) * 100 if eff_a_val else 0

    rank_order = {"real_severe": 0, "real_minor": 1, "statistical": 2, "none": 3}
    bw_type_verdicts = []  # [(type, avg_drop%, eff_drop%, verdict)]
    for t in sorted(set([b.get("transport_type", b.get("band_type", "?")) for b in bwa + bwb])):
        ea = next((b for b in bwa if b.get("transport_type", b.get("band_type", "?")) == t), None)
        eb = next((b for b in bwb if b.get("transport_type", b.get("band_type", "?")) == t), None)
        avg_drop = abs(safe_div(float(eb.get("avg_bw", 0) or 0) - float(ea.get("avg_bw", 0) or 0),
                                float(ea.get("avg_bw", 0) or 0)) * 100) if (ea and eb) else 0
        eff_drop = abs(safe_div(get_eff_bw(eb) - get_eff_bw(ea), get_eff_bw(ea)) * 100) if (ea and eb and get_eff_bw(ea)) else 0
        verdict = classify_bw_change(avg_drop, eff_drop) if (ea and eb) else "none"
        bw_type_verdicts.append((t, avg_drop, eff_drop, verdict))
    bw_verdict = min((v[3] for v in bw_type_verdicts), key=lambda x: rank_order[x]) if bw_type_verdicts else "none"

    # 判定文案与配色（供 KPI 卡与带宽图表区使用）
    bw_verdict_map = {
        "real_severe": ("text-red-400", "真实链路劣化：有效吞吐显著下降，需排查硬件"),
        "real_minor": ("text-orange-400", "轻微吞吐劣化：有效吞吐小幅下降，建议关注"),
        "statistical": ("text-emerald-400", "均值带宽下降系流量构成差异，有效吞吐一致，非链路劣化"),
        "none": ("text-emerald-400", "带宽水平基本一致"),
    }
    bw_verdict_color, bw_verdict_text = bw_verdict_map.get(bw_verdict, bw_verdict_map["none"])

    # 状态文字
    if stage_delta_pct > 10:
        status_text = "严重劣化"
        stage_delta_color = "text-red-400"
    elif stage_delta_pct > 5:
        status_text = "中度劣化"
        stage_delta_color = "text-orange-400"
    elif stage_delta_pct > 0:
        status_text = "轻微劣化"
        stage_delta_color = "text-orange-400"
    else:
        status_text = "性能改善"
        stage_delta_color = "text-emerald-400"

    # 修复: KPI 颜色按有效吞吐判定（均值带宽受流量构成影响，不作劣化依据）
    if bw_verdict == "real_severe":
        bw_delta_color = "text-red-400"
    elif bw_verdict == "real_minor":
        bw_delta_color = "text-orange-400"
    elif bw_verdict == "statistical":
        bw_delta_color = "text-emerald-400"
    else:
        bw_delta_color = "text-emerald-400" if bw_delta_pct >= 0 else "text-orange-400"
    load_type_text = f"计算 {a_comp_pct}% → {b_comp_pct}% | 通信 {a_comm_pct}% → {b_comm_pct}%"

    # 诊断摘要
    diagnosis = (f"对比集群 B 的 Stage 总耗时{'增加' if delta_stage > 0 else '减少'} "
                 f"{abs(us_to_ms(delta_stage)):.1f} ms（{stage_delta_pct:+.1f}%）")
    if not stage_significant:
        # 修复: Stage 净变化不显著时, 分项贡献度互相抵消不可解释, 改为提示绝对变化
        diagnosis += "，整体差异不显著（<1%），分项贡献度不具统计意义；波动多为分项间相互转移。"
    elif comm_contrib is not None and comm_contrib > 50:
        diagnosis += f"，通信是主要{'劣化' if delta_comm > 0 else '改善'}来源（贡献度 {comm_contrib:.1f}%）。"
    elif compute_contrib is not None and compute_contrib > 50:
        diagnosis += f"，计算是主要{'劣化' if delta_compute > 0 else '改善'}来源（贡献度 {compute_contrib:.1f}%）。"
    else:
        diagnosis += "，劣化来源分散。"

    compute_text, compute_contrib_color, compute_contrib_note = contrib_kpi(compute_contrib, delta_compute)
    comm_text, comm_contrib_color, comm_contrib_note = contrib_kpi(comm_contrib, delta_comm)

    r = {
        "{{PATH_A}}": data_a.get("data_dir", "?"),
        "{{PATH_B}}": data_b.get("data_dir", "?"),
        "{{GENERATE_TIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "{{STATUS_TEXT}}": status_text,
        "{{RANK_A}}": str(len(ra)), "{{RANK_B}}": str(len(rb)),
        "{{STEP_A}}": str(len(sa)), "{{STEP_B}}": str(len(sb)),
        "{{STAGE_A_MS}}": str(us_to_ms(avg_a_stage)), "{{STAGE_B_MS}}": str(us_to_ms(avg_b_stage)),
        "{{STAGE_DELTA_PERCENT}}": str(round(abs(stage_delta_pct), 1)),
        "{{STAGE_DELTA_SIGN}}": "+" if stage_delta_pct >= 0 else "",
        "{{STAGE_DELTA_COLOR}}": stage_delta_color,
        # 修复: 差值符号不再丢失（负值正确显示 "-"）
        "{{DELTA_COMPUTE_MS}}": str(us_to_ms(abs(delta_compute))),
        "{{DELTA_COMPUTE_SIGN}}": "+" if delta_compute >= 0 else "-",
        "{{DELTA_COMM_MS}}": str(us_to_ms(abs(delta_comm))),
        "{{DELTA_COMM_SIGN}}": "+" if delta_comm >= 0 else "-",
        # 贡献度文本含 %（"—" 表示 ΔStage 极小降级）; 颜色/注释由 contrib_kpi 按方向与显著性生成
        "{{COMPUTE_CONTRIB}}": compute_text,
        "{{COMPUTE_CONTRIB_COLOR}}": compute_contrib_color,
        "{{COMPUTE_CONTRIB_NOTE}}": compute_contrib_note,
        "{{COMM_CONTRIB}}": comm_text,
        "{{COMM_CONTRIB_COLOR}}": comm_contrib_color,
        "{{COMM_CONTRIB_NOTE}}": comm_contrib_note,
        "{{BW_A}}": str(round(bw_a_val, 2)), "{{BW_B}}": str(round(bw_b_val, 2)),
        "{{BW_DELTA_PERCENT}}": str(round(abs(bw_delta_pct), 1)),
        "{{BW_DELTA_SIGN}}": "+" if bw_delta_pct >= 0 else "",
        "{{BW_DELTA_COLOR}}": bw_delta_color,
        "{{BW_EFF_A}}": str(round(eff_a_val, 2)), "{{BW_EFF_B}}": str(round(eff_b_val, 2)),
        "{{BW_EFF_DELTA}}": f"{eff_delta_pct:+.1f}",
        "{{BW_VERDICT_COLOR}}": bw_verdict_color,
        "{{BW_VERDICT_TEXT}}": bw_verdict_text,
        "{{BW_TYPE_VERDICTS_HTML}}": ("".join(
            f'<div class="flex items-center justify-between text-xs py-1 border-b border-slate-700/40">'
            f'<span class="text-slate-300 font-mono">{t}</span>'
            f'<span class="text-slate-400">均值差 {a:.1f}% · 吞吐差 {e:.1f}% · '
            + (f'<span class="text-red-400 font-semibold">真实劣化</span>' if v == "real_severe"
               else f'<span class="text-orange-400 font-semibold">轻微劣化</span>' if v == "real_minor"
               else f'<span class="text-emerald-400 font-semibold">口径差异</span>' if v == "statistical"
               else f'<span class="text-slate-500">一致</span>')
            + '</span></div>'
            for t, a, e, v in bw_type_verdicts) if bw_type_verdicts
            else '<div class="text-xs text-slate-500">无逐类型带宽数据</div>'),
        "{{DIAGNOSIS_TEXT}}": diagnosis,
        "{{LOAD_TYPE_TEXT}}": load_type_text,
        "{{STEP_LABELS}}": json.dumps(steps),
        "{{A_STAGE_BAR}}": json.dumps([us_to_ms(sa.get(s, {"avg_stage": 0})["avg_stage"]) for s in steps]),
        "{{B_STAGE_BAR}}": json.dumps([us_to_ms(sb.get(s, {"avg_stage": 0})["avg_stage"]) for s in steps]),
        "{{A_PIE_COMPUTE_PCT}}": str(a_comp_pct), "{{A_PIE_COMM_PCT}}": str(a_comm_pct), "{{A_PIE_FREE_PCT}}": str(a_free_pct),
        "{{B_PIE_COMPUTE_PCT}}": str(b_comp_pct), "{{B_PIE_COMM_PCT}}": str(b_comm_pct), "{{B_PIE_FREE_PCT}}": str(b_free_pct),
    }

    # 关键定位点（contrib 可能为 None: Stage 差异不显著时降级，仅展示绝对变化）
    key_points = ""
    if delta_comm != 0:
        comm_contrib_str = f"，贡献度 {comm_contrib:.1f}%" if comm_contrib is not None else ""
        comm_verdict = ("通信为主要劣化来源" if delta_comm > 0 and comm_contrib is not None and comm_contrib > 50
                        else "通信有改善" if delta_comm < 0 else "通信变化不显著")
        key_points += f'<li class="flex items-start"><span class="mr-2 text-xl">📡</span><p>通信时间{"增加" if delta_comm > 0 else "减少"} <strong>{abs(us_to_ms(delta_comm)):.1f} ms</strong>{comm_contrib_str}<br><span class="text-white font-semibold">{comm_verdict}</span></p></li>'
    if a_comm_pct != b_comm_pct:
        if b_comm_pct > b_comp_pct and a_comm_pct <= a_comp_pct:
            dominant = "通信反超计算"
        elif b_comm_pct > a_comm_pct:
            dominant = "通信占比上升"
        elif b_comm_pct < a_comm_pct:
            dominant = "通信占比下降"
        else:
            dominant = "通信占比持平"
        key_points += f'<li class="flex items-start"><span class="mr-2 text-xl">📊</span><p>负载转换：通信占比 {a_comm_pct}% → {b_comm_pct}%<br><span class="text-orange-400 font-semibold">{dominant}</span></p></li>'
    # 修复: 带宽要点按 verdict 分级，均值下降但有效吞吐一致时不再误报链路劣化
    if bw_verdict == "real_severe":
        key_points += f'<li class="flex items-start"><span class="mr-2 text-xl">📉</span><p>有效吞吐下降 {abs(eff_delta_pct):.1f}%（{eff_a_val:.1f} → {eff_b_val:.1f} GB/s）<br><span class="text-red-400 font-semibold">真实链路劣化，需排查硬件</span></p></li>'
    elif bw_verdict == "real_minor":
        key_points += f'<li class="flex items-start"><span class="mr-2 text-xl">📉</span><p>有效吞吐下降 {abs(eff_delta_pct):.1f}%（{eff_a_val:.1f} → {eff_b_val:.1f} GB/s）<br><span class="text-orange-400 font-semibold">轻微吞吐劣化，建议关注</span></p></li>'
    elif bw_verdict == "statistical":
        key_points += f'<li class="flex items-start"><span class="mr-2 text-xl">📊</span><p>均值带宽 {bw_delta_pct:+.1f}%（{bw_a_val:.1f} → {bw_b_val:.1f} GB/s），有效吞吐 {eff_delta_pct:+.1f}%<br><span class="text-emerald-400 font-semibold">均值下降系流量构成差异，非链路劣化</span></p></li>'
    if delta_free > 0 and (free_contrib is None or free_contrib > 10):
        free_contrib_str = f"（贡献度 {free_contrib:.1f}%）" if free_contrib is not None else ""
        key_points += f'<li class="flex items-start"><span class="mr-2 text-xl">💤</span><p>空闲时间增加 {us_to_ms(delta_free):.1f} ms{free_contrib_str}<br><span class="text-orange-400 font-semibold">可能存在 Host 下发瓶颈</span></p></li>'
    if not key_points:
        key_points = '<li class="flex items-start"><span class="mr-2 text-xl">✅</span><p>未发现显著性能差异</p></li>'
    r["{{KEY_POINTS_HTML}}"] = key_points

    # 瀑布图数据（contrib 为 None 时置空串，前端 tooltip 已有容错不显示贡献度）
    def contrib_txt(c):
        return "" if c is None else str(round(c, 1))

    wf = [
        {"name": "基准 Stage", "value": us_to_ms(avg_a_stage), "contrib": ""},
        {"name": "计算变化", "value": round(us_to_ms(delta_compute), 1), "contrib": contrib_txt(compute_contrib)},
        {"name": "通信变化", "value": round(us_to_ms(delta_comm), 1), "contrib": contrib_txt(comm_contrib)},
        {"name": "空闲变化", "value": round(us_to_ms(delta_free), 1), "contrib": contrib_txt(free_contrib)},
        {"name": "对比 Stage", "value": us_to_ms(avg_b_stage), "contrib": ""},
    ]
    r["{{WATERFALL_DATA}}"] = json.dumps(wf, ensure_ascii=False)

    # 通信算子差异 — 需要统一单位到 ms
    is_a_text = data_a.get("format") == "text"
    is_b_text = data_b.get("format") == "text"
    # 修复: 防御性过滤 Total 汇总行（旧版提取器产出的 JSON 可能仍含该行）
    ops_a_raw = filter_total_ops(filter_total_ops(data_a.get("comm_time_ops", []), "op_name"), "hccl_op_name")
    ops_b_raw = filter_total_ops(filter_total_ops(data_b.get("comm_time_ops", []), "op_name"), "hccl_op_name")
    ops_a = {op.get("op_name", op.get("hccl_op_name", "?")): op for op in ops_a_raw}
    ops_b = {op.get("op_name", op.get("hccl_op_name", "?")): op for op in ops_b_raw}
    all_ops = set(list(ops_a.keys()) + list(ops_b.keys()))
    diff_list = []
    for op in all_ops:
        ea_raw = ops_a.get(op, {}).get("avg_elapsed", ops_a.get(op, {}).get("avg_elapsed_time", 0))
        eb_raw = ops_b.get(op, {}).get("avg_elapsed", ops_b.get(op, {}).get("avg_elapsed_time", 0))
        # 统一到 ms: TEXT 模式已经是 ms，DB 模式需要 μs→ms
        ea = float(ea_raw or 0) if is_a_text else us_to_ms(float(ea_raw or 0))
        eb = float(eb_raw or 0) if is_b_text else us_to_ms(float(eb_raw or 0))
        diff = eb - ea
        diff_list.append((op, diff))
    diff_list.sort(key=lambda x: x[1], reverse=True)
    top_diff = diff_list[:10]
    if top_diff:
        r["{{COMM_DIFF_HAS_DATA}}"] = "true"
        r["{{COMM_DIFF_LABELS}}"] = json.dumps([d[0][:30] for d in top_diff])
        r["{{COMM_DIFF_VALUES}}"] = json.dumps([round(d[1], 2) for d in top_diff])
    else:
        r["{{COMM_DIFF_HAS_DATA}}"] = "false"
        r["{{COMM_DIFF_LABELS}}"] = "[]"
        r["{{COMM_DIFF_VALUES}}"] = "[]"

    # 带宽对比
    if bwa or bwb:
        bw_labels = list(set([b.get("transport_type", b.get("band_type", "?")) for b in bwa + bwb]))
        a_bw_map = {b.get("transport_type", b.get("band_type", "?")): b.get("avg_bw", b.get("bandwidth_size", 0)) for b in bwa}
        b_bw_map = {b.get("transport_type", b.get("band_type", "?")): b.get("avg_bw", b.get("bandwidth_size", 0)) for b in bwb}
        r["{{BW_HAS_DATA}}"] = "true"
        r["{{BW_LABELS}}"] = json.dumps(bw_labels)
        r["{{A_BW_BAR}}"] = json.dumps([round(float(a_bw_map.get(l, 0)), 2) for l in bw_labels])
        r["{{B_BW_BAR}}"] = json.dumps([round(float(b_bw_map.get(l, 0)), 2) for l in bw_labels])
    else:
        r["{{BW_HAS_DATA}}"] = "false"
        r["{{BW_LABELS}}"] = "[]"
        r["{{A_BW_BAR}}"] = "[]"
        r["{{B_BW_BAR}}"] = "[]"

    # Rank 差异表
    all_ranks = sorted(set(list(ra.keys()) + list(rb.keys())), key=lambda x: int(x) if x.isdigit() else 999)
    rank_diff_rows = ""
    for rid in all_ranks:
        sa_val = ra.get(rid, {}).get("avg_stage")
        sb_val = rb.get(rid, {}).get("avg_stage")
        if sa_val is None or sb_val is None:
            # 两集群 Rank 集合不重叠时无法按 rank 配对，避免出现假的 ±100% 差值
            sa_txt = us_to_ms(sa_val) if sa_val is not None else "-"
            sb_txt = us_to_ms(sb_val) if sb_val is not None else "-"
            rank_diff_rows += f'<tr><td class="p-3 font-bold text-white">{rid}</td><td class="text-right p-3 text-slate-300">{sa_txt}</td><td class="text-right p-3 text-slate-300">{sb_txt}</td><td class="text-right p-3 delta-neutral">-</td><td class="text-right p-3 delta-neutral">-</td><td class="text-center p-3"><span class="delta-neutral">→ 单侧数据</span></td></tr>'
            continue
        diff = sb_val - sa_val
        pct = safe_div(diff, sa_val) * 100 if sa_val else 0
        if diff > 0:
            trend = '<span class="delta-up">↑ 劣化</span>'
        elif diff < 0:
            trend = '<span class="delta-down">↓ 改善</span>'
        else:
            trend = '<span class="delta-neutral">→ 持平</span>'
        rank_diff_rows += f'<tr><td class="p-3 font-bold text-white">{rid}</td><td class="text-right p-3 text-slate-300">{us_to_ms(sa_val)}</td><td class="text-right p-3 text-slate-300">{us_to_ms(sb_val)}</td><td class="text-right p-3 {"delta-up" if diff>0 else "delta-down" if diff<0 else "delta-neutral"}">{fmt_signed(us_to_ms(diff))}</td><td class="text-right p-3 {"delta-up" if pct>0 else "delta-down" if pct<0 else "delta-neutral"}">{fmt_signed(round(pct, 1))}%</td><td class="text-center p-3">{trend}</td></tr>'
    r["{{RANK_DIFF_ROWS}}"] = rank_diff_rows

    # 劣化根因列表
    deg_items = ""
    if stage_delta_pct > 10:
        deg_items += f'<div class="flex items-center justify-between p-3 bg-red-900/20 rounded-lg"><div><span class="text-red-400 font-bold mr-2">P0</span><span class="text-white">Stage 总耗时增幅 {stage_delta_pct:.1f}%</span></div><span class="text-red-400 font-mono">+{us_to_ms(delta_stage):.1f} ms</span></div>'
    if comm_contrib is not None and comm_contrib > 50 and delta_comm > 0:
        deg_items += f'<div class="flex items-center justify-between p-3 bg-orange-900/20 rounded-lg"><div><span class="text-orange-400 font-bold mr-2">P0</span><span class="text-white">通信劣化主导 ({comm_contrib:.1f}%)</span></div><span class="text-orange-400 font-mono">+{us_to_ms(delta_comm):.1f} ms</span></div>'
    # 修复: 劣化根因列表按 verdict 分级
    if bw_verdict == "real_severe":
        deg_items += f'<div class="flex items-center justify-between p-3 bg-red-900/20 rounded-lg"><div><span class="text-red-400 font-bold mr-2">P0</span><span class="text-white">有效吞吐暴跌 {abs(eff_delta_pct):.1f}%（真实链路劣化）</span></div><span class="text-red-400 font-mono">{eff_a_val:.1f}→{eff_b_val:.1f}</span></div>'
    elif bw_verdict == "real_minor":
        deg_items += f'<div class="flex items-center justify-between p-3 bg-orange-900/20 rounded-lg"><div><span class="text-orange-400 font-bold mr-2">P1</span><span class="text-white">有效吞吐轻微下降 {abs(eff_delta_pct):.1f}%</span></div><span class="text-orange-400 font-mono">{eff_a_val:.1f}→{eff_b_val:.1f}</span></div>'
    elif bw_verdict == "statistical":
        deg_items += f'<div class="flex items-center justify-between p-3 bg-emerald-900/20 rounded-lg"><div><span class="text-emerald-400 font-bold mr-2">OK</span><span class="text-white">均值带宽下降为统计口径差异（流量构成变化），有效吞吐一致</span></div><span class="text-emerald-400 font-mono">吞吐 {eff_delta_pct:+.1f}%</span></div>'
    if compute_contrib is not None and compute_contrib > 50 and delta_compute > 0:
        deg_items += f'<div class="flex items-center justify-between p-3 bg-blue-900/20 rounded-lg"><div><span class="text-blue-400 font-bold mr-2">P1</span><span class="text-white">计算劣化 ({compute_contrib:.1f}%)</span></div><span class="text-blue-400 font-mono">+{us_to_ms(delta_compute):.1f} ms</span></div>'
    if free_contrib is not None and free_contrib > 10 and delta_free > 0:
        deg_items += f'<div class="flex items-center justify-between p-3 bg-orange-900/20 rounded-lg"><div><span class="text-orange-400 font-bold mr-2">P1</span><span class="text-white">空闲增加 ({free_contrib:.1f}%)</span></div><span class="text-orange-400 font-mono">+{us_to_ms(delta_free):.1f} ms</span></div>'
    if not deg_items:
        deg_items = '<div class="p-3 bg-emerald-900/20 rounded-lg text-emerald-400">未发现显著劣化</div>'
    r["{{DEGRADATION_ITEMS}}"] = deg_items

    # 行动建议
    actions = ""
    if bw_verdict == "real_severe":
        actions += '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">1</div><div><h4 class="text-sm font-semibold text-white">网络拓扑与硬件检查</h4><p class="text-xs text-slate-400 mt-1">有效吞吐显著下降，重点检查 HCCS 链路状态、交换机拥塞、光模块降级。</p></div></li>'
    elif bw_verdict == "real_minor":
        actions += '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">1</div><div><h4 class="text-sm font-semibold text-white">网络链路跟踪观察</h4><p class="text-xs text-slate-400 mt-1">有效吞吐轻微下降，建议持续跟踪各带宽类型趋势，暂无需硬件排查。</p></div></li>'
    elif bw_verdict == "statistical":
        actions += '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">1</div><div><h4 class="text-sm font-semibold text-white">带宽口径确认</h4><p class="text-xs text-slate-400 mt-1">均值带宽下降源于算子流量构成差异（包大小分布变化），有效吞吐一致，无需排查硬件。</p></div></li>'
    if comm_contrib is not None and comm_contrib > 50:
        actions += '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">2</div><div><h4 class="text-sm font-semibold text-white">集合通信算法调优</h4><p class="text-xs text-slate-400 mt-1">对比通信域切分策略、AllReduce/AllGather 算法参数。</p></div></li>'
    if compute_contrib is not None and compute_contrib > 50:
        actions += '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">3</div><div><h4 class="text-sm font-semibold text-white">算子性能排查</h4><p class="text-xs text-slate-400 mt-1">检查计算算子变化、精度配置、Kernel 编译优化。</p></div></li>'
    if free_contrib is not None and free_contrib > 10:
        actions += '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">4</div><div><h4 class="text-sm font-semibold text-white">Host 下发效率</h4><p class="text-xs text-slate-400 mt-1">检查算子下发、流同步、CPU 负载。</p></div></li>'
    if not actions:
        actions = '<li class="flex items-start"><div class="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold mr-3">1</div><div><h4 class="text-sm font-semibold text-white">持续监控</h4><p class="text-xs text-slate-400 mt-1">性能差异在可接受范围，建议持续监控。</p></div></li>'
    r["{{ACTION_ITEMS}}"] = actions

    # 进阶分析区块: 合并内嵌结果（提取单集群 JSON 时已同步执行, 比对前即已获得）
    # 与 --advanced 外部结果; 两者皆无时渲染引导卡, 说明缺失原因与补救方式
    merged_advanced = merge_advanced(data_a, data_b, advanced_json)
    if merged_advanced:
        r["{{ADVANCED_ANALYSIS_HTML}}"] = render_advanced_html(merged_advanced)
    else:
        r["{{ADVANCED_ANALYSIS_HTML}}"] = (
            '<div class="mb-8"><h3 class="text-xl font-bold text-white mb-1">🔬 进阶分析（msprof-analyze -m 专项）</h3>'
            '<div class="card p-5"><p class="text-sm text-slate-400">本次无进阶分析数据。默认情况下, 提取器'
            '（cluster_data_extractor.py）生成单集群 JSON 时会同步执行 free_analysis / communication_bottleneck'
            ' 并嵌入 JSON 的 advanced_analysis 字段；出现此提示说明提取时使用了'
            ' <code class="px-1.5 py-0.5 rounded bg-slate-800 text-sky-400 text-xs">--no-advanced</code>'
            ' 或该环境 msprof-analyze 不可用。补救：去掉 --no-advanced 重新提取集群 JSON，或运行'
            ' <code class="px-1.5 py-0.5 rounded bg-slate-800 text-sky-400 text-xs">run_advanced_analysis.py --cluster-a &lt;A目录&gt; --cluster-b &lt;B目录&gt; --output advanced_analysis.json</code>'
            ' 后用 <code class="px-1.5 py-0.5 rounded bg-slate-800 text-sky-400 text-xs">--advanced advanced_analysis.json</code>'
            ' 重新生成本报告（可补充双集群时间拆解对比）。</p></div></div>'
        )

    for key, val in r.items():
        html = html.replace(key, val)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"比对报告已生成: {output_path}")

# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="集群比对报告生成器（仅支持双集群比对）")
    parser.add_argument("--data-a", required=True, help="比对: 集群A（基准/正常）JSON")
    parser.add_argument("--data-b", required=True, help="比对: 集群B（对比/异常）JSON")
    parser.add_argument("--template-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--advanced", default=None,
                        help="可选: run_advanced_analysis.py 输出的进阶分析 JSON（提供双集群 time_cmp 对比; 内嵌结果缺失时亦作回退来源）")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tpl_dir = args.template_dir or os.path.join(script_dir, "..", "templates")

    with open(args.data_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(args.data_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)
    advanced = None
    if args.advanced:
        with open(args.advanced, "r", encoding="utf-8") as f:
            advanced = json.load(f)
    generate_compare_report(data_a, data_b, os.path.join(tpl_dir, "cluster_compare_report.html"), args.output, advanced)

if __name__ == "__main__":
    main()

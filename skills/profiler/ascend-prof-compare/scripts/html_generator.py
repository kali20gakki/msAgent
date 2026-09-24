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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HTML 报告生成器

将中间件 JSON 数据渲染为自包含的 HTML 报告。
"""
from datetime import datetime


def _fmt(val, suffix='', decimals=2):
    """格式化数值"""
    if val is None:
        return '-'
    try:
        if isinstance(val, float) and (val == float('inf') or val == float('-inf')):
            return 'inf'
        return f'{float(val):.{decimals}f}{suffix}'
    except (ValueError, TypeError):
        return str(val)


def _ratio_color(ratio):
    """根据比率返回颜色类名"""
    if ratio is None:
        return 'neutral'
    try:
        r = float(ratio)
        if r > 1.1:
            return 'bad'
        elif r < 0.9:
            return 'good'
        else:
            return 'neutral'
    except (ValueError, TypeError):
        return 'neutral'


def _diff_color(diff):
    """根据差异值返回颜色类名"""
    if diff is None:
        return 'neutral'
    try:
        d = float(diff)
        if d > 0:
            return 'bad'
        elif d < 0:
            return 'good'
        else:
            return 'neutral'
    except (ValueError, TypeError):
        return 'neutral'


def generate_html_report(results: dict, output_path: str):
    """生成 HTML 报告"""
    meta = results.get('metadata', {})
    summary = results.get('analysis_summary', {})

    # 可用 Sheet 列表
    available = summary.get('available_sheets', [])
    unavailable = summary.get('unavailable_sheets', [])

    # 总体性能
    om = results.get('overall_metrics', {})
    e2e = om.get('e2e', {})
    dims = om.get('dimensions', [])
    sub_cats = om.get('top_sub_categories', [])

    # 算子统计
    op_stat = results.get('operator_statistic', {})

    # 模块统计
    mod_stat = results.get('module_statistic', {})

    # 通信
    comm = results.get('communication', {})

    # 内存
    mem_stat = results.get('memory_statistic', {})

    # Kernel
    kc = results.get('kernel_compare', {})

    # API
    api = results.get('api_compare', {})

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>性能比对分析报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', 'Noto Sans CJK SC', sans-serif; background: #f0f2f5; color: #1a1d23; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px 24px 24px 200px; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 40px 32px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(102,126,234,0.3); }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .meta {{ font-size: 14px; opacity: 0.85; }}
.section {{ background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.section-title {{ font-size: 20px; font-weight: 700; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid #e2e5ea; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.kpi-card {{ background: #f8f9fb; border: 1px solid #e2e5ea; border-radius: 10px; padding: 16px; }}
.kpi-card .label {{ font-size: 13px; color: #6b7280; margin-bottom: 4px; }}
.kpi-card .value {{ font-size: 24px; font-weight: 700; }}
.kpi-card .sub {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
.bad {{ color: #dc2626; }}
.good {{ color: #16a34a; }}
.neutral {{ color: #6b7280; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.table-wrap {{ overflow-x: auto; margin-bottom: 8px; }}
th {{ background: #f3f4f6; padding: 10px 12px; text-align: left; font-weight: 600; color: #374151; border-bottom: 2px solid #e2e5ea; white-space: nowrap; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }}
tr:hover td {{ background: #f8f9fb; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 500; }}
.tag-bad {{ background: #fee2e2; color: #dc2626; }}
.tag-good {{ background: #dcfce7; color: #16a34a; }}
.tag-neutral {{ background: #f3f4f6; color: #6b7280; }}
.findings {{ list-style: none; }}
.findings li {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 14px; }}
.chart-container {{ position: relative; height: 300px; margin: 16px 0; }}
.unavailable {{ background: #f9fafb; color: #9ca3af; padding: 40px; text-align: center; border-radius: 8px; }}
.col-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.bar-container {{ display: flex; align-items: center; gap: 8px; }}
.bar {{ height: 8px; border-radius: 4px; background: #e5e7eb; flex: 1; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 4px; }}
.footer {{ text-align: center; padding: 24px; color: #9ca3af; font-size: 13px; }}
.dim-card {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #e2e5ea; }}
.dim-card.primary {{ border-color: #7c3aed; background: #f5f3ff; }}
/* 悬浮侧边导航 */
.nav-sidebar {{ position: fixed; top: 24px; left: 24px; width: 168px; max-height: calc(100vh - 48px); overflow-y: auto; background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); font-size: 13px; }}
.nav-sidebar .nav-title {{ font-weight: 700; font-size: 14px; margin-bottom: 10px; color: #1a1d23; }}
.nav-sidebar a {{ display: block; padding: 4px 0 4px 12px; color: #374151; text-decoration: none; border-left: 2px solid #e2e5ea; margin-bottom: 2px; }}
.nav-sidebar a:hover {{ color: #7c3aed; border-left-color: #7c3aed; }}
.nav-sidebar a.nav-sub {{ font-size: 12px; color: #6b7280; padding-left: 24px; }}
.nav-sidebar .nav-sep {{ height: 1px; background: #e2e5ea; margin: 8px 0; }}
/* Excel 样式表 */
.excel-table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-family: 'Segoe UI', sans-serif; }}
.excel-table th {{ background: #7c3aed; color: #fff; padding: 6px 10px; text-align: center; font-weight: 600; border: 1px solid #5b3fae; white-space: nowrap; font-size: 11px; }}
.excel-table td {{ padding: 5px 10px; border: 1px solid #e2e5ea; text-align: center; }}
.excel-table td.col-label {{ text-align: left; font-weight: 500; background: #f8f9fb; }}
.excel-table tr:nth-child(even) td {{ background: #fafafa; }}
.excel-table tr:nth-child(even) td.col-label {{ background: #f3f4f6; }}
.excel-table td.bad {{ color: #dc2626; font-weight: 600; }}
.excel-table td.good {{ color: #16a34a; font-weight: 600; }}
.excel-table td.indent {{ padding-left: 24px; }}
.excel-table td.indent2 {{ padding-left: 40px; }}
.excel-table tr.row-key td {{ background: #ede9fe; font-weight: 600; }}
.excel-table tr.row-key td.col-label-bold {{ background: #ddd6fe; font-weight: 700; color: #5b21b6; }}
</style>
</head>
<body>
<div class="container">

<!-- 悬浮侧边导航 -->
<div class="nav-sidebar">
  <div class="nav-title">目录导航</div>
  <a href="#overall">1. 总体性能比对</a>
  <a href="#conclusion">2. 分析结论</a>
  <div class="nav-sep"></div>
  <a href="#operator">3. 算子统计</a>
  <a href="#module">4. 模块统计</a>
  <a href="#communication">5. 通信比对</a>
  <a href="#memory">6. 内存统计</a>
  <a href="#kernel">7. Kernel比对</a>
  <a href="#kernel-data" class="nav-sub">- 两卡数据量比对</a>
  <a href="#kernel-top10" class="nav-sub">- Top10计算算子</a>
  <a href="#kernel-imbalance" class="nav-sub">- 负载不均预警</a>
  <a href="#api">8. API比对</a>
  <a href="#sheets">9. Sheet可用性</a>
</div>

<!-- 头部 -->
<div class="header">
  <h1>性能比对分析报告</h1>
  <div class="meta">
    源文件: {meta.get('source_file', '-')} ｜ 分析时间: {meta.get('analysis_time', '-')} ｜
    Sheet 数: {meta.get('total_sheets', '-')} ｜ 分析方法: analysis_methodology.md
  </div>
</div>

<!-- 总体性能 -->
<div class="section" id="overall">
  <div class="section-title">1. 总体性能比对 (OverallMetrics)</div>
  {_render_overall_metrics(om, e2e, dims, sub_cats) if om.get('available') else '<div class="unavailable">该 Sheet 不存在或为空</div>'}
</div>

<!-- 分析结论 -->
<div class="section" id="conclusion">
  <div class="section-title">2. 分析结论</div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
    <div>
      <h4 style="color: #16a34a; margin-bottom: 8px;">[改善] 改善亮点</h4>
      <ul class="findings" style="list-style: none;">
        {''.join(f'<li style="background:#dcfce7;border-left-color:#16a34a;">{f}</li>' for f in summary.get('improvements', [])) if summary.get('improvements') else '<li style="background:#f3f4f6;border-left-color:#9ca3af;">暂无改善项</li>'}
      </ul>
    </div>
    <div>
      <h4 style="color: #dc2626; margin-bottom: 8px;">[风险] 劣化风险</h4>
      <ul class="findings" style="list-style: none;">
        {''.join(f'<li style="background:#fee2e2;border-left-color:#dc2626;">{f}</li>' for f in summary.get('risks', [])) if summary.get('risks') else '<li style="background:#f3f4f6;border-left-color:#9ca3af;">暂无劣化项</li>'}
      </ul>
    </div>
  </div>
</div>

<!-- 算子统计 -->
<div class="section" id="operator">
  <div class="section-title">3. 算子统计比对 (OperatorCompareStatistic)</div>
  {_render_operator_statistic(op_stat) if op_stat.get('available') else '<div class="unavailable">该 Sheet 不存在或为空</div>'}
</div>

<!-- 模块统计 -->
<div class="section" id="module">
  <div class="section-title">4. 模块统计比对 (ModuleCompareStatistic)</div>
  {_render_module_statistic(mod_stat) if mod_stat.get('available') else '<div class="unavailable">该 Sheet 不存在或为空（需采集 Python Function 事件）</div>'}
</div>

<!-- 通信比对 -->
<div class="section" id="communication">
  <div class="section-title">5. 通信比对 (CommunicationCompare)</div>
  {_render_communication(comm) if comm.get('available') else '<div class="unavailable">该 Sheet 不存在或为空</div>'}
</div>

<!-- 内存统计 -->
<div class="section" id="memory">
  <div class="section-title">6. 内存统计比对 (MemoryCompareStatistic)</div>
  {_render_memory_statistic(mem_stat) if mem_stat.get('available') else '<div class="unavailable">该 Sheet 不存在或为空</div>'}
</div>

<!-- Kernel 比对 -->
<div class="section" id="kernel">
  <div class="section-title">7. Kernel 比对 (KernelCompare)</div>
  {_render_kernel_compare(kc) if kc.get('available') else '<div class="unavailable">该 Sheet 不存在或为空（仅 NPU vs NPU 场景）</div>'}
</div>

<!-- API 比对 -->
<div class="section" id="api">
  <div class="section-title">8. API 比对 (ApiCompare)</div>
  {_render_api_compare(api) if api.get('available') else '<div class="unavailable">该 Sheet 不存在或为空</div>'}
</div>

<!-- Sheet 可用性 -->
<div class="section" id="sheets">
  <div class="section-title">9. Sheet 可用性总览</div>
  <table>
    <tr><th>状态</th><th>Sheet 名称</th></tr>
    {''.join(f'<tr><td><span class="tag tag-good">可用</span></td><td>{s}</td></tr>' for s in available)}
    {''.join(f'<tr><td><span class="tag tag-neutral">不可用</span></td><td>{s}</td></tr>' for s in unavailable)}
  </table>
</div>

<div class="footer">
  Generated by Compare Analyzer ｜ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _render_overall_metrics(om, e2e, dims, sub_cats):
    """渲染总体性能部分"""
    parts = []

    # E2E KPI + 趋势
    trend = om.get('overall_trend', '')
    trend_color = '#16a34a' if trend == '改善' else '#dc2626' if trend == '劣化' else '#6b7280'
    if e2e:
        parts.append(f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="label">E2E 总耗时（基准）</div>
            <div class="value">{_fmt(e2e.get('base'), 'ms')}</div>
          </div>
          <div class="kpi-card">
            <div class="label">E2E 总耗时（比对）</div>
            <div class="value {_diff_color(e2e.get('diff'))}">{_fmt(e2e.get('comp'), 'ms')}</div>
          </div>
          <div class="kpi-card">
            <div class="label">E2E 差异</div>
            <div class="value {_diff_color(e2e.get('diff'))}">{_fmt(e2e.get('diff'), 'ms')}</div>
            <div class="sub">比率: {_fmt(e2e.get('ratio'))}</div>
          </div>
          <div class="kpi-card">
            <div class="label">整体趋势</div>
            <div class="value" style="color:{trend_color};">{trend}</div>
          </div>
        </div>""")

    # 维度拆解（区分改善/劣化）
    if dims:
        parts.append('<div style="margin-top:16px;"><h4 style="margin-bottom:4px;">维度拆解</h4>')
        parts.append('<div style="font-size:12px;color:#6b7280;margin-bottom:12px;">比率 = 比对值/基准值（>1劣化，<1改善）。贡献占比仅对劣化维度计算，分母为所有劣化维度差异之和。</div>')
        for d in dims:
            is_imp = d.get('is_improved')
            is_deg = d.get('is_degraded')
            if is_imp:
                tag = '<span class="tag tag-good" style="margin-left:8px;">改善</span>'
            elif is_deg:
                tag = '<span class="tag tag-bad" style="margin-left:8px;">劣化</span>'
            else:
                tag = ''
            parts.append(f"""
            <div class="dim-card">
              <div>
                <strong>{d['name']}</strong> ({d['label']})
                {tag}
              </div>
              <div style="text-align:right;">
                <span class="{_diff_color(d['diff_duration'])}">{_fmt(d['diff_duration'], 'ms')}</span>
                <span class="neutral" style="margin-left:12px;">比率 {_fmt(d['diff_ratio'])}</span>
                <span class="neutral" style="margin-left:12px;">劣化贡献 {d.get('contribution_str', '-')}</span>
              </div>
            </div>""")
        parts.append('</div>')

    # 子类别 Top
    if sub_cats:
        parts.append('<div style="margin-top:16px;"><h4 style="margin-bottom:12px;">子类别变化 Top 8</h4><table>')
        parts.append('<tr><th>子类别</th><th class="col-num">基准(ms)</th><th class="col-num">比对(ms)</th><th class="col-num">差异(ms)</th><th class="col-num">比率</th><th>状态</th></tr>')
        for s in sub_cats:
            cls = _ratio_color(s.get('diff_ratio'))
            if s.get('is_improved'):
                tag = '<span class="tag tag-good">改善</span>'
            elif s.get('is_degraded'):
                tag = '<span class="tag tag-bad">劣化</span>'
            else:
                tag = '<span class="tag tag-neutral">持平</span>'
            parts.append(f'<tr><td>{s["name"]}</td><td class="col-num">{_fmt(s.get("base_duration"))}</td><td class="col-num">{_fmt(s.get("comp_duration"))}</td><td class="col-num {_diff_color(s.get("diff_duration"))}">{_fmt(s.get("diff_duration"), "ms")}</td><td class="col-num {cls}">{_fmt(s.get("diff_ratio"))}</td><td>{tag}</td></tr>')
        parts.append('</table></div>')

    # Not minimal profiling 警告
    if om.get('has_minimal_warning'):
        parts.append('<div style="margin-top:12px;padding:12px;background:#fef3c7;border-radius:8px;color:#92400e;font-size:13px;">[警告] 检测到 Not minimal profiling，E2E 时间可能存在性能膨胀，影响通信和调度耗时判断。</div>')

    # OverallMetrics 原始数据表（Excel 样式 + 中文翻译）
    raw_metrics = om.get('raw_metrics', [])
    if raw_metrics:
        parts.append('<h4 style="margin:16px 0 8px;">总体性能原始数据</h4>')
        parts.append('<div class="table-wrap"><table class="excel-table">')
        parts.append('<tr><th>指标</th><th>基准耗时(ms)</th><th>基准占比</th><th>基准数量</th><th>比对耗时(ms)</th><th>比对占比</th><th>比对数量</th><th>差异(ms)</th><th>差异比率</th></tr>')
        for m in raw_metrics:
            idx = m['index']
            # 缩进处理：一级 \t 用 indent，二级 \t\t 用 indent2
            indent_cls = ''
            clean_idx = idx
            if idx.startswith('\t\t'):
                indent_cls = 'indent2'
                clean_idx = idx.strip()
            elif idx.startswith('\t'):
                indent_cls = 'indent'
                clean_idx = idx.strip()
            diff_cls = 'bad' if m['diff_duration'] > 0 else 'good' if m['diff_duration'] < 0 else ''
            ratio_cls = 'bad' if m['diff_ratio'] > 1.05 else 'good' if m['diff_ratio'] < 0.95 else ''
            # 四大核心维度行强调
            key_metrics = ('Computing Time', 'Uncovered Communication Time', 'Free Time', 'E2E Time')
            row_cls = ' class="row-key"' if clean_idx in key_metrics else ''
            label_cls = 'col-label-bold' if clean_idx in key_metrics else 'col-label'
            parts.append(f'<tr{row_cls}><td class="{label_cls} {indent_cls}">{clean_idx}</td><td>{_fmt(m["base_duration"])}</td><td>{_fmt(m["base_ratio"])}</td><td>{m["base_number"]}</td><td>{_fmt(m["comp_duration"])}</td><td>{_fmt(m["comp_ratio"])}</td><td>{m["comp_number"]}</td><td class="{diff_cls}">{_fmt(m["diff_duration"])}</td><td class="{ratio_cls}">{_fmt(m["diff_ratio"])}</td></tr>')
        parts.append('</table></div>')

    return ''.join(parts)


def _render_operator_statistic(op_stat):
    """渲染算子统计部分"""
    parts = []

    parts.append(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">参与比对算子总数</div><div class="value">{op_stat.get('total_operators', 0)}</div></div>
      <div class="kpi-card"><div class="label">劣化算子数</div><div class="value bad">{op_stat.get('degraded_count', 0)}</div></div>
      <div class="kpi-card"><div class="label">改善算子数</div><div class="value good">{op_stat.get('improved_count', 0)}</div></div>
      <div class="kpi-card"><div class="label">Top10 集中度</div><div class="value">{_fmt(op_stat.get('concentration'), '%', 1)}</div><div class="sub">{'集中' if op_stat.get('is_concentrated') else '分散'}</div></div>
    </div>""")

    top10 = op_stat.get('top10', [])
    if top10:
        parts.append('<table><tr><th>排名</th><th>算子名称</th><th class="col-num">基准耗时(ms)</th><th class="col-num">比对耗时(ms)</th><th class="col-num">差异(ms)</th><th class="col-num">比率</th></tr>')
        for i, op in enumerate(top10, 1):
            parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;">{op["name"]}</td><td class="col-num">{_fmt(op["base_duration"])}</td><td class="col-num">{_fmt(op["comp_duration"])}</td><td class="col-num {_diff_color(op["diff_duration"])}">{_fmt(op["diff_duration"])}</td><td class="col-num {_ratio_color(op["diff_ratio"])}">{_fmt(op["diff_ratio"])}</td></tr>')
        parts.append('</table>')

    return ''.join(parts)


def _render_module_statistic(mod_stat):
    """渲染模块统计部分"""
    parts = []

    parts.append(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">参与比对模块总数</div><div class="value">{mod_stat.get('total_modules', 0)}</div></div>
      <div class="kpi-card"><div class="label">有调用栈</div><div class="value">{'是' if mod_stat.get('has_call_stack') else '否'}</div></div>
    </div>""")

    top10 = mod_stat.get('top10', [])
    if top10:
        parts.append('<table><tr><th>排名</th><th>模块类</th><th>层级</th><th>模块名</th><th class="col-num">基准总耗时(ms)</th><th class="col-num">比对总耗时(ms)</th><th class="col-num">差异(ms)</th><th class="col-num">比率</th></tr>')
        for i, m in enumerate(top10, 1):
            parts.append(f'<tr><td>{i}</td><td>{m["module_class"]}</td><td>{m["module_level"]}</td><td style="font-family:monospace;font-size:12px;">{m["module_name"]}</td><td class="col-num">{_fmt(m["base_total_time"])}</td><td class="col-num">{_fmt(m["comp_total_time"])}</td><td class="col-num {_diff_color(m["diff_total_time"])}">{_fmt(m["diff_total_time"])}</td><td class="col-num {_ratio_color(m["diff_total_ratio"])}">{_fmt(m["diff_total_ratio"])}</td></tr>')
        parts.append('</table>')

    return ''.join(parts)


def _render_communication(comm):
    """渲染通信部分"""
    parts = []

    parts.append(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">通信算子总数</div><div class="value">{comm.get('total_comm_ops', 0)}</div></div>
      <div class="kpi-card"><div class="label">改善数</div><div class="value good">{comm.get('improved_count', 0)}</div></div>
      <div class="kpi-card"><div class="label">劣化数</div><div class="value bad">{comm.get('degraded_count', 0)}</div></div>
      <div class="kpi-card"><div class="label">极端改善</div><div class="value">{'是' if comm.get('has_extreme_improvement') else '否'}</div></div>
    </div>""")

    # 变化最大的算子（包含改善和劣化）
    top_changed = comm.get('top_changed', [])
    if top_changed:
        parts.append('<h4 style="margin:12px 0 8px;">变化最大的通信算子</h4><table>')
        parts.append('<tr><th>排名</th><th>通信算子名</th><th class="col-num">基准(us)</th><th class="col-num">比对(us)</th><th class="col-num">差异(us)</th><th class="col-num">比率</th><th>状态</th></tr>')
        for i, c in enumerate(top_changed, 1):
            if c['diff_duration'] < 0:
                tag = '<span class="tag tag-good">改善</span>'
                if c.get('is_extreme_improvement'):
                    tag += ' <span class="tag tag-good">极端</span>'
            elif c['diff_duration'] > 0:
                tag = '<span class="tag tag-bad">劣化</span>'
            else:
                tag = '<span class="tag tag-neutral">持平</span>'
            parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;">{c["comm_name"]}</td><td class="col-num">{_fmt(c["base_total"])}</td><td class="col-num">{_fmt(c["comp_total"])}</td><td class="col-num {_diff_color(c["diff_duration"])}">{_fmt(c["diff_duration"])}</td><td class="col-num {_ratio_color(c["diff_ratio"])}">{_fmt(c["diff_ratio"])}</td><td>{tag}</td></tr>')
        parts.append('</table>')

    return ''.join(parts)


def _render_memory_statistic(mem_stat):
    """渲染内存统计部分"""
    parts = []

    if mem_stat.get('all_zero_memory'):
        parts.append(f"""
        <div class="kpi-card" style="margin-bottom:12px;background:#f9fafb;border-color:#e5e7eb;">
          <div class="label">参与比对算子数</div>
          <div class="value">{mem_stat.get('total_operators', 0)}</div>
        </div>
        <div style="padding:16px;background:#fef3c7;border-radius:8px;color:#92400e;font-size:14px;">
          [说明] {mem_stat.get('zero_memory_note', '所有算子内存无差异')}
        </div>""")
        return ''.join(parts)

    parts.append(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">参与比对算子数</div><div class="value">{mem_stat.get('total_operators', 0)}</div></div>
      <div class="kpi-card"><div class="label">总内存差异</div><div class="value {_diff_color(mem_stat.get('total_diff_memory'))}">{_fmt(mem_stat.get('total_diff_memory'), 'MB')}</div></div>
      <div class="kpi-card"><div class="label">内存增长算子数</div><div class="value bad">{mem_stat.get('increased_count', 0)}</div></div>
      <div class="kpi-card"><div class="label">内存减少算子数</div><div class="value good">{mem_stat.get('decreased_count', 0)}</div></div>
    </div>""")

    top10 = mem_stat.get('top10', [])
    if top10:
        parts.append('<table><tr><th>排名</th><th>算子名称</th><th class="col-num">基准内存(MB)</th><th class="col-num">比对内存(MB)</th><th class="col-num">差异(MB)</th><th class="col-num">比率</th></tr>')
        for i, m in enumerate(top10, 1):
            parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;">{m["name"]}</td><td class="col-num">{_fmt(m["base_memory"])}</td><td class="col-num">{_fmt(m["comp_memory"])}</td><td class="col-num {_diff_color(m["diff_memory"])}">{_fmt(m["diff_memory"])}</td><td class="col-num {_ratio_color(m["diff_ratio"])}">{_fmt(m["diff_ratio"])}</td></tr>')
        parts.append('</table>')

    return ''.join(parts)


def _render_kernel_compare(kc):
    """渲染 Kernel 比对部分"""
    parts = []

    trend = kc.get('net_trend', '')
    trend_color = '#16a34a' if trend == '改善' else '#dc2626' if trend == '劣化' else '#6b7280'
    parts.append(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">Kernel 总数</div><div class="value">{kc.get('total_kernels', 0)}</div><div class="sub">计算 {kc.get('compute_kernel_count', 0)} / 通信 {kc.get('comm_kernel_count', 0)}</div></div>
      <div class="kpi-card"><div class="label">劣化总量</div><div class="value bad">+{_fmt(kc.get('total_degraded_us', 0), 'us')}</div><div class="sub">{kc.get('degraded_count', 0)} 个</div></div>
      <div class="kpi-card"><div class="label">改善总量</div><div class="value good">-{_fmt(kc.get('total_improved_us', 0), 'us')}</div><div class="sub">{kc.get('improved_count', 0)} 个</div></div>
      <div class="kpi-card"><div class="label">净趋势</div><div class="value" style="color:{trend_color};">{trend}</div><div class="sub">显著劣化: {kc.get('significant_degraded_count', 0)} 个</div></div>
    </div>""")

    # 显著劣化 Kernel
    sig_degraded = kc.get('significant_degraded', [])
    if sig_degraded:
        parts.append('<h4 style="margin:12px 0 8px;color:#dc2626;">显著劣化 Kernel（比率 > 1.05）</h4><table>')
        parts.append('<tr><th>Kernel名称</th><th>输入Shape</th><th class="col-num">基准(us)</th><th class="col-num">比对(us)</th><th class="col-num">差异(us)</th><th class="col-num">比率</th><th>原因</th></tr>')
        for k in sig_degraded:
            parts.append(f'<tr><td style="font-family:monospace;font-size:12px;">{k["kernel"]}</td><td style="font-size:12px;">{k["input_shape"]}</td><td class="col-num">{_fmt(k["base_total"])}</td><td class="col-num">{_fmt(k["comp_total"])}</td><td class="col-num bad">+{_fmt(k["diff_total"])}</td><td class="col-num bad">{_fmt(k["total_ratio"])}</td><td><span class="tag tag-bad">{k.get("degradation_type", "-")}</span></td></tr>')
        parts.append('</table>')

    # === 两卡数据量比对分析（在 Top 10 变化最大 之前） ===
    dvc = kc.get('data_volume_compare', {})
    if dvc:
        elements_changed = dvc.get('elements_changed', False)
        calls_changed = dvc.get('calls_changed', False)
        balance_tag = '<span class="tag tag-good">负载均衡</span>' if not elements_changed and not calls_changed else '<span class="tag tag-bad">负载变化</span>'
        parts.append(f"""
        <h4 id="kernel-data" style="margin:16px 0 8px;color:#2563eb;">两卡输入数据量比对分析 {balance_tag}</h4>
        <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">总数据量 = 各算子单次数据量 × calls 之和。若两卡总数据量和总 calls 完全一致，说明计算负载均衡；若不一致，说明模型结构或执行路径发生变化。</div>
        <table>
        <tr><th>指标</th><th class="col-num">基准卡</th><th class="col-num">比对卡</th><th class="col-num">差异</th></tr>
        <tr><td>计算算子种类数</td><td class="col-num">{kc.get('compute_kernel_count', 0)}</td><td class="col-num">{kc.get('compute_kernel_count', 0)}</td><td class="col-num">-</td></tr>
        <tr><td>总调用次数</td><td class="col-num">{dvc.get('base_total_calls', 0)}</td><td class="col-num">{dvc.get('comp_total_calls', 0)}</td><td class="col-num {_diff_color(dvc.get('comp_total_calls', 0) - dvc.get('base_total_calls', 0))}">{dvc.get('comp_total_calls', 0) - dvc.get('base_total_calls', 0):+d}</td></tr>
        <tr><td>总输入数据量</td><td class="col-num">{dvc.get('base_total_elements_fmt', '-')}</td><td class="col-num">{dvc.get('comp_total_elements_fmt', '-')}</td><td class="col-num">{'一致' if not elements_changed else '变化'}</td></tr>
        <tr><td>总耗时(us)</td><td class="col-num">{_fmt(dvc.get('base_total_us', 0))}</td><td class="col-num">{_fmt(dvc.get('comp_total_us', 0))}</td><td class="col-num {_diff_color(dvc.get('comp_total_us', 0) - dvc.get('base_total_us', 0))}">{_fmt(dvc.get('comp_total_us', 0) - dvc.get('base_total_us', 0))}</td></tr>
        </table>""")

        # Top10 计算算子的数据量明细对比
        top10_data = kc.get('top10_data_compare', [])
        if top10_data:
            parts.append('<h4 id="kernel-top10" style="margin:12px 0 8px;color:#2563eb;">Top 10 计算算子数据量明细</h4>')
            parts.append('<div class="table-wrap"><table>')
            parts.append('<tr><th>排名</th><th>算子名</th><th>单次数据量</th><th class="col-num">calls(基准/比对)</th><th class="col-num">总数据量(基准/比对)</th><th class="col-num">耗时(us)(基准/比对)</th></tr>')
            for i, k in enumerate(top10_data, 1):
                calls_str = f'{k["base_calls"]} / {k["comp_calls"]}'
                data_str = f'{k["base_total_elements_fmt"]} / {k["comp_total_elements_fmt"]}'
                dur_str = f'{_fmt(k["base_total_us"])} / {_fmt(k["comp_total_us"])}'
                parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;font-size:12px;" title="{k["shape"]}">{k["kernel"]}</td><td class="col-num">{k["elements_per_call_fmt"]}</td><td class="col-num">{calls_str}</td><td class="col-num">{data_str}</td><td class="col-num">{dur_str}</td></tr>')
            parts.append('</table></div>')

    top10 = kc.get('top10', [])
    if top10:
        parts.append('<h4 style="margin:16px 0 8px;">Top 10 变化最大</h4><table>')
        parts.append('<tr><th>排名</th><th>Kernel名称</th><th>输入Shape</th><th class="col-num">基准(us)</th><th class="col-num">比对(us)</th><th class="col-num">差异(us)</th><th class="col-num">比率</th><th>原因</th></tr>')
        for i, k in enumerate(top10, 1):
            parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;font-size:12px;">{k["kernel"]}</td><td style="font-size:12px;">{k["input_shape"]}</td><td class="col-num">{_fmt(k["base_total"])}</td><td class="col-num">{_fmt(k["comp_total"])}</td><td class="col-num {_diff_color(k["diff_total"])}">{_fmt(k["diff_total"])}</td><td class="col-num {_ratio_color(k["total_ratio"])}">{_fmt(k["total_ratio"])}</td><td><span class="tag {_ratio_color(k["total_ratio"]).replace("bad","tag-bad").replace("good","tag-good").replace("neutral","tag-neutral")}">{k.get("degradation_type", "-")}</span></td></tr>')
        parts.append('</table>')

    # === Top 10 计算算子绝对耗时（排除通信算子和空Shape） ===
    top10_compute = kc.get('top10_compute', [])
    if top10_compute:
        parts.append('<h4 style="margin:16px 0 4px;color:#7c3aed;">Top 10 计算算子绝对耗时（排除通信算子）</h4>')
        parts.append('<div style="font-size:12px;color:#6b7280;margin-bottom:8px;">输入数据量 = Shape 各维度相乘后所有 tensor 求和。基准与比对共用同一 Shape，故数据量相同；关注 calls 变化和耗时方差判断负载均衡。</div>')
        parts.append('<div class="table-wrap"><table>')
        parts.append('<tr><th>排名</th><th>Kernel名称</th><th class="col-num">数据量</th><th class="col-num">calls(基准/比对)</th><th class="col-num">耗时(us) 总/均/最大/最小</th><th class="col-num">方差</th></tr>')
        for i, k in enumerate(top10_compute, 1):
            calls_changed = k.get('calls_changed', False)
            calls_str = f'{k["base_calls"]} / {k["comp_calls"]}'
            if calls_changed:
                calls_str += ' <span class="tag tag-bad">变化</span>'
            variance = k.get('variance_ratio', 0)
            variance_str = f'{variance}x' if variance > 0 else '-'
            variance_color = 'bad' if variance > 2 else 'neutral'
            dur_str = f'{_fmt(k["comp_total"])} / {_fmt(k["comp_avg"])} / {_fmt(k["comp_max"])} / {_fmt(k["comp_min"])}'
            parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;font-size:12px;" title="{k["input_shape"]}">{k["kernel"]}</td><td class="col-num">{k.get("total_elements_fmt", "-")}</td><td class="col-num">{calls_str}</td><td class="col-num">{dur_str}</td><td class="col-num {variance_color}">{variance_str}</td></tr>')
        parts.append('</table></div>')

    # === 计算算子负载均衡分析（仅calls>10的计算算子） ===
    imbalance = kc.get('load_imbalance', [])
    if imbalance:
        parts.append(f'<h4 id="kernel-imbalance" style="margin:16px 0 4px;color:#dc2626;">[预警] 计算算子负载不均预警（max/min > 2x，共 {kc.get("load_imbalance_count", 0)} 个）</h4>')
        parts.append('<div style="font-size:12px;color:#6b7280;margin-bottom:8px;">负载不均 = 同一 Kernel 在多次调用中耗时差异大（最大值/最小值 > 2x）。通常由输入数据量不均匀（如 MoE 专家路由不均衡）或系统调度抖动导致。方差倍数越高，说明部分调用的耗时远超平均水平。</div>')
        parts.append('<div class="table-wrap"><table>')
        parts.append('<tr><th>Kernel名称</th><th class="col-num">输入数据量</th><th class="col-num">平均(us)</th><th class="col-num">最大(us)</th><th class="col-num">最小(us)</th><th class="col-num">方差倍数</th><th class="col-num">调用次数</th><th>严重程度</th></tr>')
        for k in imbalance:
            sev = k.get('imbalance_severity', '')
            sev_tag = f'<span class="tag tag-bad">{sev}</span>' if sev in ('严重', '显著') else f'<span class="tag tag-neutral">{sev}</span>'
            parts.append(f'<tr><td style="font-family:monospace;font-size:12px;" title="{k["input_shape"]}">{k["kernel"]}</td><td class="col-num">{k.get("total_elements_fmt", "-")}</td><td class="col-num">{_fmt(k["comp_avg"])}</td><td class="col-num bad">{_fmt(k["comp_max"])}</td><td class="col-num good">{_fmt(k["comp_min"])}</td><td class="col-num bad">{k.get("variance_ratio", 0)}x</td><td class="col-num">{k["comp_calls"]}</td><td>{sev_tag}</td></tr>')
        parts.append('</table></div>')
    else:
        parts.append('<div style="margin:16px 0;padding:12px;background:#dcfce7;border-radius:8px;color:#16a34a;font-size:14px;">[正常] 计算算子未检测到负载不均（所有 calls>10 的计算 Kernel 的 max/min 比值 < 2x）</div>')

    # === 多 Shape 计算算子分布 ===
    multi_shape = kc.get('multi_shape_kernels', {})
    if multi_shape:
        parts.append(f'<h4 style="margin:16px 0 4px;color:#2563eb;">同名计算算子多 Shape 分布（共 {kc.get("multi_shape_kernel_count", 0)} 个）</h4>')
        parts.append('<div style="font-size:12px;color:#6b7280;margin-bottom:8px;">同一 Kernel 名称对应不同输入 Shape，反映模型中不同层/专家的数据量分配差异。数据量差异大说明各层负载不均。</div>')
        for name, entries in multi_shape.items():
            parts.append(f'<div style="margin-bottom:12px;"><div style="font-weight:600;font-family:monospace;font-size:13px;margin-bottom:4px;">{name}</div><table>')
            parts.append('<tr><th>Shape</th><th class="col-num">输入数据量</th><th class="col-num">总耗时(us)</th><th class="col-num">平均(us)</th><th class="col-num">调用次数</th></tr>')
            for e in entries:
                parts.append(f'<tr><td style="font-size:11px;">{e["shape"][:40]}</td><td class="col-num">{e.get("total_elements_fmt", "-")}</td><td class="col-num">{_fmt(e["comp_total"])}</td><td class="col-num">{_fmt(e["comp_avg"])}</td><td class="col-num">{e["comp_calls"]}</td></tr>')
            parts.append('</table></div>')

    # === 通信算子概览（单独区域，不分析Shape和方差） ===
    comm_kernels = kc.get('comm_kernels', [])
    if comm_kernels:
        parts.append('<details style="margin-top:16px;"><summary style="cursor:pointer;font-weight:600;color:#6b7280;font-size:14px;margin-bottom:8px;">通信算子耗时概览（点击展开）</summary><table>')
        parts.append('<tr><th>通信算子</th><th class="col-num">基准(us)</th><th class="col-num">比对(us)</th><th class="col-num">差异(us)</th><th class="col-num">比率</th><th class="col-num">平均(us)</th><th class="col-num">最大(us)</th><th class="col-num">最小(us)</th><th class="col-num">调用次数</th></tr>')
        for k in comm_kernels:
            parts.append(f'<tr><td style="font-family:monospace;font-size:12px;">{k["kernel"]}</td><td class="col-num">{_fmt(k.get("base_total", 0))}</td><td class="col-num">{_fmt(k["comp_total"])}</td><td class="col-num {_diff_color(k["diff_total"])}">{_fmt(k["diff_total"])}</td><td class="col-num {_ratio_color(k["total_ratio"])}">{_fmt(k["total_ratio"])}</td><td class="col-num">{_fmt(k["comp_avg"])}</td><td class="col-num">{_fmt(k["comp_max"])}</td><td class="col-num">{_fmt(k["comp_min"])}</td><td class="col-num">{k["comp_calls"]}</td></tr>')
        parts.append('</table></details>')

    return ''.join(parts)


def _render_api_compare(api):
    """渲染 API 比对部分"""
    parts = []

    trend = api.get('net_trend', '')
    trend_color = '#16a34a' if trend == '改善' else '#dc2626' if trend == '劣化' else '#6b7280'
    parts.append(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="label">API 总数</div><div class="value">{api.get('total_apis', 0)}</div></div>
      <div class="kpi-card"><div class="label">劣化总量</div><div class="value bad">+{_fmt(api.get('total_degraded_ms', 0), 'ms')}</div><div class="sub">{api.get('degraded_count', 0)} 个</div></div>
      <div class="kpi-card"><div class="label">改善总量</div><div class="value good">-{_fmt(api.get('total_improved_ms', 0), 'ms')}</div><div class="sub">{api.get('improved_count', 0)} 个</div></div>
      <div class="kpi-card"><div class="label">净趋势</div><div class="value" style="color:{trend_color};">{trend}</div><div class="sub">纯耗时变化: {api.get('pure_duration_change_count', 0)} / 调用次数变化: {api.get('call_count_change_count', 0)}</div></div>
    </div>""")

    top10 = api.get('top10', [])
    if top10:
        parts.append('<table><tr><th>排名</th><th>API名称</th><th class="col-num">基准(ms)</th><th class="col-num">比对(ms)</th><th class="col-num">差异(ms)</th><th class="col-num">Self比率</th><th class="col-num">次数比率</th><th>原因</th></tr>')
        for i, a in enumerate(top10, 1):
            parts.append(f'<tr><td>{i}</td><td style="font-family:monospace;font-size:12px;">{a["name"]}</td><td class="col-num">{_fmt(a["base_total"])}</td><td class="col-num">{_fmt(a["comp_total"])}</td><td class="col-num {_diff_color(a["diff_total"])}">{_fmt(a["diff_total"])}</td><td class="col-num {_ratio_color(a["self_ratio"])}">{_fmt(a["self_ratio"])}</td><td class="col-num {_ratio_color(a["calls_ratio"])}">{_fmt(a["calls_ratio"])}</td><td><span class="tag tag-neutral">{a.get("degradation_type", "-")}</span></td></tr>')
        parts.append('</table>')

    return ''.join(parts)

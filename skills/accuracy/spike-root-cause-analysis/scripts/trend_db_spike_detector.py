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

"""
Phase 1 — 异常前反向定位: 从梯度监控数据检测 spike 候选坐标。

功能:
  - 自动切分分析（PP/micro_step 累积窗口检测）
  - 按数据类型分叉: step 级（全局基线 IQR）/ micro_step 累积（top-3 suspect → delta 突变）/ dump（绝对值排序）

输入:
  - trend.db (SQLite): msprobe 梯度趋势，含 trend_data / monitoring_targets / monitoring_metrics 表
  - monitor CSV: vpp_stage,name,step,micro_step,min,max,mean,norm,shape,dtype
  - dump_statistic 目录: dump.json 中的 parameters_grad 条目

输出: JSON with sharding_analysis, summary, anomalies

用法:
  python trend_db_spike_detector.py <data> [--csv|--dump] [-o result.json]
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from statistics import median

from monitor_data_loader import load_trend_db, load_dump_statistic, load_monitor_csv
from step_level_detector import compute_per_target_baselines, detect_spikes


def detect_dump_spikes(trend_rows, targets, top_n=20):
    """
    dump parameters_grad 数据异常检测。

    单 step（所有 row step 相同）: 无时间维度，按绝对值 top-N
    多 step: 有 step 维度，按参数计算跨 step 基线（MAD/IQR），动态阈值检测
    """
    steps = sorted(set(r['step'] for r in trend_rows))
    has_multi_step = len(steps) > 1

    if not has_multi_step:
        anomalies = []
        for r in trend_rows:
            anomalies.append({
                'rank': r['rank'],
                'step': r['step'],
                'target_id': r['target_id'],
                'target_name': targets[r['target_id']]['name'],
                'metric': 'grad_norm',
                'norm': r['norm'],
                'min': r['min'],
                'max': r['max'],
                'mean': r['mean'],
                'deviation_ratio': 0,
                'trigger': 'dump_abs_norm'
            })
        anomalies.sort(key=lambda x: x['norm'], reverse=True)
        anomalies = anomalies[:top_n]

        if anomalies:
            med = sorted([a['norm'] for a in anomalies])[len(anomalies) // 2]
            for a in anomalies:
                a['baseline_median'] = med
                a['deviation_ratio'] = a['norm'] / med if med > 0 else 0
        return anomalies

    baselines = compute_per_target_baselines(trend_rows, targets)
    anomalies = detect_spikes(trend_rows, targets, baselines,
                              metric_id=1, iqr_multiplier=5.0, mad_multiplier=10.0)
    return anomalies[:top_n]


def detect_accumulation_window(trend_rows, num_samples=5, drop_threshold=3.0):
    """
    检测 trend.db 中 step 列是否实际是 micro_step (梯度累积)。

    方法: 采样若干参数，找 norm 的周期性重置点。
    重置点 = norm 突然大幅下降 (相对前一步)，意味着梯度累积清空，进入下一个 optimizer step。
    如果多个参数的 reset 间隔一致 → step 实际是 micro_step，间隔 = 累积窗口大小。

    返回: (is_micro_step: bool, window_size: int, optimizer_step_count: int, reset_boundary: int|None)
    """
    if not trend_rows:
        return False, 1, 0, None

    # 多 metric 混合会破坏逐 step 覆盖判定，只用行数最多的 metric
    metric_counts = Counter(r['metric_id'] for r in trend_rows)
    if len(metric_counts) > 1:
        trend_rows = [r for r in trend_rows
                      if r['metric_id'] == metric_counts.most_common(1)[0][0]]

    steps = sorted(set(r['step'] for r in trend_rows))
    # 放宽下限: 至少 5 个 step 且存在周期性重置才判定为累积
    if len(steps) < 5:
        return False, 1, 0, None

    rank_counts = Counter(r['rank'] for r in trend_rows)
    best_rank = max(rank_counts, key=rank_counts.get)

    tid_steps = defaultdict(list)
    for r in trend_rows:
        if r['rank'] == best_rank:
            tid_steps[r['target_id']].append((r['step'], r['norm']))

    full_coverage = [tid for tid, data in tid_steps.items() if len(data) == len(steps)]

    if not full_coverage:
        return False, 1, 0, None

    sample_tids = full_coverage[:num_samples] if len(full_coverage) <= num_samples else \
        [full_coverage[i * len(full_coverage) // num_samples] for i in range(num_samples)]

    all_intervals = []
    reset_points = []  # 所有参数的重置点集合（用于短序列判定）
    for tid in sample_tids:
        data = sorted(tid_steps[tid], key=lambda x: x[0])
        norms = [d[1] for d in data]

        reset_steps = [
            data[i][0] for i in range(1, len(norms))
            if norms[i - 1] > 0 and norms[i] > 0 and norms[i - 1] / norms[i] > drop_threshold
        ]

        reset_points.append(set(reset_steps))
        if len(reset_steps) >= 2:
            intervals = [reset_steps[j + 1] - reset_steps[j] for j in range(len(reset_steps) - 1)]
            all_intervals.extend(intervals)

    # 短序列降级: step 数 < 10 时，用「多个参数共同的重置点」判定
    # CSV 是 trend.db 的切片，窗口边界重置会在所有 rank 同时出现
    if len(steps) < 10:
        reset_counter = Counter()
        for rp in reset_points:
            for s in rp:
                reset_counter[s] += 1
        # 至少 2 个采样参数在同一个 step 重置 → 该 step 是累积窗口边界
        common_resets = [s for s, c in reset_counter.items() if c >= 2]
        if common_resets:
            # 取最靠后的共同重置点作为窗口边界:
            # 单个参数的尖刺回落可能误报 (如 s13-15), 但所有 rank 共同重置
            # 只发生在真实 opt_step 边界 (如 s18: 8/8 rank 重置)
            boundary = max(common_resets)
            # 窗口大小 = 边界之前的步数 + 1（让前面所有 step 落入同一 opt_step）
            window_size = max(boundary - steps[0] + 1, 2)
            optimizer_steps = 2 if max(steps) >= boundary else 1
            return True, window_size, optimizer_steps, boundary
        return False, 1, 0, None

    if not all_intervals:
        return False, 1, 0, None

    interval_counts = Counter(all_intervals)
    most_common_interval, count = interval_counts.most_common(1)[0]

    if count >= 2 and most_common_interval > 1:
        window_size = most_common_interval
        optimizer_steps = (max(steps) + 1) // window_size
        return True, window_size, optimizer_steps, None

    return False, 1, 0, None


def analyze_sharding(data):
    """
    分析数据的并行切分方式，确定前反向的基本单位。

    检测逻辑:
      - PP: 检查 vpp_stage 是否有多个值
      - micro_step:
          1. targets/CSV 中有 micro_step 字段 → 直接使用
          2. DB 数据无 micro_step 字段 → 通过 norm 重置点检测梯度累积窗口
      - 确定前反向粒度

    返回: {
        'pp_stages': int,           # vpp_stage 数量
        'has_micro_step': bool,     # 是否有 micro_step 维度
        'micro_step_range': list,   # micro_step 范围
        'accumulation_window': int, # 梯度累积窗口大小 (0 表示无累积)
        'optimizer_step_count': int,# optimizer step 数量
        'pass_unit': str,           # 前反向单位描述
        'conclusion': str           # 一句话结论
    }
    """
    targets = data.get('targets', {})
    trend_rows = data.get('trend_rows', [])

    vpp_stages = {t.get('vpp_stage', 0) for t in targets.values()}
    pp_stages = len(vpp_stages)

    micro_steps = {t.get('micro_step', 0) for t in targets.values()}
    micro_steps.update(r['micro_step'] for r in trend_rows if 'micro_step' in r)
    has_explicit_micro_step = len(micro_steps) > 1

    accumulation_window = 0
    optimizer_step_count = 0
    has_accumulation = False
    reset_boundary = None

    if not has_explicit_micro_step and len(trend_rows) > 0:
        has_accumulation, accumulation_window, optimizer_step_count, reset_boundary = \
            detect_accumulation_window(trend_rows)

    if has_accumulation or has_explicit_micro_step:
        has_micro_step = True
        micro_step_range = (list(range(0, accumulation_window)) if has_accumulation
                            else sorted(micro_steps))
    else:
        has_micro_step = False
        micro_step_range = [0]

    pass_parts = []
    if has_micro_step:
        if has_accumulation:
            pass_parts.append(f"(optimizer_step, micro_step), 累积窗口={accumulation_window}")
        else:
            pass_parts.append("(step, micro_step)")
    else:
        pass_parts.append("(step,)")
    pass_unit = " + ".join(pass_parts)

    pp_note = f"PP={pp_stages}"

    if has_accumulation:
        micro_note = (f"DB step 实际为 micro_step (梯度累积), 窗口={accumulation_window}, "
                      f"{optimizer_step_count} 个 optimizer step")
    elif has_micro_step:
        micro_note = f"micro_step={micro_step_range[0]}..{micro_step_range[-1]}"
    else:
        micro_note = "无 micro_step"

    conclusion = (
        f"{pp_note}. {micro_note}. "
        f"前反向单位: {pass_unit}. "
        f"每个 rank 有完整模型，独立构成一次前反向。"
    )

    return {
        'pp_stages': pp_stages,
        'vpp_stages': sorted(vpp_stages),
        'has_micro_step': has_micro_step,
        'micro_step_range': micro_step_range,
        'accumulation_window': accumulation_window,
        'optimizer_step_count': optimizer_step_count,
        'accumulation_detected': has_accumulation,
        'reset_boundary': reset_boundary,
        'pass_unit': pass_unit,
        'conclusion': conclusion
    }


def detect_micro_step_spikes(trend_rows, targets, top_per_step=3,
                             delta_iqr_mult=5.0):
    """
    两步法检测 micro_step 累积数据中的 spike:

    Step 1 — 最终态 Top-N:
      对每个 optimizer step, 取最终 micro_step（最后 1 个）的累积 norm。
      按 norm 绝对值降序排列，取 top N 个 (rank, target)。

    Step 2 — 累积曲线 delta 检测:
      对每个 top suspect，展开 micro_step 累积曲线，组内检测 delta 突变点。

    返回每个 optimizer step 的 top 异常列表。
    """
    if not trend_rows:
        return []

    max_ms_per_opt = {}
    for r in trend_rows:
        opt = r.get('optimizer_step', 0)
        max_ms_per_opt[opt] = max(max_ms_per_opt.get(opt, 0), r.get('micro_step', 0))

    final_rows = defaultdict(list)  # optimizer_step -> [(norm, rank, target_id, row)]
    for r in trend_rows:
        opt = r.get('optimizer_step', 0)
        last_ms = max_ms_per_opt.get(opt, 0)
        if r.get('micro_step', 0) == last_ms:
            final_rows[opt].append((r['norm'], r['rank'], r['target_id'], r))

    suspects = []
    for opt_step in sorted(final_rows.keys()):
        items = sorted(final_rows[opt_step], key=lambda x: x[0], reverse=True)
        for norm, rank, tid, row in items[:top_per_step]:
            suspects.append({
                'rank': rank, 'optimizer_step': opt_step,
                'target_id': tid, 'target_name': targets[tid]['name'],
                'final_norm': norm, 'final_micro_step': max_ms_per_opt.get(opt_step, 0)
            })

    # ── Step 2: 累积曲线 delta 检测 ──
    # 短序列（每个 opt_step 内 micro_step 数少）时全量展开所有 (target, rank) 的曲线，
    max_ms = max(max_ms_per_opt.values(), default=0)
    is_short = max_ms <= 12

    if is_short:
        suspect_keys = {(r['target_id'], r['rank'], r.get('optimizer_step', 0)) for r in trend_rows}
    else:
        suspect_keys = {(s['target_id'], s['rank'], s['optimizer_step']) for s in suspects}

    ms_index = defaultdict(list)
    for r in trend_rows:
        k = (r['target_id'], r['rank'], r.get('optimizer_step', 0))
        if k in suspect_keys:
            ms_index[k].append((r.get('micro_step', 0), r))

    suspect_map = {(s['target_id'], s['rank'], s['optimizer_step']): s for s in suspects}
    result = []

    for key, items in ms_index.items():
        tid, rank, opt_step = key
        items.sort(key=lambda x: x[0])
        sus_info = suspect_map.get(key)

        # delta 序列
        deltas = []
        for i, (ms, r) in enumerate(items):
            if i == 0:
                delta = r['norm']
            else:
                prev_norm = items[i - 1][1]['norm']
                delta = max(0, r['norm'] - prev_norm)
            deltas.append(delta)

        if len(deltas) < 3:
            continue

        # 组内 delta 突变检测
        sd = sorted([d for d in deltas if d > 0])
        if len(sd) < 2:
            continue
        n = len(sd)
        q1 = sd[n // 4]
        q3 = sd[(3 * n) // 4]
        iqr = q3 - q1
        med = sd[n // 2]

        # 短序列（delta 数 < 8）时 IQR 不可靠: 尖刺本身会撑大 q3/iqr，
        # 改用 delta 相对中位数倍数 + 绝对下限判定（尖刺通常 > 10x 组内 median）
        # 长序列仍用 IQR 自适应阈值
        if len(sd) < 8:
            if med <= 0:
                continue
            delta_threshold = max(med * 10.0, 0.01)
        else:
            if iqr <= 0:
                continue
            delta_threshold = q3 + delta_iqr_mult * iqr

        # 收集该 suspect 的所有 delta 突变，取 top 2
        suspect_spikes = []
        for i, delta in enumerate(deltas):
            if delta > delta_threshold and delta > 0:
                r = items[i][1]
                ms = items[i][0]
                deviation = delta / med if med > 0 else 0
                suspect_spikes.append((deviation, r, ms, delta))

        suspect_spikes.sort(key=lambda x: x[0], reverse=True)
        for deviation, r, ms, delta in suspect_spikes[:2]:
            anomaly = {
                'rank': r['rank'], 'step': r['step'],
                'micro_step': ms, 'optimizer_step': opt_step,
                'target_id': tid, 'target_name': targets[tid]['name'],
                'metric': 'grad_unreduced',
                'norm': r['norm'], 'delta': delta,
                'min': r['min'], 'max': r['max'], 'mean': r['mean'],
                'group_delta_median': med, 'group_delta_iqr': iqr,
                'deviation_ratio': deviation,
                'trigger': 'micro_step_delta'
            }
            if sus_info:
                anomaly['suspect_final_norm'] = sus_info['final_norm']
            result.append(anomaly)

    # 短序列（CSV 切片）时 delta 检测会产出大量小值 noise，
    # 按每个 opt_step 取 delta 最大的 top-N 收敛输出
    if is_short:
        per_opt = defaultdict(list)
        for a in result:
            per_opt[a['optimizer_step']].append(a)
        result = []
        for opt in sorted(per_opt):
            items = sorted(per_opt[opt], key=lambda x: x['delta'], reverse=True)
            result.extend(items[:max(top_per_step * 3, 10)])

    return result


def collect_target_rank_norms(trend_rows, anomalies, window, targets=None):
    """对 top suspect target，收集每个 opt_step 的所有 rank 最终 micro_step norm。"""
    step_top_target = {}
    for a in anomalies:
        os_step = a.get('optimizer_step', 0)
        fn = a.get('suspect_final_norm', 0)
        tname = a.get('target_name', '')
        if os_step not in step_top_target or fn > step_top_target[os_step][1]:
            step_top_target[os_step] = (tname, fn)

    if not step_top_target:
        return None

    tid_to_name = {tid: t['name'] for tid, t in targets.items()}

    max_ms_per_opt = {}
    for r in trend_rows:
        opt = r.get('optimizer_step', 0)
        max_ms_per_opt[opt] = max(max_ms_per_opt.get(opt, 0), r.get('micro_step', 0))

    result = {}
    for os_step, (tname, _) in step_top_target.items():
        last_ms = max_ms_per_opt.get(os_step, window - 1)
        rank_norms = {}
        for r in trend_rows:
            if (r.get('micro_step') == last_ms and r.get('optimizer_step') == os_step
                    and tid_to_name.get(r['target_id'], '') == tname):
                rank_norms[str(r['rank'])] = r['norm']
        result[str(os_step)] = {'target_name': tname, 'ranks': rank_norms}

    return result

def main():
    parser = argparse.ArgumentParser(description='Phase 1 — Spike 异常检测')
    parser.add_argument('db_path', nargs='?', help='数据路径（trend.db / CSV / dump 目录）')
    parser.add_argument('--csv', action='store_true', help='CSV 格式')
    parser.add_argument('--dump', action='store_true', help='dump_statistic 目录')
    parser.add_argument('--metric', type=int, default=1, help='metric_id')
    parser.add_argument('--iqr-mult', type=float, default=5.0, help='IQR 倍数阈值')
    parser.add_argument('--mad-mult', type=float, default=10.0, help='MAD 倍数阈值')
    parser.add_argument('--output', '-o', help='输出 JSON')
    parser.add_argument('--summary', action='store_true', help='截断异常列表')
    args = parser.parse_args()

    if not args.db_path:
        parser.print_help()
        sys.exit(1)

    if args.dump:
        loader = load_dump_statistic
    elif args.csv:
        loader = load_monitor_csv
    else:
        loader = load_trend_db
    data = loader(args.db_path)
    if not data:
        sys.exit(1)

    sharding = analyze_sharding(data)

    if args.dump:
        anomalies = detect_dump_spikes(data['trend_rows'], data['targets'])
    elif sharding.get('accumulation_detected'):
        window = sharding['accumulation_window']
        boundary = sharding.get('reset_boundary')
        if boundary is not None:
            # 短序列: 重置边界前的 step 归 opt0，边界及之后归 opt1
            # micro_step = 相对窗口起点的偏移
            start = min(r['step'] for r in data['trend_rows'])
            for r in data['trend_rows']:
                r['optimizer_step'] = 0 if r['step'] < boundary else 1
                r['micro_step'] = r['step'] - start
        else:
            for r in data['trend_rows']:
                r['optimizer_step'] = r['step'] // window
                r['micro_step'] = r['step'] % window
        anomalies = detect_micro_step_spikes(
            data['trend_rows'], data['targets'], delta_iqr_mult=args.iqr_mult)
    elif sharding.get('has_micro_step'):
        # 显式 micro_step（如 CSV）: 无需窗口切分，按已有 micro_step 检测
        for r in data['trend_rows']:
            r.setdefault('optimizer_step', 0)
        anomalies = detect_micro_step_spikes(
            data['trend_rows'], data['targets'], delta_iqr_mult=args.iqr_mult)
    else:
        baselines = compute_per_target_baselines(data['trend_rows'], data['targets'])
        anomalies = detect_spikes(data['trend_rows'], data['targets'], baselines,
                                  metric_id=args.metric, iqr_multiplier=args.iqr_mult,
                                  mad_multiplier=args.mad_mult)

    target_rank_norms = None
    if sharding.get('has_micro_step') and anomalies:
        target_rank_norms = collect_target_rank_norms(
            data['trend_rows'], anomalies, sharding['accumulation_window'], data['targets'])

    result = {
        'phase': 1, 'db_path': args.db_path,
        'sharding_analysis': sharding,
        'target_rank_norms': target_rank_norms,
        'summary': {
            'total_targets': len(data['targets']),
            'total_rows': len(data['trend_rows']),
            'steps': sorted(set(r['step'] for r in data['trend_rows'])),
            'ranks': sorted(set(r['rank'] for r in data['trend_rows'])),
            'pass_unit': sharding['pass_unit'],
            'anomaly_count': len(anomalies)
        },
        'anomalies': anomalies if not args.summary else anomalies[:50]
    }

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()

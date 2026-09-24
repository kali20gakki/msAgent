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

"""自适应阈值级联检测模块（序列变点 → 锚定回溯 → Delta-NRE 离群 → 分布间隙 → 统计兜底）

实现 design_try.md 中描述的五级级联回退算法，自动从比对数据中确定
NRE 阈值，无需用户手动指定。

用法:
    from cascade_threshold import auto_detect_threshold, compute_max_jump_supplement

    result = auto_detect_threshold(output_nodes, op_groups)
    # result: {threshold, method, confidence, stats: {noise_ceiling, p25, p50, p01}}

    supplements = compute_max_jump_supplement(all_nodes, result['stats'], result['threshold'])

级联顺序:
    序列变点检测（SICD） → 第一优先级
    锚定回溯             → 未找到变点时回退
    Delta-NRE 离群       → 锚定回溯不可靠时回退
    分布间隙检测          → 无离群/样本不足时回退
    统计兜底              → 最终兜底

全流程阈值保护: 下限 max(..., 0.1%), 上限 min(..., 100%)
"""

import math
import statistics


MIN_THRESHOLD = 0.1   # 下限 0.1%
MAX_THRESHOLD = 100.0  # 上限 100%

def _clamp_threshold(t):
    """将阈值限制在 [0.1%, 100%] 范围内。"""
    return max(MIN_THRESHOLD, min(MAX_THRESHOLD, t))


def _sicd_change_point(output_nodes):
    """在 log(NRE+1) 空间做序列增量变点检测。

    核心策略: 找**第一个**结构断裂点（不是最大的），对应"哪里开始出问题"。

    Args:
        output_nodes: list of dict, 按执行序 (idx) 排列。
                      每个 node 含 {'idx', 'nre', ...}

    Returns:
        (threshold, method_suffix, stats_dict) 或 (None, None, {})
        stats_dict 至少包含 noise_ceiling, p25, p50, p01
    """
    nre_seq = [(n['idx'], n['nre']) for n in output_nodes
               if n.get('nre') is not None and not math.isnan(n['nre'])]
    if len(nre_seq) < 3:
        return None, None, {}

    all_nres = sorted([nre for _, nre in nre_seq])
    n_all = len(all_nres)
    p01 = all_nres[max(0, n_all // 100)]
    p25 = all_nres[n_all // 4]
    p50 = all_nres[n_all // 2]
    noise_ceiling = p25

    stats = {
        'noise_ceiling': noise_ceiling,
        'p25': p25,
        'p50': p50,
        'p01': p01
    }

    _lead_nres = [nre for _, nre in nre_seq[:100]]
    if _lead_nres:
        _lead_sorted = sorted(_lead_nres)
        _lead_p25 = _lead_sorted[len(_lead_sorted) // 4]
    else:
        _lead_p25 = 0.0
    near_zero_eps = max(_lead_p25 / 2.0 if _lead_p25 > 0 else 1e-6, 1e-6)

    near_zero_run = 0
    for _, nre in nre_seq:
        if nre <= near_zero_eps:
            near_zero_run += 1
        else:
            break

    if near_zero_run > 0 and near_zero_run < len(nre_seq):
        cp_idx_in_seq = near_zero_run
        cp_nre = nre_seq[cp_idx_in_seq][1]
        if cp_nre > near_zero_eps:
            threshold = _clamp_threshold(cp_nre * 0.5)
            mode = 'zero_baseline' if near_zero_eps <= 1e-6 else 'near_zero_baseline'
            stats['noise_ceiling'] = near_zero_eps
            stats['sicd_cp_nre'] = cp_nre
            stats['sicd_cp_idx'] = nre_seq[cp_idx_in_seq][0]
            stats['sicd_mode'] = mode
            return threshold, 'SICD ({})'.format(mode), stats

    log_vals = [math.log1p(nre) for _, nre in nre_seq]
    m = len(log_vals)

    MIN_BASELINE = max(5, min(50, m // 10))
    first_cp = None

    for i in range(MIN_BASELINE, m - 3):
        before = log_vals[:i]
        after_window = log_vals[i:i + 3]

        mean_before = statistics.mean(before)
        std_before = statistics.stdev(before) if len(before) > 1 else 0.001

        threshold_log = mean_before + max(2.0 * std_before, 0.01)
        if all(v > threshold_log for v in after_window):
            first_cp = i
            break  # 找到第一个变点即停止

    global_threshold = None
    global_method = None
    global_first_cp = None

    if first_cp is not None:
        before_nres = [nre for _, nre in nre_seq[:first_cp]]
        mean_nre = statistics.mean(before_nres)
        std_nre = statistics.stdev(before_nres) if len(before_nres) > 1 else 0
        global_threshold = _clamp_threshold(mean_nre + 3.0 * std_nre)
        global_method = 'SICD (sicd_log_space)'
        global_first_cp = first_cp
        stats['noise_ceiling'] = max(p01, mean_nre + std_nre)
        stats['sicd_cp_idx'] = nre_seq[first_cp][0]
        stats['sicd_cp_nre'] = nre_seq[first_cp][1]
        stats['sicd_mode'] = 'sicd_log_space'

    mw_threshold, mw_window_size, mw_stats = _sicd_multi_window(output_nodes, nre_seq)

    if mw_threshold is not None:
        stats.update(mw_stats)
        if global_threshold is None or mw_threshold < global_threshold:
            stats['multi_window_applied'] = True
            stats['multi_window_threshold'] = mw_threshold
            stats['multi_window_size'] = mw_window_size
            return mw_threshold, 'SICD (multi_window_sicd, w={})'.format(mw_window_size), stats
        else:
            stats['multi_window_applied'] = False
            stats['multi_window_threshold'] = mw_threshold
            stats['multi_window_size'] = mw_window_size

    if global_threshold is not None:
        stats['multi_window_applied'] = False
        stats['multi_window_threshold'] = None
        return global_threshold, global_method, stats

    stats['multi_window_applied'] = False
    stats['multi_window_threshold'] = None
    return None, None, stats


def _sicd_multi_window(output_nodes, nre_seq=None):
    """多窗口滑动 SICD 变点检测。

    在全局 SICD 基础上增加滑动窗口（窗口大小 200/500/1000，步长 50%），
    每个窗口独立执行 SICD 变点检测。取所有有效窗口检测的最小阈值，
    解决全局窗口被远端大幅振荡拉高的问题。

    Args:
        output_nodes: 原始 output_nodes (保留兼容)
        nre_seq: list of (idx, nre) tuples (复用已有序列，避免重复构建)

    Returns:
        (threshold, window_size, stats_dict) 或 (None, None, {})
    """
    WINDOW_SIZES = [200, 500, 1000]

    if nre_seq is None:
        nre_seq = [(n['idx'], n['nre']) for n in output_nodes
                   if n.get('nre') is not None and not math.isnan(n['nre'])]

    if len(nre_seq) < 50:
        return None, None, {}

    best_threshold = None
    best_window_size = None
    best_stats = {}

    for window_size in WINDOW_SIZES:
        if len(nre_seq) <= window_size:
            continue  # 序列太短，跳过此窗口大小

        step = max(window_size // 2, 1)

        for start in range(0, len(nre_seq) - window_size + 1, step):
            window_seq = nre_seq[start:start + window_size]
            if len(window_seq) < 50:
                continue

            window_nres = [nre for _, nre in window_seq]
            if len(window_nres) < 2:
                continue
            std_nre = statistics.stdev(window_nres)
            if std_nre <= 0:
                continue  # 窗口内无变化，跳过

            log_vals = [math.log1p(nre) for _, nre in window_seq]
            m = len(log_vals)
            MIN_BASELINE = max(5, min(50, m // 10))

            for i in range(MIN_BASELINE, m - 3):
                before = log_vals[:i]
                after_window_log = log_vals[i:i + 3]

                mean_before = statistics.mean(before)
                std_before = statistics.stdev(before) if len(before) > 1 else 0.001

                threshold_log = mean_before + max(2.0 * std_before, 0.01)
                if all(v > threshold_log for v in after_window_log):
                    before_nres = [nre for _, nre in window_seq[:i]]
                    mean_nre = statistics.mean(before_nres)
                    std_nre_local = statistics.stdev(before_nres) if len(before_nres) > 1 else 0
                    candidate = _clamp_threshold(mean_nre + 3.0 * std_nre_local)

                    if best_threshold is None or candidate < best_threshold:
                        best_threshold = candidate
                        best_window_size = window_size
                        best_stats = {
                            'multi_window_cp_idx': window_seq[i][0],
                            'multi_window_cp_nre': window_seq[i][1],
                            'multi_window_cp_before_mean': mean_nre,
                            'multi_window_cp_before_std': std_nre_local,
                        }
                    break  # 找到此窗口的第一个变点，跳到下一个窗口

    if best_threshold is not None:
        return best_threshold, best_window_size, best_stats
    return None, None, {}


def _anchored_backtrack(output_nodes):
    """从 Top-5 最差节点回溯 300 行，找最早跳变点。

    算法:
      1. 取 NRE 最高的 5 个 output 节点
      2. 对每个，往前面 300 行范围内找 ratio ≥ 5 的跳变点
      3. left_max=0 → 使用 p01 兜底值
      4. 过滤传播链内部跳变 (baseline > p25 且 > 1%)
      5. threshold = median(baselines) × 3

    Returns:
        (threshold, method_suffix, stats_dict, is_reliable)
    """
    valid_nodes = [n for n in output_nodes
                   if n.get('nre') is not None and not math.isnan(n['nre'])]

    if len(valid_nodes) < 5:
        return None, None, {}, False

    sorted_nodes = sorted(valid_nodes, key=lambda n: n['idx'])

    all_nres = sorted([n['nre'] for n in valid_nodes])
    n = len(all_nres)
    p01 = all_nres[max(0, n // 100)]
    p25 = all_nres[n // 4]
    p50 = all_nres[n // 2]

    worst_nodes = sorted(valid_nodes, key=lambda n: n['nre'], reverse=True)[:5]

    idx_to_pos = {n['idx']: i for i, n in enumerate(sorted_nodes)}

    all_baselines = []  # 所有检测到的跳变点的 "左侧" NRE 值

    for worst in worst_nodes:
        worst_pos = idx_to_pos.get(worst['idx'])
        if worst_pos is None:
            continue

        backtrack_start = max(0, worst_pos - 300)
        backtrack_nodes = sorted_nodes[backtrack_start:worst_pos + 1]

        if len(backtrack_nodes) < 2:
            continue

        jump_points = []
        for j in range(1, len(backtrack_nodes)):
            prev_nre = backtrack_nodes[j - 1]['nre']
            curr_nre = backtrack_nodes[j]['nre']

            if prev_nre == 0:
                if curr_nre >= max(p01 * 5, 0.01):
                    jump_points.append((j, max(p01, 0.001), curr_nre))
            elif curr_nre / prev_nre >= 5:
                jump_points.append((j, prev_nre, curr_nre))

        if not jump_points:
            continue

        earliest_jp = jump_points[0]
        baseline_nre = earliest_jp[1]
        all_baselines.append(baseline_nre)

    if not all_baselines:
        return None, None, {}, False

    filter_threshold = max(p25, 1.0)
    filtered_baselines = [b for b in all_baselines if b <= filter_threshold]

    filter_ratio = (len(all_baselines) - len(filtered_baselines)) / len(all_baselines) if all_baselines else 1.0
    is_reliable = True
    if len(filtered_baselines) < 2 or filter_ratio > 0.6:
        is_reliable = False

    baselines_for_threshold = filtered_baselines if filtered_baselines else all_baselines

    threshold = _clamp_threshold(statistics.median(baselines_for_threshold) * 3.0)

    stats = {
        'noise_ceiling': p25,
        'p25': p25,
        'p50': p50,
        'p01': p01,
        'hv2_n_jumps': len(all_baselines),
        'hv2_filtered_ratio': filter_ratio,
        'hv2_is_reliable': is_reliable
    }

    return threshold, 'AnchoredBacktrack', stats, is_reliable


def _delta_nre_outlier(op_groups):
    """基于算子组 delta = output_NRE - max_input_NRE 的 IQR 离群检测。

    要求 op_groups ≥ 30，否则自动跳过。

    Returns:
        (threshold, method_suffix, stats_dict, has_outliers)
    """
    if not op_groups or len(op_groups) < 30:
        return None, None, {}, False

    deltas = []
    for g in op_groups:
        out_nre = g.get('worst_output_nre')
        inp_nre = g.get('worst_input_nre')

        if out_nre is None or math.isnan(out_nre):
            continue
        if inp_nre is None or math.isnan(inp_nre):
            inp_nre = 0.0

        delta = out_nre - inp_nre
        deltas.append(delta)

    if len(deltas) < 4:
        return None, None, {}, False

    sorted_deltas = sorted(deltas)
    n_d = len(sorted_deltas)
    q1 = sorted_deltas[n_d // 4]
    q3 = sorted_deltas[3 * n_d // 4]
    iqr = q3 - q1

    if iqr <= 0:
        return None, None, {}, False

    upper_fence = q3 + 3.0 * iqr

    outliers = [d for d in deltas if d > upper_fence]
    non_outliers = [d for d in deltas if d <= upper_fence]

    if not outliers:
        return None, None, {}, False

    median_non_outlier = statistics.median(non_outliers) if non_outliers else 0
    threshold = _clamp_threshold(min(median_non_outlier * 3.0, upper_fence))

    all_out_nres = sorted([g['worst_output_nre'] for g in op_groups
                           if g.get('worst_output_nre') is not None
                           and not math.isnan(g['worst_output_nre'])])
    na = len(all_out_nres)
    stats = {
        'noise_ceiling': all_out_nres[na // 4] if na > 0 else 0.1,
        'p25': all_out_nres[na // 4] if na > 0 else 0.1,
        'p50': all_out_nres[na // 2] if na > 0 else 0.1,
        'p01': all_out_nres[max(0, na // 100)] if na > 0 else 0.1,
        'f_n_outliers': len(outliers),
        'f_upper_fence': upper_fence
    }

    return threshold, 'DeltaNREOutlier (delta_iqr)', stats, True


def _distribution_gap(output_nodes):
    """排序 NRE 值，检测最大相邻比值缺口。

    三守卫:
      - min_samples ≥ 10
      - max_ratio ≥ 3
      - signal ≥ 3 (max_ratio / median_ratio)

    threshold = gap_lower × 2.0

    Returns:
        (threshold, method_suffix, stats_dict, found_gap)
    """
    nres = sorted([n['nre'] for n in output_nodes
                   if n.get('nre') is not None
                   and not math.isnan(n['nre'])
                   and n['nre'] > 0])

    if len(nres) < 10:
        return None, None, {}, False

    ratios = []
    for i in range(1, len(nres)):
        if nres[i - 1] > 0:
            ratios.append((i, nres[i] / nres[i - 1], nres[i - 1], nres[i]))

    if not ratios:
        return None, None, {}, False

    max_ratio_entry = max(ratios, key=lambda x: x[1])
    max_ratio = max_ratio_entry[1]

    if max_ratio < 3:
        return None, None, {}, False

    all_ratios = [r[1] for r in ratios]
    median_ratio = statistics.median(all_ratios)
    if median_ratio <= 0:
        return None, None, {}, False
    signal = max_ratio / median_ratio
    if signal < 3:
        return None, None, {}, False


    gap_lower = max_ratio_entry[2]  # 缺口下界 NRE
    threshold = _clamp_threshold(gap_lower * 2.0)

    # 统计量
    n = len(nres)
    stats = {
        'noise_ceiling': max(gap_lower, nres[n // 4]) if n > 0 else 0.1,
        'p25': nres[n // 4] if n > 0 else 0.1,
        'p50': nres[n // 2] if n > 0 else 0.1,
        'p01': nres[max(0, n // 100)] if n > 0 else 0.1,
        'a_gap_lower': gap_lower,
        'a_max_ratio': max_ratio,
        'a_signal': signal
    }

    return threshold, 'DistributionGap (gap)', stats, True


def _statistical_fallback(output_nodes):
    """threshold = max(p50 × 3, 0.1%)"""
    valid_nres = sorted([n['nre'] for n in output_nodes
                         if n.get('nre') is not None
                         and not math.isnan(n['nre'])])

    n = len(valid_nres)
    if n == 0:
        return MIN_THRESHOLD, 'StatisticalFallback (empty)', {
            'noise_ceiling': MIN_THRESHOLD,
            'p25': MIN_THRESHOLD,
            'p50': MIN_THRESHOLD,
            'p01': MIN_THRESHOLD
        }

    p01 = valid_nres[max(0, n // 100)]
    p25 = valid_nres[n // 4]
    p50 = valid_nres[n // 2]

    threshold = _clamp_threshold(max(p50 * 3.0, MIN_THRESHOLD))

    stats = {
        'noise_ceiling': p25,
        'p25': p25,
        'p50': p50,
        'p01': p01
    }

    return threshold, 'StatisticalFallback', stats


def _detect_segments(output_nodes):
    """检测数据中的结构性断点，将数据分成多个段。

    分段依据:
      1. Dtype 变化: 相邻节点的 dtype 不一致
      2. NRE 量级跃迁: 相邻节点 NRE 比值 >= 10×

    Shape 变化不作为分段判据——真实 dump 中相邻算子 shape 几乎必然不同，
    若按 shape 判据分段会产生数千微段，失去"按结构聚类"的意义。

    每段至少包含 30 个节点，否则与前一段合并。

    Args:
        output_nodes: list of dict, 按执行序排列。
                      每个 node 含 {'idx', 'nre', 'shape', 'dtype', ...}

    Returns:
        segments: list of (start_idx, end_idx, break_reason) tuples
        segment_count: int
    """
    if len(output_nodes) < 60:
        return [(0, len(output_nodes) - 1, 'single_segment')], 1

    segments = []
    current_start = 0
    current_reason = None

    for i in range(1, len(output_nodes)):
        prev_node = output_nodes[i - 1]
        curr_node = output_nodes[i]

        reason = None

        prev_dtype = prev_node.get('dtype', '').strip()
        curr_dtype = curr_node.get('dtype', '').strip()
        if prev_dtype and curr_dtype and prev_dtype != curr_dtype:
            reason = 'dtype_change'

        if reason is None:
            prev_nre = prev_node.get('nre', 0) or 0
            curr_nre = curr_node.get('nre', 0) or 0
            if prev_nre > 1e-8 and curr_nre > 1e-8:
                ratio = max(prev_nre, curr_nre) / max(min(prev_nre, curr_nre), 1e-8)
                if ratio >= 10.0:
                    reason = 'nre_jump_{:.0f}x'.format(ratio)

        if reason is not None:
            segment_end = i - 1
            segments.append((current_start, segment_end, current_reason or 'start'))
            current_start = i
            current_reason = reason

    if current_start < len(output_nodes):
        segments.append((current_start, len(output_nodes) - 1, current_reason or 'tail'))

    max_iterations = max(len(segments) * 2, 100)
    merged = list(segments)
    for _ in range(max_iterations):
        all_big_enough = all((end - start + 1) >= 30 for start, end, _reason in merged)
        if all_big_enough or len(merged) <= 1:
            break
        new_merged = []
        i = 0
        while i < len(merged):
            start, end, reason = merged[i]
            seg_size = end - start + 1
            if seg_size < 30 and i + 1 < len(merged):
                next_start, next_end, next_reason = merged[i + 1]
                new_merged.append((start, next_end, reason + '+merged'))
                i += 2
            elif seg_size < 30 and i > 0:
                if new_merged:
                    prev = new_merged[-1]
                    new_merged[-1] = (prev[0], end, prev[2] + '+merged_tail')
                else:
                    new_merged.append((start, end, reason))
                i += 1
            else:
                new_merged.append((start, end, reason))
                i += 1
        merged = new_merged

    return merged, len(merged)

def _low_signal_fallback(output_nodes, segments, global_threshold, cascade_stats):
    """低信号回退机制 (P0-1)。

    当全局阈值 > 参考值 (5%) 时自动触发。对阈值前的数据段，
    若段内 shape/dtype 一致且存在 NRE 超过局部基线的节点，
    将这些节点作为附加候选输出。

    Args:
        output_nodes: list of dict
        segments: list of (start, end, reason) tuples
        global_threshold: 全局阈值 (%)
        cascade_stats: dict, cascade 统计量

    Returns:
        low_signal_nodes: list of dict 或空数组
    """
    if global_threshold <= 5.0:
        return []

    low_signal_nodes = []
    p50 = cascade_stats.get('p50', 0.1)
    local_baseline = min(0.1, p50)

    for seg_idx, (start, end, reason) in enumerate(segments):
        if not reason or reason in ('start', 'single_segment', 'tail'):
            continue  # First segment or single segment

        seg_nodes = output_nodes[start:end + 1]
        if not seg_nodes:
            continue

        shapes = set(n.get('shape', '').strip() for n in seg_nodes if n.get('shape', '').strip())
        dtypes = set(n.get('dtype', '').strip() for n in seg_nodes if n.get('dtype', '').strip())

        if len(shapes) > 1 or len(dtypes) > 1:
            continue  # Segment with mixed shapes/dtypes, skip

        for n in seg_nodes:
            nre = n.get('nre', 0) or 0
            if nre > local_baseline and nre < global_threshold:
                low_signal_nodes.append({
                    "prefix": n.get('name', ''),
                    "nre": round(nre, 4),
                    "row": n.get('idx', 0),
                    "segment": seg_idx,
                    "segment_reason": reason
                })

    return low_signal_nodes


def auto_detect_threshold(output_nodes, op_groups=None):
    """自适应阈值级联检测（SICD 序列变点 → 锚定回溯 → DeltaNRE 离群 → 分布间隙 → 统计兜底，含分段检测 P0-1）。

    Args:
        output_nodes: list of dict, 按执行序 (idx) 排列。
                      每个 node 含 {'idx', 'nre', 'name', 'shape', 'dtype', ...}
        op_groups: list of dict, 可选。
                   每个 group 含 {'prefix', 'direction', 'worst_input_nre', 'worst_output_nre', ...}

    Returns:
        dict: {
            'threshold': float,         # 阈值 (%)
            'method': str,              # 检测方法描述
            'confidence': str,          # 'high' | 'medium' | 'low'
            'stats': {
                'noise_ceiling': float,
                'p25': float, 'p50': float, 'p01': float, ...
            },
            'per_segment_thresholds': [...],  # P0-1: 分段阈值信息
            'segment_count': int,             # P0-1: 分段数
            'low_signal_nodes': [...]         # P0-1: 低信号回退节点
        }
    """
    if op_groups is None:
        op_groups = []

    # ==== 内部函数: 对给定节点集运行级联 (不含分段逻辑) ====
    def _run_cascade(nodes, groups):
        """Core cascade logic, returns (threshold, method, confidence, stats)."""
        t, m, s = _sicd_change_point(nodes)
        if t is not None:
            return t, m, 'high', s

        t, m, hs, reliable = _anchored_backtrack(nodes)
        if hs:
            s.update(hs)
        if t is not None and reliable:
            return t, m, 'high', s

        ft, fm, fs, has_outliers = _delta_nre_outlier(groups)
        if fs:
            s.update(fs)
        if ft is not None and has_outliers:
            return ft, fm, 'medium', s

        at, am, a_stats, found_gap = _distribution_gap(nodes)
        if a_stats:
            s.update(a_stats)
        if at is not None and found_gap:
            return at, am, 'medium', s

        if t is not None:
            return t, m + ' (unreliable)', 'low', s

        ft, fm, fs = _statistical_fallback(nodes)
        s.update(fs)
        return ft, fm, 'low', s

    # ==== P0-1: 分段检测 ====
    segments, segment_count = _detect_segments(output_nodes)

    if segment_count > 500:
        t, m, c, s = _run_cascade(output_nodes, op_groups)
        return {
            'threshold': round(t, 4),
            'method': m,
            'confidence': c,
            'stats': s,
            'per_segment_thresholds': [{
                'segment_index': 0,
                'start_row': output_nodes[0].get('idx', 0) if output_nodes else 0,
                'end_row': output_nodes[-1].get('idx', 0) if output_nodes else 0,
                'threshold': round(t, 4),
                'method': m,
                'confidence': c,
                'fallback': False
            }],
            'segment_count': 1,
            'low_signal_nodes': [],
            'segment_overflow': True  # signal that >500 segments were detected
        }

    if segment_count <= 1:
        t, m, c, s = _run_cascade(output_nodes, op_groups)
        return {
            'threshold': round(t, 4),
            'method': m,
            'confidence': c,
            'stats': s,
            'per_segment_thresholds': [{
                'segment_index': 0,
                'start_row': output_nodes[0].get('idx', 0) if output_nodes else 0,
                'end_row': output_nodes[-1].get('idx', 0) if output_nodes else 0,
                'threshold': round(t, 4),
                'method': m,
                'confidence': c,
                'fallback': False
            }],
            'segment_count': 1,
            'low_signal_nodes': []
        }

    per_segment = []
    full_seq_t, full_seq_m, full_seq_c, full_seq_s = _run_cascade(output_nodes, op_groups)

    i = 0
    while i < len(segments):
        start, end, reason = segments[i]
        seg_nodes = output_nodes[start:end + 1]

        if len(seg_nodes) < 30 and i + 1 < len(segments):
            next_start, next_end, _ = segments[i + 1]
            merged_nodes = output_nodes[start:next_end + 1]
            seg_start_row = output_nodes[start].get('idx', 0)
            seg_end_row = output_nodes[next_end].get('idx', 0)
            seg_groups = [g for g in op_groups
                          if seg_start_row <= g.get('min_row', seg_start_row) <= seg_end_row]
            t, m, c, s = _run_cascade(merged_nodes, seg_groups)
            per_segment.append({
                'segment_index': len(per_segment),
                'start_row': seg_start_row,
                'end_row': seg_end_row,
                'threshold': round(t, 4),
                'method': m,
                'confidence': c,
                'fallback': c == 'low',
                'merged_micro_segment': True
            })
            i += 2
            continue
        elif len(seg_nodes) < 30:
            i += 1
            continue

        seg_start_row = output_nodes[start].get('idx', 0)
        seg_end_row = output_nodes[end].get('idx', 0)
        seg_groups = [g for g in op_groups
                      if seg_start_row <= g.get('min_row', seg_start_row) <= seg_end_row]

        t, m, c, s = _run_cascade(seg_nodes, seg_groups)
        per_segment.append({
            'segment_index': len(per_segment),
            'start_row': seg_start_row,
            'end_row': seg_end_row,
            'threshold': round(t, 4),
            'method': m,
            'confidence': c,
            'fallback': c == 'low'
        })
        i += 1

    valid_thresholds = [ps['threshold'] for ps in per_segment
                        if ps['threshold'] is not None and not ps.get('fallback', False)]
    if valid_thresholds:
        global_threshold = min(valid_thresholds)
        best_seg = min((ps for ps in per_segment
                        if ps['threshold'] is not None and not ps.get('fallback', False)),
                       key=lambda x: x['threshold'])
        best_method = best_seg.get('method', 'segment_cascade')
        best_confidence = best_seg.get('confidence', 'medium')
    else:
        global_threshold = round(full_seq_t, 4)
        best_method = '{} (full_sequence_fallback)'.format(full_seq_m)
        best_confidence = full_seq_c

    is_clamped = (abs(global_threshold - MIN_THRESHOLD) < 0.001)
    if is_clamped:
        if 'full_sequence_fallback' not in best_method:
            best_method = '{} (clamped to {:.1f}%)'.format(best_method, MIN_THRESHOLD)
        if best_confidence == 'high':
            best_confidence = 'medium'
        elif best_confidence == 'medium':
            best_confidence = 'low'

    low_signal_nodes = _low_signal_fallback(output_nodes, segments,
                                            global_threshold, full_seq_s)

    return {
        'threshold': round(global_threshold, 4),
        'method': best_method,
        'confidence': best_confidence,
        'stats': full_seq_s,
        'per_segment_thresholds': per_segment,
        'segment_count': segment_count,
        'low_signal_nodes': low_signal_nodes
    }


def compute_max_jump_supplement(all_nodes, cascade_stats, threshold):
    """全自适应最大跳变补充规则。

    当阈值被手动设高或自适应阈值偏高时，补充可能遗漏的低 NRE 但高跳变
    的根因算子。所有门限值均来自 cascade 输出的统计量，无硬编码值。

    Args:
        all_nodes: list of dict, 所有节点 (含传播分析分类结果)。
                   每个 node 含 {'name', 'input_nre', 'output_nre', 'jump', ...}
        cascade_stats: dict, cascade 输出的统计量
        threshold: float, 主阈值 (%)

    Returns:
        list of dict: 补充候选节点
    """
    noise_ceiling = cascade_stats.get('noise_ceiling', 0.1)
    p25 = cascade_stats.get('p25', 0.1)
    p01 = cascade_stats.get('p01', 0.001)

    clean_input_threshold = max(noise_ceiling, p25)
    min_jump = max(p01 * 3.0, 0.1)
    min_output_nre = min_jump

    supplements = []

    for node in all_nodes:
        input_nre = node.get('input_nre')
        output_nre = node.get('output_nre')
        jump = node.get('jump')

        if input_nre is None or output_nre is None:
            continue

        if input_nre > clean_input_threshold:
            continue

        if output_nre < min_output_nre:
            continue

        if jump is not None and jump < min_jump:
            continue

        supplements.append(node)

    seen = set()
    unique_supplements = []
    for s in supplements:
        name = s.get('name', '')
        if name not in seen:
            seen.add(name)
            unique_supplements.append(s)

    return unique_supplements


def auto_threshold_from_rows(rows, op_groups=None):
    """从 CSV rows 计算自适应阈值 (便捷封装)。

    自动提取 output_nodes 并调用 auto_detect_threshold。

    Args:
        rows: list of dict, CSV 行数据
        op_groups: list of dict, 可选

    Returns:
        dict: cascade 结果
    """
    output_nodes = []
    for r in rows:
        name = r.get('NPU Name', '')
        if '.output.' not in name and '.parameters_grad.' not in name:
            continue

        from _common import safe_float
        nre = safe_float(r.get('NormRelativeErr', ''))
        if nre is None:
            continue

        row_index = r.get('RowIndex', 0)
        output_nodes.append({
            'idx': row_index,
            'name': name,
            'nre': nre,
        })

    output_nodes.sort(key=lambda n: n['idx'])

    return auto_detect_threshold(output_nodes, op_groups)

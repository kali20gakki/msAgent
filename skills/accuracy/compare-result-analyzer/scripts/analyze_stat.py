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
"""msProbe 统计量模式精度比对分析脚本 — 主入口

用法:
    python analyze_stat.py <compare_result.csv|xlsx> [--format json] [--keep-only <kw>]
    python analyze_stat.py <compare_result.csv|xlsx> --format json -o result.json

功能: 自适应阈值级联检测 + 传播跳变分析 + JSON 结构化输出
"""

import argparse
import json
import os
import sys
import time as _time
from collections import defaultdict

from _common import safe_float, extract_op_prefix, get_param_type
from _noise_filter import (DTYPE_EPS_BOUNDARY, _ensure_output_dir, _default_output_path,
                            classify_near_zero_noise)
from _data_io import (_setup_encoding, load_rows, filter_na_rows,
                      collect_stat_nodes, na_summary, meta_errors, _build_row_index,
                      output_nodes_detail, all_bad_nodes_detail)
from _propagation import first_problem_point, propagation_analysis
from _pooling import (dedup_root_causes_by_nre_l2,
                      trace_execution_chain, detect_data_coverage_gaps)
from cascade_threshold import compute_max_jump_supplement


def _precompute_amplifier_metadata(root_cause_list, op_groups_list, threshold):
    """预计算放大器候选元数据，供 Agent C-ANALYSIS-025 判定。"""
    EPS = 1e-12
    _worst_input = {}
    _rr_lookup = {}
    for g in (op_groups_list or []):
        wi = g.get('worst_input_nre')
        if wi is not None:
            _worst_input[(g['prefix'], g['direction'])] = wi
        _rr_lookup[(g['prefix'], g['direction'])] = g.get('row_range', [0, 0])

    by_key = {}
    for rc in root_cause_list:
        prefix, direction = rc[0], rc[1]
        if len(rc) > 10 and rc[10]:
            continue
        key = (prefix, direction)
        by_key.setdefault(key, []).append(rc)

    candidates = []
    for key, entries in by_key.items():
        prefix, direction = key
        worst_in = _worst_input.get(key)
        for rc in entries:
            inp_nre, out_nre = rc[2], rc[3] or 0
            if inp_nre is not None and inp_nre > EPS:
                ratio = out_nre / inp_nre
            elif worst_in is not None and worst_in > EPS:
                ratio = out_nre / worst_in
            else:
                ratio = float('inf') if out_nre > EPS else 0.0
            candidates.append({
                "prefix": prefix, "direction": direction,
                "input_nre": inp_nre, "output_nre": out_nre,
                "jump": rc[4] or 0,
                "amplification_ratio": round(ratio, 2) if ratio != float('inf') else 'inf',
                "worst_input_nre": worst_in,
                "all_inputs_clean": (worst_in is not None and worst_in < threshold),
                "has_param_grad_output": (len(rc) > 10 and rc[10]),
                "category": rc[7], "trigger": rc[8] if len(rc) > 8 else "",
                "row_range": _rr_lookup.get(key, [0, 0]),
            })
    candidates.sort(key=lambda x: abs(x['jump']), reverse=True)
    return candidates

def _precompute_spike_indicators(nodes, all_prop_categories, pre_filter_snapshot=None,
                                  op_groups_list=None):
    """预计算 grad_norm_spike 判定指标。"""
    has_extreme = False
    extreme_count = 0
    if pre_filter_snapshot:
        for be in pre_filter_snapshot:
            if be.get('output_nre', 0) > 100:
                has_extreme = True
                extreme_count += 1
    if not has_extreme:
        for cat_items in (all_prop_categories or {}).values():
            for item in cat_items:
                if item[1] == 'backward' and item[3] is not None and item[3] > 100:
                    has_extreme = True
                    extreme_count += 1
    if not has_extreme and op_groups_list:
        for g in op_groups_list:
            if g.get('direction') == 'backward' and (g.get('worst_output_nre') or 0) > 100:
                has_extreme = True
                extreme_count += 1

    fw_rc = sum(1 for cat_items in (all_prop_categories or {}).values()
                for item in cat_items if item[1] == 'forward')
    bw_rc = sum(1 for cat_items in (all_prop_categories or {}).values()
                for item in cat_items if item[1] == 'backward')
    total_fw = sum(1 for n in (nodes or []) if extract_op_prefix(n.get('name', ''))[1] == 'forward')
    total_bw = sum(1 for n in (nodes or []) if extract_op_prefix(n.get('name', ''))[1] == 'backward')
    ratio = (total_fw / max(total_bw, 1)) if total_bw > 0 else float('inf')

    return {
        "has_extreme_backward": has_extreme, "extreme_backward_count": extreme_count,
        "forward_rc_count": fw_rc, "backward_rc_count": bw_rc,
        "total_forward_nodes": total_fw, "total_backward_nodes": total_bw,
        "forward_backward_ratio": round(ratio, 2) if ratio != float('inf') else 'inf',
        # 与 detect_grad_norm_spike(main) 对齐：存在 backward NRE>100% 极端节点即触发，
        # 不要求 forward/backward 比值>10——完整前向+反向 dump 比值准则不适用。
        "spike_condition_met": has_extreme,
    }

def _precompute_fb_candidates(first_point, all_prop_categories, threshold,
                                pre_filter_snapshot=None):
    """预计算 FB 关联候选对（含置信度，与 main detect_fb_association 对齐）。"""
    if first_point is None:
        return []
    fp_direction = first_point.get('direction', 'forward')
    fp_prefix = first_point.get('prefix', '')

    def _confidence(nre, fp_has_jump):
        """置信度：纯信号强度判定（与 detect_fb_association main 一致）。"""
        if fp_has_jump and nre > 100:
            return 'high'
        if fp_has_jump and nre > 50:
            return 'medium'
        if nre > 1000:
            return 'medium'
        if nre > 100:
            return 'medium' if fp_has_jump else 'low'
        return 'low'

    if fp_direction == 'backward':
        backward_extreme = []
        seen = set()
        for cat_items in (all_prop_categories or {}).values():
            for item in cat_items:
                if item[1] != 'backward':
                    continue
                prefix, out_nre = item[0], item[3]
                if out_nre is None or prefix in seen:
                    continue
                if out_nre > 50:
                    seen.add(prefix)
                    backward_extreme.append({
                        "forward_prefix": None, "backward_prefix": prefix,
                        "backward_nre": out_nre, "forward_jump": None,
                        "direction": "backward_dominant", "source_category": "propagation",
                        "confidence": _confidence(out_nre, False),
                        "inference": "首问题点在 backward 方向，基于 backward 信号强度判定",
                    })
        backward_extreme.sort(key=lambda x: x['backward_nre'] or 0, reverse=True)
        return backward_extreme

    forward_jump = 0.0
    for cat_items in (all_prop_categories or {}).values():
        for item in cat_items:
            if item[0] == fp_prefix and item[1] == 'forward':
                forward_jump = max(forward_jump, item[4] or 0)
    fp_has_jump = forward_jump > (threshold or 0)

    candidates, seen = [], set()
    for cat_items in (all_prop_categories or {}).values():
        for item in cat_items:
            if item[1] != 'backward':
                continue
            prefix, out_nre = item[0], item[3]
            if out_nre is None or prefix in seen or out_nre <= 50:
                continue
            seen.add(prefix)
            candidates.append({
                "forward_prefix": fp_prefix, "backward_prefix": prefix,
                "backward_nre": out_nre, "forward_jump": forward_jump,
                "direction": "forward_dominant", "source_category": "propagation",
                "confidence": _confidence(out_nre, fp_has_jump),
                "inference": "前向首问题点 {} 存在跳变 (Jump={:.2f}%), backward 方向 {} NRE={:.2f}% — 其 backward 实现可能存在问题".format(
                    fp_prefix, forward_jump, prefix, out_nre),
            })
    if pre_filter_snapshot:
        for be in pre_filter_snapshot:
            if be.get('output_nre', 0) > 50 and be.get('prefix') not in seen:
                seen.add(be['prefix'])
                candidates.append({
                    "forward_prefix": fp_prefix, "backward_prefix": be['prefix'],
                    "backward_nre": be['output_nre'], "forward_jump": forward_jump,
                    "direction": "forward_dominant", "source_category": "pre_filter_snapshot",
                    "confidence": _confidence(be['output_nre'], fp_has_jump),
                    "inference": "前向首问题点 {} 存在跳变, backward 方向 {} (过滤前快照) NRE={:.2f}% — 其 backward 实现可能存在问题".format(
                        fp_prefix, be['prefix'], be['output_nre']),
                })
    candidates.sort(key=lambda x: (0 if x.get('confidence') == 'high' else 1, -x['backward_nre']))
    return candidates

def _precompute_pool_external_indicators(all_prop, top_root_causes, op_groups_list, threshold,
                                         first_point=None):
    """预计算池外候选指标。"""
    _rr_lookup = {}
    for g in (op_groups_list or []):
        _rr_lookup[(g['prefix'], g['direction'])] = g.get('row_range', [0, 0])

    covered = set()
    for direction in ['forward', 'backward']:
        for entry in (top_root_causes or {}).get(direction, []):
            covered.add((entry.get('prefix', ''), entry.get('direction', direction)))
    # 首问题点不参与补充候选（已在 §3 置顶呈现）
    if first_point:
        covered.add((first_point.get('prefix', ''), first_point.get('direction', 'backward')))

    all_entries = []
    for cat_name in ['root_cause', 'propagation', 'pass_through', 'input_propagation']:
        for entry in (all_prop or {}).get(cat_name, []):
            prefix, direction = entry[0], entry[1]
            if (prefix, direction) in covered:
                continue
            output_nre = entry[3]
            if output_nre is None or output_nre < threshold:
                continue
            all_entries.append({
                'prefix': prefix, 'direction': direction,
                'output_nre': output_nre, 'input_nre': entry[2], 'jump': entry[4],
                'row_start': _rr_lookup.get((prefix, direction), [0, 0])[0],
                'row_range': _rr_lookup.get((prefix, direction), [0, 0]),
                'category': entry[7] if len(entry) > 7 else cat_name,
            })

    if not all_entries:
        return []

    def _family_key(prefix):
        parts = prefix.split('.')
        if parts and parts[0] == 'Module':
            return '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        return parts[0] if parts else prefix

    families = defaultdict(list)
    for e in all_entries:
        families[_family_key(e['prefix'])].append(e)
    family_earliest = {}
    for fk, members in families.items():
        members.sort(key=lambda x: x['row_start'])
        family_earliest[fk] = members[0]['prefix']

    result = []
    for e in all_entries:
        fk = _family_key(e['prefix'])
        result.append({
            'prefix': e['prefix'], 'direction': e['direction'],
            'output_nre': e['output_nre'], 'input_nre': e.get('input_nre'),
            'jump': e.get('jump'), 'row_range': e['row_range'],
            'family_key': fk,
            'is_earliest_in_family': (e['prefix'] == family_earliest.get(fk)),
            'is_param_grad_no_input': ('.parameters_grad.' in e['prefix'] and e.get('input_nre') is None),
            'category': e['category'],
        })
    result.sort(key=lambda x: x['row_range'][0])
    return result


def _precompute_param_grad_three_category(all_prop, op_groups_list, threshold):
    """参数梯度三分类候选（C-ANALYSIS-030）：同块成堆 / 孤立大NRE / 执行序靠前。

    数据源：所有 NRE ≥ threshold 的 backward 参数梯度输出（`.parameters_grad.` 前缀）。
    三类并列呈现、不互斥；各条目保留既有传播分类标注（inheritance / param_grad_output）。
    每类至多 5 条，按 NRE 降序。
    """
    _rr_lookup = {}
    for g in (op_groups_list or []):
        _rr_lookup[(g['prefix'], g['direction'])] = g.get('row_range', [0, 0])

    candidates = []
    for cat_name in ['root_cause', 'input_propagation', 'propagation', 'pass_through', 'absorbed']:
        for entry in (all_prop or {}).get(cat_name, []):
            prefix, direction = entry[0], entry[1]
            if direction != 'backward' or '.parameters_grad.' not in prefix:
                continue
            output_nre = entry[3]
            if output_nre is None or output_nre < threshold:
                continue
            candidates.append({
                'prefix': prefix, 'direction': 'backward',
                'output_nre': output_nre, 'input_nre': entry[2],
                'jump': entry[4],
                'row_range': _rr_lookup.get((prefix, direction), [0, 0]),
                'row_start': _rr_lookup.get((prefix, direction), [0, 0])[0],
                'category': entry[7] if len(entry) > 7 else cat_name,
                'param_grad_output': entry[10] if len(entry) > 10 else False,
                'inheritance': entry[12] if len(entry) > 12 else None,
            })

    empty = {"same_block_cluster": [], "isolated_large_nre": [], "execution_order_first": []}
    if not candidates:
        return empty

    def _family_key(prefix):
        parts = prefix.split('.')
        if parts and parts[0] == 'Module':
            return '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        return parts[0] if parts else prefix

    # 1. 同块成堆：同一子模块块（同一模块路径前缀）内 ≥2 个参数梯度超标
    families = defaultdict(list)
    for c in candidates:
        families[_family_key(c['prefix'])].append(c)
    same_block_cluster = []
    for fk, members in families.items():
        if len(members) >= 2:
            members.sort(key=lambda x: -x['output_nre'])
            same_block_cluster.extend(members)
    same_block_cluster.sort(key=lambda x: -x['output_nre'])
    same_block_cluster = same_block_cluster[:5]

    # 2. 孤立大NRE：单点绝对 NRE 最大
    isolated_large_nre = sorted(candidates, key=lambda x: -x['output_nre'])[:5]

    # 3. 执行序靠前：row_start 最小（执行序最早出现的超标参数梯度）
    execution_order_first = sorted(candidates, key=lambda x: (x['row_start'], -x['output_nre']))[:5]

    return {
        "same_block_cluster": same_block_cluster,
        "isolated_large_nre": isolated_large_nre,
        "execution_order_first": execution_order_first,
    }

def build_pool_input(root_cause_list, op_groups_list, threshold, first_point_row_start=0):
    """构建池输入——元数据 + 标签，不做排名/合并。Agent 按 C-ANALYSIS-027 执行三池合并。"""
    EPS = 1e-12
    _meta_lookup = {}
    _worst_input_lookup = {}
    if op_groups_list:
        for g in op_groups_list:
            key = (g['prefix'], g['direction'])
            params = g.get('params', {})
            _bench_l2, _max_diff, _row_start = None, None, g.get('row_range', [0, 0])[0]
            for pk, pv in params.items():
                pt = get_param_type(pk)
                if pt == 'output':
                    bl2 = safe_float(pv.get('Bench l2norm', ''))
                    md = safe_float(pv.get('Max diff', ''))
                    if bl2 is not None:
                        _bench_l2 = max(_bench_l2 or 0, bl2)
                    if md is not None:
                        _max_diff = max(_max_diff or 0, abs(md))
            _meta_lookup[key] = {'bench_l2': _bench_l2, 'max_diff': _max_diff,
                                  'row_start': _row_start, 'row_range': g.get('row_range', [0, 0])}
            wi = g.get('worst_input_nre')
            if wi is not None:
                _worst_input_lookup[key] = wi

    def _tagging(rc):
        meta = _meta_lookup.get((rc[0], rc[1]), {})
        bl2, md = meta.get('bench_l2'), meta.get('max_diff')
        if bl2 and bl2 > 0 and md is not None:
            if abs(md) <= DTYPE_EPS_BOUNDARY.get('torch.float32', 1.2e-7) * bl2:
                return 'denominator_effect'
        if bl2 and bl2 < 1e-3 and md is not None and abs(md) < 0.01:
            return 'small_magnitude'
        return 'none'

    def _is_true_rc(rc):
        if threshold is None:
            return False
        inp, out = rc[2], rc[3]
        if inp is None or out is None or inp >= threshold or out < threshold:
            return False
        worst_in = _worst_input_lookup.get((rc[0], rc[1]))
        ratio = (out / inp) if inp > EPS else ((out / worst_in) if (worst_in and worst_in > EPS) else (float('inf') if out > EPS else 0.0))
        return ratio > 2.0

    def _serialize(rc):
        meta = _meta_lookup.get((rc[0], rc[1]), {})
        return {
            "prefix": rc[0], "direction": rc[1],
            "row_range": meta.get('row_range', [0, 0]),
            "row_start": meta.get('row_start', 0),
            "input_nre": rc[2], "output_nre": rc[3], "jump": rc[4],
            "input_mean_re": rc[5], "output_mean_re": rc[6],
            "category": rc[7], "trigger": rc[8] if len(rc) > 8 else "",
            "inheritance": rc[12] if len(rc) > 12 else None,
            "bench_l2": meta.get('bench_l2'), "max_diff": meta.get('max_diff'),
            "tagging": _tagging(rc),
            "is_true_root_cause_feature": _is_true_rc(rc),
            "abs_magnitude": abs(rc[4] or 0) if rc[2] is not None else abs(rc[3] or 0),
        }

    fp_start = first_point_row_start or 0
    def _filter(items):
        eligible = [it for it in items
                    if _meta_lookup.get((it[0], it[1]), {}).get('row_start', 0) >= fp_start]
        return [_serialize(it) for it in eligible]

    fwd = [rc for rc in root_cause_list if rc[1] == 'forward' and '_foreach_norm' not in rc[0]]
    bwd = [rc for rc in root_cause_list if rc[1] == 'backward' and '_foreach_norm' not in rc[0]]
    return {"forward": _filter(fwd), "backward": _filter(bwd), "first_point_row_start": fp_start}


def _merge_pool_top(pool_input):
    """按 C-ANALYSIS-027 三池合并（保底 ∪ 执行序 ∪ 量级），写入 top_root_causes。

    整表（forward+backward 合并）总上限 15：保底 ≤5 + 常规 ≤10，不区分方向。
    保底保证入选、不改变位置；整表统一按 row_start 升序。条目保留 direction 标注。
    """
    combined = (pool_input.get('forward') or []) + (pool_input.get('backward') or [])

    def _key(e):
        return (e.get('prefix'), e.get('direction'))

    # 去重：同一 (prefix, direction) 多输入路径合并为一行（保留首现）
    seen = set()
    deduped = []
    for e in combined:
        k = _key(e)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(e)
    combined = deduped
    if not combined:
        return {"forward": [], "backward": []}

    # 保底池：is_true_root_cause_feature == true，按 row_start 升序取前 5
    ground = [e for e in combined if e.get('is_true_root_cause_feature')]
    ground.sort(key=lambda e: (e.get('row_start') or 0, e.get('prefix') or ''))
    ground = ground[:5]

    # 执行序池：按 row_start 升序取前 10
    exec_order = sorted(combined,
                        key=lambda e: (e.get('row_start') or 0, e.get('prefix') or ''))[:10]

    # 量级池：排除 denominator_effect，按 abs_magnitude 降序取前 10
    magnitude = [e for e in combined if e.get('tagging') != 'denominator_effect']
    magnitude.sort(key=lambda e: e.get('abs_magnitude') or 0, reverse=True)
    magnitude = magnitude[:10]

    # 保底保证入选（整表上限内优先纳入保底，不改变排序位置）
    selected = list(ground)
    seen = {_key(e) for e in selected}
    regular = []
    for e in exec_order + magnitude:
        k = _key(e)
        if k in seen:
            continue
        seen.add(k)
        regular.append(e)
    regular.sort(key=lambda e: (e.get('row_start') or 0, e.get('prefix') or ''))

    # 常规补齐到整表上限 15
    remaining = 15 - len(selected)
    if remaining > 0:
        selected += regular[:remaining]

    # 整表统一按 row_start 升序（出现顺序为最高优先级）
    merged = sorted(selected, key=lambda e: (e.get('row_start') or 0, e.get('prefix') or ''))

    # 按方向拆分存储（§5.1 单张表含方向列）
    return {
        "forward": [e for e in merged if e.get('direction') == 'forward'],
        "backward": [e for e in merged if e.get('direction') == 'backward'],
    }


def main():
    parser = argparse.ArgumentParser(description='msProbe 统计量模式精度比对分析 (CSV/XLSX)')
    parser.add_argument('csv_path', nargs='?', help='compare_result CSV 或 XLSX 文件路径')
    parser.add_argument('--keep-na', action='store_true', help='保留 N/A 行')
    parser.add_argument('--format', choices=['json'], default='json',
                        help='输出格式: json (结构化)')
    parser.add_argument('--summary-only', action='store_true',
                        help='仅输出分析结论，跳过 §D/§E 明细和 op_groups per-param 明细')
    parser.add_argument('-o', '--output', help='JSON 输出文件路径 (默认: <csv_dir>/.compare_result_analyzer/<csv_stem>_result.json)')
    parser.add_argument('--analysis-range', type=int, default=500, metavar='N',
                        help='上下游分析范围 (行数，默认 500)')
    parser.add_argument('--keep-only', default=None,
                        help='场景定向过滤: 仅保留 NPU Name 含指定关键词的行 (如 --keep-only parameters_grad)')

    args = parser.parse_args()

    # Normal analysis mode: require csv_path
    if not args.csv_path:
        parser.print_help()
        sys.exit(1)

    format_mode = args.format

    # 默认启用自适应阈值级联检测
    rows_for_cascade = load_rows(args.csv_path)
    filtered_for_cascade = filter_na_rows(rows_for_cascade) if not args.keep_na else rows_for_cascade
    cascade_nodes = collect_stat_nodes(filtered_for_cascade)

    from cascade_threshold import auto_detect_threshold as _cascade
    cascade_output_nodes = [n for n in cascade_nodes
                            if '.output.' in n.get('name', '') or '.parameters_grad.' in n.get('name', '')]
    cascade_output_nodes.sort(key=lambda n: n.get('idx', 0))

    from _common import extract_op_prefix as _extract
    op_prefixes_temp = {}
    for r in filtered_for_cascade:
        name = r.get('NPU Name', '')
        prefix, direction = _extract(name)
        if direction == 'unknown':
            continue
        if prefix not in op_prefixes_temp:
            op_prefixes_temp[prefix] = {}
        if direction not in op_prefixes_temp[prefix]:
            op_prefixes_temp[prefix][direction] = {'inputs': [], 'outputs': []}
        from _common import get_param_key as _gk, get_param_type as _gt
        pk = _gk(name, prefix, direction)
        pt = _gt(pk)
        nre = safe_float(r.get('NormRelativeErr', ''))
        if pt == 'input':
            op_prefixes_temp[prefix][direction]['inputs'].append(nre)
        else:
            op_prefixes_temp[prefix][direction]['outputs'].append(nre)

    cascade_op_groups = []
    for prefix, dirs in op_prefixes_temp.items():
        for direction, io in dirs.items():
            worst_in = max((n for n in io['inputs'] if n is not None), default=None)
            worst_out = max((n for n in io['outputs'] if n is not None), default=None)
            cascade_op_groups.append({
                'prefix': prefix,
                'direction': direction,
                'worst_input_nre': worst_in,
                'worst_output_nre': worst_out
            })

    cascade_result = _cascade(cascade_output_nodes, cascade_op_groups)
    threshold = cascade_result['threshold']
    csv_path = args.csv_path

    rows = load_rows(csv_path)

    _keep_only_applied = False
    if args.keep_only:
        _kw = args.keep_only.lower()
        rows = [r for r in rows if _kw in r.get('NPU Name', '').lower()]
        _keep_only_applied = True
        if not rows:
            print("⚠ --keep-only '{}' 过滤后无数据行 (共 0 行)，请检查过滤关键词或使用全量分析。".format(
                args.keep_only), file=sys.stderr)
            # 输出空结果 JSON 避免静默失败
            _empty_result = {
                "meta": {"file_path": os.path.abspath(csv_path),
                         "threshold": threshold, "total_rows": 0,
                         "valid_rows": 0, "keep_only_filter": args.keep_only},
                "error": "keep-only filter returned 0 rows",
                "top_root_causes": {"forward": [], "backward": []},
                "significant_amplifiers": [],
                "fb_association": [],
                "scenario_flags": {"grad_norm_spike": False, "spike_has_extreme_backward": False}
            }
            _out_dir = _ensure_output_dir(csv_path)
            _kw_suffix = args.keep_only.lower().replace(' ', '_').replace('/', '_')
            csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
            _out_path = os.path.join(_out_dir, '{}_{}_result.json'.format(csv_stem, _kw_suffix))
            with open(_out_path, 'w', encoding='utf-8') as f:
                json.dump(_empty_result, f, ensure_ascii=False, indent=2)
            print("空结果已写入: {}".format(_out_path), file=sys.stderr)
            from _common import flush_unparseable_names as _flush_unparseable
            _flush_unparseable()
            return

    if not rows:
        return

    # Progress output to stderr
    import time as _time
    _t0 = _time.time()
    def _progress(stage, extra=''):
        _elapsed = _time.time() - _t0
        print("[{}/4] {}... ({:.1f}s){}".format(stage, extra, _elapsed, ''),
              file=sys.stderr)

    _progress(1, 'Loading data')

    filtered_rows = filter_na_rows(rows) if not args.keep_na else rows

    # ==== Collect all return values from analysis functions ====
    meta_result = na_summary(rows, filtered_rows, threshold, csv_path, analysis_range=args.analysis_range)

    _elapsed = _time.time() - _t0
    _t_stage1 = _time.time()
    print("[1/4] Loading data... ({} rows, {:.1f}s)".format(len(filtered_rows), _elapsed),
          file=sys.stderr)

    nodes = collect_stat_nodes(filtered_rows)
    meta_errors_result = meta_errors(filtered_rows)

    # P1-5: 数据覆盖缺口检测
    data_coverage_gaps = detect_data_coverage_gaps(filtered_rows, threshold)
    meta_errors_result['data_coverage_gaps'] = data_coverage_gaps

    # ==== P0-2: Build pre-filter root_cause_snapshot BEFORE noise filtering ====
    _pre_filter_snapshot = []  # list of {prefix, direction, nre, jump, bench_l2}
    _grad_norm_spike_pre = False
    _backward_extreme_pre = []
    _forward_count_pre = 0
    _backward_count_pre = 0
    for n in nodes:
        name = n.get('name', '')
        nre = n.get('nre')
        if nre is None:
            continue
        prefix, direction = extract_op_prefix(name)
        if direction == 'unknown':
            continue
        if direction == 'backward':
            _backward_count_pre += 1
            if nre > 50:
                _backward_extreme_pre.append({
                    'prefix': prefix,
                    'direction': direction,
                    'output_nre': nre,
                    'jump': 0.0,  # no input NRE context yet, set as param-grad
                    'bench_l2': n.get('bench_l2norm'),
                    'max_diff': n.get('max_diff')
                })
        elif direction == 'forward':
            _forward_count_pre += 1

    # Dedup pre-filter extreme backward by prefix (keep highest NRE)
    _seen_pf = set()
    _deduped_pf = []
    for be in sorted(_backward_extreme_pre, key=lambda x: x['output_nre'], reverse=True):
        if be['prefix'] not in _seen_pf:
            _seen_pf.add(be['prefix'])
            _deduped_pf.append(be)
    _pre_filter_snapshot = _deduped_pf

    if _backward_count_pre > 0:
        _extreme_count = sum(1 for be in _pre_filter_snapshot if be['output_nre'] > 100)
        _grad_norm_spike_pre = (_extreme_count > 0
                               and _forward_count_pre / _backward_count_pre > 10.0)

    # 近零噪声过滤
    noise_nodes, cutoff_info, noise_filter_result = classify_near_zero_noise(
        nodes, threshold, grad_norm_spike=_grad_norm_spike_pre)

    # to skip, ensuring consistency with first_problem_point data source.
    noise_node_names = {n['name'] for n in noise_nodes}

    output_nodes, output_nodes_result = output_nodes_detail(nodes)

    _build_row_index(filtered_rows)

    # 首问题点
    first, first_skipped, first_point_result = first_problem_point(
        output_nodes, threshold, all_rows=filtered_rows, analysis_range=args.analysis_range)

    # 传播跳变分析
    root_cause, absorbed, propagation, input_propagation, pass_through, op_groups_list = \
        propagation_analysis(filtered_rows, threshold, noise_node_names=noise_node_names)

    _elapsed = _time.time() - _t_stage1
    print("[2/4] Cascade threshold... (method={}, threshold={:.4f}%, {:.1f}s)".format(
        cascade_result['method'], threshold, _elapsed), file=sys.stderr)
    _t_stage2 = _time.time()

    # Task 9.4: 参数梯度空间阈值校准
    # Calculate p95 of backward parameters_grad NRE values.
    # If full-row threshold < p95, report threshold discrepancy.
    _pgrad_nres = []
    for n in nodes:
        name = n.get('name', '')
        nre = n.get('nre')
        # parameters_grad 行名不含 .backward 段
        if nre is not None and '.parameters_grad.' in name:
            _pgrad_nres.append(nre)
    _pgrad_p95 = None
    _pgrad_threshold_discrepancy = False
    if _pgrad_nres:
        _sorted_pgrad = sorted(_pgrad_nres)
        _p95_idx = int(len(_sorted_pgrad) * 0.95)
        _pgrad_p95 = _sorted_pgrad[min(_p95_idx, len(_sorted_pgrad) - 1)]
        _pgrad_threshold_discrepancy = threshold < _pgrad_p95

    # 超阈值节点明细
    bad_nodes, bad_nodes_result = all_bad_nodes_detail(nodes, threshold)

    # ==== P0-1: Get first_point row_start for C-ANALYSIS-013 A-pool filter ====
    _fp_confirmed_pre = first_point_result.get('confirmed')
    _first_point_row_start = 0
    if _fp_confirmed_pre:
        _fp_prefix_pre = _fp_confirmed_pre.get('prefix', '')
        _fp_name_pre = _fp_confirmed_pre.get('name', '')
        _fp_dir_pre = extract_op_prefix(_fp_name_pre)[1]
        if _fp_dir_pre == 'unknown':
            _fp_dir_pre = 'forward'
        for g in op_groups_list:
            if g['prefix'] == _fp_prefix_pre and g['direction'] == _fp_dir_pre:
                _first_point_row_start = g.get('row_range', [0, 0])[0]
                break

    # 方向分池：脚本按 C-ANALYSIS-027 三池合并写入 top_root_causes（Agent 直接读取）
    _root_cause_for_pool = dedup_root_causes_by_nre_l2(root_cause)
    pool_input = build_pool_input(_root_cause_for_pool, op_groups_list, threshold, _first_point_row_start)

    all_prop = {
        'root_cause': root_cause,
        'propagation': propagation,
        'pass_through': pass_through,
        'input_propagation': input_propagation,
        'absorbed': absorbed
    }

    # Pre-computed metadata fields — Agent 按 constraints.md 规则消费
    amplifier_candidates = _precompute_amplifier_metadata(root_cause, op_groups_list, threshold)
    spike_indicators = _precompute_spike_indicators(nodes, all_prop, _pre_filter_snapshot, op_groups_list)
    fb_candidates = _precompute_fb_candidates(first_point_result.get('confirmed'), all_prop, threshold,
                                               _pre_filter_snapshot)

    def _is_amp_gt_2(a):
        r = a.get('amplification_ratio')
        return r == 'inf' or (isinstance(r, (int, float)) and r > 2.0)

    significant_amplifiers = [a for a in amplifier_candidates
                              if a.get('all_inputs_clean') and _is_amp_gt_2(a)
                              and a.get('output_nre', 0) < threshold][:5]

    top_root_causes = _merge_pool_top(pool_input)
    scenario_flags = {"grad_norm_spike": spike_indicators.get('spike_condition_met', False),
                      "spike_has_extreme_backward": spike_indicators.get('has_extreme_backward', False)}

    # ==== Grad Norm Spike 场景：FB 关联同族 backward 置顶 ====
    # main 分支通过 high_conf_fb 后处理把与首点同族的 backward 关联置顶进候选池，
    # 否则 PROPAGATION 分类的 backward（如 SDPA.backward）永远进不了 pool_input。
    # 此逻辑恢复该能力：spike 场景下，从 fb_candidates 中挑与首点同族、
    # backward_nre 最大的关联 backward，置顶进 top_root_causes.backward。
    if scenario_flags.get('grad_norm_spike'):
        top_root_causes['forward'] = []  # C-ANALYSIS-016: spike 下前向候选清空
        _fp_c = first_point_result.get('confirmed') or {}
        _fp_prefix = _fp_c.get('prefix', '')
        _fp_family = _fp_prefix.rsplit('.forward', 1)[0].rsplit('.backward', 1)[0] if _fp_prefix else ''
        _rr_lookup_fb = {(g['prefix'], g['direction']): g.get('row_range', [0, 0])
                         for g in op_groups_list}
        _ordered_fb = sorted([fb for fb in fb_candidates if fb.get('backward_prefix')],
                             key=lambda x: x.get('backward_nre') or 0, reverse=True)
        _fb_seen = set()
        _fb_top_entries = []
        for _fb_item in _ordered_fb:
            _bw = _fb_item['backward_prefix']
            if _bw in _fb_seen:
                continue
            _bw_family = _bw.rsplit('.forward', 1)[0].rsplit('.backward', 1)[0]
            if _fp_family and _bw_family != _fp_family:
                continue  # 只置顶与首点同族的 backward（首点是该误差链的直接入口）
            _fb_seen.add(_bw)
            for _sl in (root_cause, propagation):
                _rc = next((rc for rc in _sl if rc[0] == _bw and rc[1] == 'backward'), None)
                if _rc:
                    _fb_top_entries.append({
                        "prefix": _rc[0], "direction": _rc[1],
                        "row_range": _rr_lookup_fb.get((_rc[0], _rc[1]), [0, 0]),
                        "input_nre": _rc[2], "output_nre": _rc[3],
                        "jump": (_rc[3] or 0) - (_rc[2] or 0),
                        "input_mean_re": _rc[5], "output_mean_re": _rc[6],
                        "category": "ROOT_CAUSE (FB high confidence)",
                        "trigger": _rc[8] if len(_rc) > 8 else "",
                        "fb_associated": True,
                    })
                    break
            else:
                # 同族 backward 未进传播分类（如 input 全缺失导致跳过）——
                # 用 fb_candidate 自身数据置顶，该类节点在 spike 场景同样是关键信号
                _fb_top_entries.append({
                    "prefix": _bw, "direction": 'backward',
                    "row_range": _rr_lookup_fb.get((_bw, 'backward'), [0, 0]),
                    "input_nre": None, "output_nre": _fb_item.get('backward_nre'),
                    "jump": None,
                    "input_mean_re": None, "output_mean_re": None,
                    "category": "ROOT_CAUSE (FB high confidence)",
                    "trigger": "",
                    "fb_associated": True,
                })
            if len(_fb_top_entries) >= 3:
                break
        top_root_causes['backward'] = _fb_top_entries + top_root_causes['backward']

    fb_association = fb_candidates
    pool_external_indicators = _precompute_pool_external_indicators(
        all_prop, top_root_causes, op_groups_list, threshold, first_point_result.get('confirmed'))
    param_grad_three_category = _precompute_param_grad_three_category(
        all_prop, op_groups_list, threshold)

    summary_result = {
        "first_point": {
            "name": first['name'] if first else None,
            "trigger": first.get('_trigger', 'NRE') if first else None,
            "nre": first['nre'] if first else None,
            "mean_re": first.get('mean_re') if first else None
        },
        "root_cause_count": len(root_cause),
        "propagation_count": len(propagation),
        "pass_through_count": len(pass_through),
        "input_propagation_count": len(input_propagation),
        "absorbed_count": len(absorbed),
        "noise_filtered_count": len(noise_nodes) if noise_nodes else 0,
        "noise_cutoff_info": cutoff_info or {},
        "threshold": threshold,
    }

    # ==== 执行顺序链追溯 (P5) ====
    first_problem_trace = trace_execution_chain(
        first_point_result.get('confirmed'), filtered_rows, threshold, max_range=args.analysis_range
    )

    # 全自适应最大跳变补充
    max_jump_supplements = []
    if cascade_result:
        cascade_stats = cascade_result['stats']
        # 合并 root_cause + propagation 条目，转换为 node dict 格式供 supplement 扫描
        all_prop_nodes = []
        for entry in root_cause + propagation + pass_through:
            prefix, direction, inp_nre, out_nre, jump, inp_mean, out_mean, category = entry[:8]
            trigger = entry[8] if len(entry) > 8 else ''
            all_prop_nodes.append({
                'name': '{}:{}'.format(prefix, direction),
                'input_nre': inp_nre,
                'output_nre': out_nre,
                'jump': jump,
                'category': category,
                'trigger': trigger
            })
        max_jump_supplements = compute_max_jump_supplement(all_prop_nodes, cascade_stats, threshold)

    # ==== JSON output ====
    _elapsed = _time.time() - _t_stage2
    _root_cause_count = len(root_cause)
    print("[3/4] Propagation analysis... ({} root causes, {:.1f}s)".format(
            _root_cause_count, _elapsed), file=sys.stderr)
    _t_stage3 = _time.time()

    # Build propagation dict
    _row_range_lookup = {
            (g['prefix'], g['direction']): g.get('row_range', [0, 0])
            for g in op_groups_list
    }

    def _serialize_prop(items):
            return [{
                "prefix": it[0],
                "direction": it[1],
                "row_range": _row_range_lookup.get((it[0], it[1]), [0, 0]),
                "input_nre": it[2],
                "output_nre": it[3],
                "jump": it[4],
                "input_mean_re": it[5],
                "output_mean_re": it[6],
                # Task 9.1: Mark _foreach_norm monitoring nodes
                "category": "monitor" if '_foreach_norm' in it[0] else it[7],
                "trigger": it[8] if len(it) > 8 else "",
                "is_param": it[9] if len(it) > 9 else False,
                "param_grad_output": it[10] if len(it) > 10 else False,
                "no_downstream_consumer": it[11] if len(it) > 11 else False,
                "inheritance": it[12] if len(it) > 12 else None
            } for it in items]

    propagation_result = {
            "root_cause": _serialize_prop(root_cause),
            "propagation": _serialize_prop(propagation),
            "pass_through": _serialize_prop(pass_through),
            "input_propagation": _serialize_prop(input_propagation),
            "absorbed": _serialize_prop(absorbed)
    }

    # P5#10: Mark shape-mismatched entries as nre_unreliable when critical
    if meta_errors_result.get('shape_mismatch_level') == 'critical':
            shape_mismatched_prefixes = set()
            for r in filtered_rows:
                if r.get('shape_inconsistent'):
                    prefix, _ = extract_op_prefix(r.get('NPU Name', ''))
                    if prefix:
                        shape_mismatched_prefixes.add(prefix)
            for cat_name in ['root_cause', 'propagation', 'pass_through']:
                for entry in propagation_result.get(cat_name, []):
                    if entry['prefix'] in shape_mismatched_prefixes:
                        entry['nre_unreliable'] = True

    if args.summary_only:
            output_nodes_result = []
            bad_nodes_result = []
            # Strip per-param detail from op_groups
            for g in op_groups_list:
                g.pop('params', None)

    analysis_result = {
            "meta": meta_result,
            "first_point": first_point_result,
            "propagation": propagation_result,
            "noise_filter": noise_filter_result,
            # Task 4.3: Mark that propagation analysis uses the same filtered
            # data as first_problem_point discovery
            "noise_filter_applied_to_propagation": True,
            # P0-2: Attach pre_filter_root_cause_snapshot to noise_filter for
            # fb_association and grad_norm_spike traceability
            "pre_filter_root_cause_snapshot": _pre_filter_snapshot,
            "meta_errors": meta_errors_result,
            "op_groups": op_groups_list,
            "output_nodes": output_nodes_result,
            "all_bad_nodes": bad_nodes_result,
            "summary": summary_result,
            "pool_input": pool_input,
            "top_root_causes": top_root_causes,
            "amplifier_candidates": amplifier_candidates,
            "significant_amplifiers": significant_amplifiers,
            "spike_indicators": spike_indicators,
            "fb_association_candidates": fb_candidates,
            "pool_external_indicators": pool_external_indicators,
            "param_grad_three_category": param_grad_three_category,
            "fb_association": fb_association,
            "scenario_flags": scenario_flags,
            # Task 9.4: Parameter gradient space threshold calibration
            "parameter_gradient_calibration": {
                "p95_nre": round(_pgrad_p95, 6) if _pgrad_p95 is not None else None,
                "sample_count": len(_pgrad_nres),
                "threshold_discrepancy": _pgrad_threshold_discrepancy,
                "note": ("⚠️ 全行集阈值低于参数梯度空间噪声底 p95" if _pgrad_threshold_discrepancy
                         else None)
            },
            "first_problem_trace": first_problem_trace
    }

    if cascade_result:
            # P5#13: Compute noise_ceiling from DTYPE_EPS_BOUNDARY minimum
            noise_ceiling = min(DTYPE_EPS_BOUNDARY.values()) * 100  # convert to percentage
            threshold_conf = cascade_result.get('confidence', 'medium')
            warnings_list = []
            if threshold < noise_ceiling / 100:
                threshold_conf = 'low'
                warnings_list.append("threshold_far_below_noise_ceiling")

            analysis_result["auto_threshold"] = {
                "threshold": cascade_result['threshold'],
                "method": cascade_result['method'],
                "confidence": threshold_conf,
                "warnings": warnings_list if warnings_list else None,
                "noise_ceiling_pct": round(noise_ceiling, 8),
                "stats": {k: v for k, v in cascade_result['stats'].items()
                          if isinstance(v, (int, float, str, bool, type(None)))},
                # P0-1: 分段阈值检测
                "per_segment_thresholds": cascade_result.get('per_segment_thresholds', []),
                "segment_count": cascade_result.get('segment_count', 1),
                "low_signal_nodes": cascade_result.get('low_signal_nodes', [])
            }
            analysis_result["max_jump_supplements"] = [
                {"name": s['name'], "input_nre": s.get('input_nre'),
                 "output_nre": s.get('output_nre'), "jump": s.get('jump'),
                 "category": s.get('category', ''), "trigger": s.get('trigger', '')}
                for s in max_jump_supplements
            ]

    json_str = json.dumps(analysis_result, indent=2, ensure_ascii=False, allow_nan=True)

    if args.output:
            output_path = args.output
    elif _keep_only_applied:
            # P2-7: --keep-only 使用独立输出路径，避免覆盖全量结果
            _kw_suffix = args.keep_only.lower().replace(' ', '_').replace('/', '_')
            csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
            _out_dir = _ensure_output_dir(csv_path)
            output_path = os.path.join(_out_dir, '{}_{}_result.json'.format(csv_stem, _kw_suffix))
    else:
            output_path = _default_output_path(csv_path)
    _ensure_output_dir(csv_path)

    with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str + '\n')

    _elapsed = _time.time() - _t_stage3
    print("[4/4] Writing JSON... ({:.1f}s)".format(_elapsed), file=sys.stderr)
    _total_elapsed = _time.time() - _t0
    print("JSON written to: {} (total: {:.1f}s)".format(output_path, _total_elapsed))

    # 汇总无法解析的 NPU Name 命名格式告警
    from _common import flush_unparseable_names as _flush_unparseable
    _flush_unparseable()

if __name__ == '__main__':
    try:
        main()
    except UnicodeEncodeError:
        print("ERROR: Unicode encode failure - try setting PYTHONIOENCODING=utf-8", file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        sys.stderr.close()
        sys.exit(0)

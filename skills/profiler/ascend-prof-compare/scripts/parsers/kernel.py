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

"""Kernel 比对解析器（合并 kernel_compare + kernel_type）

分析思路：
- Compare：按 Type+Shape 分组，统计劣化/改善总量，识别显著劣化
- Type：按 Type+CoreType 分组，输出简化结果
- 负载均衡分析：仅针对计算算子（排除通信算子和空Shape），分析数据量和耗时方差
"""
from parsers.common import safe_float, safe_int, get_sheet_data
from collections import defaultdict


# 通信算子前缀列表
COMM_PREFIXES = ('hcom_',)


def _is_comm_kernel(name: str) -> bool:
    """判断是否为通信算子"""
    name_lower = name.lower()
    return any(name_lower.startswith(p) for p in COMM_PREFIXES)


def _is_empty_shape(shape: str) -> bool:
    """判断Shape是否为空"""
    s = shape.strip().strip('"').strip()
    return s in ('', 'N/A', 'None', 'n/a', 'null')


def _calc_elements(shape: str) -> int:
    """从Shape字符串计算总输入数据量（所有tensor elements之和）和最大tensor elements

    Shape格式: "128,4096;4096,10240;10240;;128"
    分号分隔tensor，逗号分隔维度
    """
    total_elements = 0
    max_tensor_elements = 0
    parts = shape.strip().strip('"').split(';')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        dims = part.split(',')
        elements = 1
        valid = True
        for d in dims:
            d = d.strip()
            if d.isdigit():
                elements *= int(d)
            else:
                valid = False
                break
        if valid and elements > 0:
            total_elements += elements
            if elements > max_tensor_elements:
                max_tensor_elements = elements
    return total_elements, max_tensor_elements


def _fmt_elements(n: int) -> str:
    """格式化elements数量为人类可读"""
    if n >= 1_000_000_000:
        return f'{n / 1_000_000_000:.1f}B'
    elif n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    elif n >= 1_000:
        return f'{n / 1_000:.1f}K'
    else:
        return str(n)


def parse_compare(wb) -> dict:
    """KernelCompare Kernel比对（按 Type+Shape 分组）"""
    headers, rows = get_sheet_data(wb, 'KernelCompare')
    if not rows:
        return {'sheet_name': 'KernelCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

    kernels = []
    for row in rows:
        if len(row) < 15:
            continue
        kernel_name = row[1].strip() if row[1] else ''
        if not kernel_name or kernel_name == 'Kernel':
            continue
        shape_str = str(row[2]).strip() if row[2] else ''
        is_comm = _is_comm_kernel(kernel_name)
        is_empty = _is_empty_shape(shape_str)

        entry = {
            'kernel': kernel_name,
            'input_shape': shape_str,
            'base_total': safe_float(row[3]),
            'base_avg': safe_float(row[4]),
            'base_max': safe_float(row[5]),
            'base_min': safe_float(row[6]),
            'base_calls': safe_int(row[7]),
            'comp_total': safe_float(row[8]),
            'comp_avg': safe_float(row[9]),
            'comp_max': safe_float(row[10]),
            'comp_min': safe_float(row[11]),
            'comp_calls': safe_int(row[12]),
            'total_ratio': safe_float(row[13]),
            'avg_ratio': safe_float(row[14]),
            'is_comm': is_comm,
            'is_empty_shape': is_empty,
        }

        # 计算输入数据量（仅对有Shape的计算算子）
        if not is_comm and not is_empty:
            total_elems, max_elems = _calc_elements(shape_str)
            entry['total_elements'] = total_elems
            entry['max_tensor_elements'] = max_elems
            entry['total_elements_fmt'] = _fmt_elements(total_elems)
            # 基准侧也需要计算（用于比率比较）
            # 注意：基准侧Shape在比对场景中通常相同，这里直接用基准的total计算
            # 如果基准Shape不同，比率会反映在total_ratio上
        else:
            entry['total_elements'] = 0
            entry['max_tensor_elements'] = 0
            entry['total_elements_fmt'] = '-'

        kernels.append(entry)

    for k in kernels:
        k['diff_total'] = k['comp_total'] - k['base_total']

    # === 全局统计（含通信算子，排除基准或比对为0的inf比率项） ===
    valid_kernels = [k for k in kernels if k['base_total'] > 0 and k['comp_total'] > 0]
    degraded = [k for k in valid_kernels if k['diff_total'] > 0]
    improved = [k for k in valid_kernels if k['diff_total'] < 0]
    total_degraded = sum(k['diff_total'] for k in degraded)
    total_improved = sum(abs(k['diff_total']) for k in improved)

    # Top 10 变化最大：仅包含基准和比对都非空的 Kernel
    valid_kernels.sort(key=lambda x: x['diff_total'], reverse=True)
    top10 = valid_kernels[:10]

    for k in top10:
        calls_changed = k['base_calls'] != k['comp_calls']
        avg_changed = k['avg_ratio'] > 1.05 if k['avg_ratio'] > 0 else False
        if calls_changed and avg_changed:
            k['degradation_type'] = '耗时和调用次数均增加'
        elif calls_changed:
            k['degradation_type'] = '调用次数增加'
        elif avg_changed:
            k['degradation_type'] = '单次耗时显著增加'
        else:
            k['degradation_type'] = '轻微变化'

    # 显著劣化：也排除inf比率，并为每个元素添加 degradation_type
    significant_degraded = [k for k in degraded if k['total_ratio'] > 1.05 and k['total_ratio'] != float('inf')]
    for k in significant_degraded:
        if 'degradation_type' not in k:
            calls_changed = k['base_calls'] != k['comp_calls']
            avg_changed = k['avg_ratio'] > 1.05 if k['avg_ratio'] > 0 else False
            if calls_changed and avg_changed:
                k['degradation_type'] = '耗时和调用次数均增加'
            elif calls_changed:
                k['degradation_type'] = '调用次数增加'
            elif avg_changed:
                k['degradation_type'] = '单次耗时显著增加'
            else:
                k['degradation_type'] = '轻微变化'

    # === 分离通信算子和计算算子 ===
    comm_kernels = [k for k in kernels if k['is_comm']]
    compute_kernels = [k for k in kernels if not k['is_comm'] and not k['is_empty_shape']]

    # === Top 10 计算算子绝对耗时（排除通信算子和空Shape） ===
    top10_compute = sorted(compute_kernels, key=lambda x: x['comp_total'], reverse=True)[:10]
    for k in top10_compute:
        k['calls_changed'] = k['base_calls'] != k['comp_calls']
        k['calls_diff'] = k['comp_calls'] - k['base_calls']
        if k['comp_calls'] > 1 and k['comp_min'] > 0:
            k['variance_ratio'] = round(k['comp_max'] / k['comp_min'], 1) if k['comp_min'] > 0 else 0
        else:
            k['variance_ratio'] = 0

    # === 计算算子负载均衡分析（仅calls>10的计算算子） ===
    load_imbalance = []
    for k in compute_kernels:
        if k['comp_calls'] > 10 and k['comp_max'] > 0 and k['comp_min'] > 0:
            variance_ratio = k['comp_max'] / k['comp_min'] if k['comp_min'] > 0 else 0
            if variance_ratio > 2.0:
                k['variance_ratio'] = round(variance_ratio, 1)
                k['imbalance_severity'] = '严重' if variance_ratio > 5 else '显著' if variance_ratio > 3 else '轻微'
                load_imbalance.append(k)
    load_imbalance.sort(key=lambda x: x.get('variance_ratio', 0), reverse=True)

    # === 多Shape计算算子分布 ===
    shape_dist = defaultdict(list)
    for k in compute_kernels:
        shape_dist[k['kernel']].append(k)
    multi_shape_kernels = {name: entries for name, entries in shape_dist.items() if len(entries) > 1}

    # === 两卡数据量汇总对比 ===
    base_total_elements = sum(k.get('total_elements', 0) * k['base_calls'] for k in compute_kernels)
    comp_total_elements = sum(k.get('total_elements', 0) * k['comp_calls'] for k in compute_kernels)
    base_total_calls = sum(k['base_calls'] for k in compute_kernels)
    comp_total_calls = sum(k['comp_calls'] for k in compute_kernels)
    base_total_us = sum(k['base_total'] for k in compute_kernels)
    comp_total_us = sum(k['comp_total'] for k in compute_kernels)

    # Top10 计算算子的数据量对比（按比对侧总耗时排序）
    top10_data_compare = []
    for k in top10_compute:
        top10_data_compare.append({
            'kernel': k['kernel'],
            'shape': k['input_shape'],
            'elements_per_call': k.get('total_elements', 0),
            'elements_per_call_fmt': k.get('total_elements_fmt', '-'),
            'base_calls': k['base_calls'],
            'comp_calls': k['comp_calls'],
            'base_total_elements': k.get('total_elements', 0) * k['base_calls'],
            'comp_total_elements': k.get('total_elements', 0) * k['comp_calls'],
            'base_total_us': k['base_total'],
            'comp_total_us': k['comp_total'],
        })
        # 格式化总数据量
        for prefix in ['base_total_elements', 'comp_total_elements']:
            val = top10_data_compare[-1][prefix]
            top10_data_compare[-1][prefix + '_fmt'] = _fmt_elements(val)

    return {
        'sheet_name': 'KernelCompare',
        'available': True,
        'total_kernels': len(kernels),
        'comm_kernel_count': len(comm_kernels),
        'compute_kernel_count': len(compute_kernels),
        'degraded_count': len(degraded),
        'improved_count': len(improved),
        'total_degraded_us': round(total_degraded, 2),
        'total_improved_us': round(total_improved, 2),
        'net_trend': '劣化' if total_degraded > total_improved else '改善' if total_improved > total_degraded else '持平',
        'significant_degraded_count': len(significant_degraded),
        'top10': top10,
        'significant_degraded': significant_degraded[:5],
        # 负载均衡分析（新增）
        'top10_compute': top10_compute,
        'load_imbalance': load_imbalance[:10],
        'load_imbalance_count': len(load_imbalance),
        'multi_shape_kernel_count': len(multi_shape_kernels),
        'multi_shape_kernels': {
            name: [{'shape': e['input_shape'], 'comp_total': e['comp_total'], 'comp_avg': e['comp_avg'],
                     'comp_calls': e['comp_calls'], 'total_elements': e.get('total_elements', 0),
                     'total_elements_fmt': e.get('total_elements_fmt', '-')}
                    for e in entries]
            for name, entries in list(multi_shape_kernels.items())[:5]
        },
        # 通信算子概览
        'comm_kernels': [{'kernel': k['kernel'], 'comp_total': k['comp_total'], 'comp_avg': k['comp_avg'],
                          'comp_max': k['comp_max'], 'comp_min': k['comp_min'], 'comp_calls': k['comp_calls'],
                          'diff_total': k['diff_total'], 'total_ratio': k['total_ratio']}
                         for k in comm_kernels],
        # 两卡数据量汇总对比（新增）
        'data_volume_compare': {
            'base_total_elements': base_total_elements,
            'base_total_elements_fmt': _fmt_elements(base_total_elements),
            'comp_total_elements': comp_total_elements,
            'comp_total_elements_fmt': _fmt_elements(comp_total_elements),
            'base_total_calls': base_total_calls,
            'comp_total_calls': comp_total_calls,
            'base_total_us': round(base_total_us, 2),
            'comp_total_us': round(comp_total_us, 2),
            'elements_changed': base_total_elements != comp_total_elements,
            'calls_changed': base_total_calls != comp_total_calls,
        },
        'top10_data_compare': top10_data_compare,
    }


def parse_type(wb) -> dict:
    """KernelTypeCompare Kernel类型比对（按 Type+CoreType 分组）"""
    headers, rows = get_sheet_data(wb, 'KernelTypeCompare')
    if not rows:
        return {'sheet_name': 'KernelTypeCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

    types = []
    for row in rows:
        if len(row) < 15:
            continue
        kernel_type = row[1].strip() if row[1] else ''
        if not kernel_type or kernel_type == 'Kernel Type':
            continue
        types.append({
            'kernel_type': kernel_type,
            'core_type': row[2].strip() if row[2] else '',
            'base_total': safe_float(row[3]),
            'base_avg': safe_float(row[4]),
            'base_max': safe_float(row[5]),
            'base_min': safe_float(row[6]),
            'base_calls': safe_int(row[7]),
            'comp_total': safe_float(row[8]),
            'comp_avg': safe_float(row[9]),
            'comp_max': safe_float(row[10]),
            'comp_min': safe_float(row[11]),
            'comp_calls': safe_int(row[12]),
            'total_ratio': safe_float(row[13]),
            'avg_ratio': safe_float(row[14]),
        })

    for t in types:
        t['diff_total'] = t['comp_total'] - t['base_total']
    types.sort(key=lambda x: x['diff_total'], reverse=True)
    top10 = types[:10]

    type_summary = {}
    for t in types:
        kt = t['kernel_type']
        if kt not in type_summary:
            type_summary[kt] = {'count': 0, 'diff_total': 0}
        type_summary[kt]['count'] += 1
        type_summary[kt]['diff_total'] += t['diff_total']

    return {
        'sheet_name': 'KernelTypeCompare',
        'available': True,
        'total_types': len(types),
        'top10': top10,
        'type_summary': type_summary,
    }

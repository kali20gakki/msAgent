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

"""算子比对解析器（合并 operator_statistic + operator_compare）

分析思路：
- Statistic：按算子名聚合，按 Diff Duration 降序，提取 Top 10，计算集中度
- Detail：关联统计页 Top 算子，提取 Kernel 详情和 Input Shape
"""
from parsers.common import safe_float, safe_int, get_sheet_data


def parse_statistic(wb) -> dict:
    """OperatorCompareStatistic 算子统计比对"""
    headers, rows = get_sheet_data(wb, 'OperatorCompareStatistic')
    if not rows:
        return {'sheet_name': 'OperatorCompareStatistic', 'available': False, 'reason': 'Sheet不存在或为空'}

    operators = []
    for row in rows:
        if len(row) < 8:
            continue
        op_name = row[1].strip() if row[1] else ''
        if not op_name or op_name == 'Operator Name':
            continue
        diff_dur = safe_float(row[6])
        diff_ratio = safe_float(row[7])
        operators.append({
            'name': op_name,
            'base_duration': safe_float(row[2]),
            'base_number': safe_int(row[3]),
            'comp_duration': safe_float(row[4]),
            'comp_number': safe_int(row[5]),
            'diff_duration': diff_dur,
            'diff_ratio': diff_ratio,
            'degraded': diff_dur > 0,
        })

    operators.sort(key=lambda x: x['diff_duration'], reverse=True)
    top10 = operators[:10]

    total_diff = sum(o['diff_duration'] for o in operators)
    top10_diff = sum(o['diff_duration'] for o in top10)
    concentration = abs(top10_diff) / abs(total_diff) * 100 if total_diff != 0 else 0

    degraded_count = sum(1 for o in operators if o['diff_duration'] > 0)
    improved_count = sum(1 for o in operators if o['diff_duration'] < 0)

    return {
        'sheet_name': 'OperatorCompareStatistic',
        'available': True,
        'total_operators': len(operators),
        'degraded_count': degraded_count,
        'improved_count': improved_count,
        'total_diff_duration': round(total_diff, 2),
        'top10': top10,
        'concentration': round(concentration, 1),
        'is_concentrated': concentration > 80,
    }


def parse_compare(wb, top_op_names=None) -> dict:
    """OperatorCompare 算子明细比对"""
    headers, rows = get_sheet_data(wb, 'OperatorCompare')
    if not rows:
        return {'sheet_name': 'OperatorCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

    all_details = []
    for row in rows:
        if len(row) < 13:
            continue
        base_name = row[1].strip() if row[1] else ''
        comp_name = row[6].strip() if row[6] else ''
        op_name = base_name or comp_name
        if not op_name:
            continue
        all_details.append({
            'order': row[0],
            'base_name': base_name,
            'base_shape': row[2],
            'base_type': row[3],
            'base_kernel': row[4],
            'base_duration': safe_float(row[5]),
            'comp_name': comp_name,
            'comp_shape': row[7],
            'comp_type': row[8],
            'comp_kernel': row[9],
            'comp_duration': safe_float(row[10]),
            'diff_duration': safe_float(row[11]),
            'diff_ratio': safe_float(row[12]),
        })

    if top_op_names:
        top_set = set(top_op_names)
        filtered = [d for d in all_details if d['base_name'] in top_set or d['comp_name'] in top_set]
    else:
        all_details.sort(key=lambda x: x['diff_duration'], reverse=True)
        filtered = all_details[:10]

    for d in filtered:
        base_kernels = d['base_kernel'].split('\n') if d['base_kernel'] else []
        comp_kernels = d['comp_kernel'].split('\n') if d['comp_kernel'] else []
        d['base_kernel_count'] = len([k for k in base_kernels if k.strip()])
        d['comp_kernel_count'] = len([k for k in comp_kernels if k.strip()])
        d['kernel_count_changed'] = d['base_kernel_count'] != d['comp_kernel_count']

    return {
        'sheet_name': 'OperatorCompare',
        'available': True,
        'total_rows': len(all_details),
        'top_details': filtered[:10],
    }

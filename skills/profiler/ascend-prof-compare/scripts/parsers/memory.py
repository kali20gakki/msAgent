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

"""内存比对解析器（合并 memory_statistic + memory_compare）

分析思路：
- Statistic：按 Diff Memory(MB) 降序，检测全零内存场景
- Detail：关联统计页 Top 算子，提取内存分配详情
"""
from parsers.common import safe_float, safe_int, get_sheet_data


def parse_statistic(wb) -> dict:
    """MemoryCompareStatistic 内存统计比对"""
    headers, rows = get_sheet_data(wb, 'MemoryCompareStatistic')
    if not rows:
        return {'sheet_name': 'MemoryCompareStatistic', 'available': False, 'reason': 'Sheet不存在或为空'}

    operators = []
    for row in rows:
        if len(row) < 10:
            continue
        op_name = row[1].strip() if row[1] else ''
        if not op_name or op_name == 'Operator Name':
            continue
        operators.append({
            'name': op_name,
            'base_duration': safe_float(row[2]),
            'base_memory': safe_float(row[3]),
            'base_number': safe_int(row[4]),
            'comp_duration': safe_float(row[5]),
            'comp_memory': safe_float(row[6]),
            'comp_number': safe_int(row[7]),
            'diff_memory': safe_float(row[8]),
            'diff_ratio': safe_float(row[9]),
        })

    all_zero = all(o['base_memory'] == 0 and o['comp_memory'] == 0 for o in operators)

    operators.sort(key=lambda x: abs(x['diff_memory']), reverse=True)
    top10 = operators[:10]

    if not all_zero:
        increased = [o for o in operators if o['diff_memory'] > 0]
        decreased = [o for o in operators if o['diff_memory'] < 0]
        total_diff = sum(o['diff_memory'] for o in operators)
    else:
        increased = []
        decreased = []
        total_diff = 0

    return {
        'sheet_name': 'MemoryCompareStatistic',
        'available': True,
        'total_operators': len(operators),
        'total_diff_memory': round(total_diff, 2),
        'increased_count': len(increased),
        'decreased_count': len(decreased),
        'top10': top10,
        'all_zero_memory': all_zero,
        'zero_memory_note': '所有算子内存占用均为 0，可能未开启 profile_memory 采集或为 NPU vs NPU 无内存差异场景' if all_zero else None,
    }


def parse_compare(wb, top_op_names=None) -> dict:
    """MemoryCompare 内存明细比对"""
    headers, rows = get_sheet_data(wb, 'MemoryCompare')
    if not rows:
        return {'sheet_name': 'MemoryCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

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
            'base_details': row[4],
            'base_size': safe_float(row[5]),
            'comp_name': comp_name,
            'comp_shape': row[7],
            'comp_type': row[8],
            'comp_details': row[9],
            'comp_size': safe_float(row[10]),
            'diff_size': safe_float(row[11]),
            'diff_ratio': safe_float(row[12]),
        })

    if top_op_names:
        top_set = set(top_op_names)
        filtered = [d for d in all_details if d['base_name'] in top_set or d['comp_name'] in top_set]
    else:
        all_details.sort(key=lambda x: x['diff_size'], reverse=True)
        filtered = all_details[:10]

    return {
        'sheet_name': 'MemoryCompare',
        'available': True,
        'total_rows': len(all_details),
        'top_details': filtered[:10],
    }

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

"""模块比对解析器（合并 module_statistic + module_compare）

分析思路：
- Statistic：筛选 [ TOTAL ] 行，按总耗时差异降序，提取 Top 10 模块
- Detail：关联统计页 Top 模块，提取调用栈定位代码位置
"""
from parsers.common import safe_float, safe_int, get_sheet_data


def parse_statistic(wb) -> dict:
    """ModuleCompareStatistic 模块统计比对"""
    headers, rows = get_sheet_data(wb, 'ModuleCompareStatistic')
    if not rows:
        return {'sheet_name': 'ModuleCompareStatistic', 'available': False, 'reason': 'Sheet不存在或为空'}

    total_rows = []
    detail_rows = []
    for row in rows:
        if len(row) < 18:
            continue
        op_name = row[4].strip() if row[4] else ''
        module_class = row[1].strip() if row[1] else ''
        module_name = row[3].strip() if row[3] else ''
        if not module_name:
            continue

        entry = {
            'module_class': module_class,
            'module_level': safe_int(row[2]),
            'module_name': module_name,
            'operator_name': op_name,
            'base_self_time': safe_float(row[6]),
            'base_number': safe_int(row[7]),
            'base_total_time': safe_float(row[8]),
            'comp_self_time': safe_float(row[10]),
            'comp_number': safe_int(row[11]),
            'comp_total_time': safe_float(row[12]),
            'diff_total_time': safe_float(row[13]),
            'diff_self_time': safe_float(row[14]),
            'diff_total_ratio': safe_float(row[15]),
            'base_call_stack': row[16],
            'comp_call_stack': row[17],
        }

        if op_name == '[ TOTAL ]':
            total_rows.append(entry)
        else:
            detail_rows.append(entry)

    total_rows.sort(key=lambda x: x['diff_total_time'], reverse=True)
    top10 = total_rows[:10]

    level_dist = {}
    for m in total_rows:
        lvl = m['module_level']
        level_dist[lvl] = level_dist.get(lvl, 0) + 1

    return {
        'sheet_name': 'ModuleCompareStatistic',
        'available': True,
        'total_modules': len(total_rows),
        'top10': top10,
        'level_distribution': level_dist,
        'has_call_stack': any(m['base_call_stack'] or m['comp_call_stack'] for m in top10),
    }


def parse_compare(wb, top_module_names=None) -> dict:
    """ModuleCompare 模块明细比对"""
    headers, rows = get_sheet_data(wb, 'ModuleCompare')
    if not rows:
        return {'sheet_name': 'ModuleCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

    all_details = []
    for row in rows:
        if len(row) < 17:
            continue
        module_name = row[3].strip() if row[3] else ''
        op_name = row[4].strip() if row[4] else ''
        if not module_name:
            continue
        all_details.append({
            'module_class': row[1].strip() if row[1] else '',
            'module_level': row[2],
            'module_name': module_name,
            'base_op_name': row[4].strip() if row[4] else '',
            'base_kernel': row[5],
            'base_self_time': safe_float(row[6]),
            'base_total_time': safe_float(row[7]),
            'comp_op_name': row[8].strip() if row[8] else '',
            'comp_kernel': row[9],
            'comp_self_time': safe_float(row[10]),
            'comp_total_time': safe_float(row[11]),
            'diff_total_time': safe_float(row[12]),
            'diff_self_time': safe_float(row[13]),
            'diff_total_ratio': safe_float(row[14]),
            'base_call_stack': row[15],
            'comp_call_stack': row[16],
        })

    if top_module_names:
        top_set = set(top_module_names)
        filtered = [d for d in all_details if d['module_name'] in top_set]
    else:
        filtered = all_details[:15]

    degraded_with_stack = [
        d for d in filtered
        if d['diff_self_time'] > 0 and (d['base_call_stack'] or d['comp_call_stack'])
    ]

    return {
        'sheet_name': 'ModuleCompare',
        'available': True,
        'total_rows': len(all_details),
        'top_details': filtered[:15],
        'degraded_with_stack': degraded_with_stack[:5],
    }

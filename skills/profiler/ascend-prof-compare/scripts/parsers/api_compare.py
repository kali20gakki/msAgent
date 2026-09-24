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

"""ApiCompare API比对解析器（优化版）

分析思路：
1. 按总耗时差异排序
2. 区分纯耗时变化（calls_ratio=1.0）vs 调用次数变化
3. 统计整体劣化/改善趋势
"""
from parsers.common import safe_float, safe_int, get_sheet_data


def parse(wb) -> dict:
    headers, rows = get_sheet_data(wb, 'ApiCompare')
    if not rows:
        return {'sheet_name': 'ApiCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

    apis = []
    for row in rows:
        if len(row) < 14:
            continue
        api_name = row[1].strip() if row[1] else ''
        if not api_name or api_name == 'api name':
            continue
        apis.append({
            'name': api_name,
            'base_total': safe_float(row[2]),
            'base_self': safe_float(row[3]),
            'base_avg': safe_float(row[4]),
            'base_calls': safe_int(row[5]),
            'comp_total': safe_float(row[6]),
            'comp_self': safe_float(row[7]),
            'comp_avg': safe_float(row[8]),
            'comp_calls': safe_int(row[9]),
            'total_ratio': safe_float(row[10]),
            'self_ratio': safe_float(row[11]),
            'avg_ratio': safe_float(row[12]),
            'calls_ratio': safe_float(row[13]),
        })

    for a in apis:
        a['diff_total'] = a['comp_total'] - a['base_total']

    apis.sort(key=lambda x: x['diff_total'], reverse=True)
    top10 = apis[:10]

    # 分析劣化原因
    for a in top10:
        calls_changed = a['calls_ratio'] > 1.05 if a['calls_ratio'] > 0 else False
        self_changed = a['self_ratio'] > 1.05 if a['self_ratio'] > 0 else False
        if calls_changed and self_changed:
            a['degradation_type'] = 'Self耗时和调用次数均增加'
        elif calls_changed:
            a['degradation_type'] = '调用次数增加'
        elif self_changed:
            a['degradation_type'] = 'Self耗时增加'
        elif a['diff_total'] > 0:
            a['degradation_type'] = '轻微劣化'
        else:
            a['degradation_type'] = '轻微变化'

    # 统计整体趋势
    degraded = [a for a in apis if a['diff_total'] > 0]
    improved = [a for a in apis if a['diff_total'] < 0]
    total_degraded = sum(a['diff_total'] for a in degraded)
    total_improved = sum(abs(a['diff_total']) for a in improved)

    # 区分纯耗时变化 vs 调用次数变化
    pure_duration_change = [a for a in degraded if a['calls_ratio'] == 1.0]
    call_count_change = [a for a in degraded if a['calls_ratio'] != 1.0]

    return {
        'sheet_name': 'ApiCompare',
        'available': True,
        'total_apis': len(apis),
        'degraded_count': len(degraded),
        'improved_count': len(improved),
        'total_degraded_ms': round(total_degraded, 2),
        'total_improved_ms': round(total_improved, 2),
        'net_trend': '劣化' if total_degraded > total_improved else '改善' if total_improved > total_degraded else '持平',
        'pure_duration_change_count': len(pure_duration_change),
        'call_count_change_count': len(call_count_change),
        'top10': top10,
    }

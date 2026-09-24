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

"""CommunicationCompare 通信比对解析器（优化版）

分析思路：
1. 区分 summary 行（Task Name为空）和 detail 行
2. 提取所有通信算子，按差异绝对值排序
3. 分别识别改善最大和劣化最大的算子
4. 分析极端改善（如 allreduce 从 33113us → 417us）
"""
from parsers.common import safe_float, safe_int, get_sheet_data


def parse(wb) -> dict:
    headers, rows = get_sheet_data(wb, 'CommunicationCompare')
    if not rows:
        return {'sheet_name': 'CommunicationCompare', 'available': False, 'reason': 'Sheet不存在或为空'}

    summary_rows = []
    task_detail_count = 0
    for row in rows:
        if len(row) < 17:
            continue
        comm_name = row[1].strip() if row[1] else ''
        task_name = row[2].strip() if row[2] else ''
        if not comm_name:
            continue
        if comm_name == '|':
            task_detail_count += 1
            continue
        if not task_name:
            summary_rows.append({
                'comm_name': comm_name,
                'base_calls': safe_int(row[3]),
                'base_total': safe_float(row[4]),
                'base_avg': safe_float(row[5]),
                'base_max': safe_float(row[6]),
                'base_min': safe_float(row[7]),
                'comp_calls': safe_int(row[8]),
                'comp_total': safe_float(row[9]),
                'comp_avg': safe_float(row[10]),
                'comp_max': safe_float(row[11]),
                'comp_min': safe_float(row[12]),
                'diff_duration': safe_float(row[15]),
                'diff_ratio': safe_float(row[16]),
            })

    # 按差异绝对值排序
    summary_rows.sort(key=lambda x: abs(x['diff_duration']), reverse=True)

    # 区分改善和劣化
    improved = [s for s in summary_rows if s['diff_duration'] < 0]
    degraded = [s for s in summary_rows if s['diff_duration'] > 0]

    # 标记极端改善（比率 < 0.1）
    for s in improved:
        s['is_extreme_improvement'] = s['diff_ratio'] < 0.1 and s['base_total'] > 1000

    return {
        'sheet_name': 'CommunicationCompare',
        'available': True,
        'total_comm_ops': len(summary_rows),
        'improved_count': len(improved),
        'degraded_count': len(degraded),
        'top_changed': summary_rows[:10],
        'top_improved': improved[:5],
        'top_degraded': degraded[:5],
        'has_task_detail': task_detail_count > 0,
        'task_detail_count': task_detail_count,
        'has_extreme_improvement': any(s.get('is_extreme_improvement') for s in improved),
    }

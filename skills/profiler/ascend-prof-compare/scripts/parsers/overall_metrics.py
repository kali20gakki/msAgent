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

"""OverallMetrics 总体性能比对解析器（优化版）

分析思路：
1. 提取四大维度（计算/通信/调度/E2E）的基准和比对值
2. 区分改善维度和劣化维度，分别识别各自最大贡献者
3. 计算各维度对总变化的贡献占比（区分正向/负向）
4. 下钻提取子类别，分改善/劣化两组排序
5. 检查 Not minimal profiling 警告
"""
from parsers.common import safe_float, get_sheet_data


def parse(wb) -> dict:
    headers, rows = get_sheet_data(wb, 'OverallMetrics', strip=False)
    if not rows:
        return {'sheet_name': 'OverallMetrics', 'available': False, 'reason': 'Sheet不存在或为空'}

    sheet_name_actual = ''
    for name in wb.sheetnames:
        if 'overallmetrics' in name.lower():
            sheet_name_actual = name
            break
    has_minimal_warning = 'minimal' in sheet_name_actual.lower() and 'not' not in sheet_name_actual.lower()

    # 用列索引精确提取（headers可能有重名）
    metric_list = []
    for row in rows:
        if len(row) < 9:
            continue
        index_val = row[0] if row[0] else ''
        index_val = index_val.strip(' ') if isinstance(index_val, str) else str(index_val)
        if not index_val.strip():
            continue
        metric_list.append({
            'index': index_val,
            'base_duration': safe_float(row[1]),
            'base_ratio': safe_float(row[2]),
            'base_number': row[3],
            'comp_duration': safe_float(row[4]),
            'comp_ratio': safe_float(row[5]),
            'comp_number': row[6],
            'diff_duration': safe_float(row[7]),
            'diff_ratio': safe_float(row[8]),
        })

    def find_metric(keyword):
        for m in metric_list:
            if keyword.lower() in m['index'].lower():
                return m
        return None

    e2e = find_metric('E2E Time')
    computing = find_metric('Computing Time')
    communication = find_metric('Uncovered Communication Time')
    free_time = find_metric('Free Time')

    e2e_diff = e2e['diff_duration'] if e2e else 0
    e2e_base = e2e['base_duration'] if e2e else 0
    e2e_comp = e2e['comp_duration'] if e2e else 0

    dimensions = []
    for name, metric, label in [
        ('计算', computing, 'Computing Time'),
        ('通信', communication, 'Uncovered Communication Time'),
        ('调度', free_time, 'Free Time'),
    ]:
        if metric:
            is_degraded = metric['diff_duration'] > 0
            is_improved = metric['diff_duration'] < 0
            # 贡献占比：仅劣化维度参与百分比计算，分母为所有劣化维度差异之和
            # 改善维度不计算百分比，显示为 "-"
            if is_degraded:
                total_degraded = sum(m['diff_duration'] for m in [computing, communication, free_time] if m and m['diff_duration'] > 0)
                contribution = abs(metric['diff_duration']) / total_degraded * 100 if total_degraded > 0 else 0
            else:
                contribution = None  # 改善维度不显示百分比
            dimensions.append({
                'name': name,
                'label': label,
                'base_duration': metric['base_duration'],
                'comp_duration': metric['comp_duration'],
                'diff_duration': metric['diff_duration'],
                'diff_ratio': metric['diff_ratio'],
                'contribution': round(contribution, 1) if contribution is not None else None,
                'contribution_str': f'{contribution:.1f}%' if contribution is not None else '-',
                'is_improved': is_improved,
                'is_degraded': is_degraded,
            })

    # 分别找出改善最大和劣化最大的维度
    improved_dims = sorted([d for d in dimensions if d['diff_duration'] < 0],
                           key=lambda x: abs(x['diff_duration']), reverse=True)
    degraded_dims = sorted([d for d in dimensions if d['diff_duration'] > 0],
                           key=lambda x: abs(x['diff_duration']), reverse=True)

    primary_improved = improved_dims[0] if improved_dims else None
    primary_degraded = degraded_dims[0] if degraded_dims else None

    # 判断整体趋势
    if e2e_diff < 0:
        overall_trend = '改善'
    elif e2e_diff > 0:
        overall_trend = '劣化'
    else:
        overall_trend = '持平'

    # 提取子类别（一级缩进 \t 但非二级 \t\t）
    sub_categories = []
    for m in metric_list:
        idx = m['index']
        if idx.startswith('\t') and not idx.startswith('\t\t'):
            sub_categories.append({
                'name': idx.strip(),
                'base_duration': m['base_duration'],
                'comp_duration': m['comp_duration'],
                'diff_duration': m['diff_duration'],
                'diff_ratio': m['diff_ratio'],
                'is_improved': m['diff_duration'] < 0,
                'is_degraded': m['diff_duration'] > 0,
            })

    # 按绝对差异排序
    sub_categories.sort(key=lambda x: abs(x['diff_duration']), reverse=True)

    # 分改善和劣化两组
    improved_subs = [s for s in sub_categories if s['diff_duration'] < 0][:5]
    degraded_subs = [s for s in sub_categories if s['diff_duration'] > 0][:5]

    return {
        'sheet_name': 'OverallMetrics',
        'available': True,
        'has_minimal_warning': not has_minimal_warning,
        'overall_trend': overall_trend,
        'e2e': {
            'base': e2e_base,
            'comp': e2e_comp,
            'diff': e2e_diff,
            'ratio': e2e['diff_ratio'] if e2e else 0,
        } if e2e else None,
        'dimensions': dimensions,
        'primary_improved': primary_improved,
        'primary_degraded': primary_degraded,
        'top_sub_categories': sub_categories[:8],
        'top_improved_subs': improved_subs,
        'top_degraded_subs': degraded_subs,
        'total_metrics_count': len(metric_list),
        'raw_metrics': metric_list,
    }

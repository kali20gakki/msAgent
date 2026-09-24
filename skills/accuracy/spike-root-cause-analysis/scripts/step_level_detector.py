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

"""step 级梯度数据异常检测: 每 step 独立完整梯度，用全局基线（MAD/IQR）动态阈值检测。"""

from collections import defaultdict
from statistics import median


def compute_per_target_baselines(trend_rows, targets):
    """
    对每个 target 学习其梯度的正常基线分布。
    使用 MAD-based robust statistics 排除异常 step 的干扰。
    """
    groups = defaultdict(list)
    for r in trend_rows:
        key = (r['target_id'], r['metric_id'])
        groups[key].append(r['norm'])

    baselines = {}
    for (tid, mid), norms in groups.items():
        n = len(norms)
        if n < 3:
            # 样本太少，直接用简单统计
            baselines[(tid, mid)] = {
                'median': median(norms),
                'q1': min(norms),
                'q3': max(norms),
                'iqr': max(norms) - min(norms),
                'mad': 0,
                'sample_count': n
            }
            continue

        sorted_norms = sorted(norms)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = sorted_norms[q1_idx]
        q3 = sorted_norms[q3_idx]
        iqr = q3 - q1
        med = median(norms)

        abs_devs = sorted(abs(v - med) for v in norms)
        mad = abs_devs[n // 2] if abs_devs else 0

        baselines[(tid, mid)] = {
            'median': med,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'mad': mad,
            'sample_count': n
        }

    return baselines


def detect_spikes(trend_rows, targets, baselines, metric_id=1,
                  iqr_multiplier=5.0, mad_multiplier=10.0):
    """
    基于学习到的基线分布，检测异常 spike。

    Args:
        iqr_multiplier: IQR 倍数阈值（超过 Q3 + k*IQR 视为异常）
        mad_multiplier: MAD 倍数阈值
    """
    anomalies = []
    for r in trend_rows:
        if r['metric_id'] != metric_id:
            continue

        tid = r['target_id']
        baseline = baselines.get((tid, metric_id))
        if not baseline or baseline['sample_count'] < 3:
            continue

        norm = r['norm']
        med = baseline['median']
        iqr = baseline['iqr']
        mad = baseline['mad']

        if med < 1e-10 and norm < 1e-10:
            continue

        # 动态阈值判定
        if iqr > 0:
            iqr_threshold = baseline['q3'] + iqr_multiplier * iqr
        else:
            iqr_threshold = med * 10 if med > 0 else 1e-10

        if mad > 0:
            mad_threshold = med + mad_multiplier * mad
        else:
            mad_threshold = med * mad_multiplier if med > 0 else 1e-10

        is_spike_iqr = norm > iqr_threshold and norm > 0
        is_spike_mad = norm > mad_threshold and norm > 0

        # 至少一个阈值触发
        if is_spike_iqr or is_spike_mad:
            if med > 0:
                deviation_ratio = norm / med
            elif norm > 0:
                deviation_ratio = float('inf')
            else:
                deviation_ratio = 1.0

            if iqr > 0:
                z_score_like = (norm - med) / (iqr / 1.349)  # IQR/1.349 ≈ 正态分布标准差
            else:
                z_score_like = float('inf') if norm > 0 else 0

            anomaly = {
                'rank': r['rank'],
                'step': r['step'],
                'target_id': tid,
                'target_name': targets[tid]['name'],
                'metric': 'grad_unreduced' if metric_id == 1 else 'grad_reduced',
                'norm': norm,
                'min': r['min'],
                'max': r['max'],
                'mean': r['mean'],
                'baseline_median': med,
                'baseline_iqr': iqr,
                'deviation_ratio': deviation_ratio,
                'z_score_like': z_score_like,
                'trigger': 'iqr+mad' if (is_spike_iqr and is_spike_mad) else
                           ('iqr' if is_spike_iqr else 'mad')
            }
            if 'micro_step' in r:
                anomaly['micro_step'] = r['micro_step']
            anomalies.append(anomaly)

    anomalies.sort(key=lambda x: x['deviation_ratio'], reverse=True)
    return anomalies


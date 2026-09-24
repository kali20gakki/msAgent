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
"""
Phase 1 脚本单元测试: step_level_detector / trend_db_spike_detector
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "skills", "accuracy", "spike-root-cause-analysis", "scripts"
    ),
)

from step_level_detector import compute_per_target_baselines, detect_spikes
from trend_db_spike_detector import detect_accumulation_window, detect_dump_spikes


def _make_step_rows(normal=20, spike_norm=100.0):
    """构造 step 级数据: normal 个正常 step + 1 个 spike。"""
    rows = []
    for i in range(normal):
        rows.append(
            {
                'rank': 0,
                'step': i,
                'target_id': 0,
                'metric_id': 1,
                'norm': 1.0 + i * 0.01,
                'min': 0.0,
                'max': 2.0,
                'mean': 1.0,
            }
        )
    rows.append(
        {
            'rank': 0,
            'step': normal,
            'target_id': 0,
            'metric_id': 1,
            'norm': spike_norm,
            'min': 0.0,
            'max': 200.0,
            'mean': 100.0,
        }
    )
    return rows


class TestDetectSpikes(unittest.TestCase):
    def test_spike_detected_normal_not(self):
        rows = _make_step_rows()
        targets = {0: {'name': 'param0'}}
        baselines = compute_per_target_baselines(rows, targets)
        anomalies = detect_spikes(rows, targets, baselines)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]['step'], 20)
        self.assertGreater(anomalies[0]['deviation_ratio'], 50)
        self.assertEqual(anomalies[0]['target_name'], 'param0')


class TestDetectDumpSpikes(unittest.TestCase):
    def test_single_step_top_n(self):
        rows = [
            {'rank': 0, 'step': 0, 'target_id': i, 'metric_id': 1, 'norm': 1.0 + i, 'min': 0.0, 'max': 5.0, 'mean': 1.0}
            for i in range(5)
        ]
        targets = {i: {'name': f'p{i}'} for i in range(5)}
        anomalies = detect_dump_spikes(rows, targets, top_n=2)
        self.assertEqual([a['norm'] for a in anomalies], [5.0, 4.0])
        self.assertEqual(anomalies[0]['trigger'], 'dump_abs_norm')


class TestDetectAccumulationWindow(unittest.TestCase):
    def test_window_detected(self):
        rows = []
        # 9 个 step (<10 走短序列分支), 2 个 target 在 step 4->5 同时重置 (10 -> 0.5)
        for tid in (0, 1):
            for step, norm in enumerate([1, 2, 3, 4, 10, 0.5, 1.5, 2.5, 3.5]):
                rows.append(
                    {
                        'rank': 0,
                        'step': step,
                        'target_id': tid,
                        'metric_id': 1,
                        'norm': norm,
                        'min': 0.0,
                        'max': 20.0,
                        'mean': 1.0,
                    }
                )
        is_ms, window, opt_steps, boundary = detect_accumulation_window(rows)
        self.assertTrue(is_ms)
        self.assertEqual(window, 6)  # 窗口 = 边界之前步数 + 1
        self.assertEqual(boundary, 5)


if __name__ == '__main__':
    unittest.main()

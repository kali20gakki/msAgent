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
Phase 2/3 脚本单元测试: phase2_root_cause_selector / phase3_trace_analyzer
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

from phase2_root_cause_selector import cross_step_root_cause, pick_root_coordinate
from phase3_trace_analyzer import normalize_op_key, ratio


class TestCrossStepRootCause(unittest.TestCase):
    def test_growth_2x_majority_roots_previous_step(self):
        p1 = {
            'target_rank_norms': {
                '0': {'target_name': 'layers.0.weight', 'ranks': {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0}},
                '1': {'target_name': 'layers.0.weight', 'ranks': {'0': 5.0, '1': 5.0, '2': 5.0, '3': 5.0}},
            },
            'anomalies': [],
        }
        root, _ = cross_step_root_cause(p1, {})
        self.assertEqual(root, 0)


class TestPickRootCoordinate(unittest.TestCase):
    def test_no_baseline_picks_max_delta(self):
        anomalies = [
            {
                'rank': 3,
                'target_name': 'a',
                'norm': 30.0,
                'delta': 3.0,
                'micro_step': 1,
                'optimizer_step': 0,
                'deviation_ratio': 3.0,
            },
            {
                'rank': 1,
                'target_name': 'a',
                'norm': 10.0,
                'delta': 5.0,
                'micro_step': 2,
                'optimizer_step': 1,
                'deviation_ratio': 5.0,
            },
            {
                'rank': 2,
                'target_name': 'a',
                'norm': 20.0,
                'delta': 8.0,
                'micro_step': 3,
                'optimizer_step': 1,
                'deviation_ratio': 8.0,
            },
        ]
        coord, reason = pick_root_coordinate(anomalies, root_opt_step=1)
        self.assertEqual(coord['rank'], 2)
        self.assertEqual(coord['delta'], 8.0)
        self.assertIn('无标杆', reason)


class TestRatio(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(ratio(10, 2), 5)

    def test_invalid_inputs(self):
        self.assertIsNone(ratio(1, 0))
        self.assertIsNone(ratio(None, 2))
        self.assertIsNone(ratio(float('inf'), 2))


class TestNormalizeOpKey(unittest.TestCase):
    def test_te_prefix_and_recompute_index(self):
        self.assertEqual(
            normalize_op_key('layers.0.self_attention.TEMLASelfAttention.forward.0'),
            'layers.0.self_attention.MLASelfAttention.forward',
        )

    def test_flash_fused_attention(self):
        self.assertEqual(normalize_op_key('mlp.FlashAttention.forward.1'), 'mlp.Attention.forward')


if __name__ == '__main__':
    unittest.main()

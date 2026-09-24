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
cascade_threshold.py 单元测试
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "skills", "accuracy", "compare-result-analyzer", "scripts"
    ),
)

from cascade_threshold import (
    MIN_THRESHOLD,
    MAX_THRESHOLD,
    _clamp_threshold,
    _sicd_change_point,
    _sicd_multi_window,
    _anchored_backtrack,
    _delta_nre_outlier,
    _distribution_gap,
    _statistical_fallback,
    _detect_segments,
    _low_signal_fallback,
    auto_detect_threshold,
    compute_max_jump_supplement,
    auto_threshold_from_rows,
)


def _make_nodes(values, dtype="float32", shape="(1,)"):
    """构造 output_nodes 列表，values 为 NRE 序列。"""
    return [
        {"idx": i, "nre": v, "name": "op.output.{}".format(i), "dtype": dtype, "shape": shape}
        for i, v in enumerate(values)
    ]


class TestClampThreshold(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(_clamp_threshold(0.05), MIN_THRESHOLD)
        self.assertEqual(_clamp_threshold(50.0), 50.0)
        self.assertEqual(_clamp_threshold(200.0), MAX_THRESHOLD)
        self.assertEqual(_clamp_threshold(5.0), 5.0)


class TestSicdChangePoint(unittest.TestCase):
    def test_zero_baseline(self):
        # 前 5 个 NRE 为 0，后 5 个为 10 → 触发 zero_baseline 变点
        nodes = _make_nodes([0.0] * 5 + [10.0] * 5)
        threshold, method, stats = _sicd_change_point(nodes)
        self.assertAlmostEqual(threshold, 5.0)
        self.assertIn("SICD", method)
        self.assertIn("zero_baseline", method)
        self.assertEqual(stats["sicd_mode"], "zero_baseline")
        self.assertEqual(stats["sicd_cp_nre"], 10.0)


class TestSicdMultiWindow(unittest.TestCase):
    def test_detect_jump(self):
        # 200 个低 NRE 基线（带轻微变化使 std>0），150 个高 NRE 跳变
        # 需足够长以产生多个滑动窗口，使某窗口覆盖跳变点
        baseline = [1.0 + 0.001 * i for i in range(200)]
        spike = [50.0] * 150
        nodes = _make_nodes(baseline + spike)
        threshold, window_size, stats = _sicd_multi_window(nodes)
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, 0.0)
        self.assertEqual(window_size, 200)
        self.assertIn("multi_window_cp_idx", stats)


class TestAnchoredBacktrack(unittest.TestCase):
    def test_jump_detection(self):
        # 5 个低 NRE(0.1)，5 个高 NRE(5.0)，比值 50 → 跳变
        nodes = _make_nodes([0.1] * 5 + [5.0] * 5)
        threshold, method, stats, reliable = _anchored_backtrack(nodes)
        self.assertAlmostEqual(threshold, 0.3)
        self.assertEqual(method, "AnchoredBacktrack")
        self.assertTrue(reliable)
        self.assertEqual(stats["hv2_n_jumps"], 5)
        self.assertLessEqual(stats["hv2_filtered_ratio"], 0.6)


class TestDeltaNreOutlier(unittest.TestCase):
    def test_outlier_detection(self):
        # 15 组 delta=1, 14 组 delta=5, 1 组 delta=100 → IQR>0 且有离群
        groups = []
        for _ in range(15):
            groups.append({"worst_output_nre": 1.0, "worst_input_nre": 0.0})
        for _ in range(14):
            groups.append({"worst_output_nre": 5.0, "worst_input_nre": 0.0})
        groups.append({"worst_output_nre": 100.0, "worst_input_nre": 0.0})
        threshold, method, stats, has_outliers = _delta_nre_outlier(groups)
        self.assertTrue(has_outliers)
        self.assertIn("DeltaNREOutlier", method)
        self.assertGreater(threshold, 0.0)
        self.assertEqual(stats["f_n_outliers"], 1)


class TestDistributionGap(unittest.TestCase):
    def test_gap_detection(self):
        # 9 个 0.1，1 个 5.0 → 比值 50 的分布间隙
        nodes = _make_nodes([0.1] * 9 + [5.0])
        threshold, method, stats, found_gap = _distribution_gap(nodes)
        self.assertTrue(found_gap)
        self.assertIn("DistributionGap", method)
        self.assertAlmostEqual(threshold, 0.2)
        self.assertEqual(stats["a_gap_lower"], 0.1)
        self.assertGreaterEqual(stats["a_max_ratio"], 3.0)


class TestStatisticalFallback(unittest.TestCase):
    def test_threshold(self):
        nodes = _make_nodes([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold, method, stats = _statistical_fallback(nodes)
        # p50=3.0 → max(3*3, 0.1) = 9.0
        self.assertAlmostEqual(threshold, 9.0)
        self.assertEqual(method, "StatisticalFallback")
        self.assertEqual(stats["p50"], 3.0)
        self.assertEqual(stats["p25"], 2.0)


class TestDetectSegments(unittest.TestCase):
    def test_dtype_change(self):
        # 70 节点，第 35 处 dtype 变化 → 2 段
        nodes = [
            {"idx": i, "nre": 1.0, "dtype": "float32" if i < 35 else "float16", "shape": "(1,)"} for i in range(70)
        ]
        segments, count = _detect_segments(nodes)
        self.assertEqual(count, 2)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], (0, 34, "start"))
        self.assertIn("dtype_change", segments[1][2])


class TestLowSignalFallback(unittest.TestCase):
    def test_low_signal_nodes(self):
        # 全局阈值 > 5，第二段统一 dtype，NRE 在 (local_baseline, threshold) 之间
        nodes = [
            {"idx": i, "nre": 0.0, "name": "a.{}".format(i), "dtype": "float32", "shape": "(1,)"} for i in range(35)
        ] + [
            {"idx": 35 + i, "nre": 1.0, "name": "b.{}".format(35 + i), "dtype": "float16", "shape": "(1,)"}
            for i in range(35)
        ]
        segments = [(0, 34, "start"), (35, 69, "dtype_change")]
        cascade_stats = {"p50": 0.1}
        result = _low_signal_fallback(nodes, segments, 10.0, cascade_stats)
        self.assertEqual(len(result), 35)
        self.assertAlmostEqual(result[0]["nre"], 1.0)
        self.assertEqual(result[0]["segment"], 1)


class TestAutoDetectThreshold(unittest.TestCase):
    def test_cascade_sicd(self):
        # 5 个零基线 + 5 个高 NRE → SICD zero_baseline
        nodes = _make_nodes([0.0] * 5 + [10.0] * 5)
        result = auto_detect_threshold(nodes, [])
        self.assertEqual(result["threshold"], 5.0)
        self.assertIn("SICD", result["method"])
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(len(result["per_segment_thresholds"]), 1)
        self.assertEqual(result["low_signal_nodes"], [])


class TestComputeMaxJumpSupplement(unittest.TestCase):
    def test_supplement(self):
        cascade_stats = {"noise_ceiling": 0.1, "p25": 0.1, "p01": 0.001}
        all_nodes = [
            {"name": "keep", "input_nre": 0.05, "output_nre": 1.0, "jump": 1.0},
            {"name": "skip_high_input", "input_nre": 0.5, "output_nre": 1.0, "jump": 1.0},
            {"name": "skip_low_output", "input_nre": 0.05, "output_nre": 0.01, "jump": 0.01},
            {"name": "skip_low_jump", "input_nre": 0.05, "output_nre": 1.0, "jump": 0.01},
        ]
        result = compute_max_jump_supplement(all_nodes, cascade_stats, 10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "keep")


class TestAutoThresholdFromRows(unittest.TestCase):
    def test_from_rows(self):
        rows = []
        for i in range(5):
            rows.append({"NPU Name": "layer.output.{}".format(i), "NormRelativeErr": "0.0", "RowIndex": i})
        for i in range(5):
            rows.append({"NPU Name": "layer.output.{}".format(5 + i), "NormRelativeErr": "10.0", "RowIndex": 5 + i})
        result = auto_threshold_from_rows(rows, [])
        self.assertIn("threshold", result)
        self.assertEqual(result["threshold"], 5.0)
        self.assertIn("SICD", result["method"])


if __name__ == "__main__":
    unittest.main()

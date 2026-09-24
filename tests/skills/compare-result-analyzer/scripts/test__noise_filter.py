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
_noise_filter.py 单元测试
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "skills", "accuracy", "compare-result-analyzer", "scripts"
    ),
)

from _noise_filter import (
    OUTPUT_SUBDIR,
    _ensure_output_dir,
    _default_output_path,
    find_gap_cutoff,
    _shape_str_to_tuple,
    _is_divergence_legitimate,
    _check_nre_relative,
    _is_problem_node,
    _check_tensor_consistency,
    classify_near_zero_noise,
)


class TestEnsureOutputDir(unittest.TestCase):
    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            out_dir = _ensure_output_dir(csv_path)
            self.assertTrue(os.path.isdir(out_dir))
            self.assertEqual(os.path.basename(out_dir), OUTPUT_SUBDIR)


class TestDefaultOutputPath(unittest.TestCase):
    def test_path_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "mydata.csv")
            expected = os.path.join(tmpdir, OUTPUT_SUBDIR, "mydata_result.json")
            self.assertEqual(_default_output_path(csv_path), expected)


class TestFindGapCutoff(unittest.TestCase):
    def test_insufficient_samples(self):
        # < 10 positive values -> None
        self.assertIsNone(find_gap_cutoff([1, 2, 3, 4, 5]))

    def test_clear_gap(self):
        # 10 small distinct values then a big jump -> cutoff at the small cluster end
        vals = [float(i) * 1e-9 for i in range(1, 11)] + [1e-2]
        cutoff = find_gap_cutoff(vals)
        self.assertIsNotNone(cutoff)
        self.assertLess(cutoff, 1e-3)


class TestShapeStrToTuple(unittest.TestCase):
    def test_list_shape(self):
        self.assertEqual(_shape_str_to_tuple("[1, 192]"), (1, 192))


class TestIsDiverenceLegitimate(unittest.TestCase):
    def test_shape_mismatch(self):
        node = {
            "bench_l2norm": 0.5,
            "row": {"Bench Tensor Shape": "[1, 2]"},
            "shape": "[3, 4]",
        }
        self.assertFalse(_is_divergence_legitimate(node))

    def test_legitimate_node(self):
        node = {
            "bench_l2norm": 0.5,
            "row": {"Bench Tensor Shape": "[1, 2]"},
            "shape": "[1, 2]",
        }
        self.assertTrue(_is_divergence_legitimate(node))


class TestCheckNreRelative(unittest.TestCase):
    def test_within_tolerance(self):
        self.assertTrue(_check_nre_relative(1.0, 1.001))


class TestIsProblemNode(unittest.TestCase):
    def test_nre_above_threshold(self):
        self.assertEqual(_is_problem_node(2.0, 0.0, 1.0), (True, "NRE"))

    def test_mean_bias_trigger(self):
        # NRE below threshold but mean_bias >= 1.2 * threshold / 100
        # threshold=100 -> 1.2 * 100 / 100 = 1.2; mean_bias=1.5 -> trigger
        self.assertEqual(_is_problem_node(0.5, 1.5, 100.0), (True, "MeanBias"))


class TestCheckTensorConsistency(unittest.TestCase):
    def test_consistent(self):
        self.assertTrue(_check_tensor_consistency(1.0, 0.01, 0.02, "torch.float32", 1.001, 0.01, 0.02, "torch.float32"))

    def test_nre_inconsistent(self):
        self.assertFalse(_check_tensor_consistency(1.0, 0.01, 0.02, "torch.float32", 5.0, 0.01, 0.02, "torch.float32"))


class TestClassifyNearZeroNoise(unittest.TestCase):
    def _make_node(self, name, dtype, bench_l2, npu_l2, nre, **kw):
        node = {
            "name": name,
            "dtype": dtype,
            "bench_l2norm": bench_l2,
            "npu_l2norm": npu_l2,
            "nre": nre,
            "mean_re": kw.get("mean_re", 0.0),
            "max_re": kw.get("max_re", 0.0),
            "min_re": kw.get("min_re", 0.0),
            "mean_bias": kw.get("mean_bias", 0.0),
            "idx": kw.get("idx", 0),
            "row": kw.get("row", {"Bench Tensor Shape": "[1, 2]"}),
            "shape": kw.get("shape", "[1, 2]"),
        }
        return node

    def test_divergence_signal_preserved(self):
        # bench near zero but NPU has real signal -> divergence, not noise
        node = self._make_node("Module.a.forward.0", "torch.float32", 1e-9, 1.0, 50.0)
        nodes = [node]
        noise_nodes, cutoff_info, result = classify_near_zero_noise(nodes, 1.0)
        self.assertEqual(len(noise_nodes), 0)
        self.assertTrue(node.get("divergence_signal"))
        self.assertGreater(result["divergence_signal_count"], 0)
        self.assertGreater(len(result["divergence_signal_details"]), 0)

    def test_normal_node_not_marked(self):
        node = self._make_node("Module.a.forward.0", "torch.float32", 1.0, 1.0, 0.5)
        noise_nodes, _, result = classify_near_zero_noise([node], 1.0)
        self.assertEqual(len(noise_nodes), 0)
        self.assertFalse(node.get("is_noise"))
        self.assertEqual(result["total_noise_nodes"], 0)


if __name__ == "__main__":
    unittest.main()

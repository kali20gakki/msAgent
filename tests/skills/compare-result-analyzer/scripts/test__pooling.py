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
_pooling.py 单元测试
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

from _pooling import (
    dedup_root_causes_by_nre_l2,
    _is_shape_transform_op,
    trace_execution_chain,
    detect_data_coverage_gaps,
)


class TestDedupRootCausesByNreL2(unittest.TestCase):
    def test_dedup_duplicates(self):
        rc1 = ("op1", "forward", 0.1, 0.5, 0.3)
        rc2 = ("op2", "forward", 0.1, 0.5, 0.3)
        rc3 = ("op3", "backward", 0.2, 0.6, 0.4)
        result = dedup_root_causes_by_nre_l2([rc1, rc2, rc3])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], rc1)
        self.assertEqual(result[1], rc3)

    def test_none_values_treated_as_zero(self):
        rc1 = ("op1", "forward", None, None, None)
        rc2 = ("op2", "forward", None, None, None)
        result = dedup_root_causes_by_nre_l2([rc1, rc2])
        self.assertEqual(len(result), 1)


class TestIsShapeTransformOp(unittest.TestCase):
    def test_reshape(self):
        self.assertTrue(_is_shape_transform_op("torch.reshape"))

    def test_non_shape_op_relu(self):
        self.assertFalse(_is_shape_transform_op("relu"))


class TestTraceExecutionChain(unittest.TestCase):
    def test_clean_input_no_trace(self):
        first_point = {
            "prefix": "Module.A.backward.0",
            "row_index": 5,
            "name": "Module.A.parameters_grad.0.weight",
            "nre": 0.5,
            "input_nres": [{"nre": 0.01}],
        }
        result = trace_execution_chain(first_point, [], 0.1)
        self.assertEqual(len(result["trace_path"]), 1)
        self.assertEqual(result["trace_path"][0]["role"], "first_point")
        self.assertIn("input 干净", result["trace_path"][0]["note"])
        self.assertFalse(result["trace_boundary_reached"])

    def test_dirty_input_traces_upstream(self):
        rows = [
            {
                "NPU Name": "Module.A.parameters_grad.0.weight",
                "NormRelativeErr": "0.5",
                "RowIndex": 10,
            },
            {
                "NPU Name": "Module.B.parameters_grad.0.weight",
                "NormRelativeErr": "0.5",
                "RowIndex": 12,
            },
            {
                "NPU Name": "Module.C.parameters_grad.0.weight",
                "NormRelativeErr": "0.01",
                "RowIndex": 15,
            },
        ]
        first_point = {
            "prefix": "Module.A.backward.0",
            "row_index": 10,
            "name": "Module.A.parameters_grad.0.weight",
            "nre": 0.5,
            "input_nres": [{"nre": 0.5}],
        }
        result = trace_execution_chain(first_point, rows, 0.1)
        self.assertGreaterEqual(len(result["trace_path"]), 2)
        self.assertEqual(result["trace_path"][0]["role"], "first_point")
        self.assertEqual(result["trace_path"][1]["role"], "upstream_source")
        self.assertEqual(result["trace_path"][1]["prefix"], "Module.B.backward.0")
        # NRE-inconsistent node (Module.C) should be skipped
        self.assertEqual(len(result["skipped_inconsistent"]), 1)
        self.assertEqual(result["skipped_inconsistent"][0]["prefix"], "Module.C.backward.0")
        self.assertIsNotNone(result["trace_note"])


class TestDetectDataCoverageGaps(unittest.TestCase):
    def test_gap_detected(self):
        rows = [
            {
                "NPU Name": "Module.A.forward.0.output.0",
                "NormRelativeErr": "0.001",
                "RowIndex": 1,
            },
            {
                "NPU Name": "Module.B.forward.0.input.0",
                "NormRelativeErr": "0.5",
                "RowIndex": 5,
            },
        ]
        gaps = detect_data_coverage_gaps(rows, 0.1)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["from_op"], "Module.A.forward.0")
        self.assertEqual(gaps[0]["to_op"], "Module.B.forward.0")
        self.assertEqual(gaps[0]["from_row"], 1)
        self.assertEqual(gaps[0]["to_row"], 5)
        self.assertEqual(gaps[0]["gap_size"], 4)
        self.assertAlmostEqual(gaps[0]["from_output_nre"], 0.001)
        self.assertAlmostEqual(gaps[0]["to_input_nre"], 0.5)

    def test_no_gap_dirty_output(self):
        rows = [
            {
                "NPU Name": "Module.A.forward.0.output.0",
                "NormRelativeErr": "0.5",
                "RowIndex": 1,
            },
            {
                "NPU Name": "Module.B.forward.0.input.0",
                "NormRelativeErr": "0.5",
                "RowIndex": 5,
            },
        ]
        gaps = detect_data_coverage_gaps(rows, 0.1)
        self.assertEqual(len(gaps), 0)


if __name__ == "__main__":
    unittest.main()

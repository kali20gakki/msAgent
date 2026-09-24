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
"""analyze_stat.py 单元测试"""

import json
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

from analyze_stat import (
    _precompute_amplifier_metadata,
    _precompute_spike_indicators,
    _precompute_fb_candidates,
    _precompute_pool_external_indicators,
    _precompute_param_grad_three_category,
    build_pool_input,
    _merge_pool_top,
)


def _rc(
    prefix,
    direction,
    inp_nre,
    out_nre,
    jump=0.0,
    inp_mean=None,
    out_mean=None,
    category="ROOT_CAUSE",
    trigger="",
    is_param=False,
    param_grad_output=False,
    no_downstream=False,
    inheritance=None,
):
    """构造传播分类条目元组（与 propagation_analysis 输出格式一致）。"""
    return (
        prefix,
        direction,
        inp_nre,
        out_nre,
        jump,
        inp_mean,
        out_mean,
        category,
        trigger,
        is_param,
        param_grad_output,
        no_downstream,
        inheritance,
    )


def _op_group(prefix, direction, worst_in=None, worst_out=None, row_range=None, params=None):
    """构造 op_groups 条目。"""
    return {
        'prefix': prefix,
        'direction': direction,
        'worst_input_nre': worst_in,
        'worst_output_nre': worst_out,
        'row_range': row_range or [0, 0],
        'params': params or {},
    }


class TestPrecomputeAmplifierMetadata(unittest.TestCase):
    def test_all_inputs_clean_and_high_ratio(self):
        # input_nre below threshold (clean), output_nre small, ratio > 2
        rcs = [_rc("Module.A.forward.0", "forward", 0.1, 0.5, jump=0.4)]
        groups = [_op_group("Module.A.forward.0", "forward", worst_in=0.1)]
        result = _precompute_amplifier_metadata(rcs, groups, 1.0)
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertEqual(c["prefix"], "Module.A.forward.0")
        self.assertEqual(c["direction"], "forward")
        self.assertTrue(c["all_inputs_clean"])
        self.assertEqual(c["amplification_ratio"], 5.0)

    def test_sorted_by_jump_desc(self):
        rcs = [
            _rc("Module.A.forward.0", "forward", 0.1, 0.5, jump=0.4),
            _rc("Module.B.forward.0", "forward", 0.1, 0.9, jump=0.8),
        ]
        groups = [
            _op_group("Module.A.forward.0", "forward", worst_in=0.1),
            _op_group("Module.B.forward.0", "forward", worst_in=0.1),
        ]
        result = _precompute_amplifier_metadata(rcs, groups, 1.0)
        self.assertEqual(result[0]["prefix"], "Module.B.forward.0")
        self.assertGreater(abs(result[0]["jump"]), abs(result[1]["jump"]))


class TestPrecomputeSpikeIndicators(unittest.TestCase):
    def test_extreme_from_pre_filter_snapshot(self):
        snap = [{"output_nre": 150, "prefix": "X", "direction": "backward"}]
        result = _precompute_spike_indicators([], {}, pre_filter_snapshot=snap)
        self.assertTrue(result["has_extreme_backward"])
        self.assertEqual(result["extreme_backward_count"], 1)
        self.assertTrue(result["spike_condition_met"])

    def test_forward_backward_ratio(self):
        nodes = [
            {"name": "Module.A.forward.0.output.0"},
            {"name": "Module.A.backward.0.output.0"},
        ]
        result = _precompute_spike_indicators(nodes, {})
        self.assertEqual(result["total_forward_nodes"], 1)
        self.assertEqual(result["total_backward_nodes"], 1)
        self.assertEqual(result["forward_backward_ratio"], 1.0)


class TestPrecomputeFbCandidates(unittest.TestCase):
    def test_backward_dominant(self):
        first_point = {"direction": "backward", "prefix": "M.backward.0"}
        all_prop = {
            "propagation": [
                _rc("M2.backward.0", "backward", 1.0, 80.0, jump=79.0),
                _rc("M3.backward.0", "backward", 1.0, 30.0, jump=29.0),
            ]
        }
        result = _precompute_fb_candidates(first_point, all_prop, 1.0)
        # only NRE > 50 kept; sorted desc by backward_nre
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["backward_prefix"], "M2.backward.0")
        self.assertEqual(result[0]["direction"], "backward_dominant")

    def test_forward_dominant_with_jump(self):
        first_point = {"direction": "forward", "prefix": "F.forward.0"}
        all_prop = {
            "propagation": [
                _rc("F.forward.0", "forward", 1.0, 2.0, jump=1.0),
                _rc("B.backward.0", "backward", 1.0, 120.0, jump=119.0),
            ]
        }
        result = _precompute_fb_candidates(first_point, all_prop, 0.5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["backward_prefix"], "B.backward.0")
        self.assertEqual(result[0]["confidence"], "high")


class TestPrecomputePoolExternalIndicators(unittest.TestCase):
    def test_returns_entries_above_threshold(self):
        all_prop = {
            "propagation": [_rc("M.forward.0", "forward", 0.1, 2.0, jump=1.9)],
            "root_cause": [],
            "pass_through": [],
            "input_propagation": [],
        }
        groups = [_op_group("M.forward.0", "forward", row_range=[5, 10])]
        result = _precompute_pool_external_indicators(all_prop, {}, groups, 1.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prefix"], "M.forward.0")
        self.assertEqual(result[0]["row_range"], [5, 10])
        self.assertEqual(result[0]["family_key"], "M")

    def test_excludes_covered_prefixes(self):
        all_prop = {
            "propagation": [_rc("M.forward.0", "forward", 0.1, 2.0, jump=1.9)],
        }
        top_rc = {"forward": [{"prefix": "M.forward.0", "direction": "forward"}]}
        result = _precompute_pool_external_indicators(all_prop, top_rc, [], 1.0)
        self.assertEqual(result, [])


class TestPrecomputeParamGradThreeCategory(unittest.TestCase):
    def test_classifies_param_grad(self):
        # prefixes contain .parameters_grad. (matches the function's filter);
        # op_groups use the SAME prefix so the row_range lookup succeeds.
        all_prop = {
            "root_cause": [
                _rc("Module.A.parameters_grad.0", "backward", None, 10.0, jump=10.0, param_grad_output=True),
                _rc("Module.B.parameters_grad.0", "backward", None, 5.0, jump=5.0, param_grad_output=True),
            ],
        }
        groups = [
            _op_group("Module.A.parameters_grad.0", "backward", row_range=[3, 5]),
            _op_group("Module.B.parameters_grad.0", "backward", row_range=[1, 2]),
        ]
        result = _precompute_param_grad_three_category(all_prop, groups, 1.0)
        self.assertEqual(len(result["isolated_large_nre"]), 2)
        self.assertEqual(result["isolated_large_nre"][0]["prefix"], "Module.A.parameters_grad.0")
        self.assertEqual(result["execution_order_first"][0]["prefix"], "Module.B.parameters_grad.0")


class TestBuildPoolInput(unittest.TestCase):
    def test_splits_forward_backward(self):
        rcs = [
            _rc("M.forward.0", "forward", 0.1, 2.0, jump=1.9),
            _rc("M.backward.0", "backward", 0.1, 3.0, jump=2.9),
        ]
        groups = [
            _op_group("M.forward.0", "forward", row_range=[5, 10]),
            _op_group("M.backward.0", "backward", row_range=[6, 12]),
        ]
        result = build_pool_input(rcs, groups, 1.0)
        self.assertEqual(len(result["forward"]), 1)
        self.assertEqual(len(result["backward"]), 1)
        self.assertEqual(result["forward"][0]["prefix"], "M.forward.0")
        self.assertEqual(result["forward"][0]["row_range"], [5, 10])


class TestMergePoolTop(unittest.TestCase):
    def test_dedup_by_prefix_direction(self):
        fwd = [
            {
                "prefix": "M",
                "direction": "forward",
                "row_start": 0,
                "abs_magnitude": 1.0,
                "tagging": "none",
                "is_true_root_cause_feature": False,
            }
        ]
        fwd2 = [
            {
                "prefix": "M",
                "direction": "forward",
                "row_start": 0,
                "abs_magnitude": 2.0,
                "tagging": "none",
                "is_true_root_cause_feature": False,
            }
        ]
        result = _merge_pool_top({"forward": fwd + fwd2, "backward": []})
        self.assertEqual(len(result["forward"]), 1)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "compare_result.csv")
        header = (
            "NPU Name,Bench Name,NormRelativeErr,MeanRelativeErr,"
            "MaxRelativeErr,MinRelativeErr,Max diff,Mean diff,"
            "L2norm diff,Bench l2norm,NPU l2norm,Result,"
            "NPU Dtype,Bench Dtype,NPU Tensor Shape,Bench Tensor Shape,"
            "Requires_grad Consistent\n"
        )
        rows = [
            header,
            "Module.layer1.forward.0.input.0,Bench,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,pass,torch.float32,torch.float32,[1,1],[1,1],True\n",
            "Module.layer1.forward.0.output.0,Bench,5.0,5.0,5.0,5.0,0.05,0.05,0.05,1.0,1.0,pass,torch.float32,torch.float32,[1,1],[1,1],True\n",
            "Module.layer1.backward.0.output.0,Bench,0.2,0.2,0.2,0.2,0.0,0.0,0.0,1.0,1.0,pass,torch.float32,torch.float32,[1,1],[1,1],True\n",
        ]
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.writelines(rows)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_main_keep_only_zero_rows(self):
        # --keep-only with a non-matching keyword produces 0 rows; main writes
        # an empty-result JSON to a keep-only-specific path (not -o) and returns.
        rc = main_shim_keep_only(self.csv_path, "nonexistent_kw_xyz", None)
        self.assertEqual(rc, 0)
        # The empty result is written under .compare_result_analyzer/ next to csv
        csv_dir = os.path.dirname(self.csv_path)
        out_dir = os.path.join(csv_dir, ".compare_result_analyzer")
        expected = os.path.join(out_dir, "compare_result_nonexistent_kw_xyz_result.json")
        self.assertTrue(os.path.exists(expected))
        with open(expected, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["meta"]["total_rows"], 0)
        self.assertIn("error", data)


def main_shim_keep_only(csv_path, keyword, output_path=None):
    """以 --keep-only 调用 main。"""
    import importlib
    import analyze_stat

    importlib.reload(analyze_stat)
    from analyze_stat import main as real_main

    old_argv = sys.argv
    argv = ["analyze_stat.py", csv_path, "--format", "json", "--keep-only", keyword]
    if output_path:
        argv += ["-o", output_path]
    sys.argv = argv
    rc = 0
    try:
        real_main()
    except SystemExit as e:
        rc = e.code if e.code is not None else 0
    finally:
        sys.argv = old_argv
    return rc


if __name__ == "__main__":
    unittest.main()

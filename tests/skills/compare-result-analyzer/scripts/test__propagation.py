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
_propagation.py 单元测试
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

from _propagation import (
    first_problem_point,
    _nre_close,
    propagation_analysis,
)


def _make_row(
    name, nre='', mean_re='', max_re='', mean_diff='', bench_l2norm='', shape='', dtype='', row_index=0, result=''
):
    """构造一行 CSV/XLSX 风格的 row dict（供 propagation_analysis / all_rows 使用）。"""
    return {
        'NPU Name': name,
        'NormRelativeErr': str(nre) if nre != '' else '',
        'MeanRelativeErr': str(mean_re) if mean_re != '' else '',
        'MaxRelativeErr': str(max_re) if max_re != '' else '',
        'Mean diff': str(mean_diff) if mean_diff != '' else '',
        'Bench l2norm': str(bench_l2norm) if bench_l2norm != '' else '',
        'NPU Tensor Shape': shape,
        'NPU Dtype': dtype,
        'RowIndex': row_index,
        'Result': result,
    }


def _make_node(
    idx, name, nre, mean_re=None, max_re=None, mean_bias=None, dtype='torch.float32', shape='[1]', result=''
):
    """构造一个 output 节点 dict（供 first_problem_point 使用）。"""
    return {
        'idx': idx,
        'name': name,
        'nre': nre,
        'mean_re': mean_re,
        'max_re': max_re,
        'mean_bias': mean_bias,
        'dtype': dtype,
        'shape': shape,
        'result': result,
    }


class TestNreClose(unittest.TestCase):
    def test_none_returns_false(self):
        self.assertFalse(_nre_close(None, 5))
        self.assertFalse(_nre_close(5, None))

    def test_close_values(self):
        # |10-12|/12 ≈ 0.167 <= 0.5
        self.assertTrue(_nre_close(10, 12))

    def test_custom_tolerance(self):
        # |10-11|/11 ≈ 0.09 > 0.05
        self.assertFalse(_nre_close(10, 11, tol=0.05))
        # |10-10.1|/10.1 ≈ 0.0099 <= 0.05
        self.assertTrue(_nre_close(10, 10.1, tol=0.05))


class TestFirstProblemPoint(unittest.TestCase):
    def test_simple_first(self):
        nodes = [
            _make_node(2, 'Module.A.forward.0.output.0', nre=1.0, mean_re=1.0, max_re=1.0),
            _make_node(5, 'Module.B.forward.0.output.0', nre=10.0, mean_re=10.0, max_re=10.0),
        ]
        first, skipped, result = first_problem_point(nodes, 5)
        self.assertIsNotNone(first)
        self.assertEqual(first['name'], 'Module.B.forward.0.output.0')
        self.assertEqual(skipped, [])
        self.assertEqual(result['confirmed']['name'], 'Module.B.forward.0.output.0')
        self.assertEqual(result['confirmed']['nre'], 10.0)
        self.assertEqual(first['_trigger'], 'NRE')

    def test_input_propagation_skip(self):
        nodes = [
            _make_node(2, 'Module.A.forward.0.output.0', nre=10.0, mean_re=10.0, max_re=10.0),
        ]
        all_rows = [
            _make_row(
                'Module.A.forward.0.input.0',
                nre=10.0,
                mean_re=10.0,
                max_re=10.0,
                dtype='torch.float32',
                shape='[1]',
                row_index=1,
            ),
            _make_row(
                'Module.A.forward.0.output.0',
                nre=10.0,
                mean_re=10.0,
                max_re=10.0,
                dtype='torch.float32',
                shape='[1]',
                row_index=2,
            ),
        ]
        first, skipped, result = first_problem_point(nodes, 5, all_rows)
        self.assertIsNone(first)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][2], 'INPUT_PROPAGATION')
        self.assertEqual(skipped[0][1], 'input.0')
        self.assertIsNone(result['confirmed'])

    def test_downstream_absorbed_skip(self):
        nodes = [
            _make_node(2, 'Module.A.forward.0.output.0', nre=10.0, mean_re=10.0, max_re=10.0),
            _make_node(5, 'Module.B.forward.0.output.0', nre=1.0, mean_re=1.0, max_re=1.0),
        ]
        all_rows = [
            _make_row(
                'Module.A.forward.0.output.0',
                nre=10.0,
                mean_re=10.0,
                max_re=10.0,
                dtype='torch.float32',
                shape='[1]',
                row_index=2,
            ),
            _make_row(
                'Module.B.forward.0.input.0',
                nre=10.0,
                mean_re=10.0,
                max_re=10.0,
                dtype='torch.float32',
                shape='[1]',
                row_index=4,
            ),
            _make_row(
                'Module.B.forward.0.output.0',
                nre=1.0,
                mean_re=1.0,
                max_re=1.0,
                dtype='torch.float32',
                shape='[1]',
                row_index=5,
            ),
        ]
        first, skipped, result = first_problem_point(nodes, 5, all_rows)
        self.assertIsNone(first)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][2], 'DOWNSTREAM_ABSORBED')
        self.assertEqual(skipped[0][1], 'Module.B.forward.0')


class TestPropagationAnalysis(unittest.TestCase):
    def test_root_cause(self):
        rows = [
            _make_row('Module.A.forward.0.input.0', nre=1.0, mean_re=1.0, max_re=1.0, row_index=2),
            _make_row('Module.A.forward.0.output.0', nre=10.0, mean_re=10.0, max_re=10.0, row_index=3),
        ]
        rc, ab, pg, ip, pt, og = propagation_analysis(rows, 5)
        self.assertEqual(len(rc), 1)
        entry = rc[0]
        self.assertEqual(entry[0], 'Module.A.forward.0')
        self.assertEqual(entry[1], 'forward')
        self.assertEqual(entry[3], 10.0)  # out_nre
        self.assertEqual(entry[7], 'ROOT_CAUSE')
        self.assertEqual(entry[8], 'NRE')
        self.assertEqual(ab, [])
        self.assertEqual(pg, [])

    def test_absorbed(self):
        rows = [
            _make_row('Module.A.forward.0.input.0', nre=10.0, mean_re=10.0, max_re=10.0, row_index=2),
            _make_row('Module.A.forward.0.output.0', nre=1.0, mean_re=1.0, max_re=1.0, row_index=3),
        ]
        rc, ab, pg, ip, pt, og = propagation_analysis(rows, 5)
        self.assertEqual(len(ab), 1)
        entry = ab[0]
        self.assertEqual(entry[0], 'Module.A.forward.0')
        self.assertEqual(entry[1], 'forward')
        self.assertEqual(entry[3], 1.0)  # out_nre
        self.assertEqual(entry[7], 'ABSORBED')
        self.assertEqual(entry[8], 'NRE')

    def test_propagation_amplified(self):
        rows = [
            _make_row('Module.A.forward.0.input.0', nre=10.0, mean_re=10.0, max_re=10.0, row_index=2),
            _make_row('Module.A.forward.0.output.0', nre=20.0, mean_re=20.0, max_re=20.0, row_index=3),
        ]
        rc, ab, pg, ip, pt, og = propagation_analysis(rows, 5)
        self.assertEqual(len(pg), 1)
        entry = pg[0]
        self.assertEqual(entry[0], 'Module.A.forward.0')
        self.assertEqual(entry[3], 20.0)  # out_nre
        self.assertEqual(entry[7], 'PROPAGATION')
        self.assertEqual(entry[8], 'NRE')

    def test_input_propagation(self):
        rows = [
            _make_row('Module.A.forward.0.input.0', nre=10.0, mean_re=10.0, max_re=10.0, row_index=2),
            _make_row('Module.A.forward.0.output.0', nre=10.0, mean_re=10.0, max_re=10.0, row_index=3),
        ]
        rc, ab, pg, ip, pt, og = propagation_analysis(rows, 5)
        self.assertEqual(len(ip), 1)
        entry = ip[0]
        self.assertEqual(entry[0], 'Module.A.forward.0')
        self.assertEqual(entry[1], 'forward')
        self.assertEqual(entry[7], 'INPUT_PROPAGATION')
        self.assertEqual(entry[8], 'input.0')
        # No root_cause / propagation when fully inherited
        self.assertEqual(rc, [])
        self.assertEqual(pg, [])

    def test_op_groups_structure(self):
        rows = [
            _make_row('Module.A.forward.0.input.0', nre=1.0, mean_re=1.0, max_re=1.0, row_index=2),
            _make_row('Module.A.forward.0.output.0', nre=10.0, mean_re=10.0, max_re=10.0, row_index=3),
        ]
        rc, ab, pg, ip, pt, og = propagation_analysis(rows, 5)
        self.assertEqual(len(og), 1)
        grp = og[0]
        self.assertEqual(grp['prefix'], 'Module.A.forward.0')
        self.assertEqual(grp['direction'], 'forward')
        self.assertEqual(grp['row_range'], [2, 3])
        self.assertEqual(grp['n_inputs'], 1)
        self.assertEqual(grp['n_outputs'], 1)
        self.assertEqual(grp['worst_input_nre'], 1.0)
        self.assertEqual(grp['worst_output_nre'], 10.0)
        self.assertEqual(grp['dirty_inputs'], [])
        self.assertEqual(grp['input_subtype'], 'INPUT_ALL_CLEAN')


if __name__ == "__main__":
    unittest.main()

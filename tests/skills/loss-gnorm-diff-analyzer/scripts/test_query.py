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
"""query.py 核心工具函数单元测试（族名/槽位路径/四元组匹配/模式匹配/可比性）。"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "accuracy",
        "loss-gnorm-diff-analyzer",
        "scripts",
    ),
)

from query import _quad_matches, _slot_cmp, family_of, match_names, resolve_slot


class TestFamilyOf(unittest.TestCase):
    def test_family_of(self):
        self.assertEqual(family_of("Tensor.permute.0.forward"), "Tensor.permute.forward")
        self.assertEqual(family_of("Module.a.Linear.forward.0"), "Module.a.Linear.forward")
        self.assertEqual(family_of("plain.node"), "plain.node")


class TestResolveSlot(unittest.TestCase):
    def test_dict_list_and_missing(self):
        node = {
            "output": {"0": {"type": "torch.Tensor", "Norm": 1.0}},
            "input_args": [{"type": "torch.Tensor", "Norm": 2.0}],
        }
        ok, v = resolve_slot(node, "output.0")
        self.assertTrue(ok)
        self.assertEqual(v["Norm"], 1.0)
        ok2, v2 = resolve_slot(node, "input_args.0")
        self.assertTrue(ok2)
        self.assertEqual(v2["Norm"], 2.0)
        self.assertFalse(resolve_slot(node, "output.9")[0])


class TestQuadMatches(unittest.TestCase):
    def test_exact_vs_tolerance(self):
        q = (1.0, -1.0, 0.5, 2.0)
        self.assertTrue(_quad_matches(q, q, 0))
        self.assertTrue(_quad_matches(q, (1.0 + 1e-9,) + q[1:], 1e-6))
        self.assertFalse(_quad_matches(q, (1.1,) + q[1:], 1e-6))

    def test_nan_semantics(self):
        q = (float("nan"), 0.0, 0.0, 0.0)
        self.assertFalse(_quad_matches(q, q, 0))
        self.assertTrue(_quad_matches(q, q, 1e-6))


class TestMatchNames(unittest.TestCase):
    def test_exact_and_substring(self):
        om = {"Tensor.permute.0.forward": 0, "Tensor.cast.1.backward": 1}
        self.assertEqual(match_names(om, ["Tensor.permute.0.forward"]), ["Tensor.permute.0.forward"])
        self.assertEqual(match_names(om, ["cast"]), ["Tensor.cast.1.backward"])


class TestSlotCmp(unittest.TestCase):
    def _leaf(self, **kw):
        leaf = {
            "type": "torch.Tensor",
            "dtype": "torch.float32",
            "shape": [2],
            "Max": 1.0,
            "Min": -1.0,
            "Mean": 0.0,
            "Norm": 1.0,
        }
        leaf.update(kw)
        return leaf

    def test_kinds(self):
        self.assertEqual(_slot_cmp(self._leaf(), self._leaf()), ("nre", 0.0))
        self.assertAlmostEqual(_slot_cmp(self._leaf(), self._leaf(Norm=2.0))[1], 0.5)
        self.assertEqual(_slot_cmp(self._leaf(), self._leaf(dtype="torch.float16"))[0], "dtype")
        self.assertEqual(_slot_cmp(self._leaf(), self._leaf(shape=[3]))[0], "shape")
        self.assertEqual(_slot_cmp(self._leaf(), self._leaf(Norm=float("nan")))[0], "nonfinite")
        self.assertEqual(_slot_cmp(None, self._leaf())[0], "na")


if __name__ == "__main__":
    unittest.main()

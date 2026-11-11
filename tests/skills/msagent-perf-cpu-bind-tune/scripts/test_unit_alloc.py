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
# MERCHANTABILITY OR FIT FOR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""unit_alloc.py：角色分核、收缩下限与线程绑定语义。"""

import unittest

import testutils  # noqa: F401
import unit_alloc


def _unit(threads, framework="vllm"):
    return {"unit_id": 0, "framework": framework, "thread_roles": dict(threads)}


class TestAllocate(unittest.TestCase):
    def test_role_split_from_thread_counts(self):
        """A2 实机画像：comm 24 → 2 核，io 4 → 1 核，main 15 → 1 核。"""
        unit = unit_alloc.allocate(_unit({"comm": 24, "io": 4, "main": 15, "compute": 47}), list(range(144, 155)))
        self.assertEqual(unit["cpuset"], "144-154")
        self.assertEqual(unit["main_cpuset"], "154")
        self.assertEqual(unit["compute_cpuset"], "144-150")
        self.assertEqual(unit["work_cpuset"], "144-150,154")
        self.assertEqual(
            {k: v["cores"] for k, v in unit["roles"].items()}, {"compute": 7, "comm": 2, "io": 1, "main": 1}
        )
        self.assertEqual(unit["env"]["OMP_NUM_THREADS"], "7")

    def test_framework_reserves_comm_and_io(self):
        """无 comm/io 线程时，vllm 仍各保留 1 核。"""
        unit = unit_alloc.allocate(_unit({"comm": 0, "io": 0, "main": 0, "compute": 10}), list(range(0, 8)))
        self.assertEqual(unit["roles"]["comm"]["cores"], 1)
        self.assertEqual(unit["roles"]["io"]["cores"], 1)
        self.assertEqual(unit["roles"]["compute"]["cores"], 5)

    def test_unknown_framework_keeps_zero(self):
        unit = unit_alloc.allocate(_unit({"comm": 0, "io": 0, "main": 0}, framework="unknown"), list(range(0, 6)))
        self.assertEqual(unit["roles"]["comm"]["cores"], 0)
        self.assertEqual(unit["thread_binding"][1]["apply"], "skip")
        self.assertEqual(unit["thread_binding"][2]["apply"], "skip")

    def test_compute_bound_by_default(self):
        unit = unit_alloc.allocate(_unit({"comm": 2, "io": 1, "main": 1}), list(range(0, 10)))
        compute = [b for b in unit["thread_binding"] if b["role"] == "compute"][0]
        self.assertEqual(compute["apply"], "exec")
        self.assertEqual(compute["cpuset"], unit["compute_cpuset"])
        self.assertEqual(compute["cpuset"], unit["roles"]["compute"]["cpus"])
        self.assertNotIn("toggle", compute)
        self.assertEqual(
            compute["match_comm"], ["vllm", "caffe", "python", "pt_autograd", "acl_thread", "compl", "omp"]
        )

    def test_shrink_when_cores_insufficient(self):
        """4 核时 comm 由 2 缩到 1，保证 compute 至少 1 核。"""
        unit = unit_alloc.allocate(_unit({"comm": 24, "io": 4, "main": 15}), [0, 1, 2, 3])
        self.assertEqual(unit["roles"]["comm"]["cores"], 1)
        self.assertEqual(unit["roles"]["io"]["cores"], 1)
        self.assertEqual(unit["roles"]["main"]["cores"], 1)
        self.assertEqual(unit["roles"]["compute"]["cores"], 1)

    def test_shrink_to_minimum_on_three_cores(self):
        """3 核时收缩到 comm1 + main1 + compute1、io 归零，核集仍完整覆盖。"""
        unit = unit_alloc.allocate(_unit({"comm": 24, "io": 4, "main": 15}), [0, 1, 2])
        self.assertEqual(unit["roles"]["comm"]["cores"], 1)
        self.assertEqual(unit["roles"]["io"]["cores"], 0)
        self.assertEqual(unit["roles"]["main"]["cores"], 1)
        self.assertEqual(unit["roles"]["compute"]["cores"], 1)
        self.assertEqual(unit["cpuset"], "0-2")

    def test_single_cpu_fallback(self):
        unit = unit_alloc.allocate(_unit({"comm": 1, "io": 0, "main": 0}), [7])
        self.assertEqual(unit["cpuset"], "7")
        self.assertTrue(unit["main_cpuset"])
        self.assertEqual(unit["roles"]["compute"]["cores"], 1)

    def test_main_shrink_keeps_roles_consistent(self):
        """main 收缩到 0 核时兜底核集需同时反映在 roles 与 thread_binding。"""
        unit = unit_alloc.allocate(_unit({"comm": 1, "io": 0, "main": 0}), [7])
        self.assertEqual(unit["roles"]["main"]["cpus"], unit["main_cpuset"])
        main_binding = [b for b in unit["thread_binding"] if b["role"] == "main"][0]
        self.assertEqual(main_binding["cpuset"], unit["main_cpuset"])


if __name__ == "__main__":
    unittest.main()

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

"""discover.py：进程发现、unit 聚类、偏差判定与线程角色统计。"""

import unittest
from unittest import mock

import testutils
import discover
import procfs


class TestClassify(unittest.TestCase):
    def test_framework_patterns(self):
        self.assertEqual(discover.classify_framework("vllm serve", "VLLM::Worker"), "vllm")
        self.assertEqual(discover.classify_framework("sglang.launch_server", ""), "sglang")
        self.assertEqual(discover.classify_framework("torchrun train.py", ""), "megatron")
        self.assertEqual(discover.classify_framework("random", "unknown-proc"), "unknown")

    def test_classify_thread_delegates_to_roles(self):
        self.assertEqual(discover.classify_thread("hccl_watchdog_t"), "comm")
        self.assertEqual(discover.classify_thread("WatchdogMonitor"), "main")


class TestCollectUnits(unittest.TestCase):
    def setUp(self):
        self.info, topo_info = testutils.load_a2_npu_raw()
        self.topo = testutils.a2_topo(self.info, topo_info)

    def test_units_and_node_attribution(self):
        with testutils.mock_uid(), testutils.MockProcfs():
            discovered = discover.collect_units(self.info, self.topo)
        units = discovered["units"]
        self.assertEqual(len(units), 3)
        self.assertEqual(
            [(u["pid"], u["primary_npu"], u["npu_node"], u["current_node"], u["current_affinity"]) for u in units],
            [(293354, 0, 6, 6, "146-165"), (293350, 1, 6, 7, "170-189"), (293351, 6, 2, 2, "50-69")],
        )

    def test_deviations_and_evidence(self):
        with testutils.mock_uid(), testutils.MockProcfs():
            units = discover.collect_units(self.info, self.topo)["units"]
        kinds = [[d["kind"] for d in u["deviations"]] for u in units]
        self.assertIn("process_node_vs_npu_node", kinds[1])
        self.assertIn("affinity_outside_npu_range", kinds[1])
        self.assertIn("last_cpu_node_mismatch", kinds[1])
        self.assertIn("numa_maps_unavailable", kinds[0])
        self.assertIn("cross_user_control", kinds[0])
        self.assertTrue(all(u["evidence_level"] == "affinity+last_cpu" for u in units))
        self.assertEqual(units[1]["last_cpu_node"], 7)
        self.assertEqual(units[1]["thread_last_cpu_nodes"], {7: 90})

    def test_thread_roles_and_framework(self):
        with testutils.mock_uid(), testutils.MockProcfs():
            units = discover.collect_units(self.info, self.topo)["units"]
        self.assertEqual(units[0]["thread_roles"], {"compute": 47, "comm": 24, "io": 4, "main": 15})
        self.assertEqual({u["framework"] for u in units}, {"vllm"})

    def test_view_limited_propagates(self):
        with testutils.mock_uid(), testutils.MockProcfs():
            discovered = discover.collect_units(self.info, self.topo)
        self.assertTrue(discovered["view_limited"])

    def test_other_uid_marks_cross_user(self):
        with testutils.mock_uid(), testutils.MockProcfs(uid=testutils.OTHER_UID):
            units = discover.collect_units(self.info, self.topo)["units"]
        self.assertTrue(all(u["is_own_process"] is False for u in units))
        self.assertTrue(all(u["target_uid"] == testutils.OTHER_UID for u in units))

    def test_uid_fields_independent_of_runtime_user(self):
        """自身进程判定由注入的 getuid 决定，不依赖跑测用户（CI 可能 root）。"""
        with testutils.mock_uid(testutils.OTHER_UID), testutils.MockProcfs(uid=testutils.OTHER_UID):
            units = discover.collect_units(self.info, self.topo)["units"]
        self.assertTrue(all(u["is_own_process"] is True for u in units))
        self.assertTrue(all("cross_user_control" not in [d["kind"] for d in u["deviations"]] for u in units))

    def test_unit_id_stays_continuous_after_skip(self):
        """无法采集的进程被跳过时，unit_id 仍连续（不复用输入下标）。"""
        real = discover._collect_proc_basics
        calls = {"n": 0}

        def skip_first(proc, *args):
            calls["n"] += 1
            return None if calls["n"] == 1 else real(proc, *args)

        with (
            testutils.mock_uid(),
            testutils.MockProcfs(),
            mock.patch.object(discover, "_collect_proc_basics", skip_first),
        ):
            units = discover.collect_units(self.info, self.topo)["units"]
        self.assertEqual([u["unit_id"] for u in units], [0, 1])


class TestGlobalAffinity(unittest.TestCase):
    def setUp(self):
        self.info, topo_info = testutils.load_a2_npu_raw()
        self.topo = testutils.a2_topo(self.info, topo_info)

    def test_global_affinity_is_not_attributed_to_node(self):
        """全机亲和表示未绑核：不退回 cgroup 推断，避免误报 node 偏差。"""
        all_cpus = [c["cpu"] for c in self.topo["cpus"]]
        self.assertIsNone(discover._proc_node(all_cpus, all_cpus, self.topo))

    def test_partial_affinity_falls_back_to_cgroup(self):
        self.assertEqual(discover._proc_node([], [144, 145], self.topo), 6)


class TestNumaPages(unittest.TestCase):
    def test_unknown_npu_node_is_not_observable(self):
        """npu_node 未知时远端页占比应为 None，而非伪报 0.0。"""
        numa = {"ok": True, "reason": None, "pages": {0: 100, 6: 300}, "total": 400}
        with mock.patch.object(procfs, "read_numa_maps", lambda pid: numa):
            _, remote_ratio, limited = discover._collect_numa_pages(1, None)
        self.assertIsNone(remote_ratio)
        self.assertFalse(limited)


if __name__ == "__main__":
    unittest.main()

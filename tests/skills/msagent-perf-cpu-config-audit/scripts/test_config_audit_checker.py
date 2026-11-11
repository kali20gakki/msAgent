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
checker.py 单元测试：NPU 域双口径（H1）、拓扑双形态（H4）与解析工具函数。
"""

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
        "host-cpu",
        "msagent-perf-cpu-config-audit",
        "scripts",
    ),
)

import checker


def _facts(**items):
    return [{"item": key, "value": value} for key, value in items.items()]


DOUBLE_DIE_FACTS = _facts(
    npu_available=True,
    npu_count=8,
    npu_die_count=16,
    davinci_devices=16,
    npu_topo_fabric="hccs_sw",
    npu_topo_rows=16,
    npu_topo_form="die",
    npu_die_groups=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    npu_sio_pairs=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
)

SINGLE_DIE_FACTS = _facts(
    npu_available=True,
    npu_count=8,
    npu_die_count=8,
    davinci_devices=8,
    npu_topo_fabric="hccs",
    npu_topo_rows=8,
    npu_topo_form="npu",
    npu_die_groups=[[0], [1], [2], [3], [4], [5], [6], [7]],
)


def _run_npu(facts, expected_cards=0):
    instance = checker.Checker()
    instance.check_npu(facts, quick=False, expected_cards=expected_cards)
    return {result["id"]: result for result in instance.results}


class TestH1DualGranularity(unittest.TestCase):
    def test_fact_when_no_expected(self):
        result = _run_npu(DOUBLE_DIE_FACTS)["H1"]
        self.assertEqual(result["status"], "FACT")
        self.assertEqual(result["value"]["packages"], 8)
        self.assertEqual(result["value"]["logical_dies"], 16)
        self.assertEqual(result["value"]["davinci_devices"], 16)
        self.assertIn("8 封装", result["value_text"])
        self.assertIn("16 die", result["value_text"])

    def test_ok_when_package_matches(self):
        result = _run_npu(DOUBLE_DIE_FACTS, expected_cards=8)["H1"]
        self.assertEqual(result["status"], "OK")
        self.assertIn("逻辑 die 16", result["evidence"])

    def test_crit_when_package_mismatch(self):
        result = _run_npu(DOUBLE_DIE_FACTS, expected_cards=4)["H1"]
        self.assertEqual(result["status"], "CRIT")


class TestH4Topo(unittest.TestCase):
    def test_double_die_sio_and_equidistant(self):
        result = _run_npu(DOUBLE_DIE_FACTS)["H4"]
        self.assertEqual(result["status"], "FACT")
        self.assertEqual(result["value"]["form"], "die")
        self.assertEqual(len(result["value"]["die_groups"]), 8)
        self.assertIn("同封装 die 对 8 组", result["value_text"])
        self.assertIn("SIO 片内直连", result["value_text"])
        self.assertIn("等距", result["evidence"])
        self.assertIn("同封装 die 对不可跨 node 拆分", result["evidence"])

    def test_single_die_says_one_die_per_package(self):
        result = _run_npu(SINGLE_DIE_FACTS)["H4"]
        self.assertIn("每封装 1 die（单 die 机型）", result["value_text"])
        self.assertNotIn("die 对", result["value_text"])
        self.assertNotIn("等距", result["evidence"])

    def test_no_topo_facts(self):
        self.assertNotIn("H4", _run_npu(_facts(npu_available=False)))


class TestCgroupQuota(unittest.TestCase):
    """cpu.max 解析：正常配额判型、max 不判型、畸形值降级为 FACT（不静默丢弃）。"""

    def _run(self, cpu_max, requested_cpus=0):
        instance = checker.Checker()
        instance.check_cgroup(_facts(cpu_max=cpu_max), quick=False, requested_cpus=requested_cpus)
        return {result["id"]: result for result in instance.results}

    def test_quota_below_requested_warns(self):
        result = self._run("400000 100000", requested_cpus=8)["F1"]
        self.assertEqual(result["status"], "WARN")
        self.assertIn("quota=4", result["evidence"])

    def test_quota_enough_is_ok(self):
        result = self._run("1600000 100000", requested_cpus=8)["F1"]
        self.assertEqual(result["status"], "OK")

    def test_unlimited_is_not_reported(self):
        self.assertNotIn("F1", self._run("max 100000"))

    def test_malformed_value_becomes_fact(self):
        result = self._run("abc 100000")["F1"]
        self.assertEqual(result["status"], "FACT")
        self.assertIn("不可解析", result["evidence"])


class TestB2Naming(unittest.TestCase):
    """B2 的三个分支必须使用同一检查项名称（name 位不能写成数据源路径）。"""

    def test_no_expected_value_uses_check_item_name(self):
        instance = checker.Checker()
        instance.check_bind(_facts(cpuset_cpus_effective="0-63"), "0-63", quick=False)
        result = {item["id"]: item for item in instance.results}["B2"]
        self.assertEqual(result["name"], "cpuset 截断")
        self.assertEqual(result["source"], "cpuset.cpus.effective")


class TestCpuSetHelpers(unittest.TestCase):
    def test_parse_cpu_set(self):
        self.assertEqual(checker._parse_cpu_set("0-3,8"), [0, 1, 2, 3, 8])
        self.assertEqual(checker._parse_cpu_set("5"), [5])
        self.assertEqual(checker._parse_cpu_set(""), [])

    def test_extract_isolcpus(self):
        cmdline = "BOOT_IMAGE=vmlinuz isolcpus=8-15 nohz_full=16"
        self.assertEqual(checker._extract_isolcpus(cmdline), "8-15")
        self.assertEqual(checker._extract_isolcpus("quiet"), "")


def _by_id(instance):
    return {result["id"]: result for result in instance.results}


class TestA1NodeMemDistribution(unittest.TestCase):
    def test_free_only_not_summed_with_total(self):
        """余量按 MemFree 比较：total 与 free 相加会掩盖真实分布差。"""
        instance = checker.Checker()
        instance.check_numa(
            _facts(
                node0_mem_total_kb=1000,
                node0_mem_free_kb=100,
                node1_mem_total_kb=1000,
                node1_mem_free_kb=200,
            ),
            quick=False,
        )
        self.assertIn("50.0%", _by_id(instance)["A1"]["value"])


class TestC1HugepagesFloor(unittest.TestCase):
    def test_threshold_override_consumed(self):
        instance = checker.Checker(thresholds={"C1_hugepages_zero": 10})
        instance.check_memory(_facts(hugepages_total=5, hugepages_free=5), quick=False)
        self.assertEqual(_by_id(instance)["C1"]["status"], "WARN")


class TestB2ExpectedAffinity(unittest.TestCase):
    def test_unparseable_expected_is_fact(self):
        instance = checker.Checker()
        instance.check_bind(_facts(cpuset_cpus_effective="0-7"), "0-7", False, "abc")
        result = _by_id(instance)["B2"]
        self.assertEqual(result["status"], "FACT")
        self.assertIn("不可解析", result["evidence"])

    def test_subset_is_ok(self):
        instance = checker.Checker()
        instance.check_bind(_facts(cpuset_cpus_effective="0-7"), "0-7", False, "0-3")
        self.assertEqual(_by_id(instance)["B2"]["status"], "OK")


class TestF1QuotaPeriod(unittest.TestCase):
    def test_period_from_cpu_max(self):
        """quota 4 核（800000/200000）< 申请 8 核时按实际 period 判 WARN。"""
        instance = checker.Checker()
        instance.check_cgroup(_facts(cpu_max="800000 200000"), quick=False, requested_cpus=8)
        result = _by_id(instance)["F1"]
        self.assertEqual(result["status"], "WARN")
        self.assertIn("quota=4", result["evidence"])


class TestG1OmpLimit(unittest.TestCase):
    def test_numeric_limit_override(self):
        warn = checker.Checker(thresholds={"G1_omp_threads": 64})
        warn.check_env(_facts(pid_environ_picked={"OMP_NUM_THREADS": "96"}), quick=False, num_cpus=128)
        self.assertEqual(_by_id(warn)["G1"]["status"], "WARN")

        ok = checker.Checker(thresholds={"G1_omp_threads": 128})
        ok.check_env(_facts(pid_environ_picked={"OMP_NUM_THREADS": "96"}), quick=False, num_cpus=128)
        self.assertEqual(_by_id(ok)["G1"]["status"], "OK")


class TestQuickFilter(unittest.TestCase):
    def test_quick_keeps_quick_items_only(self):
        instance = checker.Checker()
        instance.check_npu(DOUBLE_DIE_FACTS, quick=True)
        instance.check_freq(DOUBLE_DIE_FACTS, quick=True)
        results = instance.run(DOUBLE_DIE_FACTS, quick=True, online_cpus="0-639")
        ids = {result["id"] for result in results}
        self.assertTrue(ids <= set(checker.QUICK_ITEMS))
        self.assertIn("H1", ids)


if __name__ == "__main__":
    unittest.main()

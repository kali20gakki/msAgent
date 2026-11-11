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
rules_locality.py 单元测试：R001 随 fabric 联动（topo_affinity / global_slice）
及 R002/R005/R007/R010 命中条件。
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

import rules_locality


def _facts(**items):
    return [{"item": key, "value": value} for key, value in items.items()]


def _results(**items):
    return [{"id": key, "status": value[0], "value": value[1], "name": key} for key, value in items.items()]


class TestSetHelpers(unittest.TestCase):
    def test_parse_set(self):
        self.assertEqual(rules_locality._parse_set("0-2,5"), {0, 1, 2, 5})
        self.assertEqual(rules_locality._parse_set(""), set())

    def test_fmt_set(self):
        self.assertEqual(rules_locality._fmt_set([0, 1, 2, 5]), "0-2,5")
        self.assertEqual(rules_locality._fmt_set([]), "")
        self.assertEqual(rules_locality._fmt_set([7]), "7")


class TestBalancedSlice(unittest.TestCase):
    def test_slice_per_unit(self):
        facts = {"node0_cpus": "0-79", "node1_cpus": "80-159"}
        groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        # 8 unit / 2 node = 每 node 4 unit，每 unit 20 核
        self.assertEqual(rules_locality._balanced_slice(facts, groups), "0-19")

    def test_no_nodes(self):
        self.assertEqual(rules_locality._balanced_slice({}, []), "")


class TestR001(unittest.TestCase):
    def _run(self, results, facts, target=None):
        rules = rules_locality.LocalityRules(results, facts, [target] if target else [])
        rules.rule_r001("0-191")
        return rules.findings

    def test_hit_with_affinity(self):
        results = _results(B1=("WARN", "0-191"))
        facts = _facts(
            npu_topo_fabric="hccs", npu_affinity_available=True, node0_cpus="0-23", npu_die_groups=[[0], [1]]
        )
        findings = self._run(results, facts)
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["rule"], "R001")
        self.assertEqual(finding["locality_mode"], "topo_affinity")
        self.assertEqual(finding["confidence"], 0.9)
        self.assertEqual(finding["suggested_cpuset"], "0-23")

    def test_hit_without_affinity_uses_balanced_slice(self):
        results = _results(B1=("WARN", "0-639"))
        facts = _facts(
            npu_topo_fabric="hccs_sw",
            npu_affinity_available=False,
            node0_cpus="0-79",
            node1_cpus="80-159",
            npu_die_groups=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
        )
        rules = rules_locality.LocalityRules(results, facts, [])
        rules.rule_r001("0-639")
        finding = rules.findings[0]
        self.assertEqual(finding["locality_mode"], "global_slice")
        self.assertEqual(finding["confidence"], 0.5)
        self.assertEqual(finding["suggested_cpuset"], "0-19")
        self.assertIn("hccs_sw", finding["impact"])
        self.assertTrue(any("die 组" in item for item in finding["evidence"]))

    def test_target_pid_in_evidence(self):
        results = _results(B1=("WARN", "0-191"))
        facts = _facts(npu_affinity_available=True, node0_cpus="0-23")
        findings = self._run(results, facts, {"pid": 123, "comm": "torchrun"})
        self.assertIn("pid=123", findings[0]["evidence"][0])

    def test_no_hit_when_not_fully_unbound(self):
        results = _results(B1=("WARN", "0-95"))
        self.assertEqual(self._run(results, _facts()), [])

    def test_no_hit_when_b1_ok(self):
        results = _results(B1=("OK", "0-191"))
        self.assertEqual(self._run(results, _facts()), [])


class TestOtherRules(unittest.TestCase):
    def test_r002(self):
        rules = rules_locality.LocalityRules(_results(B2=("CRIT", "0-63")), [], [])
        rules.rule_r002()
        self.assertEqual(rules.findings[0]["rule"], "R002")
        self.assertEqual(rules.findings[0]["suggested_cpuset"], "0-63")

    def test_r005(self):
        rules = rules_locality.LocalityRules(_results(B3=("WARN", "0-31")), [], [])
        rules.rule_r005()
        self.assertEqual(rules.findings[0]["rule"], "R005")

    def test_r007(self):
        rules = rules_locality.LocalityRules(_results(E1=("WARN", "cpu0 62%")), [], [])
        rules.rule_r007()
        self.assertEqual(rules.findings[0]["rule"], "R007")

    def test_r010_process_inside_isolated(self):
        """业务核完整落在隔离核内 = isolcpus 的期望状态，不报告。"""
        results = _results(B5=("FACT", "8-15"))
        facts = _facts(pid_affinity="8-15")
        rules = rules_locality.LocalityRules(results, facts, [])
        rules.rule_r010()
        self.assertEqual(rules.findings, [])

    def test_r010_process_outside_isolated(self):
        """隔离核已配置但业务完全未使用 → 隔离核闲置。"""
        results = _results(B5=("FACT", "8-15"))
        facts = _facts(pid_affinity="0-7")
        rules = rules_locality.LocalityRules(results, facts, [])
        rules.rule_r010()
        self.assertEqual(rules.findings[0]["rule"], "R010")
        self.assertIn("隔离核未使用", rules.findings[0]["problem"])

    def test_r010_partial_overlap_reported(self):
        """业务核部分落在隔离核内 → 未完全绑入（部分越界）。"""
        results = _results(B5=("FACT", "8-15"))
        facts = _facts(pid_affinity="0-15")
        rules = rules_locality.LocalityRules(results, facts, [])
        rules.rule_r010()
        self.assertEqual(rules.findings[0]["rule"], "R010")
        self.assertIn("部分越界", rules.findings[0]["problem"])

    def test_r010_unparsable_isolated_ignored(self):
        """隔离核值不可解析时不判定，避免把空核集当成「隔离核闲置」。"""
        results = _results(B5=("FACT", "N/A"))
        facts = _facts(pid_affinity="0-7")
        rules = rules_locality.LocalityRules(results, facts, [])
        rules.rule_r010()
        self.assertEqual(rules.findings, [])

    def test_r010_without_pid_affinity_ignored(self):
        """无事中进程亲和（事前模式）时不判定。"""
        results = _results(B5=("FACT", "8-15"))
        rules = rules_locality.LocalityRules(results, _facts(), [])
        rules.rule_r010()
        self.assertEqual(rules.findings, [])

    def test_run_collects_all(self):
        results = _results(
            B1=("WARN", "0-191"),
            B2=("CRIT", "0-63"),
            B3=("WARN", "0-31"),
            E1=("WARN", "cpu0 62%"),
        )
        facts = _facts(npu_affinity_available=True, node0_cpus="0-23")
        rules = rules_locality.LocalityRules(results, facts, [])
        ids = [finding["rule"] for finding in rules.run("0-191")]
        self.assertEqual(ids, ["R001", "R002", "R005", "R007"])


if __name__ == "__main__":
    unittest.main()

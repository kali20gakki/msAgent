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

"""plan.py：摘要字段、参数校验与预留核数越界处理。"""

import unittest
from unittest import mock

import testutils  # noqa: F401
import plan


class TestSummary(unittest.TestCase):
    def _summary(self, unit):
        paths = {"out_dir": "/tmp", "plan_path": "p.json", "report_path": "r.html", "verify_path": "v.sh"}
        return plan._build_summary(plan._parse_args(["--dry-run"]), "in_run", {}, [unit], paths)

    def test_unit_target_node_is_reported(self):
        """摘要必须输出模型层的 target_node，而不是恒为空的 numa_node。"""
        summary = self._summary({"unit_id": 0, "target_node": 6, "roles": {}, "deviations": []})
        self.assertEqual(summary["units"][0]["target_node"], 6)

    def test_unit_deviation_kinds(self):
        summary = self._summary(
            {
                "unit_id": 0,
                "target_node": 6,
                "roles": {},
                "deviations": [{"kind": "multi_npu_cross_node"}],
            }
        )
        self.assertEqual(summary["units"][0]["deviations"], ["multi_npu_cross_node"])


class TestArgs(unittest.TestCase):
    def test_negative_reserve_rejected(self):
        with self.assertRaises(SystemExit):
            plan._parse_args(["--irq-reserve", "-1"])

    def test_removed_plan_only_flag_rejected(self):
        """--plan-only 无实际语义，已删除，不再接受该参数。"""
        with self.assertRaises(SystemExit):
            plan._parse_args(["--plan-only"])

    def test_reserve_not_less_than_node_cpus_errors(self):
        topo = {"nodes": [{"node": 0, "cpus": list(range(4))}], "meta": {}}
        args = plan._parse_args(["--dry-run", "--irq-reserve", "3", "--interference-reserve", "1"])
        with (
            mock.patch.object(plan.preflight, "collect", return_value={}),
            mock.patch.object(plan.npu_query, "query", return_value={}),
            mock.patch.object(plan.preflight, "detect_form", return_value=("pre_run", None)),
            mock.patch.object(plan.topology, "collect", return_value=topo),
        ):
            result, _, _ = plan.run(args)
        self.assertEqual(result["status"], "error")
        self.assertIn("预留核数", result["error"])


if __name__ == "__main__":
    unittest.main()

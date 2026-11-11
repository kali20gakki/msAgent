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

"""emit_plan.py：plan 结构、命令契约（不含 pid）、验证脚本落盘。"""

import os
import unittest
from unittest import mock

import testutils
import emit_plan


class TestInRunPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.plan_path, cls.verify_path, cls.report_path, cls.topo = testutils.build_a2_plan()

    def test_basic_shape(self):
        plan = self.plan
        self.assertEqual(plan["form"], "in_run")
        self.assertEqual(plan["locality_mode"], "topo_affinity")
        self.assertEqual(len(plan["units"]), 3)
        self.assertEqual(plan["summary"]["unit_count"], 3)
        self.assertEqual(plan["summary"]["execution_mode"], "delegated")
        self.assertFalse(plan["executable"])

    def test_commands_reference_plan_not_pid(self):
        unit = self.plan["units"][0]
        command = unit["commands"][0]
        self.assertIn("apply_plan.py", command)
        self.assertIn("--roles comm,io", command)
        self.assertIn(self.plan_path, command)
        self.assertNotIn("pid", command.lower())
        self.assertIn("--rollback", unit["rollback"][0])
        apply_cmd = self.plan["apply"]["command"]
        self.assertIn("apply_plan.py", apply_cmd)
        self.assertIn("--roles comm,io", apply_cmd)
        self.assertNotIn("pid", apply_cmd.lower())
        self.assertTrue(self.plan["apply"]["dynamic_task_scan"])

    def test_verify_script_path_and_name(self):
        self.assertTrue(os.path.isabs(self.plan["verify_script_path"]))
        self.assertEqual(self.plan["verify_script"], os.path.basename(self.plan["verify_script_path"]))
        self.assertTrue(self.plan["verify_script_path"].endswith(".sh"))
        self.assertTrue(self.plan["report"].endswith(".html"))

    def test_unit_fields(self):
        unit = self.plan["units"][0]
        self.assertEqual(unit["pid"], 293354)
        self.assertEqual(unit["primary_npu"], 0)
        self.assertEqual(unit["numa_node"], 6)
        self.assertEqual(unit["cpuset"], "144-154")
        self.assertEqual(unit["main_cpuset"], "154")
        self.assertEqual(unit["mechanism"], "sched_setaffinity")
        self.assertEqual(unit["command_source"], "runtime_task_scan")
        self.assertEqual([b["role"] for b in unit["thread_binding"]], ["main", "comm", "io", "compute"])
        self.assertTrue(unit["requires_privilege"])
        self.assertEqual(unit["env"]["OMP_PROC_BIND"], "close")

    def test_privilege_block(self):
        priv = self.plan["privilege"]
        self.assertEqual(priv["execution_mode"], "delegated")
        self.assertTrue(priv["requires_privilege"])
        self.assertEqual(priv["target_uids"], [testutils.OTHER_UID])
        self.assertFalse(priv["can_read_numa_maps"])

    def test_root_can_execute_cross_user_targets(self):
        """root 具备 CAP_SYS_NICE：跨用户目标也应给出可执行结论。"""
        plan = testutils.build_a2_plan(preflight_overrides={"is_root": True})[0]
        priv = plan["privilege"]
        self.assertEqual(priv["execution_mode"], "self")
        self.assertTrue(priv["can_taskset"])
        self.assertFalse(priv["requires_privilege"])
        self.assertTrue(plan["executable"])

    def test_topology_meta_wins_over_legacy_preflight(self):
        """locality/fabric/granularity 以 topo["meta"] 为准，preflight 仅作兼容兜底。"""
        plan = testutils.build_a2_plan(
            preflight_overrides={
                "fabric": "pcie",
                "locality_mode": "global_slice",
                "affinity_granularity": "per_npu",
            }
        )[0]
        self.assertEqual(plan["fabric"], "hccs")
        self.assertEqual(plan["locality_mode"], "topo_affinity")
        self.assertEqual(plan["affinity_granularity"], "group")

    def test_plan_flags_merges_view_limited(self):
        self.assertTrue(
            emit_plan._plan_flags({"flags": {"view_limited": False}}, {"view_limited": True})["view_limited"]
        )
        self.assertTrue(emit_plan._plan_flags({"flags": {"view_limited": True}}, {})["view_limited"])

    def test_apply_block_bind_compute_matches_units(self):
        """执行块的 bind_compute 与 unit compute=exec 的绑定语义一致。"""
        self.assertTrue(self.plan["apply"]["bind_compute"])

    def test_findings_are_consumed(self):
        """model.findings 必须写入 plan，不能静默丢弃。"""
        kinds = [f.get("kind") for f in self.plan["findings"]]
        self.assertIn("shared_npu_node", kinds)
        self.assertEqual(self.plan["summary"]["finding_count"], len(self.plan["findings"]))

    def test_plan_id_includes_pid(self):
        self.assertTrue(self.plan["plan_id"].endswith("-%d" % os.getpid()))

    def test_expected_effect_and_notes(self):
        self.assertEqual(self.plan["expected_effect"]["level"], "high")
        basis = self.plan["expected_effect"]["basis"]
        self.assertTrue(basis)
        self.assertTrue(all(basis))
        self.assertLessEqual(len(self.plan["notes"]), 40)

    def test_chmod_failure_is_logged(self):
        with mock.patch("os.chmod", side_effect=OSError("denied")):
            with self.assertLogs("emit_plan", level="WARNING"):
                emit_plan.write_verify_script(self.plan, self.verify_path)

    def test_write_outputs(self):
        emit_plan.write_plan(self.plan, self.plan_path)
        emit_plan.write_verify_script(self.plan, self.verify_path)
        self.assertTrue(os.path.isfile(self.plan_path))
        with open(self.verify_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("verify_config.py", text)
        self.assertIn("--threads", text)
        self.assertIn(self.plan["plan_file"], text)


class TestPreRunPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = testutils.build_a2_plan(form="pre_run")[0]

    def test_no_commands_and_env_only(self):
        for unit in self.plan["units"]:
            self.assertEqual(unit["commands"], [])
            self.assertEqual(unit["rollback"], [])
            self.assertEqual(unit["mechanism"], "launch_wrapper")
            self.assertEqual(unit["command_source"], "env_only")
            self.assertIn("OMP_NUM_THREADS", unit["env"])

    def test_form_and_executable(self):
        self.assertEqual(self.plan["form"], "pre_run")
        self.assertFalse(self.plan["executable"])


if __name__ == "__main__":
    unittest.main()

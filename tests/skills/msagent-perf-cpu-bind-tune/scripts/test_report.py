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

"""report.py：报告结构、结论合并、偏差分组、runbook 与转义。"""

import unittest

import testutils
import report


def _unit(uid, affinity, node, target, cpuset="10-19"):
    return {
        "unit_id": uid,
        "pid": 100 + uid,
        "pids": [100 + uid],
        "process_name": "VLLM::Worker",
        "framework": "vllm",
        "primary_npu": uid,
        "numa_node": target,
        "current_node": node,
        "current_affinity": affinity,
        "cpuset": cpuset,
        "main_cpuset": cpuset.split("-")[-1],
        "thread_binding": [],
        "deviations": [],
        "capacity_gap": 0,
        "env": {},
    }


class TestUnitState(unittest.TestCase):
    def test_states(self):
        self.assertEqual(report._unit_state(_unit(0, "0-191", 0, 6), 192), "未绑核")
        self.assertEqual(report._unit_state(_unit(1, "144-165", 7, 6), 192), "跨 node")
        self.assertEqual(report._unit_state(_unit(2, "144-165", 6, 6), 192), "已绑核")
        self.assertEqual(report._unit_state(_unit(3, None, None, 6), 192), "未观测")

    def test_pending_when_no_pid(self):
        unit = _unit(0, "0-191", 0, 6)
        unit["pid"] = None
        unit["pids"] = []
        self.assertEqual(report._unit_state(unit, 192), "待启动")


class TestConclusions(unittest.TestCase):
    def setUp(self):
        self.topo = testutils.a2_topo()

    def test_merged_by_state(self):
        plan = {
            "locality_mode": "topo_affinity",
            "units": [
                _unit(0, "0-191", 0, 6),
                _unit(1, "0-191", 0, 4),
                _unit(2, "144-165", 7, 6),
                _unit(3, "144-165", 6, 6),
            ],
            "summary": {"unknown_hbm_npus": [{"npu": 1, "hbm_used_mb": 3584}]},
        }
        items = report._conclusions(plan, self.topo)
        text = "\n".join(items)
        self.assertIn("2 个单元未绑核", text)
        self.assertIn("1 个单元已绑核但跨 node", text)
        self.assertIn("1 个单元已绑核且落在 NPU 所在 node", text)
        self.assertIn("NPU1", text)
        # 合并后条数应为个位数，避免逐 unit 罗列
        self.assertLessEqual(len(items), 6)

    def test_global_slice_lead(self):
        topo = testutils.a3_topo()
        unit = _unit(0, None, None, 0)
        unit["pid"] = None
        unit["pids"] = []
        plan = {"locality_mode": "global_slice", "units": [unit], "summary": {}}
        self.assertIn("等距", report._conclusions(plan, topo)[0])

    def test_unobserved_unit_does_not_look_successful(self):
        """受限场景（亲和不可观测）不能落成「未发现需要调整的绑核问题」。"""
        plan = {"locality_mode": "topo_affinity", "units": [_unit(0, None, None, 6)], "summary": {}}
        text = "\n".join(report._conclusions(plan, self.topo))
        self.assertIn("不可观测", text)
        self.assertNotIn("未发现需要调整的绑核问题", text)

    def test_no_units(self):
        self.assertEqual(report._conclusions({"units": []}, self.topo), ["未发现运行中的 NPU 进程，无法给出绑核建议。"])

    def test_conclusion_text_is_escaped_except_bold(self):
        """结论里的动态值必须转义，只保留结论自带的 <b> 强调标签。"""
        text = report._e_conclusion("unit <script>alert(1)</script> <b>2 个单元未绑核</b>")
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("<script>", text)
        self.assertIn("<b>2 个单元未绑核</b>", text)

    def test_legacy_mode_alias(self):
        self.assertEqual(report._mode("balanced_partition"), "global_slice")
        self.assertEqual(report._mode("npu_local"), "topo_affinity")


class TestDevGroups(unittest.TestCase):
    def test_group_by_kind_severity_handling(self):
        units = []
        for uid in range(4):
            units.append(
                {
                    "unit_id": uid,
                    "deviations": [
                        {
                            "kind": "affinity_outside_npu_range",
                            "severity": "high",
                            "handling": "solve",
                            "evidence": "e%d" % uid,
                        },
                        {
                            "kind": "numa_maps_unavailable",
                            "severity": "info",
                            "handling": "view_limited",
                            "evidence": "n",
                        },
                    ],
                }
            )
        groups = report._dev_groups({"units": units})
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["kind"], "affinity_outside_npu_range")
        self.assertEqual(groups[0]["units"], [0, 1, 2, 3])


class TestRunbook(unittest.TestCase):
    def test_in_run_four_steps_without_pid(self):
        plan, plan_path, verify_path, _, _ = testutils.build_a2_plan()
        text = report._runbook(plan)
        self.assertIn("1) 预演", text)
        self.assertIn("--dry-run", text)
        self.assertIn("2) 执行", text)
        self.assertIn("3) 验证", text)
        self.assertIn("4) 回滚", text)
        self.assertIn(verify_path, text)
        self.assertIn("--rollback", text)
        self.assertNotIn("pid", text.lower())
        self.assertEqual(text.count("apply_plan.py"), 3)  # 预演 + 执行 + 回滚

    def test_pre_run_env_exports(self):
        plan = testutils.build_a2_plan(form="pre_run")[0]
        text = report._runbook(plan)
        self.assertIn("OMP_NUM_THREADS", text)
        self.assertIn("OMP_PROC_BIND=close", text)
        self.assertNotIn("apply_plan.py", text)
        self.assertEqual(text.count("OMP_NUM_THREADS"), 3)


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, _, _, _, cls.topo = testutils.build_a2_plan()
        cls.html = report.render(cls.plan, cls.topo)

    def test_sections(self):
        for title in ("结论", "NUMA 拓扑与设备亲和", "进程 → 目标核分配", "偏差处理", "执行与验证", "更多明细"):
            self.assertIn(title, self.html)
        self.assertEqual(self.html.count("<h2"), 5)
        self.assertGreaterEqual(self.html.count("<details"), 4)

    def test_conclusion_card_and_kpis(self):
        self.assertIn('class="concl"', self.html)
        # A2 fixture：unit1 已绑核但 node7→node6，其余已绑核
        self.assertIn("个单元已绑核但跨 node", self.html)
        self.assertIn("个单元已绑核且落在 NPU 所在 node", self.html)
        self.assertGreaterEqual(self.html.count('class="kpi'), 4)

    def test_status_and_handling_labels(self):
        self.assertIn("未绑核", self.html)
        self.assertIn("纳入方案", self.html)
        self.assertIn("转出", self.html)
        # 原始英文 kind 只作为 title 提示保留
        self.assertIn('title="affinity_outside_npu_range"', self.html)

    def test_no_developer_phrases(self):
        for phrase in ("命令均不含 pid", "全 PASS", "unbound", "分析视角", "期望来源"):
            self.assertNotIn(phrase, self.html)

    def test_escaping_of_process_name(self):
        plan = dict(self.plan)
        plan["units"] = [dict(u) for u in self.plan["units"]]
        plan["units"][0]["process_name"] = "<img src=x onerror=alert(1)>"
        html = report.render(plan, self.topo)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;img src=x", html)


if __name__ == "__main__":
    unittest.main()

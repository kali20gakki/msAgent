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

"""apply_plan.py：dry-run 无副作用、属主校验、线程级绑定与回滚。"""

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import testutils
import apply_plan

PID = 293354

UNIT = {
    "unit_id": 0,
    "pid": PID,
    "cpuset": "144-154",
    "main_cpuset": "154",
    "thread_binding": [
        {"role": "main", "cpuset": "154", "apply": "exec", "match_comm": []},
        {"role": "comm", "cpuset": "151-152", "apply": "exec", "match_comm": ["hccl", "aoe_rtkb"]},
        {"role": "io", "cpuset": "153", "apply": "exec", "match_comm": ["zmq"]},
        {"role": "compute", "cpuset": "144-150", "apply": "inherit", "match_comm": ["vllm"]},
    ],
    "roles": {
        "main": {"cpus": "154", "cores": 1},
        "comm": {"cpus": "151-152", "cores": 2},
        "io": {"cpus": "153", "cores": 1},
        "compute": {"cpus": "144-150", "cores": 7},
    },
    "before": {"Cpus_allowed_list": "146-165"},
}


class ApplyTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = testutils.temp_out()
        self.calls = []
        # 全平台统一装记录型 shim：Linux 上也不能真改测试进程/目标进程亲和
        self._saved_setaffinity = getattr(os, "sched_setaffinity", None)
        os.sched_setaffinity = lambda tid, cpus: self.calls.append((tid, tuple(cpus)))

    def tearDown(self):
        if self._saved_setaffinity is None:
            del os.sched_setaffinity
        else:
            os.sched_setaffinity = self._saved_setaffinity

    def write_plan(self, units=None):
        path = os.path.join(self.tmp, "plan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"units": units if units is not None else [UNIT]}, f)
        return path


class TestDryRun(ApplyTestBase):
    def test_dry_run_plans_without_side_effects(self):
        with testutils.MockProcfs(uid=1000):
            rc = apply_plan.main([self.write_plan(), "--roles", "comm,io", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(apply_plan._binding(UNIT)["comm"]["cpuset"], "151-152")

    def test_dry_run_unit_result(self):
        with testutils.MockProcfs(uid=1000):
            result = apply_plan.main([self.write_plan(), "--roles", "comm,io", "--dry-run"])
        self.assertEqual(result, 0)

    def test_dry_run_reports_pending_threads(self):
        """dry-run 也要报告待绑核线程数，作为执行前影响面评估。"""
        with testutils.MockProcfs(uid=1000):
            buf = StringIO()
            with redirect_stdout(buf):
                rc = apply_plan.main([self.write_plan(), "--roles", "comm,io", "--dry-run"])
        self.assertEqual(rc, 0)
        roles = json.loads(buf.getvalue())["results"][0]["roles"]
        self.assertEqual(roles["comm"]["threads"], 24)
        self.assertEqual(roles["io"]["threads"], 4)
        self.assertEqual(self.calls, [])


class TestOwnerCheck(ApplyTestBase):
    def test_unit_without_pid_fails(self):
        """pre_run 计划无 pid：必须显式失败，不能整体 ok=true。"""
        unit = dict(UNIT, pid=None)
        with testutils.MockProcfs(uid=1000):
            buf = StringIO()
            with redirect_stdout(buf):
                rc = apply_plan.main([self.write_plan([unit]), "--dry-run"])
        self.assertEqual(rc, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("无 pid", payload["results"][0]["error"])

    def test_cross_user_refused(self):
        """目标 uid 与当前用户不一致且非 root：拒绝执行并标记需提权。

        固定当前 uid 为非 0：CI 可能以 root 跑测，root 会跳过属主校验。
        """
        with testutils.mock_uid(1000), testutils.MockProcfs(uid=9999):
            rc = apply_plan.main([self.write_plan(), "--roles", "comm,io"])
            refusal = apply_plan._owner_refusal(PID, False)
        self.assertEqual(rc, 1)
        self.assertEqual(self.calls, [])
        self.assertIn("delegated", refusal)

    def test_root_allowed(self):
        with testutils.mock_uid(0), testutils.MockProcfs(uid=9999):
            self.assertIsNone(apply_plan._owner_refusal(PID, False))

    def test_missing_pid_reported(self):
        unit = dict(UNIT, pid=999999)
        with testutils.MockProcfs(uid=1000):
            rc = apply_plan.main([self.write_plan([unit]), "--dry-run"])
        self.assertEqual(rc, 1)

    def test_missing_required_field_is_structured_error(self):
        """unit 缺必需字段：结构化报错，不允许 KeyError/AttributeError。"""
        unit = {k: v for k, v in UNIT.items() if k != "thread_binding"}
        with testutils.MockProcfs(uid=1000):
            buf = StringIO()
            with redirect_stdout(buf):
                rc = apply_plan.main([self.write_plan([unit]), "--dry-run"])
        self.assertEqual(rc, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("thread_binding", payload["results"][0]["error"])


class TestApplyRoles(ApplyTestBase):
    def test_thread_roles_applied_by_comm(self):
        with testutils.MockProcfs(uid=1000):
            item = apply_plan._plan_unit(UNIT, {"comm", "io"}, False, dry_run=True)
        self.assertTrue(item["ok"])
        self.assertEqual(item["main"]["cpuset"], "154")
        self.assertEqual(item["roles"]["comm"]["threads"], 24)  # hccl 8 + aoe_rtkb 16
        self.assertEqual(item["roles"]["comm"]["cpuset"], "151-152")
        self.assertEqual(item["roles"]["io"]["threads"], 4)  # zmq 2 + 2
        self.assertEqual(item["roles"]["compute"]["apply"], "inherit")

    def test_bind_compute_enabled(self):
        with testutils.MockProcfs(uid=1000):
            item = apply_plan._plan_unit(UNIT, {"comm", "io"}, True, dry_run=True)
        self.assertEqual(item["roles"]["compute"]["apply"], "exec")
        self.assertEqual(item["roles"]["compute"]["threads"], 47)

    def test_inherit_compute_widens_main_to_work_cpuset(self):
        """旧计划 compute=inherit：主线程绑 work_cpuset，避免 OMP 线程挤在 1 核。"""
        unit = dict(UNIT, work_cpuset="144-150,154")
        with testutils.MockProcfs(uid=1000):
            item = apply_plan._plan_unit(unit, {"comm", "io"}, False, dry_run=True)
        self.assertEqual(item["main"]["cpuset"], "144-150,154")
        self.assertEqual(item["roles"]["compute"]["apply"], "inherit")

    def test_bind_compute_keeps_main_cpuset(self):
        """显式 --bind-compute 后主线程仍只绑 main_cpuset。"""
        unit = dict(UNIT, work_cpuset="144-150,154")
        with testutils.MockProcfs(uid=1000):
            item = apply_plan._plan_unit(unit, {"comm", "io"}, True, dry_run=True)
        self.assertEqual(item["main"]["cpuset"], "154")

    def test_cgroup_clip(self):
        target, clipped = apply_plan._clip({0, 1, 2, 100}, "0-95")
        self.assertEqual(target, {0, 1, 2})
        self.assertTrue(clipped)

    def test_partial_failure_rolls_back_applied_threads(self):
        """绑核中途失败：已成功的线程恢复原亲和，并报告失败 tid。"""
        calls = []

        def fake_set_affinity(tid, cpus):
            calls.append((tid, tuple(sorted(cpus))))
            if len(calls) == 3:  # 第 1 次是主线程，第 3 次是第 2 个 comm 线程
                raise OSError("boom")

        with (
            testutils.MockProcfs(uid=1000),
            mock.patch.object(apply_plan, "_set_affinity", side_effect=fake_set_affinity),
            mock.patch.object(apply_plan.procfs, "read_status", lambda tid: {"Cpus_allowed_list": "146-165"}),
        ):
            with self.assertRaises(OSError) as ctx:
                apply_plan._plan_unit(UNIT, {"comm"}, False, dry_run=False)
        message = str(ctx.exception)
        self.assertIn("已回滚 1 个线程", message)
        self.assertIn("tid=", message)
        # 主线程 + 2 个 comm 线程（第 2 个失败）+ 回滚第 1 个 comm 线程
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[1][0], calls[3][0])
        self.assertEqual(calls[3][1], tuple(range(146, 166)))


class TestRollback(ApplyTestBase):
    def test_rollback_restores_before(self):
        with testutils.MockProcfs(uid=1000):
            item = apply_plan._plan_rollback(UNIT, dry_run=False)
        self.assertTrue(item["ok"])
        self.assertEqual(item["cpuset"], "146-165")
        self.assertEqual(len(self.calls), 1 + item["threads"])
        self.assertTrue(all(cpus == tuple(range(146, 166)) for _, cpus in self.calls))

    def test_rollback_without_before_fails(self):
        unit = dict(UNIT, before=None)
        with testutils.MockProcfs(uid=1000):
            item = apply_plan._plan_rollback(unit, dry_run=True)
        self.assertFalse(item["ok"])
        self.assertIn("before", item["error"])


class TestPlanValidation(ApplyTestBase):
    """计划文件读取与结构校验：失败必须是结构化 JSON + 退出码 2，不允许 traceback。"""

    def test_missing_plan_file(self):
        rc = apply_plan.main([os.path.join(self.tmp, "absent.json"), "--dry-run"])
        self.assertEqual(rc, 2)

    def test_broken_json(self):
        path = os.path.join(self.tmp, "broken.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(apply_plan.main([path, "--dry-run"]), 2)

    def test_units_not_a_list(self):
        path = os.path.join(self.tmp, "plan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"units": {"pid": PID}}, f)
        self.assertEqual(apply_plan.main([path, "--dry-run"]), 2)

    def test_non_dict_unit_skipped_with_error(self):
        with testutils.MockProcfs(uid=1000):
            rc = apply_plan.main([self.write_plan([UNIT, "oops"]), "--dry-run"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()

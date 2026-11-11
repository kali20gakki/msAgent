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

"""verify_config.py：配置生效校验（主线程 / 逐线程 / cgroup 裁剪）。"""

import json
import os
import unittest
from unittest import mock

import testutils
import procfs
import verify_config

PID = 5001
COMM_TIDS = {5002: "hccl_watchdog_t", 5003: "hccl_watchdog_t"}
IO_TIDS = {5004: "ZMQbg/IO/0"}

PLAN_UNIT = {
    "unit_id": 0,
    "pid": PID,
    "cpuset": "144-154",
    "main_cpuset": "154",
    "thread_binding": [
        {"role": "main", "cpuset": "154", "apply": "exec", "match_comm": []},
        {"role": "comm", "cpuset": "151-152", "apply": "exec", "match_comm": ["hccl"]},
        {"role": "io", "cpuset": "153", "apply": "exec", "match_comm": ["zmq"]},
        {"role": "compute", "cpuset": "144-150", "apply": "inherit", "match_comm": ["vllm"]},
    ],
}


class TestVerifyConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = testutils.temp_out()
        self.plan_path = os.path.join(self.tmp, "plan.json")
        with open(self.plan_path, "w", encoding="utf-8") as f:
            json.dump({"units": [PLAN_UNIT]}, f)

    def _patch(self, main_affinity, tid_affinity=None):
        tid_affinity = tid_affinity or {}
        tids = [PID] + list(COMM_TIDS) + list(IO_TIDS)
        stack = [
            mock.patch.object(procfs, "read_cgroup_cpuset", lambda pid, cgroup=None: "0-191"),
            mock.patch.object(
                procfs,
                "read_status",
                lambda pid: {"Cpus_allowed_list": main_affinity if int(pid) == PID else tid_affinity.get(int(pid), "")},
            ),
            mock.patch.object(procfs, "list_tids", lambda pid: tids),
            mock.patch.object(
                procfs,
                "read_comm",
                lambda pid, tid=None: (COMM_TIDS.get(int(tid)) or IO_TIDS.get(int(tid)) or "VLLM::Worker"),
            ),
        ]
        return stack

    def test_main_match(self):
        stack = self._patch("154")
        for p in stack:
            p.start()
        try:
            self.assertEqual(verify_config.main([self.plan_path]), 0)
        finally:
            for p in stack:
                p.stop()

    def test_main_mismatch_returns_nonzero(self):
        """apply 前运行：亲和未收敛，如实报 mismatch。"""
        stack = self._patch("0-191")
        for p in stack:
            p.start()
        try:
            self.assertEqual(verify_config.main([self.plan_path]), 1)
        finally:
            for p in stack:
                p.stop()

    def test_threads_all_match(self):
        tid_affinity = {**{t: "151-152" for t in COMM_TIDS}, **{t: "153" for t in IO_TIDS}}
        stack = self._patch("154", tid_affinity)
        for p in stack:
            p.start()
        try:
            self.assertEqual(verify_config.main([self.plan_path, "--threads"]), 0)
        finally:
            for p in stack:
                p.stop()

    def test_threads_mismatch_detected(self):
        tid_affinity = {**{t: "0-191" for t in COMM_TIDS}, **{t: "153" for t in IO_TIDS}}
        stack = self._patch("154", tid_affinity)
        for p in stack:
            p.start()
        try:
            self.assertEqual(verify_config.main([self.plan_path, "--threads"]), 1)
        finally:
            for p in stack:
                p.stop()

    def test_unit_without_pid_fails(self):
        """pre_run 计划无可校验对象时必须显式失败，避免零校验即 PASS。"""
        path = os.path.join(self.tmp, "plan_nopid.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"units": [dict(PLAN_UNIT, pid=None)]}, f)
        self.assertEqual(verify_config.main([path]), 1)

    def test_empty_plan_fails(self):
        """空计划没有任何单元被校验，不能按「零失败」返回 0。"""
        path = os.path.join(self.tmp, "plan_empty.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"units": []}, f)
        self.assertEqual(verify_config.main([path]), 1)

    def test_cgroup_clip(self):
        clipped, changed = verify_config._clip({0, 1, 2, 100}, "0-95")
        self.assertEqual(clipped, {0, 1, 2})
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()

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

"""/proc 读取子集：字段解析与异常降级（用 patched _read_text，不依赖真机）。"""

import unittest
from unittest import mock

import testutils  # noqa: F401
import procfs

STATUS_TEXT = """Name:\tVLLM::Worker
State:\tS (sleeping)
PPid:\t291298
Threads:\t90
Uid:\t0\t0\t0\t0
Cpus_allowed_list:\t146-165
Mems_allowed_list:\t0-7
"""

# comm 含空格与括号；第 39 字段（processor）按 ')' 之后的第 36 个字段取值
STAT_TEXT = "1234 (VLLM Worker (x)) " + " ".join(str(i) for i in range(3, 45)) + "\n"


class PatchedRead(unittest.TestCase):
    def _patch(self, mapping):
        return mock.patch.object(procfs, "_read_text", side_effect=lambda path: mapping.get(path))

    def test_read_status_and_fields(self):
        with self._patch({"/proc/1234/status": STATUS_TEXT}):
            status = procfs.read_status(1234)
            self.assertEqual(status["Name"], "VLLM::Worker")
            self.assertEqual(procfs.read_uid(1234), 0)
            self.assertEqual(procfs.read_mems_allowed(1234), "0-7")

    def test_missing_process_returns_none(self):
        with self._patch({}):
            self.assertIsNone(procfs.read_status(999))
            self.assertIsNone(procfs.read_uid(999))
            self.assertIsNone(procfs.read_last_cpu(999))

    def test_read_comm_main_and_thread(self):
        with self._patch({"/proc/1234/comm": " VLLM::Worker \n", "/proc/1234/task/1235/comm": "hccl_watchdog_t\n"}):
            self.assertEqual(procfs.read_comm(1234), "VLLM::Worker")
            self.assertEqual(procfs.read_comm(1234, 1235), "hccl_watchdog_t")

    def test_last_cpu_uses_last_rparen(self):
        with self._patch({"/proc/1234/stat": STAT_TEXT}):
            self.assertEqual(procfs.read_last_cpu(1234), 39)

    def test_stat_without_enough_fields(self):
        with self._patch({"/proc/1234/stat": "1234 (x) S 1 2\n"}):
            self.assertIsNone(procfs.read_last_cpu(1234))

    def test_numa_maps_streaming_aggregation(self):
        text = "55a0 default anon=3 dirty=3 N0=3\n55b0 heap anon=2 N1=2 N0=4\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=text)):
            parsed = procfs.read_numa_maps(1234)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["pages"], {0: 7, 1: 2})
        self.assertEqual(parsed["total"], 9)

    def test_numa_maps_permission_denied(self):
        with mock.patch("builtins.open", side_effect=PermissionError()):
            parsed = procfs.read_numa_maps(1234)
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["reason"], "permission_denied")
        self.assertEqual(parsed["total"], 0)

    def test_cgroup_v1_without_cpuset_controller_is_unreadable(self):
        """v1 只有其它控制器时不能拿它的 path 兜底读 cpuset，应标为不可观测。"""
        cg = {"version": "v1", "path": "/docker/mem", "controllers": {"memory": "/docker/mem"}}
        with mock.patch.object(procfs, "_read_text", return_value="0-95"):
            self.assertIsNone(procfs.read_cgroup_cpuset(1234, cg))

    def test_read_threads_limit_stops_early(self):
        """limit 用于展示场景：读满即停，不遍历全部线程。"""
        status = "State:\tS\nCpus_allowed_list:\t0-1\n"
        with (
            mock.patch("os.listdir", return_value=[str(i) for i in range(1, 6)]),
            mock.patch.object(procfs, "_read_text", side_effect=lambda path: status),
        ):
            threads = procfs.read_threads(1234, limit=2)
        self.assertEqual([t["tid"] for t in threads], [1, 2])


if __name__ == "__main__":
    unittest.main()

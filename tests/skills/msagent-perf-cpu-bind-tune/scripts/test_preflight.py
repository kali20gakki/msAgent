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

"""preflight.py：形态判定与 blocked 分支、无可见进程的显存占用识别。"""

import unittest
from unittest import mock

import testutils  # noqa: F401
import npu_query
import preflight

PRE_OK = {"npu_smi_available": True, "has_taskset": True}
INFO_OK = {"info_ok": True, "devices": [{"id": 0}], "processes": [{"pid": 1, "npu": 0, "name": "vllm", "mem_mb": 10}]}


class TestDetectForm(unittest.TestCase):
    def test_npu_smi_missing(self):
        form, blocked = preflight.detect_form({}, {"npu_smi_available": False, "has_taskset": True})
        self.assertIsNone(form)
        self.assertEqual(blocked["reason"], "npu-smi 不可用")
        self.assertEqual(blocked["status"], "blocked")

    def test_permission_denied(self):
        form, blocked = preflight.detect_form(
            {"info_ok": False, "info_error_kind": "permission_denied", "errors": ["Permission denied"]}, PRE_OK
        )
        self.assertIsNone(form)
        self.assertEqual(blocked["reason"], "npu-smi 权限不足")
        self.assertIn("sudo", blocked["suggestion"])

    def test_timeout_and_no_device(self):
        form, blocked = preflight.detect_form({"info_ok": False, "info_error_kind": "timeout"}, PRE_OK)
        self.assertIsNone(form)
        self.assertEqual(blocked["reason"], "npu-smi 调用超时")

        form, blocked = preflight.detect_form({"info_ok": False, "info_error_kind": "no_device"}, PRE_OK)
        self.assertIsNone(form)
        self.assertEqual(blocked["reason"], "未发现 NPU 设备")

    def test_call_failed_keeps_errors(self):
        form, blocked = preflight.detect_form(
            {"info_ok": False, "info_error_kind": "call_failed", "errors": ["boom"]}, PRE_OK
        )
        self.assertIsNone(form)
        self.assertEqual(blocked["reason"], "npu-smi 调用失败")
        self.assertEqual(blocked["suggestion"], "boom")

    def test_empty_devices_blocked(self):
        form, blocked = preflight.detect_form({"info_ok": True, "devices": [], "processes": []}, PRE_OK)
        self.assertIsNone(form)
        self.assertEqual(blocked["reason"], "未发现 NPU 设备")

    def test_in_run_and_pre_run(self):
        self.assertEqual(preflight.detect_form(INFO_OK, PRE_OK), ("in_run", None))
        self.assertEqual(
            preflight.detect_form({"info_ok": True, "devices": [{"id": 0}], "processes": []}, PRE_OK), ("pre_run", None)
        )


class TestCollectViewLimited(unittest.TestCase):
    """view_limited 需综合容器、/proc/1 可见性与自身 numa_maps 可读性。"""

    def _collect(self, in_container, proc_hidden, numa_ok):
        with (
            mock.patch.object(npu_query, "find_npu_smi", return_value=None),
            mock.patch.object(preflight, "_exists", side_effect=lambda p: in_container and p == "/.dockerenv"),
            mock.patch.object(preflight, "_readable", side_effect=lambda p: not proc_hidden),
            mock.patch.object(preflight.procfs, "read_numa_maps", return_value={"ok": numa_ok}),
            mock.patch("builtins.open", side_effect=FileNotFoundError),
        ):
            return preflight.collect()

    def test_any_signal_marks_limited(self):
        self.assertTrue(self._collect(in_container=False, proc_hidden=True, numa_ok=True)["view_limited"])
        self.assertTrue(self._collect(in_container=False, proc_hidden=False, numa_ok=False)["view_limited"])
        self.assertTrue(self._collect(in_container=True, proc_hidden=False, numa_ok=True)["view_limited"])

    def test_full_view_not_limited(self):
        pre = self._collect(in_container=False, proc_hidden=False, numa_ok=True)
        self.assertFalse(pre["view_limited"])
        self.assertFalse(pre["proc_hidden"])
        self.assertTrue(pre["numa_maps_self_ok"])


class TestUnknownHbm(unittest.TestCase):
    def test_only_occupancy_without_process(self):
        """判定实现收敛到 npu_query，preflight 不再保留重复实现。"""
        info = {
            "devices": [{"id": 0, "hbm_used_mb": 3600}, {"id": 1, "hbm_used_mb": 3600}, {"id": 2, "hbm_used_mb": 0}],
            "processes": [{"pid": 1, "npu": 0, "mem_mb": 100}],
        }
        self.assertFalse(hasattr(preflight, "unknown_hbm_npus"))
        self.assertEqual(npu_query.unknown_hbm_npus(info), [{"npu": 1, "hbm_used_mb": 3600}])


if __name__ == "__main__":
    unittest.main()

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

"""preflight.py：用临时目录伪造 tracefs，验证可写性组合与不可采集提示。"""

import os
import sys
import tempfile
import unittest
from unittest import mock

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
        "msagent-perf-cpu-tracing",
        "scripts",
    ),
)

import preflight


def _make_tracefs(with_trace=True, with_instances=False, instance_trace=False):
    """在临时目录里搭一个假的 tracefs 结构并返回根路径。"""
    base = tempfile.mkdtemp(prefix="fake-tracefs-")
    if with_trace:
        with open(os.path.join(base, "trace"), "w", encoding="utf-8") as fh:
            fh.write("")
    if with_instances:
        instances = os.path.join(base, "instances")
        os.makedirs(instances)
        if instance_trace:
            with open(os.path.join(instances, "trace"), "w", encoding="utf-8") as fh:
                fh.write("")
    return base


class TestDetectTracefs(unittest.TestCase):
    def test_no_candidate_dir(self):
        missing = os.path.join(tempfile.mkdtemp(prefix="fake-none-"), "not-exists")
        with mock.patch.object(preflight, "CANDIDATES", (missing,)):
            info = preflight.detect_tracefs()
        self.assertEqual(info, {"path": None, "available": False})

    def test_trace_writable_only(self):
        base = _make_tracefs(with_trace=True, with_instances=False)
        with open(os.path.join(base, "current_tracer"), "w", encoding="utf-8") as fh:
            fh.write("nop\n")
        with open(os.path.join(base, "available_tracers"), "w", encoding="utf-8") as fh:
            fh.write("function_graph nop\n")
        with mock.patch.object(preflight, "CANDIDATES", (base,)):
            info = preflight.detect_tracefs()
        self.assertEqual(info["path"], base)
        self.assertTrue(info["trace_writable"])
        self.assertFalse(info["instance_writable"])
        self.assertTrue(info["available"])
        self.assertEqual(info["current_tracer"], "nop")
        self.assertEqual(info["available_tracers"], "function_graph nop")

    def test_instance_writable_only(self):
        base = _make_tracefs(with_trace=False, with_instances=True, instance_trace=True)
        with mock.patch.object(preflight, "CANDIDATES", (base,)):
            info = preflight.detect_tracefs()
        self.assertFalse(info["trace_writable"])
        self.assertTrue(info["instance_writable"])
        self.assertTrue(info["available"])

    def test_instances_dir_writable_without_trace_file(self):
        """instances 是目录，用目录权限判定，而不是找里面的 trace 文件。"""
        base = _make_tracefs(with_trace=False, with_instances=True, instance_trace=False)
        with mock.patch.object(preflight, "CANDIDATES", (base,)):
            info = preflight.detect_tracefs()
        self.assertFalse(info["trace_writable"])
        self.assertTrue(info["instance_writable"])
        self.assertTrue(info["available"])

    def test_prefers_writable_candidate_over_first_existing(self):
        """首个候选只读时继续尝试后续候选，命中可写的那个。"""
        first = _make_tracefs(with_trace=False)
        second = _make_tracefs(with_trace=True)
        with mock.patch.object(preflight, "CANDIDATES", (first, second)):
            info = preflight.detect_tracefs()
        self.assertEqual(info["path"], second)
        self.assertTrue(info["available"])

    def test_all_candidates_unwritable_returns_first_for_diagnosis(self):
        first = _make_tracefs(with_trace=False)
        second = _make_tracefs(with_trace=False)
        with mock.patch.object(preflight, "CANDIDATES", (first, second)):
            info = preflight.detect_tracefs()
        self.assertEqual(info["path"], first)
        self.assertFalse(info["available"])


class TestRun(unittest.TestCase):
    def test_available_has_platform_fields_and_no_hint(self):
        base = _make_tracefs(with_trace=True)
        with mock.patch.object(preflight, "CANDIDATES", (base,)):
            info = preflight.run()
        self.assertTrue(info["available"])
        self.assertNotIn("hint", info)
        for key in ("platform", "kernel", "in_container"):
            self.assertIn(key, info)

    def test_unavailable_has_hint_and_platform_fields(self):
        base = _make_tracefs(with_trace=False)
        with mock.patch.object(preflight, "CANDIDATES", (base,)):
            info = preflight.run()
        self.assertFalse(info["available"])
        self.assertIn("hint", info)
        self.assertIn("tracefs", info["hint"])
        for key in ("platform", "kernel", "in_container"):
            self.assertIn(key, info)

    def test_explicit_path_unavailable(self):
        missing = os.path.join(tempfile.mkdtemp(prefix="fake-none-"), "not-exists")
        info = preflight.run(path=missing)
        self.assertEqual(info["path"], missing)
        self.assertFalse(info["available"])
        self.assertFalse(info["trace_writable"])
        self.assertFalse(info["instance_writable"])
        self.assertIn("hint", info)

    def test_explicit_path_available(self):
        base = _make_tracefs(with_trace=True)
        info = preflight.run(path=base)
        self.assertTrue(info["available"])
        self.assertTrue(info["trace_writable"])
        self.assertNotIn("hint", info)


if __name__ == "__main__":
    unittest.main()

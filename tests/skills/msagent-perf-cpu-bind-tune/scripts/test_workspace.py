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

"""workspace.py：主机名清洗与产物目录解析（dry-run 不落盘）。"""

import os
import tempfile
import unittest

import testutils  # noqa: F401
import workspace


class TestSafeName(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(workspace.safe_name("host/../etc"), "host_.._etc".strip("._"))
        self.assertEqual(workspace.safe_name("a b"), "a_b")
        self.assertEqual(workspace.safe_name(""), "unknown")
        self.assertEqual(workspace.safe_name(None), "unknown")

    def test_truncate(self):
        self.assertEqual(len(workspace.safe_name("x" * 200)), 64)

    def test_truncate_does_not_leave_separator_tail(self):
        """截断发生在 strip 之后时，第 64 个字符是 . / _ 也要去掉尾巴。"""
        self.assertEqual(workspace.safe_name("x" * 63 + "." + "y"), "x" * 63)


class TestSessionDir(unittest.TestCase):
    def test_layout_and_create(self):
        root = tempfile.mkdtemp(prefix="hostcpu-ws-")
        path = workspace.session_dir(host="node-1", timestamp="20260101-000000", root=root)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(path, os.path.join(root, "workspaces", "node-1", "20260101-000000"))

    def test_create_false_is_side_effect_free(self):
        """--dry-run 语义：只算路径、不创建任何目录。"""
        root = tempfile.mkdtemp(prefix="hostcpu-ws-")
        path = workspace.session_dir(host="node-1", timestamp="20260101-000001", root=root, create=False)
        self.assertFalse(os.path.exists(path))

    def test_root_priority(self):
        old = os.environ.get("HOSTCPU_WORKSPACE")
        try:
            os.environ["HOSTCPU_WORKSPACE"] = tempfile.mkdtemp(prefix="hostcpu-env-")
            self.assertEqual(workspace.workspace_root(), os.path.abspath(os.environ["HOSTCPU_WORKSPACE"]))
            explicit = tempfile.mkdtemp(prefix="hostcpu-explicit-")
            self.assertEqual(workspace.workspace_root(explicit), os.path.abspath(explicit))
        finally:
            if old is None:
                os.environ.pop("HOSTCPU_WORKSPACE", None)
            else:
                os.environ["HOSTCPU_WORKSPACE"] = old


if __name__ == "__main__":
    unittest.main()

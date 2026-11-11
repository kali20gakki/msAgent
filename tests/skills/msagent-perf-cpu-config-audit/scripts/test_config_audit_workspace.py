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
workspace.py 单元测试：主机名/时间戳进路径前清洗。
"""

import os
import sys
import tempfile
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

import workspace


class TestSafeName(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(workspace.safe_name("host/../etc"), "host_.._etc")
        self.assertEqual(workspace.safe_name("a b"), "a_b")

    def test_empty_and_truncate(self):
        self.assertEqual(workspace.safe_name(None), "unknown")
        self.assertEqual(len(workspace.safe_name("x" * 200)), 64)


class TestSessionDir(unittest.TestCase):
    def test_layout(self):
        root = tempfile.mkdtemp(prefix="hostcpu-audit-")
        path = workspace.session_dir(host="node-1", timestamp="20260101-000000", root=root)
        self.assertEqual(path, os.path.join(root, "hostcpu_audit_node-1_20260101-000000"))
        self.assertTrue(os.path.isdir(path))

    def test_path_escape_is_neutralized(self):
        root = tempfile.mkdtemp(prefix="hostcpu-audit-")
        path = workspace.session_dir(host="../../tmp", timestamp="a/b", root=root)
        self.assertEqual(path, os.path.join(root, "hostcpu_audit_tmp_a_b"))
        self.assertTrue(path.startswith(root + os.sep))


if __name__ == "__main__":
    unittest.main()

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

"""engine_detect.py：采集引擎可用/不可用两态与 path 透传。"""

import os
import sys
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
        "msagent-perf-cpu-tracing",
        "scripts",
    ),
)

from convert import engine_detect


class TestDetect(unittest.TestCase):
    def test_none_input(self):
        result = engine_detect.detect(None)
        self.assertEqual(result["engine"], "none")
        self.assertFalse(result["available"])
        self.assertIsNone(result["path"])
        self.assertEqual(result["detail"], {})

    def test_empty_dict_input(self):
        result = engine_detect.detect({})
        self.assertEqual(result["engine"], "none")
        self.assertFalse(result["available"])

    def test_available(self):
        info = {"available": True, "path": "/sys/kernel/tracing"}
        result = engine_detect.detect(info)
        self.assertEqual(result["engine"], "tracefs")
        self.assertTrue(result["available"])
        self.assertEqual(result["path"], "/sys/kernel/tracing")
        self.assertIs(result["detail"], info)

    def test_truthy_available_coerced(self):
        result = engine_detect.detect({"available": "yes", "path": "/tmp/tracing"})
        self.assertEqual(result["engine"], "tracefs")
        self.assertTrue(result["available"])
        self.assertEqual(result["path"], "/tmp/tracing")


if __name__ == "__main__":
    unittest.main()

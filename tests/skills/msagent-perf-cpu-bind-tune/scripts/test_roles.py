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

"""roles.py：comm 关键字到线程角色的唯一映射。"""

import unittest

import testutils  # noqa: F401
import roles


class TestClassify(unittest.TestCase):
    def test_comm_role(self):
        self.assertEqual(roles.classify("hccl_watchdog_t"), "comm")
        self.assertEqual(roles.classify("AOE_RTKB"), "comm")

    def test_io_role(self):
        self.assertEqual(roles.classify("ZMQbg/IO/0"), "io")
        self.assertEqual(roles.classify("tokenizer"), "io")
        self.assertEqual(roles.classify("DataLoader"), "io")

    def test_compute_role(self):
        self.assertEqual(roles.classify("VLLM::Worker"), "compute")
        self.assertEqual(roles.classify("pt_autograd_1"), "compute")
        self.assertEqual(roles.classify("acl_thread"), "compute")

    def test_default_is_main(self):
        self.assertEqual(roles.classify("WatchdogMonitor"), "main")
        self.assertEqual(roles.classify(""), "main")
        self.assertEqual(roles.classify(None), "main")

    def test_order_comm_before_compute(self):
        """comm 规则优先：同时含 hccl 与 python 时归 comm。"""
        self.assertEqual(roles.classify("hccl_python_worker"), "comm")

    def test_keywords(self):
        self.assertEqual(roles.keywords("comm"), ["hccl", "aoe_rtkb"])
        self.assertEqual(roles.keywords("unknown"), [])


if __name__ == "__main__":
    unittest.main()

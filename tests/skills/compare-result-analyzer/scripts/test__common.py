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
_common.py 单元测试
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "skills", "accuracy", "compare-result-analyzer", "scripts"
    ),
)

import _common
from _common import (
    safe_float,
    _nat_key,
    extract_op_prefix,
    get_param_key,
    get_param_type,
    _record_unparseable,
    flush_unparseable_names,
)


class TestSafeFloat(unittest.TestCase):
    def test_none_input(self):
        self.assertIsNone(safe_float(None))

    def test_normal_float(self):
        self.assertAlmostEqual(safe_float("3.14"), 3.14)


class TestNatKey(unittest.TestCase):
    def test_string_with_number(self):
        self.assertEqual(_nat_key("output.2"), ["output.", 2, ""])


class TestExtractOpPrefix(unittest.TestCase):
    def test_module_forward(self):
        prefix, direction = extract_op_prefix("Module.A.B.forward.0.input.0")
        self.assertEqual(prefix, "Module.A.B.forward.0")
        self.assertEqual(direction, "forward")

    def test_unparseable(self):
        prefix, direction = extract_op_prefix("some.unknown.format")
        self.assertEqual(prefix, "some.unknown.format")
        self.assertEqual(direction, "unknown")


class TestGetParamKey(unittest.TestCase):
    def test_module_input(self):
        result = get_param_key(
            "Module.A.B.forward.0.input.0",
            "Module.A.B.forward.0",
            "forward",
        )
        self.assertEqual(result, "input.0")

    def test_no_match(self):
        result = get_param_key("some.unknown.format", "nonexistent", "unknown")
        self.assertEqual(result, "")


class TestGetParamType(unittest.TestCase):
    def test_input(self):
        self.assertEqual(get_param_type("0.input.0"), "input")

    def test_parameters_grad(self):
        self.assertEqual(get_param_type("parameters_grad.0.weight"), "output")


class TestRecordUnparseable(unittest.TestCase):
    def setUp(self):
        _common._UNPARSEABLE_SEEN = set()
        _common._UNPARSEABLE_SAMPLES = []
        _common._UNPARSEABLE_WARNED = False

    def test_records_name(self):
        with redirect_stderr(io.StringIO()):
            _record_unparseable("unknown.op.format")
        self.assertIn("unknown.op.format", _common._UNPARSEABLE_SEEN)

    def test_dedup(self):
        with redirect_stderr(io.StringIO()):
            _record_unparseable("unknown.op.format")
            _record_unparseable("unknown.op.format")
        self.assertEqual(len(_common._UNPARSEABLE_SEEN), 1)
        self.assertEqual(len(_common._UNPARSEABLE_SAMPLES), 1)


class TestFlushUnparseableNames(unittest.TestCase):
    def setUp(self):
        _common._UNPARSEABLE_SEEN = set()
        _common._UNPARSEABLE_SAMPLES = []
        _common._UNPARSEABLE_WARNED = False

    def test_with_names_prints_warning(self):
        with redirect_stderr(io.StringIO()):
            _record_unparseable("unknown.op.format")
        f = io.StringIO()
        with redirect_stderr(f):
            flush_unparseable_names()
        self.assertIn("unknown.op.format", f.getvalue())
        self.assertIn("1", f.getvalue())


if __name__ == "__main__":
    unittest.main()

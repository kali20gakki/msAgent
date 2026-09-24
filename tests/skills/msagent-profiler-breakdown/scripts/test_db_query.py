#!/usr/bin/python3
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

"""
db_query.py 单元测试
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "profiler",
        "msagent-profiler-breakdown",
        "scripts",
    ),
)

import db_query
from db_query import (
    MSTX_TYPE,
    OP_TYPE,
    QUEUE_TYPE,
    _as_list,
    _escape_like,
    _name_cond,
    open_ro,
    query_api_combined,
    query_cann_api,
    query_mstx,
    query_pytorch_api,
    string_exists,
    strings_like,
    strings_present,
)


def _build_db(path):
    """构造最小 profiling DB：STRING_IDS + MSTX_EVENTS + PYTORCH_API + CANN_API。

    对齐 references/db-usage.md 字段约定：PYTORCH_API.startNs/endNs 为 TEXT 数字串，
    MSTX_EVENTS 独立表（message 为 STRING_IDS.id）。
    """
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE STRING_IDS (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE MSTX_EVENTS (startNs INTEGER, endNs INTEGER, message INTEGER);
        CREATE TABLE PYTORCH_API (name INTEGER, type INTEGER, startNs TEXT, endNs TEXT);
        CREATE TABLE CANN_API (name INTEGER, type INTEGER, startNs TEXT, endNs TEXT);
    """)
    ids = {
        "vllm::dsa_forward": 1,
        "forward": 2,
        "prepare input": 3,
        "actor_update": 4,
        "ref_compute_log_prob": 5,
    }
    for value, sid in ids.items():
        conn.execute("INSERT INTO STRING_IDS (id, value) VALUES (?, ?)", (sid, value))
    # PYTORCH_API：forward op(50001) 两条 + forward queue(50002) 一条（应被 type 过滤排除）
    conn.execute("INSERT INTO PYTORCH_API (name, type, startNs, endNs) VALUES (2, 50001, '1000', '2000')")
    conn.execute("INSERT INTO PYTORCH_API (name, type, startNs, endNs) VALUES (2, 50001, '5000', '7000')")
    conn.execute("INSERT INTO PYTORCH_API (name, type, startNs, endNs) VALUES (2, 50002, '8000', '9000')")
    # CANN_API：一条 forward op
    conn.execute("INSERT INTO CANN_API (name, type, startNs, endNs) VALUES (2, 50001, '3000', '4000')")
    # MSTX_EVENTS：actor_update 两条（乱序插入，查询应按 startNs 升序）
    conn.execute("INSERT INTO MSTX_EVENTS (startNs, endNs, message) VALUES (6000, 8000, 4)")
    conn.execute("INSERT INTO MSTX_EVENTS (startNs, endNs, message) VALUES (2000, 3000, 4)")
    conn.commit()
    conn.close()


class TestOpenRo(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_open_ro_and_query(self):
        path = os.path.join(self.tmpdir, "test.db")
        _build_db(path)
        conn = open_ro(path)
        try:
            cur = conn.cursor()
            self.assertTrue(string_exists(cur, "forward"))
        finally:
            conn.close()

    def test_open_ro_path_with_space(self):
        path = os.path.join(self.tmpdir, "test db.db")
        _build_db(path)
        conn = open_ro(path)
        try:
            self.assertTrue(string_exists(conn.cursor(), "forward"))
        finally:
            conn.close()


class TestEscapeLike(unittest.TestCase):
    def test_special_chars_escaped(self):
        self.assertEqual(_escape_like("a%b_c\\d"), "a\\%b\\_c\\\\d")

    def test_plain_string_unchanged(self):
        self.assertEqual(_escape_like("forward"), "forward")


class TestAsList(unittest.TestCase):
    def test_single_string(self):
        self.assertEqual(_as_list("forward"), ["forward"])

    def test_dedupe_and_drop_empty(self):
        self.assertEqual(_as_list(["a", "", "a", "b"]), ["a", "b"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _as_list([])


class TestNameCond(unittest.TestCase):
    def test_exact(self):
        cond, params = _name_cond("S.value", ["forward", "sample"], "exact")
        self.assertEqual(cond, "S.value IN (?,?)")
        self.assertEqual(params, ["forward", "sample"])

    def test_prefix(self):
        cond, params = _name_cond("S.value", ["actor", "ref"], "prefix")
        self.assertIn("LIKE ? ESCAPE '\\'", cond)
        self.assertEqual(params, ["actor%", "ref%"])

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            _name_cond("S.value", ["a"], "fuzzy")


class TestStringQueries(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        _build_db(self.db)
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_string_exists(self):
        self.assertTrue(string_exists(self.cur, "forward"))
        self.assertFalse(string_exists(self.cur, "not_exist"))

    def test_strings_like_prefix(self):
        self.assertEqual(strings_like(self.cur, "vllm"), ["vllm::dsa_forward"])
        self.assertEqual(strings_like(self.cur, "nope"), [])

    def test_strings_present_subset(self):
        result = strings_present(self.cur, ["forward", "missing", "actor_update", "missing"])
        self.assertEqual(result, ["forward", "actor_update"])


class TestQueryMstx(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        _build_db(self.db)
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_exact_sorted(self):
        records = query_mstx(self.cur, "actor_update", "exact")
        self.assertEqual(len(records), 2)
        self.assertEqual([r["start"] for r in records], [2000, 6000])
        self.assertEqual(records[0]["name"], "actor_update")
        self.assertEqual(records[0]["source"], "MSTX_EVENTS")
        self.assertEqual(records[0]["duration"], 1000)

    def test_exact_list(self):
        records = query_mstx(self.cur, ["actor_update", "ref_compute_log_prob"], "exact")
        self.assertEqual(len(records), 2)

    def test_prefix(self):
        records = query_mstx(self.cur, "actor", "prefix")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["name"], "actor_update")

    def test_no_match(self):
        self.assertEqual(query_mstx(self.cur, "nonexistent", "exact"), [])


class TestQueryPyTorchApi(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        _build_db(self.db)
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_with_type_filter_excludes_queue(self):
        records = query_pytorch_api(self.cur, "forward", "exact", (OP_TYPE, MSTX_TYPE))
        self.assertEqual(len(records), 2)
        self.assertEqual([r["start"] for r in records], [1000, 5000])
        self.assertEqual(records[0]["source"], "PYTORCH_API")

    def test_without_type_filter_includes_all(self):
        records = query_pytorch_api(self.cur, "forward", "exact")
        self.assertEqual(len(records), 3)

    def test_queues_excluded_by_default_contract(self):
        # 断言 db-usage 约定：QUEUE_TYPE 单独列出且可被过滤
        records = query_pytorch_api(self.cur, "forward", "exact", (QUEUE_TYPE,))
        self.assertEqual(len(records), 1)


class TestQueryCannApi(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        _build_db(self.db)
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_query(self):
        records = query_cann_api(self.cur, "forward", "exact")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "CANN_API")
        self.assertEqual(records[0]["start"], 3000)


class TestQueryApiCombined(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        _build_db(self.db)
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merged_sorted_by_start(self):
        records = query_api_combined(self.cur, "forward", "exact")
        self.assertEqual(len(records), 4)  # PYTORCH op 2 + queue 1 + CANN 1
        starts = [r["start"] for r in records]
        self.assertEqual(starts, sorted(starts))
        sources = {r["source"] for r in records}
        self.assertEqual(sources, {"PYTORCH_API", "CANN_API"})


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        _build_db(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, argv):
        out = StringIO()
        with redirect_stdout(out):
            rc = db_query.main(argv)
        return rc, json.loads(out.getvalue())

    def test_string_exists_subcommand(self):
        rc, data = self._run(["string", "--db", self.db, "--value", "forward"])
        self.assertEqual(rc, 0)
        self.assertTrue(data["result"])

    def test_string_prefix_subcommand(self):
        rc, data = self._run(["string", "--db", self.db, "--prefix", "vllm"])
        self.assertEqual(rc, 0)
        self.assertEqual(data["result"], ["vllm::dsa_forward"])

    def test_string_list_subcommand(self):
        rc, data = self._run(["string", "--db", self.db, "--list", "forward", "missing"])
        self.assertEqual(rc, 0)
        self.assertEqual(data["result"], ["forward"])

    def test_mstx_subcommand(self):
        rc, data = self._run(["mstx", "--db", self.db, "--name", "actor_update"])
        self.assertEqual(rc, 0)
        self.assertEqual(data["count"], 2)

    def test_pytorch_subcommand_with_type(self):
        rc, data = self._run(["pytorch", "--db", self.db, "--name", "forward", "--type", "50001"])
        self.assertEqual(rc, 0)
        self.assertEqual(data["count"], 2)

    def test_cann_subcommand(self):
        rc, data = self._run(["cann", "--db", self.db, "--name", "forward"])
        self.assertEqual(rc, 0)
        self.assertEqual(data["count"], 1)

    def test_api_combined_subcommand(self):
        rc, data = self._run(["api", "--db", self.db, "--name", "forward"])
        self.assertEqual(rc, 0)
        self.assertEqual(data["count"], 4)


if __name__ == "__main__":
    unittest.main()

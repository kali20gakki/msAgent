#!/usr/bin/env python3
"""
snapshot_analyze.py 单元测试
"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "ascend-npu-snapshot-analyzer", "scripts"))

import snapshot_queries as q
from snapshot_analyze import (
    _format_bytes,
    _health_status,
    _metric_status,
    analyze_overview,
    analyze_peak,
    analyze_fragment,
    analyze_leak,
    analyze_oom,
    analyze_compare,
)


def _setup_test_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_index INTEGER UNIQUE NOT NULL,
            device_type TEXT DEFAULT 'Ascend-NPU'
        );
        CREATE TABLE IF NOT EXISTS call_stacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stack_hash TEXT UNIQUE,
            frames_json TEXT
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            address INTEGER,
            total_size INTEGER,
            allocated_size INTEGER,
            active_size INTEGER,
            requested_size INTEGER,
            stream INTEGER,
            segment_type TEXT,
            pool_id_0 INTEGER,
            pool_id_1 INTEGER,
            is_expandable INTEGER,
            stack_id INTEGER,
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (stack_id) REFERENCES call_stacks(id)
        );
        CREATE TABLE IF NOT EXISTS blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            address INTEGER,
            size INTEGER,
            requested_size INTEGER,
            state TEXT,
            stack_id INTEGER,
            FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE,
            FOREIGN KEY (stack_id) REFERENCES call_stacks(id)
        );
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            trace_index INTEGER,
            action TEXT,
            addr INTEGER,
            device_free INTEGER,
            size INTEGER,
            stream INTEGER,
            stack_id INTEGER,
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (stack_id) REFERENCES call_stacks(id)
        );
    """)
    conn.commit()
    conn.close()


class TestFormatBytes(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(_format_bytes(512), "512 B")

    def test_kb(self):
        self.assertIn("KB", _format_bytes(2048))

    def test_mb(self):
        self.assertIn("MB", _format_bytes(5 * 1024 * 1024))

    def test_gb(self):
        self.assertIn("GB", _format_bytes(3 * 1024**3))


class TestHealthStatus(unittest.TestCase):
    def test_healthy(self):
        result = _health_status(3.0, 50, False)
        self.assertEqual(result["level"], "健康")

    def test_warn_frag(self):
        result = _health_status(10.0, 50, False)
        self.assertEqual(result["level"], "需关注")

    def test_warn_segments(self):
        result = _health_status(3.0, 150, False)
        self.assertEqual(result["level"], "需关注")

    def test_err_oom(self):
        result = _health_status(3.0, 50, True)
        self.assertEqual(result["level"], "严重")

    def test_err_frag(self):
        result = _health_status(20.0, 50, False)
        self.assertEqual(result["level"], "严重")


class TestMetricStatus(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(_metric_status(3.0, (5, 15)), "[OK]")

    def test_warn(self):
        self.assertEqual(_metric_status(10.0, (5, 15)), "[WARN]")

    def test_err(self):
        self.assertEqual(_metric_status(20.0, (5, 15)), "[ERR]")


class TestAnalyzeOverview(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        _setup_test_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) VALUES (1, 0x1000, 1024*1024*1024, 800*1024*1024, 800*1024*1024, 800*1024*1024, 'large', 1)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_overview_structure(self):
        result = analyze_overview(self.db_path)
        self.assertEqual(result["mode"], "overview")
        self.assertIn("health", result)
        self.assertIn("summary", result)
        self.assertIn("devices", result)
        self.assertEqual(result["device_count"], 1)
        self.assertEqual(result["oom_count"], 0)

    def test_summary_values(self):
        result = analyze_overview(self.db_path)
        summary = result["summary"]
        self.assertGreater(summary["reserved"], 0)
        self.assertGreater(summary["allocated"], 0)
        self.assertIsNotNone(summary["frag_pct"])


class TestAnalyzePeak(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        _setup_test_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.execute(
            "INSERT INTO call_stacks (stack_hash, frames_json) VALUES ('hash1', '[\"a.py:10:func_a\", \"b.py:20:func_b\"]')"
        )
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) VALUES (1, 0x1000, 1024*1024*1024, 800*1024*1024, 800*1024*1024, 800*1024*1024, 'large', 1)"
        )
        conn.execute(
            "INSERT INTO blocks (segment_id, address, size, requested_size, state, stack_id) VALUES (1, 0x1000, 800*1024*1024, 800*1024*1024, 'active_allocated', 1)"
        )
        conn.execute(
            "INSERT INTO traces (device_id, trace_index, action, size, stack_id) VALUES (1, 0, 'segment_alloc', 1024*1024*1024, 1)"
        )
        conn.execute(
            "INSERT INTO traces (device_id, trace_index, action, size) VALUES (1, 1, 'segment_alloc', 512*1024*1024)"
        )
        conn.execute(
            "INSERT INTO traces (device_id, trace_index, action, size) VALUES (1, 2, 'segment_free', 512*1024*1024)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_peak_structure(self):
        result = analyze_peak(self.db_path)
        self.assertEqual(result["mode"], "peak")
        self.assertIn("peak", result)
        self.assertIn("baseline", result)
        self.assertIn("deltas", result)
        self.assertIn("peak_alloc_events", result)
        self.assertIn("peak_blocks", result)
        self.assertIn("devices", result)
        self.assertGreaterEqual(len(result["devices"]), 1)

    def test_peak_timeline(self):
        result = analyze_peak(self.db_path)
        self.assertGreater(result["peak"]["reserved"], 0)

    def test_peak_contributors_have_call_path(self):
        result = analyze_peak(self.db_path)
        events = result["peak_alloc_events"]["top"]
        self.assertGreater(len(events), 0)
        for evt in events:
            self.assertIn("call_path", evt)


class TestAnalyzeFragment(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        _setup_test_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) VALUES (1, 0x1000, 1024*1024*1024, 800*1024*1024, 800*1024*1024, 800*1024*1024, 'large', 1)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fragment_structure(self):
        result = analyze_fragment(self.db_path)
        self.assertEqual(result["mode"], "fragment")
        self.assertIn("overall", result)
        self.assertIn("top_fragmented", result)

    def test_frag_pct_calculation(self):
        result = analyze_fragment(self.db_path)
        expected = round((1024 * 1024 * 1024 - 800 * 1024 * 1024) / (1024 * 1024 * 1024) * 100, 1)
        self.assertEqual(result["overall"]["frag_pct"], expected)


class TestAnalyzeLeak(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        _setup_test_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) VALUES (1, 0x1000, 1024*1024*1024, 800*1024*1024, 800*1024*1024, 800*1024*1024, 'large', 1)"
        )
        for i in range(10):
            conn.execute(
                "INSERT INTO traces (device_id, trace_index, action, size, addr) VALUES (1, ?, 'segment_alloc', 1024, 0x1000)",
                (i,),
            )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_leak_structure(self):
        result = analyze_leak(self.db_path)
        self.assertEqual(result["mode"], "leak")
        self.assertIn("risk", result)
        self.assertIn("monotonic_growth", result)
        self.assertIn("suspects", result)

    def test_monotonic_detection(self):
        result = analyze_leak(self.db_path)
        self.assertTrue(result["monotonic_growth"]["detected"])


class TestAnalyzeOOM(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        _setup_test_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.execute(
            "INSERT INTO traces (device_id, trace_index, action, device_free, size) VALUES (1, 100, 'oom', 128*1024*1024, 0)"
        )
        for i in range(50, 100):
            conn.execute(
                "INSERT INTO traces (device_id, trace_index, action, size) VALUES (1, ?, 'alloc', ?)",
                (i, 256 * 1024 * 1024),
            )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_oom_detected(self):
        result = analyze_oom(self.db_path)
        self.assertTrue(result["detected"])
        self.assertEqual(len(result["events"]), 1)

    def test_no_oom(self):
        db2 = os.path.join(self.tmpdir, "no_oom.db")
        _setup_test_db(db2)
        conn = sqlite3.connect(db2)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.commit()
        conn.close()
        result = analyze_oom(db2)
        self.assertFalse(result["detected"])


class TestAnalyzeCompare(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_a = os.path.join(self.tmpdir, "a.db")
        self.db_b = os.path.join(self.tmpdir, "b.db")

        for db_path, addr, size in [(self.db_a, 0x1000, 1024), (self.db_b, 0x1000, 2048)]:
            _setup_test_db(db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO devices (device_index) VALUES (0)")
            conn.execute(
                "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) "
                "VALUES (1, ?, ?, ?, ?, ?, 'small', 0)",
                (addr, size, size, size, size),
            )
            conn.commit()
            conn.close()

        _setup_test_db(self.db_b)
        conn = sqlite3.connect(self.db_b)
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) "
            "VALUES (1, 0x2000, 512, 512, 512, 512, 'small', 0)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compare_structure(self):
        result = analyze_compare(self.db_a, self.db_b)
        self.assertEqual(result["mode"], "compare")
        self.assertIn("metrics", result)
        self.assertIn("new_segments", result)
        self.assertIn("grown_segments", result)

    def test_compare_growth(self):
        result = analyze_compare(self.db_a, self.db_b)
        reserved_metric = next(m for m in result["metrics"] if m["name"] == "Reserved")
        self.assertGreater(reserved_metric["diff"], 0)

    def test_new_segment(self):
        result = analyze_compare(self.db_a, self.db_b)
        self.assertEqual(result["new_segment_count"], 1)

    def test_ref_not_found(self):
        result = analyze_compare(self.db_a, "nonexistent.db")
        self.assertIn("error", result)


class TestSnapshotQueries(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        _setup_test_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO devices (device_index) VALUES (0)")
        conn.execute("INSERT INTO devices (device_index) VALUES (1)")
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) VALUES (1, 0x1000, 1024*1024*1024, 800*1024*1024, 800*1024*1024, 800*1024*1024, 'large', 1)"
        )
        conn.execute(
            "INSERT INTO segments (device_id, address, total_size, allocated_size, active_size, requested_size, segment_type, is_expandable) VALUES (2, 0x2000, 512*1024*1024, 400*1024*1024, 400*1024*1024, 400*1024*1024, 'small', 0)"
        )
        conn.execute("INSERT INTO traces (device_id, trace_index, action, size) VALUES (1, 0, 'segment_alloc', 1024)")
        conn.execute("INSERT INTO traces (device_id, trace_index, action, size) VALUES (1, 1, 'alloc', 512)")
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_device_overview(self):
        result = q.get_device_overview(self.db_path)
        self.assertEqual(len(result), 2)

    def test_get_device_count(self):
        self.assertEqual(q.get_device_count(self.db_path), 2)

    def test_get_segment_count(self):
        self.assertEqual(q.get_segment_count(self.db_path), 2)

    def test_get_trace_count(self):
        self.assertEqual(q.get_trace_count(self.db_path), 2)

    def test_get_block_state_dist(self):
        result = q.get_block_state_dist(self.db_path)
        self.assertIsInstance(result, list)

    def test_get_expansion_events(self):
        result = q.get_expansion_events(self.db_path)
        self.assertGreater(len(result), 0)

    def test_get_oom_events(self):
        result = q.get_oom_events(self.db_path)
        self.assertEqual(len(result), 0)

    def test_get_fragmentation_detail(self):
        result = q.get_fragmentation_detail(self.db_path)
        self.assertEqual(len(result), 2)

    def test_execute_sql(self):
        result = q.execute_sql(self.db_path, "SELECT COUNT(*) AS cnt FROM devices")
        self.assertEqual(result[0]["cnt"], 2)


if __name__ == "__main__":
    unittest.main()

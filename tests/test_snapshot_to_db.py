#!/usr/bin/env python3
"""
snapshot_to_db.py 单元测试
"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "ascend-npu-snapshot-analyzer", "scripts"))

from snapshot_to_db import (
    _stack_hash,
    _pickle_load,
    _CallStackResolver,
    _resolve_device,
    _create_tables,
    _create_indexes,
    convert_snapshot,
)


class TestStackHash(unittest.TestCase):
    def test_empty_frames(self):
        self.assertEqual(len(_stack_hash([])), 32)

    def test_same_frames_same_hash(self):
        frames = ["file.py:10:func"]
        self.assertEqual(_stack_hash(frames), _stack_hash(frames))

    def test_different_frames_different_hash(self):
        self.assertNotEqual(
            _stack_hash(["a.py:1:f"]),
            _stack_hash(["b.py:2:g"]),
        )


class TestResolveCallStack(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        _create_tables(self.conn)
        self.resolver = _CallStackResolver(self.conn.cursor())

    def tearDown(self):
        self.conn.close()

    def test_empty_frames_returns_none(self):
        result = self.resolver.resolve([])
        self.assertIsNone(result)

    def test_dedup_same_frames(self):
        frames = ["a.py:10:func_a", "b.py:20:func_b"]
        id1 = self.resolver.resolve(frames)
        id2 = self.resolver.resolve(frames)
        self.assertEqual(id1, id2)

        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM call_stacks")
        self.assertEqual(cursor.fetchone()[0], 1)


class TestResolveDevice(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        _create_tables(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_insert_device(self):
        cursor = self.conn.cursor()
        id1 = _resolve_device(cursor, 0)
        id2 = _resolve_device(cursor, 0)
        self.assertEqual(id1, id2)

    def test_multiple_devices(self):
        cursor = self.conn.cursor()
        id0 = _resolve_device(cursor, 0)
        id1 = _resolve_device(cursor, 1)
        self.assertNotEqual(id0, id1)


class TestCreateTables(unittest.TestCase):
    def test_tables_exist(self):
        conn = sqlite3.connect(":memory:")
        _create_tables(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        expected = ["blocks", "call_stacks", "devices", "segments", "traces"]
        for t in expected:
            self.assertIn(t, tables)
        conn.close()


class TestCreateIndexes(unittest.TestCase):
    def test_indexes_created(self):
        conn = sqlite3.connect(":memory:")
        _create_tables(conn)
        _create_indexes(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        indexes = [row[0] for row in cursor.fetchall()]
        self.assertGreater(len(indexes), 5)
        self.assertIn("idx_blocks_state", indexes)
        self.assertIn("idx_traces_device_action", indexes)
        conn.close()


class TestPickleLoad(unittest.TestCase):
    def setUp(self):
        import pickle  # nosec B403

        self.tmpdir = tempfile.mkdtemp()

        self.dict_path = os.path.join(self.tmpdir, "dict.pkl")
        with open(self.dict_path, "wb") as f:
            pickle.dump({"segments": [], "device_traces": []}, f)

        self.list_path = os.path.join(self.tmpdir, "list.pkl")
        with open(self.list_path, "wb") as f:
            pickle.dump([], f)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_dict_format(self):
        result = _pickle_load(self.dict_path)
        self.assertIsInstance(result, dict)
        self.assertIn("segments", result)
        self.assertIn("device_traces", result)

    def test_load_list_format(self):
        result = _pickle_load(self.list_path)
        self.assertIsInstance(result, dict)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _pickle_load("nonexistent.pkl")


class TestConvertSnapshot(unittest.TestCase):
    def setUp(self):
        import pickle  # nosec B403

        self.tmpdir = tempfile.mkdtemp()

        self.empty_path = os.path.join(self.tmpdir, "empty.pkl")
        with open(self.empty_path, "wb") as f:
            pickle.dump({"segments": [], "device_traces": []}, f)

        self.single_path = os.path.join(self.tmpdir, "single.pkl")
        with open(self.single_path, "wb") as f:
            pickle.dump(
                {
                    "segments": [
                        {
                            "device": 0,
                            "address": 0x1000,
                            "total_size": 1024 * 1024,
                            "allocated_size": 512 * 1024,
                            "active_size": 512 * 1024,
                            "requested_size": 512 * 1024,
                            "stream": 0,
                            "segment_type": "small",
                            "segment_pool_id": (0, 0),
                            "is_expandable": True,
                            "frames": ["a.py:10:alloc"],
                            "blocks": [
                                {
                                    "address": 0x1000,
                                    "size": 512 * 1024,
                                    "requested_size": 512 * 1024,
                                    "state": "active_allocated",
                                    "frames": ["a.py:10:alloc"],
                                }
                            ],
                        }
                    ],
                    "device_traces": [
                        [
                            {
                                "action": "segment_alloc",
                                "addr": 0x1000,
                                "size": 1024 * 1024,
                                "stream": 0,
                                "frames": [],
                            }
                        ]
                    ],
                },
                f,
            )

        self.multi_device_path = os.path.join(self.tmpdir, "multi_device.pkl")
        with open(self.multi_device_path, "wb") as f:
            pickle.dump(
                {
                    "segments": [
                        {
                            "device": 0,
                            "address": 0x1000,
                            "total_size": 1024 * 1024,
                            "allocated_size": 512 * 1024,
                            "active_size": 512 * 1024,
                            "requested_size": 512 * 1024,
                            "stream": 0,
                            "segment_type": "small",
                            "segment_pool_id": (),
                            "is_expandable": False,
                            "frames": [],
                            "blocks": [],
                        },
                        {
                            "device": 1,
                            "address": 0x2000,
                            "total_size": 2 * 1024 * 1024,
                            "allocated_size": 1024 * 1024,
                            "active_size": 1024 * 1024,
                            "requested_size": 1024 * 1024,
                            "stream": 0,
                            "segment_type": "large",
                            "segment_pool_id": (1, 0),
                            "is_expandable": True,
                            "frames": [],
                            "blocks": [],
                        },
                    ],
                    "device_traces": [[], []],
                },
                f,
            )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_snapshot(self):
        output = os.path.join(self.tmpdir, "empty.db")
        stats = convert_snapshot(self.empty_path, output)
        self.assertEqual(stats["segments"], 0)
        self.assertEqual(stats["traces"], 0)
        self.assertTrue(os.path.exists(output))

    def test_single_segment(self):
        output = os.path.join(self.tmpdir, "single.db")
        stats = convert_snapshot(self.single_path, output)
        self.assertEqual(stats["segments"], 1)
        self.assertEqual(stats["blocks"], 1)
        self.assertEqual(stats["traces"], 1)
        self.assertEqual(stats["devices"], 1)
        self.assertGreater(stats["call_stacks"], 0)

    def test_multi_device(self):
        output = os.path.join(self.tmpdir, "multi.db")
        stats = convert_snapshot(self.multi_device_path, output)
        self.assertEqual(stats["segments"], 2)
        self.assertEqual(stats["devices"], 2)

    def test_no_indexes(self):
        output = os.path.join(self.tmpdir, "no_idx.db")
        stats = convert_snapshot(self.single_path, output, create_indexes=False)
        self.assertEqual(stats["time_indexes_s"], 0)

    def test_auto_output_path(self):
        stats = convert_snapshot(self.single_path)
        expected = os.path.splitext(self.single_path)[0] + ".db"
        self.assertEqual(stats["output_path"], expected)
        if os.path.exists(expected):
            os.remove(expected)

    def test_db_content(self):
        output = os.path.join(self.tmpdir, "content.db")
        convert_snapshot(self.single_path, output)

        conn = sqlite3.connect(output)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM segments")
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("SELECT total_size, allocated_size, segment_type FROM segments")
        row = cursor.fetchone()
        self.assertEqual(row[0], 1024 * 1024)
        self.assertEqual(row[1], 512 * 1024)
        self.assertEqual(row[2], "small")

        cursor.execute("SELECT COUNT(*) FROM blocks")
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("SELECT COUNT(*) FROM traces")
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("SELECT action FROM traces")
        self.assertEqual(cursor.fetchone()[0], "segment_alloc")

        conn.close()


if __name__ == "__main__":
    unittest.main()

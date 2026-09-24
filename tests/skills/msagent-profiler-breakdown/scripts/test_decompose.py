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
decompose.py 单元测试
"""

import argparse
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

from db_query import OP_TYPE
from decompose import (
    FRAMEWORK_SCENARIO,
    LEVEL_STAGE,
    LEVEL_STEP,
    build_findings,
    decompose_stages,
    decompose_steps,
    detect_framework,
    get_scenario_rules,
    load_scenario_registry,
    main,
    merge_rules,
    pct,
    query_scope_events,
    render_html,
    session_span,
    summarize,
    write_outputs,
    DEFAULT_SCENARIOS_DIR,
)


def _make_db(path, session=None, strings=None, pytorch=None, cann=None, mstx=None):
    """构造 decompose 用最小 profiling DB（对齐 references/db-usage.md 字段约定）。"""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS STRING_IDS (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS MSTX_EVENTS (startNs INTEGER, endNs INTEGER, message INTEGER);
        CREATE TABLE IF NOT EXISTS PYTORCH_API (name INTEGER, type INTEGER, startNs TEXT, endNs TEXT);
        CREATE TABLE IF NOT EXISTS CANN_API (name INTEGER, type INTEGER, startNs TEXT, endNs TEXT);
        CREATE TABLE IF NOT EXISTS SESSION_TIME_INFO (startTimeNs TEXT, endTimeNs TEXT);
    """)
    for value, sid in (strings or {}).items():
        conn.execute("INSERT INTO STRING_IDS (id, value) VALUES (?, ?)", (sid, value))
    if session:
        conn.execute("INSERT INTO SESSION_TIME_INFO (startTimeNs, endTimeNs) VALUES (?, ?)", session)
    for name, typ, start, end in pytorch or []:
        conn.execute(
            "INSERT INTO PYTORCH_API (name, type, startNs, endNs) VALUES (?, ?, ?, ?)", (name, typ, start, end)
        )
    for name, typ, start, end in cann or []:
        conn.execute("INSERT INTO CANN_API (name, type, startNs, endNs) VALUES (?, ?, ?, ?)", (name, typ, start, end))
    for start, end, msg in mstx or []:
        conn.execute("INSERT INTO MSTX_EVENTS (startNs, endNs, message) VALUES (?, ?, ?)", (start, end, msg))
    conn.commit()
    conn.close()


VLLM_STRINGS = {
    "vllm::dsa_forward": 1,
    "forward": 2,
    "prepare input": 3,
    "post process": 4,
    "sample_token": 5,
    "draft_token": 6,
}


def _vllm_inference_db(path, forward_events=None, session=("0", "1000000")):
    """构造 vLLM 推理样式的 DB：PYTORCH_API 五 stage scope + SESSION_TIME_INFO。"""
    forward_events = forward_events or [
        (2, OP_TYPE, "1000", "2000"),
        (2, OP_TYPE, "3000", "4000"),
    ]
    return _make_db(
        path,
        session=session,
        strings=VLLM_STRINGS,
        pytorch=[
            (3, OP_TYPE, "500", "900"),  # prepare input
            *forward_events,
            (4, OP_TYPE, "2000", "2100"),  # post process
            (5, OP_TYPE, "2100", "2200"),  # sample_token
        ],
    )


def _verl_rl_db(path):
    """构造 verl RL 样式的 DB：MSTX_EVENTS 低开销打点。"""
    return _make_db(
        path,
        session=("0", "1000000"),
        strings={
            "gen": 10,
            "reward": 11,
            "actor_update": 12,
            "old_log_prob": 13,
            "ref": 14,
        },
        mstx=[
            (1000, 2000, 10),  # gen
            (3000, 4000, 10),  # gen
            (4000, 5000, 11),  # reward
            (5000, 6000, 12),  # actor_update
            (6000, 7000, 13),  # old_log_prob
            (7000, 8000, 14),  # ref
        ],
    )


class TestDetectFramework(unittest.TestCase):
    def test_vllm_marker(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, strings={"vllm::dsa_forward": 1})
        conn = sqlite3.connect(path)
        try:
            self.assertEqual(detect_framework(conn.cursor()), "vllm")
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_verl_marker(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, strings={"actor_update": 1})
        conn = sqlite3.connect(path)
        try:
            self.assertEqual(detect_framework(conn.cursor()), "verl")
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_vllm_takes_priority(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, strings={"vllm::x": 1, "actor_update": 2})
        conn = sqlite3.connect(path)
        try:
            self.assertEqual(detect_framework(conn.cursor()), "vllm")
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_none(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, strings={"forward": 1})
        conn = sqlite3.connect(path)
        try:
            self.assertIsNone(detect_framework(conn.cursor()))
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestLoadScenarioRegistry(unittest.TestCase):
    def test_real_registry(self):
        reg = load_scenario_registry(DEFAULT_SCENARIOS_DIR)
        for fw in ("vllm", "sglang", "verl", "slime", "megatron", "mindspeed", "fsdp"):
            self.assertIn(fw, reg)
        self.assertEqual(reg["vllm"]["scenario"], "inference")
        self.assertTrue(reg["vllm"]["path"].endswith("vllm.json"))

    def test_missing_dir_returns_empty(self):
        reg = load_scenario_registry(os.path.join(tempfile.mkdtemp(), "nope"))
        self.assertEqual(reg, {})

    def test_invalid_json_skipped(self):
        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "inference"))
            with open(os.path.join(tmpdir, "inference", "bad.json"), "w", encoding="utf-8") as f:
                f.write("{not json")
            with open(os.path.join(tmpdir, "inference", "ok.json"), "w", encoding="utf-8") as f:
                json.dump({"framework": "okfw", "scenario": "inference"}, f)
            out = StringIO()
            with redirect_stdout(out):
                reg = load_scenario_registry(tmpdir)
            self.assertIn("okfw", reg)
            self.assertNotIn("bad", reg)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMergeRules(unittest.TestCase):
    def test_child_overrides_base(self):
        base = {"framework": "megatron", "label": "M"}
        child = {"framework": "mindspeed", "skip": None, "empty": []}
        merged = merge_rules(base, child)
        self.assertEqual(merged["framework"], "mindspeed")
        self.assertEqual(merged["label"], "M")  # 未覆盖字段保留
        self.assertNotIn("skip", merged)  # None/[]/{} 不写入
        self.assertNotIn("empty", merged)

    def test_stages_and_anchor_union(self):
        # merge_rules 仅对顶层 stages / anchor 键做并集去重
        base = {"stages": [{"name": "a"}, {"name": "b"}], "anchor": ["x"]}
        child = {"stages": [{"name": "b"}, {"name": "c"}], "anchor": ["x", "y"]}
        merged = merge_rules(base, child)
        self.assertEqual([s["name"] for s in merged["stages"]], ["a", "b", "c"])
        self.assertEqual(merged["anchor"], ["x", "y"])

    def test_nested_structure_inherits_and_unions(self):
        # 子缺省键继承父（mindspeed/fsdp 式占位），stages 递归并集
        base = {
            "framework": "megatron",
            "stage_decompose": {
                "backing": "PYTORCH_API",
                "note": "megatron",
                "stages": [{"name": "fwd"}, {"name": "bwd"}],
            },
            "step_decompose": {"backing": "PYTORCH_API", "anchor": ["optimizer"]},
        }
        child = {
            "framework": "mindspeed",
            "stage_decompose": {
                "note": "mindspeed",
                "stages": [{"name": "bwd"}, {"name": "optimizer_step"}],
            },
            "step_decompose": {},
        }
        merged = merge_rules(base, child)
        sd = merged["stage_decompose"]
        self.assertEqual(sd["backing"], "PYTORCH_API")  # 子未提供 → 继承父
        self.assertEqual(sd["note"], "mindspeed")  # 子覆盖
        self.assertEqual([s["name"] for s in sd["stages"]], ["fwd", "bwd", "optimizer_step"])  # 并集去重
        self.assertEqual(merged["step_decompose"]["anchor"], ["optimizer"])  # 子 {} 缺省继承父

    def test_none_base(self):
        merged = merge_rules(None, {"a": 1})
        self.assertEqual(merged, {"a": 1})


class TestGetScenarioRules(unittest.TestCase):
    def setUp(self):
        self.reg = load_scenario_registry(DEFAULT_SCENARIOS_DIR)

    def test_vllm_inference_scenario_resource(self):
        rules, source, path = get_scenario_rules(self.reg, "vllm", "inference")
        self.assertEqual(source, "scenario-resource")
        self.assertTrue(rules["stage_decompose"]["stages"])
        self.assertEqual(rules["step_decompose"]["anchor"], ["forward"])
        self.assertTrue(path.endswith("vllm.json"))

    def test_verl_rl_scenario_resource(self):
        rules, source, path = get_scenario_rules(self.reg, "verl", "rl")
        self.assertEqual(source, "scenario-resource")
        self.assertEqual(rules["stage_decompose"]["backing"], "MSTX_EVENTS")
        self.assertTrue(path.endswith("verl.json"))

    def test_sglang_inference_empty_stages_fallback_vllm(self):
        # 有资源文件但 stages 未填充的推理框架 → 按契约回退 vllm 兜底（rules/path 同源）
        rules, source, path = get_scenario_rules(self.reg, "sglang", "inference")
        self.assertEqual(source, "fallback-vllm")
        self.assertTrue(rules["stage_decompose"]["stages"])  # 回退到 vllm 五 stage
        self.assertTrue(path.endswith("vllm.json"))

    def test_unknown_inference_fallback_vllm(self):
        rules, source, path = get_scenario_rules(self.reg, "unknown_fw", "inference")
        self.assertEqual(source, "fallback-vllm")
        self.assertIn("forward", {s["name"] for s in rules["stage_decompose"]["stages"]})
        # 规则内容与路径同源：均指向实际生效的 vllm.json
        self.assertTrue(path.endswith("vllm.json"))

    def test_train_placeholder(self):
        for fw in ("megatron", "mindspeed", "fsdp"):
            rules, source, path = get_scenario_rules(self.reg, fw, "train")
            self.assertEqual(source, "placeholder")
            self.assertEqual(rules, {})
            self.assertTrue(path.endswith("%s.json" % fw))

    def test_slime_rl_placeholder(self):
        rules, source, path = get_scenario_rules(self.reg, "slime", "rl")
        self.assertEqual(source, "placeholder")
        self.assertEqual(rules, {})
        self.assertTrue(path.endswith("slime.json"))


class TestQueryScopeEvents(unittest.TestCase):
    def test_pytorch_backing(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _vllm_inference_db(path)
        conn = sqlite3.connect(path)
        try:
            events = query_scope_events(conn.cursor(), ["forward"], "PYTORCH_API")
            self.assertEqual(len(events), 2)
            self.assertEqual([e["start"] for e in events], [1000, 3000])
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_mstx_backing(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _verl_rl_db(path)
        conn = sqlite3.connect(path)
        try:
            events = query_scope_events(conn.cursor(), ["gen", "reward"], "MSTX_EVENTS")
            self.assertEqual(len(events), 3)
            self.assertEqual(events[0]["name"], "gen")
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestSessionSpan(unittest.TestCase):
    def test_session(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, session=("123", "456"), strings={"forward": 1})
        conn = sqlite3.connect(path)
        try:
            self.assertEqual(session_span(conn.cursor()), (123, 456))
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_missing_table(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, strings={"forward": 1})
        conn = sqlite3.connect(path)
        try:
            self.assertIsNone(session_span(conn.cursor()))
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_non_operational_error_propagates(self):
        # 表缺失（OperationalError）回退；其余非预期错误不再静默吞掉
        class _BadCursor:
            def execute(self, *args, **kwargs):
                raise sqlite3.DatabaseError("db corrupted")

        with self.assertRaises(sqlite3.DatabaseError):
            session_span(_BadCursor())


class TestSummarize(unittest.TestCase):
    def test_empty(self):
        st = summarize([])
        self.assertEqual(st["count"], 0)
        self.assertEqual(st["total"], 0)
        self.assertIsNone(st["mean"])

    def test_stats(self):
        events = [{"start": 0, "end": 10}, {"start": 20, "end": 40}]
        st = summarize(events)
        self.assertEqual(st["count"], 2)
        self.assertEqual(st["total"], 30)
        self.assertEqual(st["mean"], 15)
        self.assertEqual(st["max"], 20)
        self.assertEqual(st["min"], 10)
        self.assertEqual(st["first"], events[0])


class TestPct(unittest.TestCase):
    def test_nearest_quantile(self):
        self.assertEqual(pct([1, 2, 3, 4, 5], 0.5), 3)
        self.assertEqual(pct([1, 2, 3, 4, 5], 0.0), 1)
        self.assertEqual(pct([1, 2, 3, 4, 5], 1.0), 5)

    def test_empty(self):
        self.assertIsNone(pct([], 0.5))


class TestDecomposeStages(unittest.TestCase):
    def _rules(self, fw, task):
        reg = load_scenario_registry(DEFAULT_SCENARIOS_DIR)
        rules, _, _ = get_scenario_rules(reg, fw, task)
        return rules

    def test_inference_stages(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _vllm_inference_db(path)
        conn = sqlite3.connect(path)
        try:
            children, all_events = decompose_stages(
                conn.cursor(), "inference", "vllm", self._rules("vllm", "inference"), None
            )
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

        names = [c["name"] for c in children]
        self.assertEqual(names, ["prepare_input", "forward", "post_process", "sample_token", "draft_token"])
        fwd = children[1]
        self.assertEqual(fwd["level"], LEVEL_STAGE)
        self.assertEqual(fwd["count"], 2)
        self.assertEqual(fwd["wall_time"], 2000)  # (2000-1000)+(4000-3000)
        self.assertEqual(fwd["confidence"]["level"], "high")
        self.assertEqual(fwd["first"]["start"], 1000)
        # 无数据阶段：count=0 且无 first
        self.assertEqual(children[4]["count"], 0)
        self.assertIsNone(children[4]["first"])

    def test_inference_stages_inspect_index(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _vllm_inference_db(path)
        conn = sqlite3.connect(path)
        try:
            children, _ = decompose_stages(
                conn.cursor(), "inference", "vllm", self._rules("vllm", "inference"), inspect_index=0
            )
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        fwd = children[1]
        self.assertEqual(fwd["inspect_index"], 0)
        self.assertEqual(fwd["inspect"]["start"], 1000)

    def test_inference_stages_inspect_out_of_range(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _vllm_inference_db(path)
        conn = sqlite3.connect(path)
        try:
            children, _ = decompose_stages(
                conn.cursor(), "inference", "vllm", self._rules("vllm", "inference"), inspect_index=99
            )
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertIsNone(children[1]["inspect"])

    def test_rl_stages_mstx(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _verl_rl_db(path)
        conn = sqlite3.connect(path)
        try:
            children, _ = decompose_stages(conn.cursor(), "rl", "verl", self._rules("verl", "rl"), None)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        names = [c["name"] for c in children]
        self.assertIn("rollout", names)
        self.assertIn("reward", names)
        rollout = children[names.index("rollout")]
        self.assertEqual(rollout["count"], 2)
        self.assertEqual(rollout["wall_time"], 2000)

    def test_train_placeholder(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _make_db(path, strings={"forward": 1})
        conn = sqlite3.connect(path)
        try:
            children, all_events = decompose_stages(conn.cursor(), "train", "fsdp", {}, None)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(children, [])
        self.assertEqual(all_events, [])


class TestDecomposeSteps(unittest.TestCase):
    def _rules(self):
        reg = load_scenario_registry(DEFAULT_SCENARIOS_DIR)
        rules, _, _ = get_scenario_rules(reg, "vllm", "inference")
        return rules

    def test_warmup_and_steady_stats(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "100"),  # step0: warmup graph_capture
            (2, OP_TYPE, "100", "200"),  # step1: warmup compile
            (2, OP_TYPE, "200", "300"),  # steady
            (2, OP_TYPE, "300", "400"),  # steady
            (2, OP_TYPE, "400", "500"),  # steady
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._rules(), None, 3.0, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

        self.assertEqual(steps["level"], LEVEL_STEP)
        self.assertEqual(steps["num_steps"], 5)
        self.assertEqual(steps["num_steady"], 3)
        self.assertEqual(steps["warmup_indices"], [0, 1])
        self.assertEqual(steps["prefill_indices"], [])
        self.assertEqual(steps["stats"]["mean"], 100)
        self.assertEqual(steps["stats"]["p50"], 100)
        self.assertEqual(steps["steps"][0]["warmup_reason"], "graph_capture")
        self.assertEqual(steps["steps"][1]["warmup_reason"], "compile")

    def test_threshold_warmup(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "10"),
            (2, OP_TYPE, "10", "20"),
            (2, OP_TYPE, "20", "30"),
            (2, OP_TYPE, "30", "40"),
            (2, OP_TYPE, "40", "140"),  # wall=100 > 3*median(10)
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._rules(), None, 3.0, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(steps["warmup_indices"], [0, 1, 4])
        self.assertEqual(steps["steps"][4]["warmup_reason"], "threshold")
        self.assertEqual(steps["num_steady"], 2)

    def test_skip_first_step_marks_prefill(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "100"),
            (2, OP_TYPE, "100", "200"),
            (2, OP_TYPE, "200", "300"),
            (2, OP_TYPE, "300", "400"),
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._rules(), None, 3.0, True, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertTrue(steps["steps"][0]["is_prefill"])
        self.assertEqual(steps["prefill_indices"], [0])

    def test_cli_anchor_overrides(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _vllm_inference_db(path)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", {}, "prepare input", 3.0, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        # 锚点命中 prepare input（1 次）
        self.assertEqual(steps["num_steps"], 1)
        self.assertEqual(steps["anchor"], ["prepare input"])

    def test_no_anchor_returns_none(self):
        conn = sqlite3.connect(":memory:")
        try:
            steps = decompose_steps(conn.cursor(), "verl", {}, None, 3.0, False, 2)
            self.assertIsNone(steps)
        finally:
            conn.close()

    def test_anchor_no_match_returns_none(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        _vllm_inference_db(path)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", {}, "nonexistent_scope", 3.0, False, 2)
            self.assertIsNone(steps)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def _warmup_rules(self, factor=None):
        """构造可配置 warmup_factor 的 step 规则（锚点 forward）。"""
        rules = {"step_decompose": {"backing": "PYTORCH_API", "anchor": ["forward"], "position_warmup": 2}}
        if factor is not None:
            rules["step_decompose"]["warmup_factor"] = factor
        return rules

    def test_resource_warmup_factor_consumed(self):
        # 资源 warmup_factor=1.5：wall=30 > 1.5*median(10)=15 → 第 4 步判热身；
        # 默认 3.0 时 30 > 30 不成立（对比 test_default_warmup_factor_when_resource_missing）
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "10"),
            (2, OP_TYPE, "10", "20"),
            (2, OP_TYPE, "20", "30"),
            (2, OP_TYPE, "30", "40"),
            (2, OP_TYPE, "40", "70"),  # wall=30
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._warmup_rules(1.5), None, None, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(steps["warmup_indices"], [0, 1, 4])
        self.assertEqual(steps["steps"][4]["warmup_reason"], "threshold")

    def test_cli_warmup_factor_overrides_resource(self):
        # 同一组事件 + 资源 1.5，但 CLI 显式 3.0 → 第 4 步不判热身
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "10"),
            (2, OP_TYPE, "10", "20"),
            (2, OP_TYPE, "20", "30"),
            (2, OP_TYPE, "30", "40"),
            (2, OP_TYPE, "40", "70"),
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._warmup_rules(1.5), None, 3.0, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(steps["warmup_indices"], [0, 1])

    def test_default_warmup_factor_when_resource_missing(self):
        # 资源无 warmup_factor 且 CLI 未指定 → 默认 3.0（wall=30 不判热身）
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "10"),
            (2, OP_TYPE, "10", "20"),
            (2, OP_TYPE, "20", "30"),
            (2, OP_TYPE, "30", "40"),
            (2, OP_TYPE, "40", "70"),
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._warmup_rules(), None, None, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(steps["warmup_indices"], [0, 1])

    def test_position_warmup_beyond_two_has_reason(self):
        # position_warmup=3 时第 3 步（index 2）is_warmup 且 reason="position"，
        # 不再因 elif 链缺失而显示为空
        rules = {"step_decompose": {"backing": "PYTORCH_API", "anchor": ["forward"], "position_warmup": 3}}
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "100"),
            (2, OP_TYPE, "100", "200"),
            (2, OP_TYPE, "200", "300"),
            (2, OP_TYPE, "300", "400"),
            (2, OP_TYPE, "400", "500"),
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", rules, None, 3.0, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(steps["warmup_indices"], [0, 1, 2])
        self.assertEqual(steps["steps"][0]["warmup_reason"], "graph_capture")
        self.assertEqual(steps["steps"][1]["warmup_reason"], "compile")
        self.assertEqual(steps["steps"][2]["warmup_reason"], "position")

    def test_single_steady_step_cv_is_zero(self):
        # 单步稳态 variance=0.0 时显式 cv=0.0（统一 HTML/JSON/控制台口径，而非 None）
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        events = [
            (2, OP_TYPE, "0", "100"),
            (2, OP_TYPE, "100", "200"),
            (2, OP_TYPE, "200", "300"),
        ]
        _vllm_inference_db(path, forward_events=events)
        conn = sqlite3.connect(path)
        try:
            steps = decompose_steps(conn.cursor(), "vllm", self._warmup_rules(), None, None, False, 2)
        finally:
            conn.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.assertEqual(steps["num_steady"], 1)
        self.assertEqual(steps["stats"]["variance"], 0.0)
        self.assertEqual(steps["stats"]["cv"], 0.0)


class TestBuildFindings(unittest.TestCase):
    def test_slowest_stage(self):
        stage = {
            "children": [
                {"name": "forward", "self_time": 500, "confidence": {"level": "high"}},
                {"name": "prepare_input", "self_time": 100, "confidence": {"level": "high"}},
            ],
            "wall_time": 1000,
            "conservation_gap": 0,
            "task_type_source": "cli",
        }
        steps = {"steps": [], "stats": {}}
        findings = build_findings(stage, steps)
        f1 = next(f for f in findings if "最耗时阶段" in f["问题"])
        self.assertIn("forward", f1["证据"])
        self.assertEqual(f1["置信度"], "high")

    def test_step_outlier(self):
        stage = {"children": [], "wall_time": 1000, "conservation_gap": 0, "task_type_source": "cli"}
        steps = {
            "steps": [
                {"index": 0, "wall_time": 100, "is_warmup": False, "is_prefill": False},
                {"index": 1, "wall_time": 100, "is_warmup": False, "is_prefill": False},
                {"index": 2, "wall_time": 10000, "is_warmup": False, "is_prefill": False},
            ],
            "stats": {"p50": 100, "variance": 0.0, "cv": 0.0},
        }
        findings = build_findings(stage, steps)
        f2 = next(f for f in findings if "step outlier" in f["问题"])
        self.assertIn("step#2", f2["证据"])

    def test_drift(self):
        stage = {"children": [], "wall_time": 1000, "conservation_gap": 0, "task_type_source": "cli"}
        steps = {"steps": [], "stats": {"p50": 100, "variance": 100.0, "cv": 0.5}}
        findings = build_findings(stage, steps)
        f3 = next(f for f in findings if "波动偏大" in f["问题"])
        self.assertIn("CV", f3["证据"])

    def test_conservation_gap(self):
        stage = {"children": [], "wall_time": 1000, "conservation_gap": 200, "task_type_source": "cli"}
        findings = build_findings(stage, None)
        f4 = next(f for f in findings if "守恒缺口" in f["问题"])
        self.assertIn("20.00%", f4["问题"])
        self.assertEqual(f4["置信度"], "medium")

    def test_low_confidence(self):
        stage = {
            "children": [
                {"name": "s1", "self_time": 1, "confidence": {"level": "low", "detail": "无打点"}},
            ],
            "wall_time": 1000,
            "conservation_gap": 0,
            "task_type_source": "default",
        }
        findings = build_findings(stage, None)
        f5 = next(f for f in findings if "低置信度" in f["问题"])
        self.assertIn("无打点", f5["证据"])

    def test_no_findings(self):
        stage = {
            "children": [],
            "wall_time": 0,
            "conservation_gap": 0,
            "task_type_source": "cli",
            "framework_source": "cli",
        }
        self.assertEqual(build_findings(stage, None), [])


class TestRenderHtml(unittest.TestCase):
    def test_basic_structure(self):
        stage = {
            "level": "端到端",
            "name": "端到端（会话）",
            "task_type": "inference",
            "task_type_source": "cli",
            "framework": "vllm",
            "framework_source": "cli",
            "scenario_rules_source": "scenario-resource",
            "scenario_rules_path": "",
            "wall_time": 1000,
            "self_time": 500,
            "overlap": 500,
            "conservation_gap": 500,
            "children": [
                {
                    "name": "forward",
                    "count": 2,
                    "wall_time": 500,
                    "self_time": 500,
                    "wall_mean": 250,
                    "wall_max": 300,
                    "wall_min": 200,
                    "first": {"wall": 200},
                    "confidence": {"level": "high", "detail": "x"},
                },
            ],
        }
        steps = {"steps": [], "stats": {}, "num_steady": 0}
        html_str = render_html(stage, steps, [], None)
        self.assertIn("<h1>性能拆解报告</h1>", html_str)
        self.assertIn("一句话结论", html_str)
        self.assertIn("forward", html_str)

    def test_no_steps(self):
        stage = {
            "children": [],
            "wall_time": 0,
            "self_time": 0,
            "task_type": "inference",
            "framework": "vllm",
            "task_type_source": "cli",
            "framework_source": "cli",
            "scenario_rules_source": "x",
            "scenario_rules_path": "",
        }
        html_str = render_html(stage, None, [], None)
        self.assertIn("未检出显著阶段/step 热点", html_str)


class TestWriteOutputs(unittest.TestCase):
    def test_creates_all_outputs(self):
        tmpdir = tempfile.mkdtemp()
        try:
            args = argparse.Namespace(output_dir=os.path.join(tmpdir, "out"), prefix="test")
            tree = {"level": "端到端", "name": "会话", "children": []}
            steps = {"steps": [], "num_steady": 0}
            html_str = "<html></html>"
            html_path = write_outputs(args, tree, steps, [], html_str)
            out_dir = args.output_dir
            for name in (
                "breakdown.json",
                "stage.json",
                "test.html",
                "test_findings.json",
                "test_detail.csv",
                "test_handoff.csv",
            ):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, name)), name)
            self.assertEqual(html_path, os.path.join(out_dir, "test.html"))
            with open(os.path.join(out_dir, "breakdown.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["name"], "会话")
            with open(os.path.join(out_dir, "test_findings.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"findings": []})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_main_vllm_full_run(self):
        db = os.path.join(self.tmpdir, "vllm.db")
        _vllm_inference_db(db)
        out_dir = os.path.join(self.tmpdir, "out")
        out = StringIO()
        with redirect_stdout(out):
            rc = main(
                [
                    "--db",
                    db,
                    "--framework",
                    "vllm",
                    "--task-type",
                    "inference",
                    "--scenarios-dir",
                    DEFAULT_SCENARIOS_DIR,
                    "--output-dir",
                    out_dir,
                    "--prefix",
                    "brk",
                    "--skip-first-step",
                ]
            )
        self.assertEqual(rc, 0)
        with open(os.path.join(out_dir, "breakdown.json"), encoding="utf-8") as f:
            tree = json.load(f)
        self.assertEqual(tree["framework"], "vllm")
        self.assertEqual(tree["task_type"], "inference")
        levels = {c["level"] for c in tree["children"]}
        self.assertEqual(levels, {"阶段", "单次执行"})
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "brk.html")))

    def test_main_framework_detected(self):
        db = os.path.join(self.tmpdir, "detect.db")
        _vllm_inference_db(db)
        out_dir = os.path.join(self.tmpdir, "out2")
        with redirect_stdout(StringIO()):
            rc = main(
                [
                    "--db",
                    db,
                    "--scenarios-dir",
                    DEFAULT_SCENARIOS_DIR,
                    "--output-dir",
                    out_dir,
                    "--prefix",
                    "d",
                ]
            )
        self.assertEqual(rc, 0)
        with open(os.path.join(out_dir, "stage.json"), encoding="utf-8") as f:
            tree = json.load(f)
        self.assertEqual(tree["framework_source"], "detected")
        self.assertEqual(tree["task_type_source"], "detected")

    def test_main_train_placeholder(self):
        db = os.path.join(self.tmpdir, "train.db")
        _make_db(db, session=("0", "1000"), strings={"some_train_kernel": 1})
        out_dir = os.path.join(self.tmpdir, "out3")
        with redirect_stdout(StringIO()):
            rc = main(
                [
                    "--db",
                    db,
                    "--framework",
                    "megatron",
                    "--task-type",
                    "train",
                    "--scenarios-dir",
                    DEFAULT_SCENARIOS_DIR,
                    "--output-dir",
                    out_dir,
                    "--prefix",
                    "t",
                ]
            )
        self.assertEqual(rc, 0)
        with open(os.path.join(out_dir, "breakdown.json"), encoding="utf-8") as f:
            tree = json.load(f)
        self.assertEqual(tree["children"], [])
        self.assertEqual(tree["scenario_rules_source"], "placeholder")


class TestFrameworkScenarioMap(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(FRAMEWORK_SCENARIO["vllm"], "inference")
        self.assertEqual(FRAMEWORK_SCENARIO["sglang"], "inference")
        self.assertEqual(FRAMEWORK_SCENARIO["verl"], "rl")
        self.assertEqual(FRAMEWORK_SCENARIO["slime"], "rl")
        self.assertEqual(FRAMEWORK_SCENARIO["megatron"], "train")
        self.assertIsNone(FRAMEWORK_SCENARIO.get("unknown"))


if __name__ == "__main__":
    unittest.main()

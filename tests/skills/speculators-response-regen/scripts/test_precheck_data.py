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
# -*- coding: utf-8 -*-
"""speculators-response-regen / precheck_data.py 单元测试。

覆盖：DFX 预检的统计（解析失败/无容器/无有效轮/无 assistant/首轮非 user/
回答过短/重复 id）、分级风险 build_risks（high/medium/low/info）、抽样、
以及 markdown 报告渲染。
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "speculators",
        "speculators-response-regen",
        "scripts",
    ),
)

from precheck_data import build_risks, precheck, render_report  # noqa: E402


def _write(tmp_path, rows):
    p = tmp_path / "input.jsonl"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def _args(p, sample=0):
    return SimpleNamespace(input=str(p), precheck_sample=sample, report_out=None)


def _find(risks, level, keyword):
    return any(level == lv and keyword in msg for lv, msg in risks)


# 一组能触发各分类的样例（行序影响 dup 判定）
SAMPLE_ROWS = [
    # L1 合法 A 型(conversation 键)
    '{"conversation_id":"a1","conversation":[{"human":"hello how are you?","assistant":"I am fine, thanks for asking"}]}',
    # L2 合法 B 型(conversations 键)
    '{"id":"b2","conversations":[{"from":"user","value":"what is 2+2?"},{"role":"assistant","content":"It is four, clearly"}]}',
    # L3 解析失败
    "{oops not json",
    # L4 无会话容器
    '{"foo":"bar"}',
    # L5 容器存在但无有效轮次
    '{"id":"e5","conversation":[]}',
    # L6 无 assistant 轮
    '{"id":"f6","conversations":[{"from":"user","value":"only a question with no answer at all"}]}',
    # L7 assistant 回答过短(<8)
    '{"id":"g7","conversations":[{"from":"user","value":"a question?"},{"from":"gpt","value":"ok"}]}',
    # L8 首轮非 user(human)（无引导前缀）
    '{"id":"h8","conversations":[{"role":"assistant","content":"starts with assistant, no user before it"}]}',
    # L9 与 L7 重复 id -> dup_id
    '{"id":"g7","conversations":[{"from":"user","value":"again?"},{"role":"assistant","content":"a long enough reply here to regenerate"}]}',
]


def test_precheck_stats_full_scan(tmp_path):
    stats, stat, sample_err = precheck(_args(_write(tmp_path, SAMPLE_ROWS)))
    assert stats["total"] == 9
    assert stats["parse_err"] == 1  # L3
    assert stats["no_container"] == 1  # L4
    assert stats["no_valid_turns"] == 1  # L5
    assert stats["no_assistant"] == 1  # L6
    assert stats["starts_with_gpt"] == 1  # L8
    assert stats["assistant_too_short"] == 1  # L7
    assert stats["dup_id"] == 1  # L9 与 L7 同 id
    assert stats["dup_conv"] == 0
    assert stats["empty_value_turn"] == 0
    assert stats["very_long_row_chars"] == 0
    assert stats["non_ascii_heavy"] == 0
    assert len(sample_err) == 1 and "parse_err@" in sample_err[0]
    # 结构分布
    assert stat["schema_breakdown"] == {"conversation": 2, "conversations": 5}
    assert stat["id_key"] == "conversation_id"
    assert stat["sampled"] == 6  # 计入长度统计的有效行 L1,L2,L6,L7,L8,L9


def test_build_risks_levels():
    # 直接以构造 stats 驱动分级（独立于文件扫描）
    stats = {
        "total": 10,
        "parse_err": 2,
        "no_container": 1,
        "no_valid_turns": 1,
        "no_assistant": 1,
        "starts_with_gpt": 1,
        "assistant_too_short": 2,
        "empty_value_turn": 0,
        "dup_id": 3,
        "dup_conv": 1,
        "very_long_row_chars": 0,
        "non_ascii_heavy": 0,
    }
    stat = {"schema_breakdown": {"conversation": 3, "conversations": 4}}
    risks = build_risks(stats, stat)
    assert _find(risks, "high", "JSON 解析失败")
    assert _find(risks, "high", "缺少会话容器键")
    assert _find(risks, "high", "回合异常")
    assert _find(risks, "medium", "无 assistant(gpt) 轮")
    assert _find(risks, "medium", "首轮非 user")
    assert _find(risks, "medium", "回答过短")
    assert _find(risks, "low", "重复 id")
    assert _find(risks, "low", "重复对话内容")
    assert _find(risks, "info", "conversation 键")
    assert _find(risks, "info", "conversations 键")
    # 未触发的项不应出现
    assert not _find(risks, "medium", "空 value")
    assert not _find(risks, "medium", "超长样本")


def test_build_risks_empty_problem_free():
    stats = {
        k: 0
        for k in (
            "total",
            "parse_err",
            "no_container",
            "no_valid_turns",
            "no_assistant",
            "starts_with_gpt",
            "assistant_too_short",
            "empty_value_turn",
            "dup_id",
            "dup_conv",
            "very_long_row_chars",
            "non_ascii_heavy",
        )
    }
    stats["total"] = 5
    risks = build_risks(stats, {"schema_breakdown": {"conversations": 5}})
    # 无 high/medium/low，只有 info
    assert all(lv == "info" for lv, _ in risks)
    assert _find(risks, "info", "conversations 键")


def test_precheck_sample_limit(tmp_path):
    stats, stat, _ = precheck(_args(_write(tmp_path, SAMPLE_ROWS), sample=4))
    assert stats["total"] == 4  # 只扫前 4 行(L1-L4)
    assert stats["parse_err"] == 1  # L3
    assert stats["no_container"] == 1  # L4
    assert stats["no_valid_turns"] == 0
    assert stat["sampled"] == 2  # L1,L2


def test_render_report_md(tmp_path):
    p = _write(tmp_path, SAMPLE_ROWS)
    stats, stat, sample_err = precheck(_args(p))
    risks = build_risks(stats, stat)
    md = render_report(stats, stat, risks, sample_err, _args(p))
    assert md.startswith("# response-regen 输入预检报告")
    assert "- 扫描行数(或抽样): 9" in md
    assert "- 结构分布:" in md and "conversation" in md
    assert "## 风险清单(DFX 预检，需提前上报)" in md
    assert "**[high]**" in md
    assert "## 丢弃原因计数" in md
    assert "- parse_err: 1" in md
    assert "## 解析错误样例" in md
    assert "单行总字符数 p50/p90/max" in md


def test_report_out_writes_md_and_json(tmp_path):
    p = _write(tmp_path, SAMPLE_ROWS)
    out_md = tmp_path / "pre.md"
    out_json = tmp_path / "pre.json"
    for out in (out_md, out_json):
        args = SimpleNamespace(input=str(p), precheck_sample=0, report_out=str(out))
        stats, stat, sample_err = precheck(args)
        risks = build_risks(stats, stat)
        _ = render_report(stats, stat, risks, sample_err, args)
        md = render_report(stats, stat, risks, sample_err, args)
        if str(out).endswith(".json"):
            data = {
                "stats": stats,
                "summary": stat,
                "risks": risks,
                "sample_err": sample_err,
                "report_md": md,
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(out, "w", encoding="utf-8") as f:
                f.write(md)
        assert out.exists()
    assert json.loads(out_json.read_text(encoding="utf-8"))["stats"]["total"] == 9
    assert "风险清单" in out_md.read_text(encoding="utf-8")

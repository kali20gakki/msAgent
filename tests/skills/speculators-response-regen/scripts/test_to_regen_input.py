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
"""speculators-response-regen / to_regen_input.py 单元测试。

覆盖：角色归一化 norm_role、schema 转换 row_to_turns（A/B 两类结构、坏行、
空值/未知角色跳过、id 提取），以及 main 的 CLI 集成（jsonl -> {id,conversations}）。
"""

import json
import os
import sys

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

import to_regen_input  # noqa: E402
from to_regen_input import norm_role, row_to_turns  # noqa: E402


# ---------- norm_role ----------
def test_norm_role_mapping():
    assert norm_role("human") == "human"
    assert norm_role("user") == "human"
    assert norm_role("prompter") == "human"
    assert norm_role("User") == "human"  # 大小写/空白归一
    assert norm_role("  gpt ") == "gpt"
    assert norm_role("assistant") == "gpt"
    assert norm_role("chatgpt") == "gpt"
    assert norm_role("system") == "system"  # 未映射则原样(小写)
    assert norm_role("tool") == "tool"
    assert norm_role("unknown_role") == "unknown_role"
    assert norm_role("") == ""


# ---------- row_to_turns: A) conversation:[{human,assistant}] ----------
def test_row_to_turns_pair_type():
    obj = {
        "conversation_id": "c1",
        "conversation": [
            {"human": "hello", "assistant": "hi there"},
            {"human": "how are you?", "assistant": "fine, thanks"},
        ],
    }
    turns, rid = row_to_turns(obj)
    assert rid == "c1"
    assert turns == [
        ("human", "hello"),
        ("gpt", "hi there"),
        ("human", "how are you?"),
        ("gpt", "fine, thanks"),
    ]


def test_row_to_turns_pair_skips_empty_and_missing_side():
    # assistant 缺失 -> 只保留 human；两边空 -> 整行判无效
    obj_missing = {"id": "m1", "conversation": [{"human": "only human"}]}
    turns, _ = row_to_turns(obj_missing)
    assert turns == [("human", "only human")]

    obj_empty_pair = {"id": "m2", "conversation": [{"human": "", "assistant": ""}]}
    assert row_to_turns(obj_empty_pair) == (None, "m2")

    obj_empty_pair2 = {"id": "m3", "conversation": [{}]}
    assert row_to_turns(obj_empty_pair2) == (None, "m3")


def test_row_to_turns_pair_non_dict_element_drops_row():
    obj = {"id": "n1", "conversation": ["not-a-dict"]}
    assert row_to_turns(obj) == (None, "n1")


# ---------- row_to_turns: B) conversations:[{from,value}|{role,content}] ----------
def test_row_to_turns_from_value():
    obj = {
        "id": "b1",
        "conversations": [
            {"from": "user", "value": "what is 2+2?"},
            {"from": "assistant", "value": "four"},
            {"from": "system", "value": "be brief"},
        ],
    }
    turns, rid = row_to_turns(obj)
    assert rid == "b1"
    assert turns == [
        ("human", "what is 2+2?"),
        ("gpt", "four"),
        ("system", "be brief"),  # system 保留
    ]


def test_row_to_turns_role_content():
    obj = {
        "id": "b2",
        "conversations": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
    }
    turns, rid = row_to_turns(obj)
    assert rid == "b2"
    assert turns == [("human", "q1"), ("gpt", "a1")]


def test_row_to_turns_unknown_role_and_blank_value_skipped():
    obj = {
        "id": "b3",
        "conversations": [
            {"from": "mystery", "value": "dropped"},
            {"from": "user", "value": "   "},  # 空白 -> 跳过
            {"role": "assistant", "content": "keep"},
        ],
    }
    turns, _ = row_to_turns(obj)
    assert turns == [("gpt", "keep")]

    # 全部无效 -> 判为无效行
    obj_all_bad = {"id": "b4", "conversations": [{"from": "mystery", "value": "x"}]}
    assert row_to_turns(obj_all_bad) == (None, "b4")


def test_row_to_turns_conversations_non_dict_drops_row():
    obj = {"id": "n2", "conversations": [1, 2]}
    assert row_to_turns(obj) == (None, "n2")


def test_row_to_turns_no_known_key():
    assert row_to_turns({"other": "data"}) == (None, None)


def test_row_to_turns_id_precedence():
    # 优先 conversation_id -> id -> uuid
    obj = {
        "conversation_id": "cid",
        "id": "iid",
        "uuid": "uid",
        "conversations": [
            {"from": "user", "value": "q"},
            {"role": "assistant", "content": "answer longer than eight chars"},
        ],
    }
    _, rid = row_to_turns(obj)
    assert rid == "cid"
    obj2 = {
        "uuid": "uuid2",
        "conversation": [{"human": "h", "assistant": "long enough assistant answer here"}],
    }
    assert row_to_turns(obj2)[1] == "uuid2"


# ---------- main() CLI 集成 ----------
def test_main_converts_and_reports(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    lines = [
        json.dumps(
            {
                "conversation_id": "a",
                "conversation": [{"human": "hi", "assistant": "a fairly long answer here"}],
            }
        ),
        json.dumps(
            {
                "id": "b",
                "conversations": [
                    {"from": "user", "value": "q"},
                    {"role": "assistant", "content": "another long answer here ok"},
                ],
            }
        ),
        "{broken json",
        json.dumps({"other": "no container"}),
    ]
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["to_regen_input.py", "--input", str(inp), "--output", str(out)])
    to_regen_input.main()
    captured = capsys.readouterr().out

    assert "input=4 output=2 dropped=2" in captured
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 2
    assert records[0]["id"] == "a"
    assert records[0]["conversations"] == [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "a fairly long answer here"},
    ]
    assert records[1]["id"] == "b"
    # 无 id 行应回退 row_N —— 此处两条都有 id
    assert all("conversations" in r for r in records)


def test_main_fallback_id_when_no_id_field(tmp_path, monkeypatch):
    inp = tmp_path / "in2.jsonl"
    out = tmp_path / "out2.jsonl"
    inp.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "user", "value": "q"},
                    {"role": "assistant", "content": "long enough answer here"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["to_regen_input.py", "--input", str(inp), "--output", str(out)])
    to_regen_input.main()
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["id"] == "row_1"  # 回退 row_<行号>

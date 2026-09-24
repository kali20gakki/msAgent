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
"""Convert local conversation jsonl into response_regeneration/script.py --data input.

Accepts rows that are:
  A) {"conversation_id","category","conversation":[{"human":..,"assistant":..}]}   (ShareGPT-unfiltered 派生的对话数据)
  B) {"id","conversations":[{"from":"human"|"gpt","value":..} or {"role","content"}]}
Output (speculators response_regeneration schema):
  {"id": <conv_id>, "conversations":[{"from":"human","value":..},{"from":"gpt","value":..}, ...]}

Usage:
  python3 to_regen_input.py --input <conversation.jsonl> --output rr_input.jsonl [--limit N]
"""

import argparse
import json


def norm_role(r):
    r = (r or "").strip().lower()
    return {
        "human": "human",
        "user": "human",
        "prompter": "human",
        "gpt": "gpt",
        "assistant": "gpt",
        "chatgpt": "gpt",
    }.get(r, r)


def row_to_turns(obj):
    """Return (list[(from,value)], id) or (None, id)."""
    rid = obj.get("conversation_id") or obj.get("id") or obj.get("uuid")
    # A) human/assistant pairs under 'conversation'
    conv = obj.get("conversation")
    if isinstance(conv, list):
        turns = []
        for d in conv:
            if not isinstance(d, dict):
                return None, rid
            h = d.get("human")
            a = d.get("assistant")
            if isinstance(h, str) and h.strip():
                turns.append(("human", h))
            if isinstance(a, str) and a.strip():
                turns.append(("gpt", a))
            if not turns:
                return None, rid
        return turns, rid
    # B) from/value or role/content under 'conversations'
    convs = obj.get("conversations")
    if isinstance(convs, list):
        turns = []
        for m in convs:
            if not isinstance(m, dict):
                return None, rid
            role = norm_role(m.get("from") or m.get("role") or "")
            value = m.get("value") or m.get("content") or ""
            if role in ("human", "gpt", "system", "tool"):
                if isinstance(value, str) and value.strip():
                    turns.append((role, value))
        return turns or None, rid
    return None, rid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    args = ap.parse_args()

    n_in = n_out = n_drop = 0
    # 逐行解析: 识别结构并归一化角色，写 {id,conversations:[{from,value}]}
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.limit and n_out >= args.limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                n_drop += 1
                continue
            if not isinstance(obj, dict):
                n_drop += 1
                continue
            turns, rid = row_to_turns(obj)
            if not turns:
                n_drop += 1
                continue
            rec = {
                "id": rid if rid is not None else "row_%d" % n_in,
                "conversations": [{"from": f, "value": v} for f, v in turns],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print("input=%d output=%d dropped=%d" % (n_in, n_out, n_drop))


if __name__ == "__main__":
    main()

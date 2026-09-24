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
"""response-regen 输入数据的 DFX 预检（语义面向“响应重生成”）。

在把原始多轮对话交给 to_regen_input.py 转换、或交给 response_regeneration 重生成之前，
先全量/抽样扫描，提前上报数据质量风险：
  - high  : JSON 解析失败 / 无会话容器键(conversation/conversations) / 无有效轮次（这些行无法被消费）
  - medium: 无 assistant(gpt) 轮（重生成对象缺失）/ 首轮非 user（无引导前缀）/ assistant 回答过短 /
            空值回合 / 超长上下文（会被截断，长上下文信息丢失）
  - low   : 非英文为主 / 重复 id / 重复对话内容
  - info  : 结构类型提示（{human,assistant} 对 / {from,value} 或 {role,content}）

复用同目录 to_regen_input.py 的 schema 解析（保证“预检结论 = 转换/重生成能消费的结论”）。

用法:
  python3 precheck_data.py --input <conversation.jsonl> [--precheck-sample N(0=全量)] [--report-out <precheck>.md|.json]
"""

import argparse
import hashlib
import json
import os
import sys

# 让本脚本可从任意 cwd 调用时也能 import 同目录的 to_regen_input.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from to_regen_input import row_to_turns  # noqa: E402


def load_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def content_hash(turns):
    h = hashlib.md5()
    for f, v in turns:
        h.update(("%s\0%s\0" % (f, v)).encode("utf-8", "replace"))
    return h.hexdigest()


def has_container(obj):
    return any(isinstance(obj.get(k), list) for k in ("conversation", "conversations"))


def precheck(args):
    stats = {
        "total": 0,
        "parse_err": 0,
        "no_container": 0,
        "no_valid_turns": 0,
        "no_assistant": 0,
        "starts_with_gpt": 0,
        "assistant_too_short": 0,
        "empty_value_turn": 0,
        "dup_id": 0,
        "dup_conv": 0,
        "very_long_row_chars": 0,
        "non_ascii_heavy": 0,
    }
    lengths_chars = []
    turn_counts = []
    schema_cnt = {"conversation": 0, "conversations": 0}
    seen_ids = set()
    seen_conv = set()
    sample_err = []
    id_key = None
    limit = args.precheck_sample

    for line in load_lines(args.input):
        if limit and stats["total"] >= limit:
            break
        stats["total"] += 1
        try:
            obj = json.loads(line)
        except Exception as e:  # noqa: BLE001
            stats["parse_err"] += 1
            if len(sample_err) < 10:
                sample_err.append("parse_err@%d: %s" % (stats["total"], str(e)[:120]))
            continue
        if not isinstance(obj, dict):
            stats["parse_err"] += 1
            if len(sample_err) < 10:
                sample_err.append("not_dict@%d" % stats["total"])
            continue

        if not has_container(obj):
            stats["no_container"] += 1
            continue
        if isinstance(obj.get("conversation"), list):
            schema_cnt["conversation"] += 1
        if isinstance(obj.get("conversations"), list):
            schema_cnt["conversations"] += 1

        # 探测 id 字段名（去重统一在 row_to_turns 返回的 rid 上统计，避免重复计数）
        for k in ("conversation_id", "id", "uuid"):
            if obj.get(k):
                id_key = id_key or k
                break

        turns, rid = row_to_turns(obj)
        if not turns:
            stats["no_valid_turns"] += 1
            continue
        if rid is not None:
            if rid in seen_ids:
                stats["dup_id"] += 1
            seen_ids.add(rid)

        ch = content_hash(turns)
        if ch in seen_conv:
            stats["dup_conv"] += 1
        seen_conv.add(ch)

        turn_counts.append(len(turns))
        tot_chars = sum(len(v) for _, v in turns)
        lengths_chars.append(tot_chars)
        if tot_chars > 20000:
            stats["very_long_row_chars"] += 1

        has_gpt = any(f == "gpt" for f, _ in turns)
        if not has_gpt:
            stats["no_assistant"] += 1
        if turns and turns[0][0] != "human":
            stats["starts_with_gpt"] += 1
        text_all = []
        for f, v in turns:
            if not v.strip():
                stats["empty_value_turn"] += 1
            if f == "gpt" and len(v) < 8:
                stats["assistant_too_short"] += 1
            text_all.append(v)
        joined = " ".join(text_all)
        if joined:
            ascii_ratio = sum(1 for ch in joined if ord(ch) < 128) / len(joined)
            if ascii_ratio < 0.85:
                stats["non_ascii_heavy"] += 1

    stat = {"sampled": len(lengths_chars), "schema_breakdown": schema_cnt, "id_key": id_key}
    if lengths_chars:
        lengths_chars.sort()
        n = len(lengths_chars)
        stat["rows_total_chars_p50"] = lengths_chars[min(n - 1, int(n * 0.50))]
        stat["rows_total_chars_p90"] = lengths_chars[min(n - 1, int(n * 0.90))]
        stat["rows_total_chars_max"] = lengths_chars[-1]
        tc = sorted(turn_counts)
        stat["turns_p50"] = tc[min(n - 1, int(n * 0.50))]
        stat["turns_max"] = tc[-1]
        stat["est_tokens_p50"] = max(1, int(lengths_chars[min(n - 1, int(n * 0.50))] / 4))
    return stats, stat, sample_err


def build_risks(stats, stat):
    risks = []
    total = max(1, stats["total"])

    def pct(v):
        return "%.1f%%" % (100.0 * v / total)

    if stats["parse_err"]:
        risks.append(
            ["high", "JSON 解析失败 %d 行(%s)，这些行无法进入重生成" % (stats["parse_err"], pct(stats["parse_err"]))]
        )
    if stats["no_container"]:
        risks.append(
            [
                "high",
                "缺少会话容器键(conversation/conversations) %d 行(%s)，无法提取对话"
                % (stats["no_container"], pct(stats["no_container"])),
            ]
        )
    if stats["no_valid_turns"]:
        risks.append(
            [
                "high",
                "回合异常导致整行丢弃 %d 行(%s)(空会话/非字典/无有效轮次)"
                % (stats["no_valid_turns"], pct(stats["no_valid_turns"])),
            ]
        )
    if stats["no_assistant"]:
        risks.append(
            [
                "medium",
                "无 assistant(gpt) 轮的样本 %d 行(%s)，没有可重生成的目标回答"
                % (stats["no_assistant"], pct(stats["no_assistant"])),
            ]
        )
    if stats["starts_with_gpt"]:
        risks.append(
            [
                "medium",
                "首轮非 user(human) 的样本 %d 行(%s)，重生成时缺少引导前缀"
                % (stats["starts_with_gpt"], pct(stats["starts_with_gpt"])),
            ]
        )
    if stats["assistant_too_short"]:
        risks.append(["medium", "assistant 回答过短(<8字符)的回合数 %d，重生成收益低" % stats["assistant_too_short"]])
    if stats["empty_value_turn"]:
        risks.append(
            ["medium", "存在空 value/content 回合 %d 处，将按 to_regen_input 规则跳过" % stats["empty_value_turn"]]
        )
    if stats["very_long_row_chars"]:
        risks.append(
            [
                "medium",
                "超长样本(>20000字符) %d 行(%s)，重生成/训练会被截断，长上下文信息部分丢失"
                % (stats["very_long_row_chars"], pct(stats["very_long_row_chars"])),
            ]
        )
    if stats["non_ascii_heavy"]:
        risks.append(
            [
                "low",
                "非英文为主样本 %d 行(%s)(ASCII<85%%)，若训练语料是英文需注意混入"
                % (stats["non_ascii_heavy"], pct(stats["non_ascii_heavy"])),
            ]
        )
    if stats["dup_id"]:
        risks.append(["low", "重复 id %d 处，可能数据污染，建议去重" % stats["dup_id"]])
    if stats["dup_conv"]:
        risks.append(["low", "重复对话内容 %d 处，建议去重" % stats["dup_conv"]])
    sbreak = stat.get("schema_breakdown", {})
    if sbreak.get("conversation"):
        risks.append(["info", "含 {human,assistant} 对结构(conversation 键)，将由 to_regen_input 展平为 human/gpt"])
    if sbreak.get("conversations"):
        risks.append(["info", "含 {from,value}/{role,content} 结构(conversations 键)，to_regen_input 可直接归一化"])
    return risks


def render_report(stats, stat, risks, sample_err, args):
    lines = []
    lines.append("# response-regen 输入预检报告")
    lines.append("")
    lines.append("- 输入: `%s`" % args.input)
    lines.append("- 扫描行数(或抽样): %d" % stats["total"])
    lines.append("- 有效样本(统计): %d" % stat.get("sampled", 0))
    lines.append("- 结构分布: %s" % json.dumps(stat.get("schema_breakdown", {}), ensure_ascii=False))
    if stat.get("id_key"):
        lines.append(
            "- id 字段: `%s` | 重复 id: %d | 重复对话: %d" % (stat["id_key"], stats["dup_id"], stats["dup_conv"])
        )
    lines.append("")
    lines.append("## 风险清单(DFX 预检，需提前上报)")
    lines.append("")
    if risks:
        for lv, msg in risks:
            lines.append("- **[%s]** %s" % (lv, msg))
    else:
        lines.append("- 未发现显著风险。")
    lines.append("")
    lines.append("## 丢弃原因计数")
    lines.append("")
    for k in ("parse_err", "no_container", "no_valid_turns"):
        if stats.get(k):
            lines.append("- %s: %d" % (k, stats[k]))
    lines.append("")
    if sample_err:
        lines.append("## 解析错误样例")
        lines.append("")
        for e in sample_err:
            lines.append("- `%s`" % e)
    lines.append("")
    lines.append("## 长度/轮次参考(重生成上下文)")  # noqa: W605
    lines.append("")
    lines.append(
        "- 单行总字符数 p50/p90/max: %s / %s / %s"
        % (
            stat.get("rows_total_chars_p50", "-"),
            stat.get("rows_total_chars_p90", "-"),
            stat.get("rows_total_chars_max", "-"),
        )
    )
    lines.append("- 轮次 p50/max: %s / %s" % (stat.get("turns_p50", "-"), stat.get("turns_max", "-")))
    lines.append("- 粗估 token p50: %s" % stat.get("est_tokens_p50", "-"))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="原始多轮对话 jsonl")
    ap.add_argument("--precheck-sample", type=int, default=0, help="抽样行数，0=全量")
    ap.add_argument("--report-out", default=None, help="报告输出路径(.md 或 .json)")
    args = ap.parse_args()

    # 执行预检扫描并打印分级报告
    print("===PRECHECK===")
    stats, stat, sample_err = precheck(args)
    risks = build_risks(stats, stat)
    md = render_report(stats, stat, risks, sample_err, args)
    print(md)
    print("===PRECHECK_END===")
    for lv, msg in risks:
        print("[%s] %s" % (lv, msg))
    # 写报告(按扩展名 md 或 json)
    if args.report_out:
        if args.report_out.endswith(".json"):
            data = {"stats": stats, "summary": stat, "risks": risks, "sample_err": sample_err, "report_md": md}
            with open(args.report_out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(args.report_out, "w", encoding="utf-8") as f:
                f.write(md)
        print("report_written=%s" % args.report_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_evidence.py — evidence.json 骨架 / 结构校验。

消灭末段机械修复循环的两类来源：
  1. 结构缺字段 → `new` 落的骨架自带 report_check.py 要求的全部必填顶层
     字段，先天结构合规；
  2. 末段才发现的结构问题 → `check` 随时可跑（规则=report_check.py 的
     E1–E10；--report 给出时含 R1–R4/E9），填充过程中即校验即修复。

资格表（candidates[].qualification）与排除/锚点结案台账（exclusion_ledger）
随骨架/形状契约一并落 evidence.json——报告不承载机器可查结构。

用法（python3 -I 运行）：

  new    --case caseX --out <工作目录>/evidence.json
         分析开工第一件事。骨架中候选/证据为空，此时跑 check 会报
         E4（candidates 为空）——属预期草稿态，填入后消除。
  check  --evidence <路径> [--report <report.md>]
         只校验不改文件；交付前最后一步也用它。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_check  # noqa: E402  （同目录，规则唯一权威，不复制实现）

# 形状契约：new 时打印一次，条目照抄该形状填充（消灭首稿结构违规返工）。
# 枚举与 report_check.py 常量同源（CANDIDATE_KEYS/CHAIN_STATUS/ROOT_STATUS）。
SHAPE_NOTICE = """  条目形状契约（照抄填充，勿自创结构）：
    candidates[] 条目 = {"candidate_id", "node", "rank", "basis",
      "supporting_evidence_ids": [证据ID...], "counterevidence_ids": [...],
      "causal_chain": [{"desc": str, "status": 枚举, "evidence_ids": [...]}],
      "confidence", "role", "qualification"}（链环节必须是含 status 的 dict，
      不能是裸字符串）
    qualification = HC-C8 资格表行，四格：
      {"predicate_rewrite_test", "anchor_verification",
       "background_unexcluded", "verdict"}
      （锚点验证格合并承载：生产者锚点/节点级验证/同缺陷类归一；背景级与
       未排除格含 HC-C5 迁移留痕；资格结论格含机制覆盖内/外；schema 细节
       见 evidence-and-state.md）
    exclusion_ledger[] 条目 = {"item", "type", "disposition", "basis"}
      （排除/降权项与速览锚点结案台账；同类条目合并为族级行，参考上限 5）
    evidence_log[] 条目 = {"id", "source", "metrics", "usage"}
      （usage=用途与验证结论一句；不设独立 verification 字段）
    root_cause_judgment = {"status", "statement", "confidence",
      "unverified_links"}
  枚举表（填其它值 report_check 会打回）：
    causal_chain[].status ∈ {已证实, 合理推断, 待验证}
    root_cause_judgment.status ∈ {已验证根因, 较强候选, 待验证假设}
    （位序/限定语一律写进 statement，不进 status）"""

SKELETON = {
    "case": None,
    "phenomenon": {"description": "", "evidence_sources": [],
                   "quantified": ""},
    "candidates": [],
    "evidence_log": [],
    "exclusion_ledger": [],
    "open_questions": [],
    "root_cause_judgment": {"status": "", "statement": "", "confidence": "",
                            "unverified_links": []},
    "verification_methods_used": [],
    "run_notes": "",
}


def die(msg, code=2):
    print("[make_evidence.py] 错误: %s" % msg, file=sys.stderr)
    sys.exit(code)


def load_evidence(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            ev = json.load(f)
    except OSError as e:
        die("无法读取 %s: %s" % (path, e))
    except json.JSONDecodeError as e:
        die("%s 不是合法 JSON: %s" % (path, e))
    if not isinstance(ev, dict):
        die("%s 顶层不是对象" % path)
    return ev


def save(path, ev):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=2)


def cmd_new(args):
    if os.path.exists(args.out):
        die("%s 已存在——不覆盖（防重置已填内容）；如需重建请先删除"
            % args.out)
    ev = json.loads(json.dumps(SKELETON))
    ev["case"] = args.case
    save(args.out, ev)
    print("make_evidence new: 骨架已写入 %s" % args.out)
    print("  待填：phenomenon / candidates / evidence_log / "
          "exclusion_ledger / open_questions / root_cause_judgment / "
          "verification_methods_used")
    print(SHAPE_NOTICE)
    print("  交付前最后一步：check --evidence %s --report <report.md>"
          % args.out)


def run_check(evidence_path, report_path):
    issues = []
    n_cands = None
    if report_path:
        n_cands = report_check.check_report(report_path, issues)
    report_check.check_evidence(evidence_path, issues, n_cands)
    if issues:
        print("共 %d 项问题：" % len(issues))
        for it in issues:
            print("  - %s" % it)
        return 1
    print("make_evidence check: 通过（结构完整性；语义自检仍需人工完成）")
    return 0


def cmd_check(args):
    sys.exit(run_check(args.evidence, args.report))


def build_parser():
    p = argparse.ArgumentParser(prog="make_evidence.py",
                                description="evidence.json 骨架/结构校验")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new")
    s.add_argument("--case", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("check")
    s.add_argument("--evidence", required=True)
    s.add_argument("--report")
    s.set_defaults(func=cmd_check)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

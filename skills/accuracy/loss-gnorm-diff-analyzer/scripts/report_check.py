#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_check.py — report.md / evidence.json 结构完整性机械自检。

只做确定性结构检查（不替代语义自检——证据回查、反证完备性、位序一致性仍为
分析者义务）：

report.md:
  R1 首行 case 身份锚（"本报告属于 case"）
  R2 6 个必需小节标题齐备（§1-§6；资格表/排除台账/分析限制移
     evidence.json，根因判断前移）
  R3 无『（待填…』占位残留（裸钉与带说明钉均检出）
  R4 §3 清单行声明的候选数与 "### 候选" 条目数一致（仅认 ±20 字符
     内含"候选"的"共 N 个"声明，防"共 16 个文件"类误报）
  E9 报告模板占位残留检测（"<case>"/"caseXXX"/"<本 case" 等骨架钉）
evidence.json:
  E1 可解析为 JSON
  E2 顶层必填字段（case/phenomenon/candidates/evidence_log/
     exclusion_ledger/open_questions/root_cause_judgment/
     verification_methods_used）
  E4 每个候选必填子字段 + causal_chain 每环节有 status + qualification
     对象（HC-C8 资格表行）四格齐备
  E5 root_cause_judgment 有 status/statement/confidence
  E5b root_cause_judgment.status 枚举校验（堵复合串漏网）
  E6 evidence_log 每条有 id/source/metrics/usage
  E7 候选引用的 evidence_ids 均可解析（存在于 evidence_log）
  E8 与 report.md 的候选数一致
  E10 exclusion_ledger 每条有 item/type/disposition/basis（排除/锚点
     结案台账）

用法: python3 -I report_check.py <report.md> <evidence.json>
退出码: 0=通过, 1=存在问题（问题清单打印到 stdout）
"""

import json
import re
import sys

REPORT_SECTIONS = [
    "目标异常现象", "数据与对齐质量", "候选节点总表", "根因判断",
    "候选详析", "后续验证建议",
]

EVIDENCE_TOP_KEYS = [
    "case", "phenomenon", "candidates", "evidence_log",
    "exclusion_ledger", "open_questions", "root_cause_judgment",
    "verification_methods_used",
]

CANDIDATE_KEYS = [
    "candidate_id", "node", "rank", "basis",
    "supporting_evidence_ids", "counterevidence_ids", "causal_chain",
    "confidence", "role", "qualification",
]

# HC-C8 资格表行（落 evidence.json 各候选，报告不誊入）
# 四格：谓语改写测试 / 锚点验证（生产者锚点+节点级验证+同缺陷类归一合并格）/
# 背景级与未排除（含 HC-C5 迁移留痕）/ 资格结论（含机制覆盖内/外）；
# 引入 order 不设格——node 字段含调用号+order
QUALIFICATION_KEYS = [
    "predicate_rewrite_test", "anchor_verification",
    "background_unexcluded", "verdict",
]

EXCLUSION_KEYS = ["item", "type", "disposition", "basis"]

CHAIN_STATUS = {"已证实", "合理推断", "待验证"}

ROOT_STATUS = {"已验证根因", "较强候选", "待验证假设"}


def check_report(path, issues):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        issues.append(f"R0 无法读取 report.md: {e}")
        return None
    lines = text.splitlines()
    # R1 case 身份锚（前 5 行内出现）
    if not any("本报告属于" in ln and "case" in ln for ln in lines[:5]):
        issues.append("R1 报告首部缺少 case 身份锚（『本报告属于 caseXXX…』）")
    # R2 小节齐备
    for sec in REPORT_SECTIONS:
        if not any(re.match(rf"#+\s*\d*\s*.*{sec}", ln) for ln in lines):
            issues.append(f"R2 缺少小节：{sec}")
    # R3 占位残留（含"（待填）/（待填：…）"两种占位钉形式）
    for i, ln in enumerate(lines, 1):
        if "（待填" in ln or "(待填" in ln:
            issues.append(f"R3 第 {i} 行存在『（待填…』占位残留")
    # R4 候选数一致（只认 ±20 字符内含"候选"的"共 N 个"声明，
    # 防 "共 16 个文件" 类无关数字误报）
    n_detail = len(re.findall(r"^###\s*候选\s*\d+", text, re.M))
    m = None
    for cand_m in re.finditer(r"共\s*(\d+)\s*个", text):
        ctx = text[max(0, cand_m.start() - 20):cand_m.end() + 20]
        if "候选" in ctx:
            m = cand_m
            break
    if m:
        n_declared = int(m.group(1))
        if n_detail and n_declared != n_detail:
            issues.append(
                f"R4 清单行声明候选数 {n_declared} 与详析条目数 {n_detail} 不一致")
    elif n_detail == 0:
        issues.append("R4 未找到任何『### 候选 N』条目，也未声明候选数")
    # E9 模板占位残留：骨架未剥离即拼装（骨架文本被前置在正式报告头部，
    # R1 靠后段合规标题侥幸通过）
    for i, ln in enumerate(lines, 1):
        if ("<case>" in ln or "caseXXX" in ln or "<本 case" in ln
                or "<本case" in ln):
            issues.append(
                f"E9 第 {i} 行存在模板占位残留（骨架未剥离）: {ln.strip()[:60]}")
    return n_detail


def check_evidence(path, issues, n_report_cands):
    try:
        with open(path, "r", encoding="utf-8") as f:
            ev = json.load(f)
    except OSError as e:
        issues.append(f"E0 无法读取 evidence.json: {e}")
        return
    except json.JSONDecodeError as e:
        issues.append(f"E1 evidence.json 不是合法 JSON: {e}")
        return
    for k in EVIDENCE_TOP_KEYS:
        if k not in ev:
            issues.append(f"E2 缺少顶层字段: {k}")
    cands = ev.get("candidates", [])
    if not cands:
        issues.append("E4 candidates 为空")
    ids = set()
    for c in cands:
        cid = c.get("candidate_id", "?")
        for k in CANDIDATE_KEYS:
            if k not in c:
                issues.append(f"E4 候选 {cid} 缺少字段: {k}")
        # E4 资格表行（HC-C8，落 evidence.json）
        q = c.get("qualification")
        if not isinstance(q, dict):
            issues.append(f"E4 候选 {cid} 缺少 qualification 对象（HC-C8 资格表行）")
        else:
            for k in QUALIFICATION_KEYS:
                if k not in q:
                    issues.append(f"E4 候选 {cid} qualification 缺少格: {k}")
        for step in c.get("causal_chain", []):
            if step.get("status") not in CHAIN_STATUS:
                issues.append(
                    f"E4 候选 {cid} 因果链环节 status 非法: {step.get('status')!r}")
    # E10 排除/锚点结案台账
    for e in ev.get("exclusion_ledger", []):
        for k in EXCLUSION_KEYS:
            if not e.get(k):
                issues.append(f"E10 台账条目缺少字段: {k}（{e}）")
    log = ev.get("evidence_log", [])
    for e in log:
        for k in ("id", "source", "metrics", "usage"):
            if k not in e:
                issues.append(f"E6 证据条目缺少字段: {k}（{e}）")
        ids.add(e.get("id"))
    rc = ev.get("root_cause_judgment", {})
    for k in ("status", "statement", "confidence"):
        if not rc.get(k):
            issues.append(f"E5 root_cause_judgment 缺少 {k}")
    # E5b status 枚举校验：复合串/修饰串一律打回，枚举外语义写 statement
    if rc.get("status") and rc["status"] not in ROOT_STATUS:
        issues.append(
            f"E5b root_cause_judgment.status 非法: {rc['status']!r}"
            f"（允许值: {sorted(ROOT_STATUS)}；位序/限定语写入 statement）")
    # E7 引用可解析
    for c in cands:
        cid = c.get("candidate_id", "?")
        for k in ("supporting_evidence_ids", "counterevidence_ids"):
            for ref in c.get(k, []) or []:
                if ref not in ids:
                    issues.append(f"E7 候选 {cid} 引用的证据 ID 不存在: {ref}")
        for step in c.get("causal_chain", []):
            for ref in step.get("evidence_ids", []) or []:
                if ref not in ids:
                    issues.append(f"E7 候选 {cid} 因果链引用的证据 ID 不存在: {ref}")
    # E8 候选数一致
    if n_report_cands is not None and cands and n_report_cands != len(cands):
        issues.append(
            f"E8 report.md 候选数 {n_report_cands} 与 evidence.json 候选数 "
            f"{len(cands)} 不一致")


def main():
    if len(sys.argv) != 3:
        print("用法: python3 -I report_check.py <report.md> <evidence.json>")
        sys.exit(1)
    issues = []
    n_cands = check_report(sys.argv[1], issues)
    check_evidence(sys.argv[2], issues, n_cands)
    if issues:
        print(f"共 {len(issues)} 项问题：")
        for it in issues:
            print(f"  - {it}")
        sys.exit(1)
    print("report_check: 通过（结构完整性；语义自检仍需人工完成）")


if __name__ == "__main__":
    main()

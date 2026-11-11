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
"""rules_locality.py — R001/R002/R005~R010 组合诊断（检查项事实 → Finding）。

R003/R004（NUMA 亲和类）已移交 bind-tune 建模步，编号保留不复用。
每条命中输出 Finding（含 suggested_cpuset），Action 一律转出 bind-tune。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List


def _finding(rule_id: str, problem: str, evidence: List[str], impact: str,
             confidence: float, suggested_cpuset: str = "",
             locality_mode: str = "") -> Dict[str, Any]:
    return {
        "finding_id": f"F-{rule_id}-{int(time.time())}",
        "rule": rule_id,
        "problem": problem,
        "evidence": evidence,
        "impact": impact,
        "confidence": confidence,
        "suggested_cpuset": suggested_cpuset,
        # 建议核集依据：topo_affinity（有 NPU 亲和）/ global_slice（等距，仅均衡示例）
        "locality_mode": locality_mode,
    }


def _parse_set(text: str) -> set:
    cpus: set = set()
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                cpus.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cpus.add(int(part))
    return cpus


def _fmt_set(cpus: List[int]) -> str:
    """连续核号压缩为区间（[0,1,2,5] → '0-2,5'）。"""
    if not cpus:
        return ""
    ordered = sorted(set(cpus))
    spans: List[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        spans.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    spans.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(spans)


def _balanced_slice(facts: Dict[str, Any], die_groups: List[List[int]]) -> str:
    """均衡切片示例（global_slice 降级口径）：每 node 均分 unit，返回 unit0 示例核集。

    仅在无 NPU 亲和依据（UB/HCCS_SW 等距）时使用，属示例而非亲和方案。
    unit 数取逻辑 die 组数（双 die 机型同封装 die 对为一个 unit），无 die 组时取封装数。
    """
    nodes: List[List[int]] = []
    for node in range(64):
        cpus = facts.get(f"node{node}_cpus")
        if cpus is None:
            continue
        nodes.append(sorted(_parse_set(cpus)))
    if not nodes:
        return ""
    units = len(die_groups) if die_groups else int(facts.get("npu_count") or 0)
    if units <= 0:
        units = len(nodes)
    units_per_node = max(1, -(-units // len(nodes)))  # ceil
    per_unit = max(1, len(nodes[0]) // units_per_node)
    return _fmt_set(nodes[0][:per_unit])


class LocalityRules:
    def __init__(self, results: List[Dict[str, Any]], facts: List[Dict[str, Any]],
                 targets: List[Dict[str, Any]]):
        self.results = {r["id"]: r for r in results}
        self.facts = {f["item"]: f.get("value") for f in facts}
        self.targets = targets
        self.findings: List[Dict[str, Any]] = []

    def _add(self, finding: Dict[str, Any]) -> None:
        self.findings.append(finding)

    # R001 进程完全未绑核（suggested_cpuset 随 fabric 联动）
    def rule_r001(self, online_cpus: str) -> None:
        b1 = self.results.get("B1")
        if not b1 or b1["status"] != "WARN":
            return
        if str(b1["value"]).replace(" ", "") == online_cpus.replace(" ", ""):
            pid_hint = ""
            for target in self.targets:
                pid_hint = f"pid={target.get('pid')} comm={target.get('comm')}"
                break
            fabric = self.facts.get("npu_topo_fabric", "unknown")
            die_groups = self.facts.get("npu_die_groups") or []
            affinity_available = bool(self.facts.get("npu_affinity_available"))
            evidence = [f"B1: {pid_hint} Cpus_allowed_list={b1['value']}",
                        f"全机核集={online_cpus}"]
            if affinity_available:
                # 有亲和依据：按 NPU 亲和核集建议（topo_affinity）
                locality_mode = "topo_affinity"
                suggested = self.facts.get("node0_cpus", "")
                confidence = 0.9
                impact = "线程全机漂移、cache 失效"
            else:
                # 无亲和依据（UB/HCCS_SW 等距）：只能给均衡切片示例，须降置信
                locality_mode = "global_slice"
                suggested = _balanced_slice(self.facts, die_groups)
                confidence = 0.5
                impact = ("线程全机漂移、cache 失效；本机为 "
                          f"{fabric} 等距互联，无 NPU 亲和依据，"
                          "建议核集仅为按 node 均衡切片的示例（需 bind-tune 精算）")
                evidence.append(f"fabric={fabric}，NPU 与各 node 等距"
                                "（npu_cpu_affinity 不可得）")
                if die_groups:
                    evidence.append(f"同封装 die 组={die_groups}"
                                    "（绑核不可跨 node 拆分）")
            self._add(_finding(
                "R001", "进程完全未绑核", evidence, impact, confidence,
                suggested_cpuset=suggested, locality_mode=locality_mode))

    # R002 cpuset 截断
    def rule_r002(self) -> None:
        b2 = self.results.get("B2")
        if not b2 or b2["status"] != "CRIT":
            return
        self._add(_finding(
            "R002", "cpuset 截断",
            [f"B2: 期望核集被容器截断，effective={b2['value']}"],
            "绑核参数被容器截掉（隐性失效）", 0.9,
            suggested_cpuset=str(b2["value"])))

    # R005 数据线程抢占计算核
    def rule_r005(self) -> None:
        b3 = self.results.get("B3")
        if not b3 or b3["status"] != "WARN":
            return
        self._add(_finding(
            "R005", "数据线程抢占计算核",
            [f"B3: dataloader 与计算线程允许核重叠（{b3['value']}）"],
            "数据线程抢计算核", 0.7))

    # R007 中断落在业务核
    def rule_r007(self) -> None:
        e1 = self.results.get("E1")
        if not e1 or e1["status"] != "WARN":
            return
        self._add(_finding(
            "R007", "中断集中在少数核",
            [f"E1: {e1['value']}（busiest/cpu0 占比超阈值）"],
            "中断打扰业务线程", 0.8))

    # R010 隔离核未用 / 业务核未被完整绑入隔离核
    def rule_r010(self) -> None:
        b5 = self.results.get("B5")
        if not b5 or b5["status"] != "FACT":
            return
        isolated = b5.get("value")
        if not isolated or isolated == "未配置":
            return
        isolated_set = _parse_set(str(isolated))
        pid_affinity = self.facts.get("pid_affinity", "")
        pid_set = _parse_set(pid_affinity)
        # 无隔离核、无业务核集，或业务核已完全落在隔离核内 → 隔离配置被正确使用
        if not isolated_set or not pid_set or pid_set.issubset(isolated_set):
            return
        if pid_set.isdisjoint(isolated_set):
            self._add(_finding(
                "R010", "隔离核未使用（业务进程未绑到隔离核）",
                [f"B5: isolcpus={isolated}",
                 f"业务允许核={pid_affinity} ∩ 隔离核 = ∅"],
                "隔离失效、隔离核闲置", 0.6))
            return
        self._add(_finding(
            "R010", "业务进程未完全绑入隔离核（部分越界）",
            [f"B5: isolcpus={isolated}",
             f"业务允许核={pid_affinity} 仅部分落在隔离核内"],
            "隔离边界被打破、隔离核与其他核混用", 0.6))

    def run(self, online_cpus: str) -> List[Dict[str, Any]]:
        self.rule_r001(online_cpus)
        self.rule_r002()
        self.rule_r005()
        self.rule_r007()
        self.rule_r010()
        return self.findings

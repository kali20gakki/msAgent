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
"""checker.py — 8 域检查项执行与判定（OK/WARN/CRIT/FACT，阈值三级覆盖）。

阈值优先级：check-items 默认表（内置）→ 机型 profile → 用户 --expected。
只消费 facts[]（单点快照），不做差分/时序。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

# ---- 默认阈值表（check-items.md 的机器可读形式）----
DEFAULT_THRESHOLDS = {
    "A1_node_mem_imbalance_pct": 20,
    "A2_remote_distance": 21,
    "C1_hugepages_zero": 0,
    "C4_swappiness": 10,
    "D2_freq_ratio_min": 0.8,
    "E1_single_cpu_irq_pct": 30,
    "E1_cpu0_irq_pct": 50,
    "F1_quota_vs_request": "lt",
    "G1_omp_threads": "num_cpus",
}

# F1 配额对比方式（F1_quota_vs_request 取值 → 比较函数）
_QUOTA_COMPARE = {
    "lt": lambda quota, request: quota < request,
    "le": lambda quota, request: quota <= request,
    "gt": lambda quota, request: quota > request,
    "ge": lambda quota, request: quota >= request,
    "eq": lambda quota, request: quota == request,
}

# quick 档检查项
QUICK_ITEMS = ("A1", "A2", "A5", "B1", "B2", "C1", "C2", "D1",
               "E1", "E2", "F1", "F2", "G1", "G2", "H1", "H4")

# 收益定级映射（高/中/低）
BENEFIT_MAP = {
    "D1": "high", "B2": "high", "H1": "high",
    "C1": "medium", "C2": "medium", "E1": "medium", "E2": "medium",
    "A1": "medium", "F1": "medium", "F2": "medium",
    "D2": "medium", "D3": "medium", "D4": "low", "C4": "low",
}

# 检查项元数据（人可读报告的核心）：
#   explain = 这个配置是什么、为什么影响训练/推理（通俗语言，无术语堆砌）
#   advice  = 推荐值 + 优化方向（训练/推理场景的合理设置）
ITEM_META = {
    "A1": {"explain": "多路 CPU 的内存被均匀分到几个 NUMA 组，如果某一组内存快用完，进程会去别的组借内存，速度慢好几倍。",
           "advice": "排查大内存占用来源（free 逐 node 对比），确保每 node 余量充足；可用 numactl --interleave 分散分配"},
    "A2": {"explain": "NUMA 组之间的访问距离，10 表示同组（快），≥21 表示跨组（慢）。",
           "advice": "FACT 项：进程/线程应与所用资源（卡、网卡、数据盘）放在同一 node，跨组访存要避免"},
    "A3": {"explain": "一颗物理 CPU 内部再细分成几个小组（SNC），系统默认当成一个 node，容易误判距离。",
           "advice": "FACT 项：SNC 开启时绑核应按子簇划分，node 参数要细化到子簇"},
    "A4": {"explain": "一个物理核带几个超线程（SMT），两个超线程共享执行单元。",
           "advice": "FACT 项：性能敏感线程独占物理核（只用一个超线程），避免和杂务线程挤同一物理核"},
    "A5": {"explain": "每张 NPU 卡挂在哪个 NUMA 组下。多数 PCIe 机型有这个映射，进程不在卡所在的组时，数据搬运要走跨组总线；UB（灵衢）/交换互联机型 NPU 与所有 CPU 等距，本来就没有这个映射——这是正常形态，不是故障。",
           "advice": "有亲和的机型：rank 绑到卡所在 NUMA 组（→ bind-tune 出方案）；UB/等距机型：按 node 均衡分配（global_slice），无需也无法标定亲和"},
    "A6": {"explain": "网卡挂在哪个 NUMA 组下，决定网络收发和中断的访存路径。",
           "advice": "FACT 项：通信线程应与网卡同 node（→ bind-tune）"},
    "A7": {"explain": "数据盘挂在哪个 NUMA 组下，影响数据加载速度。",
           "advice": "FACT 项：DataLoader 线程应与数据盘同 node（→ bind-tune）"},
    "A8": {"explain": "内核自动把进程内存搬到它常跑的 CPU 附近。听起来是好事，但会抵消手工绑核，还引发 TLB 抖动。",
           "advice": "推荐 0（已绑核/严格绑核场景）；仅当完全不绑核时可保留 1"},
    "A9": {"explain": "本组内存紧张时优先回收本组，会造成明显停顿。",
           "advice": "推荐 0（多 node 训练机通用）"},
    "B1": {"explain": "进程允许在哪些 CPU 上跑。如果是全部核，线程会满机器漂移，缓存反复失效。",
           "advice": "按 rank 绑定到指定核集（→ bind-tune 出方案）"},
    "B2": {"explain": "容器实际分配到的 CPU 范围。如果比期望的小，绑核参数写了也会被悄悄截掉。",
           "advice": "CRIT 项：联系平台放开 cpuset，或按实际 effective 核集重新规划"},
    "B5": {"explain": "内核启动参数里的隔离核（isolcpus 等），这些核不参与普通调度。",
           "advice": "FACT 项：业务绑核方案应显式利用或避开隔离核"},
    "C1": {"explain": "大页是给大内存训练专门预留的整块内存，能显著减少地址翻译开销（TLB miss）。",
           "advice": "训练大内存场景推荐预留（如 2MB 大页若干；需重启或 sysctl vm.nr_hugepages 生效）"},
    "C2": {"explain": "透明大页让内核自动合并大内存页。never 时全部按小页处理，TLB 压力大。",
           "advice": "推荐 madvise（应用按需使用，避免 always 带来的整理抖动）"},
    "C3": {"explain": "大页整理策略。always 时分配大页会强制同步整理内存，造成长尾停顿。",
           "advice": "推荐 madvise"},
    "C4": {"explain": "swappiness 决定内核换出内存页的积极程度。值越大，越容易把冷数据挤到磁盘。",
           "advice": "训练大内存场景推荐 ≤10"},
    "C5": {"explain": "机器有活跃 swap 分区，内存压力大时进程会被换到磁盘，造成不可预测的卡顿。",
           "advice": "训练机推荐关闭 swap（swapoff）或降低 swappiness"},
    "C6": {"explain": "overcommit=2 时内核按比例限制虚拟内存总量，大模型分配大块内存可能直接失败。",
           "advice": "大内存训练推荐 0（宽松模式）"},
    "D1": {"explain": "CPU 频率策略。powersave 时 CPU 不跑满频率，Host 侧下发/预处理的算力上限被压低。",
           "advice": "推荐 performance"},
    "D2": {"explain": "CPU 当前实际频率距最高频率的比例。长期低于 80% 说明频率被压（策略或热）。",
           "advice": "检查 governor/EPP/散热；持续降频趋势交 os-observe 确认"},
    "D3": {"explain": "能效偏好。power/balance_power 会让 CPU 偏向省电，牺牲频率。",
           "advice": "推荐 performance 或 balance_performance"},
    "D4": {"explain": "空闲时 CPU 允许睡多深。深睡省电但唤醒有延迟，造成小步延迟抖动。",
           "advice": "延迟敏感场景建议限制到浅睡（如 C2）"},
    "E1": {"explain": "中断集中在个别核（常见是 CPU0），这些核上的业务线程会被频繁打断。",
           "advice": "配置 irqbalance 或手工把高频 IRQ 迁到预留核（业务核之外）"},
    "E7": {"explain": "软中断（网络收包、定时器）在各核的累计分布。",
           "advice": "FACT 项：NET_RX/TIMER 集中时配置 RPS/RFS 分散收包；持续速率交 os-observe"},
    "F1": {"explain": "容器分到的 CPU 配额。低于申请数时，负载一高就被限流。",
           "advice": "按申请 CPU 数放开 cpu.max（联系平台调整）"},
    "F2": {"explain": "cgroup 限流计数。大于 0 说明容器确实发生过『想跑但不让跑』的时段。",
           "advice": "累计值仅作嫌疑；周期性限流确认交 os-observe（throttled 增量曲线）"},
    "F3": {"explain": "容器允许使用哪些 NUMA 组的内存。",
           "advice": "FACT 项：与 NUMA 规划不符时联系平台调整"},
    "G1": {"explain": "OpenMP 线程数。超过可用核数时线程互相抢核，效率反而下降。",
           "advice": "推荐 = 可用核数（绑核场景 = 该 rank 分到的核数）"},
    "G2": {"explain": "OpenMP 绑定开关。false 时线程随意漂移，绑核白做。",
           "advice": "推荐 spread/close + OMP_PLACES=cores（绑核场景）"},
    "G3": {"explain": "进程可见的 NPU 卡列表。",
           "advice": "FACT 项：与期望卡集不符时检查容器挂卡与环境变量"},
    "G5": {"explain": "glibc 内存池上限。不设时每线程最多 8 份池，大线程数进程的虚拟内存碎片和 TLB 压力会放大。",
           "advice": "推荐设置（如 MALLOC_ARENA_MAX=4）"},
    "H1": {"explain": "本机实际可用的 NPU 数。注意两个口径：物理封装（卡，910C 每颗封装含 2 个 die）与逻辑 die（软件设备 davinci0-15 按 die 编号）。卡数对比按封装口径，容器可见设备（ASCEND_RT_VISIBLE_DEVICES）按 die 口径。",
           "advice": "封装数与申请不符时检查容器挂卡、ASCEND_RT_VISIBLE_DEVICES 与平台分配；双 die 机型两个口径不要混用"},
    "H3": {"explain": "NPU 型号与固件版本。",
           "advice": "FACT 项：多机训练时各卡版本应保持一致（随基线对比判定）"},
    "H4": {"explain": "卡间互联拓扑。SIO 是同封装两个 die 的片内直连（最快）；UB/HCCS_SW 是交换互联（NPU 之间等距，没有『某几张卡更近』的说法）；HCCS 是直连连线（同组带宽高）。",
           "advice": "有亲和组的机型：rank 绑核落在各卡的亲和核集（→ bind-tune）；交换互联机型：同封装两 die 尽量放同一 node 的核集，绑核方案归 bind-tune"},
}


class Checker:
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None,
                 profile_extra: Optional[Dict[str, Any]] = None):
        # 阈值三级覆盖：默认 → profile/expected（外部传入合并）
        self.th = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.th.update(thresholds)
        # 机型 profile 的非阈值扩展（npu_numa_override / snc_expected / die_topo 等）
        self.profile: Dict[str, Any] = profile_extra or {}
        self.results: List[Dict[str, Any]] = []

    # ---- 工具 ----
    def _fact_value(self, facts: List[Dict[str, Any]], item: str) -> Any:
        for fact in facts:
            if fact.get("item") == item:
                return fact.get("value")
        return None

    def _add(self, item_id: str, domain: str, name: str, status: str,
             value: Any, threshold: Any, source: str, evidence: str,
             benefit: str = "", value_text: str = "") -> None:
        meta = ITEM_META.get(item_id, {})
        self.results.append({
            "id": item_id, "domain": domain, "name": name, "status": status,
            "value": value, "threshold": threshold, "source": source,
            "evidence": evidence, "benefit": benefit,
            # 结构化 value 的人可读形式（报告优先展示，避免直接摊 JSON）
            "value_text": value_text,
            # 人可读报告核心字段（通俗解释 + 优化建议）
            "explain": meta.get("explain", ""),
            "advice": meta.get("advice", ""),
        })

    # ---- A 域 ----
    def _check_node_mem(self, facts: List[Dict[str, Any]]) -> None:
        """A1：node 间可用内存（MemFree）分布。"""
        free: Dict[int, int] = {}
        total: Dict[int, int] = {}
        for fact in facts:
            m = re.fullmatch(r"node(\d+)_mem_(total|free)_kb", fact["item"])
            if m:
                node = int(m.group(1))
                bucket = free if m.group(2) == "free" else total
                bucket[node] = fact["value"] or 0
        # 余量判定取 MemFree；缺 free 事实时退回 MemTotal（不能把两者相加）
        values = list(free.values()) or list(total.values())
        if values:
            spread = 100.0 * (max(values) - min(values)) / max(values) if max(values) else 0
            status = "WARN" if spread > self.th["A1_node_mem_imbalance_pct"] else "OK"
            basis = "可用内存" if free else "总内存"
            self._add("A1", "numa", "node 间内存分布", status,
                      f"{spread:.1f}%", f"±{self.th['A1_node_mem_imbalance_pct']}%",
                      "/sys/devices/system/node/*/meminfo",
                      f"node {basis}差 {spread:.1f}%", "medium")

    def _check_distance(self, facts: List[Dict[str, Any]]) -> None:
        """A2：NUMA distance 矩阵事实（跨组链路数按阈值统计）。"""
        for fact in facts:
            if re.fullmatch(r"node\d+_distance", fact["item"]):
                threshold = self.th["A2_remote_distance"]
                distances = [d for d in (fact["value"] or []) if isinstance(d, int)]
                remote = [d for d in distances if d >= threshold]
                evidence = "拓扑事实；跨组链路（≥%s）%d/%d 条" % (
                    threshold, len(remote), len(distances))
                self._add("A2", "numa", "NUMA distance 矩阵", "FACT",
                          fact["value"], "", fact["source"], evidence)
                break

    def _check_affinity(self, facts: List[Dict[str, Any]]) -> None:
        """A5：NPU→NUMA 亲和状态机（topo → profile → UB/等距 → 未知）。"""
        # A5 NPU→NUMA 亲和：状态机（形态自适应，fabric 与 CPU Affinity 为独立信号）
        # ① topo 亲和 → ② profile 标定（仅 PCIe/Box8 可标定）→
        # ③ UB/等距（合法无亲和，不告警、不建议标定）→ ④ 未知（真降级，建议标定）
        affinity = self._fact_value(facts, "npu_cpu_affinity")
        fabric = self._fact_value(facts, "npu_topo_fabric")
        if affinity:
            aff_text = "；".join(f"NPU{e.get('npu')}→{e.get('cpus')}"
                                for e in affinity[:8])
            if len(affinity) > 8:
                aff_text += f" 等 {len(affinity)} 卡"
            self._add("A5", "numa", "NPU→NUMA 亲和", "FACT", affinity, "",
                      "npu-smi info -t topo",
                      "卡→CPU 核集事实（亲和判定归 bind-tune）",
                      value_text=aff_text)
        else:
            override = self.profile.get("npu_numa_override")
            if override:
                self._add("A5", "numa", "NPU→NUMA 亲和", "FACT", override, "",
                          "机型 profile 标定",
                          "该机型 topo 无 CPU Affinity 列，采用机型标定映射（标定一次全批复用）")
            elif fabric in ("ub", "hccs_sw"):
                self._add("A5", "numa", "NPU→NUMA 亲和", "FACT", "无（UB/等距）", "",
                          "npu-smi info -t topo",
                          f"fabric={fabric}：NPU 与所有 CPU 等距，无亲和关系"
                          "（合法形态，非降级，无需标定）；绑核按均衡分配"
                          "（global_slice，方案归 bind-tune）")
            else:
                reason = self._fact_value(facts, "npu_affinity_unavailable_reason") or "未知"
                self._add("A5", "numa", "NPU→NUMA 亲和", "FACT", "不可得", "",
                          "npu-smi info -t topo",
                          f"topo 无 CPU Affinity 列（{reason}）；"
                          "仅 PCIe/Box8 形态可标定，建议标定后写入机型 profile 的"
                          " npu_numa_override")

    def _check_numa_facts(self, facts: List[Dict[str, Any]]) -> None:
        """A3/A4/A6/A7：SNC、SMT 与网卡/存储 node 事实。"""
        snc = self._fact_value(facts, "snc")
        if snc is not None:
            snc_expected = self.profile.get("snc_expected")
            if snc_expected is not None and str(snc_expected) != "" \
                    and str(snc_expected) != str(snc):
                self._add("A3", "numa", "SNC 子簇", "WARN", snc,
                          f"={snc_expected}", "机型 profile",
                          "SNC 配置与机型期望不符（见 profile.snc_expected）")
            else:
                self._add("A3", "numa", "SNC 子簇", "FACT", snc, "",
                          "/sys/devices/system/node/*",
                          "SNC 开启时绑核需按子簇划分")
        smt = self._fact_value(facts, "smt_siblings")
        if smt is not None:
            self._add("A4", "numa", "SMT 兄弟 + L3 domain", "FACT",
                      f"{len(smt)} 物理核多线程", "", "cpu*/topology", "兄弟对事实（冲突判定归 R008）")
        for fact in facts:
            if re.fullmatch(r"nic_\w+_node", fact["item"]):
                self._add("A6", "numa", "NIC→NUMA 映射", "FACT", "见 snapshot",
                          "", fact["source"], "网卡 node 事实（判定归 bind-tune）")
                break
        for fact in facts:
            if re.fullmatch(r"block_\w+_node", fact["item"]):
                self._add("A7", "numa", "存储→NUMA 映射", "FACT", "见 snapshot",
                          "", fact["source"], "存储 node 事实（判定归 bind-tune）")
                break

    def _check_numa_sysctl(self, facts: List[Dict[str, Any]]) -> None:
        """A8/A9：numa_balancing 与 zone_reclaim_mode 判定。"""
        numa_balancing = self._fact_value(facts, "numa_balancing")
        if numa_balancing is not None:
            status = "WARN" if numa_balancing == "1" else "OK"
            self._add("A8", "numa", "numa_balancing", status, numa_balancing,
                      "0（严格绑核场景）", "/proc/sys/kernel/numa_balancing",
                      "内核自动页迁移可能抵消绑核")
        zrc = self._fact_value(facts, "zone_reclaim_mode")
        if zrc is not None:
            status = "WARN" if zrc == "1" else "OK"
            self._add("A9", "numa", "zone_reclaim_mode", status, zrc, "0",
                      "/proc/sys/vm/zone_reclaim_mode", "本地内存紧张时同步回收造成停顿")

    def check_numa(self, facts: List[Dict[str, Any]], quick: bool) -> None:
        # A1 node 内存分布
        self._check_node_mem(facts)
        # A2/A3/A4/A5/A6/A7 → FACT
        self._check_distance(facts)
        self._check_affinity(facts)
        self._check_numa_facts(facts)
        self._check_numa_sysctl(facts)

    # ---- B 域 ----
    def check_bind(self, facts: List[Dict[str, Any]], online_cpus: str,
                   quick: bool, expected_affinity: str = "") -> None:
        affinity = self._fact_value(facts, "pid_affinity")
        if affinity is not None:
            is_full_machine = affinity.replace(" ", "") == online_cpus.replace(" ", "")
            status = "WARN" if is_full_machine else "OK"
            self._add("B1", "bind", "目标进程允许核集", status, affinity,
                      "非全机核集", "/proc/<pid>/status",
                      "完全未绑核（组合结论归 R001）" if is_full_machine else "已绑核")
        effective = self._fact_value(facts, "cpuset_cpus_effective")
        if effective is not None:
            if expected_affinity:
                expected = set(_parse_cpu_set(expected_affinity))
                actual = set(_parse_cpu_set(effective))
                if not expected:
                    # 期望核集解析为空时不能当子集命中放行，落 FACT 交人工确认
                    self._add("B2", "bind", "cpuset 截断", "FACT", effective,
                              expected_affinity, "cpuset.cpus.effective",
                              "期望核集不可解析，未做截断判定", "high")
                else:
                    status = "CRIT" if not expected.issubset(actual) else "OK"
                    self._add("B2", "bind", "cpuset 截断", status, effective,
                              expected_affinity, "cpuset.cpus.effective",
                              "期望核被容器截断（组合结论归 R002）" if status == "CRIT" else "期望核集完整",
                              "high")
            else:
                self._add("B2", "bind", "cpuset 截断", "FACT", effective,
                          "", "cpuset.cpus.effective", "无期望值，仅报事实")
        cmdline = self._fact_value(facts, "cmdline_boot_params") or ""
        if cmdline:
            isolated = _extract_isolcpus(cmdline)
            self._add("B5", "bind", "内核隔离核（isolcpus/nohz_full）", "FACT",
                      isolated or "未配置", "", "/proc/cmdline", "隔离核集事实（冲突判定归 R010）")

    # ---- C 域 ----
    def check_memory(self, facts: List[Dict[str, Any]], quick: bool) -> None:
        total = self._fact_value(facts, "hugepages_total")
        free = self._fact_value(facts, "hugepages_free")
        if total is not None:
            floor = self.th["C1_hugepages_zero"]
            if total <= floor:
                status = "WARN"
                evidence = "HugePages 未配置或低于下限 %s" % floor
            elif total > 0 and free == 0:
                status = "WARN"
                evidence = "HugePages 预留不足（Free=0）"
            else:
                status = "OK"
                evidence = f"Total={total} Free={free}"
            self._add("C1", "hugepage", "HugePages 预留", status,
                      f"Total={total} Free={free}", "Total>0 且 Free>0",
                      "/proc/meminfo", evidence, "medium")
        thp = self._fact_value(facts, "thp_enabled")
        if thp is not None:
            status = "WARN" if thp == "never" else "OK"
            self._add("C2", "hugepage", "THP 模式", status, thp, "madvise/always",
                      "/sys/kernel/mm/transparent_hugepage/enabled",
                      "THP 未启用，TLB miss 风险", "medium")
        defrag = self._fact_value(facts, "thp_defrag")
        if defrag is not None:
            status = "WARN" if defrag == "always" else "OK"
            self._add("C3", "hugepage", "THP defrag", status, defrag, "madvise",
                      ".../defrag", "运行时同步内存整理引发长尾停顿")
        swap_val = self._fact_value(facts, "swappiness")
        if swap_val is not None:
            status = "WARN" if int(swap_val) > self.th["C4_swappiness"] else "OK"
            self._add("C4", "hugepage", "swappiness", status, swap_val,
                      f"≤{self.th['C4_swappiness']}", "/proc/sys/vm/swappiness",
                      "冷页换出风险")
        swaps = self._fact_value(facts, "active_swap_devices")
        if swaps is not None:
            status = "WARN" if swaps else "OK"
            swap_text = "、".join(
                "%s（%.1f GB）" % (s.get("filename") or "?",
                                 float(s.get("size_kb") or 0) / 1048576.0)
                for s in swaps) if swaps else "无"
            self._add("C5", "hugepage", "活跃 swap 设备", status, swaps,
                      "无", "/proc/swaps", "换入换出抖动" if swaps else "无活跃 swap",
                      value_text=swap_text)
        oc = self._fact_value(facts, "overcommit_memory")
        if oc is not None:
            status = "WARN" if oc == "2" else "OK"
            self._add("C6", "hugepage", "overcommit 模式", status, oc, "0/1",
                      "/proc/sys/vm/overcommit_memory", "大块分配被拒风险")

    # ---- D 域 ----
    def check_freq(self, facts: List[Dict[str, Any]], quick: bool) -> None:
        governors = self._fact_value(facts, "scaling_governor") or {}
        if governors:
            top_gov, count = max(governors.items(), key=lambda kv: kv[1])
            status = "CRIT" if top_gov == "powersave" else "OK"
            self._add("D1", "freq", "scaling_governor", status, governors,
                      "performance", "cpu*/cpufreq/scaling_governor",
                      "未跑满频，算力上限被压" if status == "CRIT" else "governor 正常",
                      "high")
        ratio = self._fact_value(facts, "freq_cur_over_max")
        if ratio is not None:
            status = "WARN" if ratio < self.th["D2_freq_ratio_min"] else "OK"
            self._add("D2", "freq", "当前频率 vs max", status, f"{ratio:.1%}",
                      f"≥{self.th['D2_freq_ratio_min']:.0%}",
                      "cpu*/cpufreq", "频率受限（持续趋势归 os-observe）", "medium")
        epp = self._fact_value(facts, "epp") or {}
        if epp:
            bad = {k: v for k, v in epp.items() if k in ("power", "balance_power")}
            status = "WARN" if bad else "OK"
            self._add("D3", "freq", "EPP", status, epp, "performance/balance_performance",
                      "cpu*/cpufreq/energy_performance_preference",
                      "偏节能策略压频率" if bad else "EPP 正常")
        cstates = self._fact_value(facts, "cstates") or []
        if cstates:
            deep = [s for s in cstates if re.fullmatch(r"C[6-9]", s)]
            status = "WARN" if deep else "OK"
            self._add("D4", "freq", "C-state 深度", status, cstates, "≤C5",
                      "cpu*/cpuidle/state*/name", "深睡唤醒延迟尖刺")

    # ---- E 域 ----
    def check_irq(self, facts: List[Dict[str, Any]], quick: bool) -> None:
        busiest = self._fact_value(facts, "irq_busiest_cpu_pct")
        cpu0 = self._fact_value(facts, "irq_cpu0_pct")
        if busiest is not None:
            status = "WARN" if (busiest > self.th["E1_single_cpu_irq_pct"]
                                or (cpu0 or 0) > self.th["E1_cpu0_irq_pct"]) else "OK"
            self._add("E1", "irq", "每核中断分布", status,
                      f"busiest={busiest}% cpu0={cpu0}%",
                      f"单核≤{self.th['E1_single_cpu_irq_pct']}%",
                      "/proc/interrupts", "中断集中", "medium")
        softirq = self._fact_value(facts, "softirq_totals") or {}
        if softirq:
            self._add("E7", "irq", "softirq 分布（累计）", "FACT", softirq,
                      "", "/proc/softirqs", "单次累计快照（速率归 os-observe）")

    # ---- F 域 ----
    def check_cgroup(self, facts: List[Dict[str, Any]], quick: bool,
                    requested_cpus: int = 0) -> None:
        cpu_max = self._fact_value(facts, "cpu_max")
        if cpu_max is not None:
            # v2: "max 100000" 或 "800000 100000"（quota period，单位 us）
            parts = str(cpu_max).split()
            if parts and parts[0] != "max":
                try:
                    quota_raw = int(parts[0])
                    period = int(parts[1]) if len(parts) > 1 else 100000
                    if period <= 0:
                        raise ValueError("period 必须为正")
                    quota = quota_raw / float(period)
                    mode = str(self.th["F1_quota_vs_request"]).lower()
                    compare = _QUOTA_COMPARE.get(mode, _QUOTA_COMPARE["lt"])
                    if requested_cpus and compare(quota, requested_cpus):
                        status, evidence = "WARN", "quota=%g 核 < 申请 %d 核" % (quota, requested_cpus)
                    else:
                        status, evidence = "OK", "quota=%g 核" % quota
                    self._add("F1", "cgroup", "CPU 配额", status, cpu_max,
                              f"≥{requested_cpus}" if requested_cpus else "",
                              "/sys/fs/cgroup/cpu.max", evidence, "medium")
                except ValueError:
                    # 格式非 "max 100000" / "<quota> <period>"：不静默丢弃，落 FACT 供人工判读
                    self._add("F1", "cgroup", "CPU 配额", "FACT", cpu_max, "",
                              "/sys/fs/cgroup/cpu.max",
                              "cpu.max 不可解析（非 max/quota period 格式）", "medium")
        throttled = self._fact_value(facts, "cpu_stat_throttled") or {}
        if throttled:
            has_throttle = any(int(v) > 0 for v in throttled.values()
                               if str(v).isdigit())
            status = "WARN" if has_throttle else "OK"
            self._add("F2", "cgroup", "throttled 计数", status, throttled, "0",
                      "cpu.stat", "实际发生限流（持续时段归 os-observe）", "medium")
        mems = self._fact_value(facts, "cpuset_mems_effective")
        if mems is not None:
            self._add("F3", "cgroup", "cpuset.mems 限定", "FACT", mems,
                      "", "cpuset.mems.effective", "内存本地性限定事实")

    # ---- G 域 ----
    def check_env(self, facts: List[Dict[str, Any]], quick: bool,
                  num_cpus: int = 0) -> None:
        env = self._fact_value(facts, "pid_environ_picked") or {}
        if not env:
            return
        omp = env.get("OMP_NUM_THREADS")
        if omp is not None and num_cpus:
            limit_spec = self.th.get("G1_omp_threads", "num_cpus")
            try:
                limit = num_cpus if str(limit_spec) == "num_cpus" else int(limit_spec)
            except (TypeError, ValueError):
                limit = num_cpus
            try:
                status = "WARN" if int(omp) > limit else "OK"
                evidence = f"OMP_NUM_THREADS={omp} vs {limit} 核"
            except ValueError:
                status, evidence = "OK", f"OMP_NUM_THREADS={omp}（非数值）"
            self._add("G1", "env", "OMP_NUM_THREADS", status, omp,
                      f"≤{limit}", "/proc/<pid>/environ", evidence)
        proc_bind = env.get("OMP_PROC_BIND")
        if proc_bind is not None:
            status = "WARN" if proc_bind.lower() in ("false", "0") else "OK"
            self._add("G2", "env", "OMP_PROC_BIND", status, proc_bind,
                      "spread/close", "/proc/<pid>/environ",
                      "OpenMP 线程漂移抵消绑核")
        visible = env.get("ASCEND_RT_VISIBLE_DEVICES")
        if visible is not None:
            self._add("G3", "env", "ASCEND_RT_VISIBLE_DEVICES", "FACT", visible,
                      "", "/proc/<pid>/environ", "可见卡集事实（与期望比对需 --expected）")
        arena = env.get("MALLOC_ARENA_MAX")
        if arena is None:
            self._add("G5", "env", "MALLOC_ARENA_MAX", "FACT", "未设置",
                      "", "/proc/<pid>/environ", "默认 8×核数，虚拟内存碎片风险")

    # ---- H 域 ----
    def check_npu(self, facts: List[Dict[str, Any]], quick: bool,
                  expected_cards: int = 0) -> None:
        available = self._fact_value(facts, "npu_available")
        if available is not None and available:
            count = self._fact_value(facts, "npu_count")
            die_count = self._fact_value(facts, "npu_die_count")
            davinci = self._fact_value(facts, "davinci_devices")
            if count is not None:
                # 双口径：封装（-l 权威）为主，逻辑 die / davinci 设备并列展示
                value: Dict[str, Any] = {"packages": count}
                text = f"{count} 封装"
                if die_count is not None:
                    value["logical_dies"] = die_count
                    text += f" / {die_count} die"
                if davinci is not None:
                    value["davinci_devices"] = davinci
                    text += f"（/dev/davinci* {davinci} 个）"
                if expected_cards:
                    status = "CRIT" if count != expected_cards else "OK"
                    self._add("H1", "npu", "设备可见数（封装/逻辑 die）", status,
                              value, f"封装={expected_cards}", "npu-smi info -l/-m",
                              ("封装数与申请卡数不符（卡数对比按封装口径）"
                               if status == "CRIT"
                               else f"封装 {count} 与申请一致；逻辑 die "
                                    f"{die_count if die_count is not None else '-'}"),
                              "high", value_text=text)
                else:
                    self._add("H1", "npu", "设备可见数（封装/逻辑 die）", "FACT",
                              value, "", "npu-smi info -l/-m",
                              "无期望值，仅报事实（卡数对比按封装口径）",
                              value_text=text)
        names = self._fact_value(facts, "npu_names")
        if names:
            self._add("H3", "npu", "NPU 型号", "FACT", names, "", "npu-smi info -l",
                      "版本一致性随基线对比判定")
        fabric = self._fact_value(facts, "npu_topo_fabric")
        topo_rows = self._fact_value(facts, "npu_topo_rows")
        if topo_rows:
            die_groups = self._fact_value(facts, "npu_die_groups") or []
            die_pairs = [g for g in die_groups if len(g) > 1]
            sio_pairs = self._fact_value(facts, "npu_sio_pairs")
            value: Dict[str, Any] = {
                "fabric": fabric, "form": self._fact_value(facts, "npu_topo_form"),
                "rows": topo_rows,
            }
            if die_pairs:
                value["die_groups"] = die_pairs
            if sio_pairs:
                value["sio_pairs"] = sio_pairs
            evidence = "拓扑事实（跨组判定需 topo 明细；方案归 bind-tune）"
            text = f"互联 {fabric} · 矩阵 {value['form']}（{topo_rows} 行）"
            if die_pairs:
                text += f" · 同封装 die 对 {len(die_pairs)} 组"
            elif die_groups:
                text += " · 每封装 1 die（单 die 机型）"
            if sio_pairs:
                text += " · SIO 片内直连"
            if fabric in ("ub", "hccs_sw"):
                evidence = (f"fabric={fabric}：交换/统一互联，NPU 间等距，"
                            "无 per-card 亲和组（合法形态）")
                if die_pairs:
                    evidence += "；同封装 die 对不可跨 node 拆分"
            self._add("H4", "npu", "互联拓扑与 fabric", "FACT", value, "",
                      "npu-smi info -t topo", evidence, value_text=text)

    # ---- 主入口 ----
    def run(self, facts: List[Dict[str, Any]], quick: bool,
            online_cpus: str = "", num_cpus: int = 0,
            expected_affinity: str = "", requested_cpus: int = 0,
            expected_cards: int = 0) -> List[Dict[str, Any]]:
        self.check_numa(facts, quick)
        self.check_bind(facts, online_cpus, quick, expected_affinity)
        self.check_memory(facts, quick)
        self.check_freq(facts, quick)
        self.check_irq(facts, quick)
        self.check_cgroup(facts, quick, requested_cpus)
        self.check_env(facts, quick, num_cpus)
        self.check_npu(facts, quick, expected_cards)
        if quick:
            keep = {item for item in QUICK_ITEMS}
            self.results = [r for r in self.results if r["id"] in keep]
        return self.results

    def extract_suspects(self) -> List[Dict[str, Any]]:
        """从 WARN 结果提取 suspect_items（供 os-observe 承接）。"""
        suspects = []
        for result in self.results:
            if result["status"] != "WARN":
                continue
            if result["id"] in ("D2", "E1", "E7", "F2", "A1"):
                suspects.append({
                    "item": result["id"], "name": result["name"],
                    "value": result["value"],
                    "suggest_observe": _OBSERVE_HINT.get(result["id"], "os-observe 持续观测"),
                })
        return suspects


_OBSERVE_HINT = {
    "D2": "scaling_cur_freq 时序（稳态降频 vs 瞬时波动）",
    "E1": "irq_rate 速率 + 落核分布（中断风暴 vs 静态集中）",
    "E7": "softirq 速率趋势",
    "F2": "throttled 增量曲线（周期性限流判定）",
    "A1": "numastat 命中趋势",
}


def _parse_cpu_set(text: str) -> List[int]:
    cpus: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                cpus.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cpus.append(int(part))
    return cpus


def _extract_isolcpus(cmdline: str) -> str:
    for token in cmdline.split():
        key, _, value = token.partition("=")
        if key in ("isolcpus", "nohz_full", "rcu_nocbs"):
            return value
    return ""


def load_threshold_overrides(profile_path: str, expected_path: str) -> Dict[str, Any]:
    """加载机型 profile 与用户 expected（后者优先）。

    兼容两种格式：{"thresholds": {...}} 包装或顶层即阈值 dict；
    下划线开头的说明键与 expected_affinity（有专门读者）不并入阈值。
    """
    merged: Dict[str, Any] = {}
    for path in (profile_path, expected_path):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for key, value in data.get("thresholds", data).items():
                    if str(key).startswith("_") or key == "expected_affinity":
                        continue
                    merged[key] = value
        except (OSError, ValueError):
            continue
    return merged


def load_profile_extra(profile_path: str) -> Dict[str, Any]:
    """加载机型 profile 的非阈值扩展（npu_numa_override / snc_expected / die_topo 等）。"""
    if not profile_path or not os.path.isfile(profile_path):
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            extra = dict(data)
            extra.pop("thresholds", None)
            extra.pop("match", None)
            extra.pop("notes", None)
            return extra
    except (OSError, ValueError):
        pass
    return {}


# 阈值键说明（--dump-thresholds 模板与文档共用）
THRESHOLD_HELP = {
    "A1_node_mem_imbalance_pct": "node 间可用内存分布差告警线（%，超过判 WARN）",
    "A2_remote_distance": "NUMA 跨组访问距离线（distance ≥ 此值视为跨组）",
    "C1_hugepages_zero": "HugePages 预留下限（Total 低于此值判 WARN）",
    "C4_swappiness": "swappiness 上限（超过判 WARN）",
    "D2_freq_ratio_min": "当前频率/最高频率 下限（低于判 WARN）",
    "E1_single_cpu_irq_pct": "单核中断占比告警线（%）",
    "E1_cpu0_irq_pct": "CPU0 中断占比告警线（%）",
    "F1_quota_vs_request": "容器配额对比方式（'lt' = 配额低于申请判 WARN）",
    "G1_omp_threads": "OMP 线程数上限口径（'num_cpus' = 可用核数）",
}


def dump_thresholds_template(thresholds: Dict[str, Any], path: str) -> str:
    """导出当前生效阈值模板（默认 + profile 合并结果），用户改数值后经 --expected 传回。"""
    payload: Dict[str, Any] = {
        "_usage": ("修改数值后通过 --expected <本文件> 传入即可覆盖判定阈值；"
                   "下划线开头的说明键会被忽略。"
                   "expected_affinity 为期望绑核核集（如 '0-79'，留空不判）。"),
        "expected_affinity": "",
    }
    for key in sorted(thresholds):
        payload[key] = thresholds[key]
        if key in THRESHOLD_HELP:
            payload["_" + key] = THRESHOLD_HELP[key]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path

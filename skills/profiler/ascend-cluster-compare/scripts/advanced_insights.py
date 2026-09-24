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
"""
advanced_insights.py
进阶分析结果的中文化解读与规则化分析判断生成器。

三个消费方共享（保证口径一致）:
1. cluster_data_extractor.py  : 提取单集群 JSON 时把结论写入 feature 的 analysis 字段;
2. generate_cluster_report.py : HTML 报告「进阶分析」每个 feature 卡片渲染「分析与判断」,
   A/B 对照组追加「综合判断（基准 vs 对比）」;
3. generate_cluster_md.py     : Markdown 比对报告渲染同样的结论。

能力:
- zh_text(): msprof-analyze 工具输出的英文成因/推断原因 → 中文化（短语映射 + 关键词映射,
  未命中或技术标识符如 AllReduce/Rank/host_bug1 保留原文）;
- free_insights / comm_bn_insights / time_cmp_insights: 基于各 feature 解析数据的
  规则化中文结论（首要成因、集中度、影响面、方向判断、优先排查建议）;
- free_compare / comm_bn_compare: 双集群同 feature 数据齐备时的跨集群综合判断;
- feature_insights(feat) / compare_insights(feat_a, feat_b): 统一入口,
  feature_insights 优先使用 feature 记录内嵌的 analysis 字段（旧 JSON 无该字段时规则计算）。
"""
import re

FEAT_NAME_TIME_CMP = "cluster_time_compare_summary"
FEAT_NAME_FREE = "free_analysis"
FEAT_NAME_COMM_BN = "communication_bottleneck"

# ==================== 术语中文化 ====================

# 短语级映射（先整短语替换，大小写不敏感）
_PHRASES_ZH = [
    ("waiting for other ranks", "等待其他 Rank 对齐"),
    ("wait for other ranks", "等待其他 Rank 对齐"),
    ("collective communication", "集合通信"),
    ("communication wait", "通信等待"),
    ("host overhead", "Host 侧开销"),
    ("device wait", "Device 侧等待"),
    ("slow rank", "慢卡 Rank"),
    ("fast rank", "快卡 Rank"),
    ("slowest rank", "最慢 Rank"),
    ("fastest rank", "最快 Rank"),
]

# 词级映射（token 全词匹配，小写后查表）
_KEYWORD_ZH = {
    "communication": "通信", "comm": "通信", "collective": "集合通信", "p2p": "点对点",
    "compute": "计算", "calculating": "计算", "calculation": "计算",
    "host": "Host侧", "device": "Device侧",
    "wait": "等待", "waiting": "等待", "await": "等待",
    "sync": "同步", "synchronization": "同步", "synchronizing": "同步",
    "dispatch": "下发", "scheduling": "调度", "schedule": "调度",
    "copy": "拷贝", "copies": "拷贝", "kernel": "算子", "kernels": "算子",
    "memory": "内存", "bandwidth": "带宽", "network": "网络",
    "link": "链路", "links": "链路", "slow": "慢", "fast": "快",
    "slower": "较慢", "faster": "较快", "slowest": "最慢", "fastest": "最快",
    "gap": "差距", "latency": "时延", "overhead": "开销",
    "idle": "空闲", "free": "空闲", "congestion": "拥塞", "congested": "拥塞",
    "serial": "串行", "serialized": "串行化", "overlap": "重叠", "overlapped": "重叠",
    "hardware": "硬件", "software": "软件", "abnormal": "异常", "normal": "正常",
    "transit": "传输", "transfer": "传输", "transmission": "传输",
    "elapsed": "耗时", "duration": "时长", "total": "总",
    "other": "其他", "than": "高于", "and": "与", "between": "在…之间",
    "infer": "推断", "inferred": "推断", "reason": "原因", "cause": "成因",
    "uneven": "不均衡", "imbalance": "不均衡", "unbalanced": "不均衡",
    "insufficient": "不足", "high": "高", "low": "低",
    "higher": "更高", "lower": "更低", "increase": "增加", "increased": "增加",
    "decrease": "减少", "decreased": "减少", "large": "大", "small": "小",
}

# 技术标识符保留原文（算子名、字段名、ID 类 token 不翻译）
_PRESERVE = {
    "allreduce", "allgather", "reducescatter", "reduce", "broadcast", "alltoall",
    "send", "receive", "recv", "rank", "ranks", "rankid", "step", "steps",
    "hccl", "hccs", "aicore", "ai_core", "sdma", "rdma", "roce", "pcie",
    "npu", "cpu", "gpu", "tp", "pp", "dp", "mb", "gb", "ms", "us",
    "id", "ids", "op", "ops", "host_bug1", "host_bug2",
}
# host_bug1/2 仅为测试样例兜底; 任意 host_xxx 形态由下划线规则保留

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def zh_text(s):
    """英文成因/描述中文化。短语映射优先, 再做全词映射; 未命中 token 保留原文。"""
    if not s or not isinstance(s, str):
        return s or ""
    txt = s
    low = txt.lower()
    for en, cn in _PHRASES_ZH:
        if en in low:
            # 大小写不敏感整短语替换
            txt = re.sub(re.escape(en), cn, txt, flags=re.IGNORECASE)
            low = txt.lower()

    def _repl(m):
        tok = m.group(0)
        t = tok.lower()
        if t in _PRESERVE:
            return tok
        # 下划线标识符（如 host_bug1 / MWU_LOAD 风格变量）视为标识符保留
        if "_" in t and not t.startswith("_") and t not in _KEYWORD_ZH:
            return tok
        zh = _KEYWORD_ZH.get(t)
        if zh is None:
            return tok
        # 保持首词原样大小写观感: 原词首字母大写且中文为纯词时直接返回中文
        return zh

    txt = _TOKEN_RE.sub(_repl, txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt


# ==================== 单 feature 分析判断 ====================

def _label_of(feat):
    tag = feat.get("cluster")
    if tag == "A":
        return "集群A（基准）"
    if tag == "B":
        return "集群B（对比）"
    return None


def free_insights(data, label=None):
    """空闲时间成因聚合 → 中文结论列表。"""
    out = []
    total = float(data.get("total_ms") or 0)
    reasons = data.get("reasons") or []
    if not reasons or total <= 0:
        return ["未采集到空闲片段，无法给出成因判断。"]
    top = reasons[0]
    tr = zh_text(top.get("reason", ""))
    pct = float(top.get("pct") or 0)
    lead = f"{label}空闲总时长 {total} ms" if label else f"空闲总时长 {total} ms"
    out.append(f"{lead}，首要成因为「{tr}」（占比 {pct}%，{top.get('count', '-')} 次，"
               f"共 {top.get('total_ms', '-')} ms）")
    if pct >= 50:
        out.append(f"判断：空闲高度集中于「{tr}」，为优先排查方向，建议结合 Host 下发与同步链路确认")
    elif pct >= 30:
        out.append(f"判断：「{tr}」为主要成因但未过半，建议按占比顺序逐项排查前 2-3 项成因")
    else:
        out.append("判断：空闲成因分散，单一原因不足以解释，建议结合 Step 级时间线定位波动区间")
    if len(reasons) > 1:
        second = reasons[1]
        out.append(f"次要成因「{zh_text(second.get('reason', ''))}」占比 {second.get('pct', '-')}%，"
                   f"与首要成因合计 {round(pct + float(second.get('pct') or 0), 1)}%")
    rc = int(top.get("rank_count") or 0)
    if rc:
        if rc <= 4:
            out.append(f"判断：该成因仅影响约 {rc} 个 Rank（{top.get('sample_ranks', [])[:8]}），"
                       "指向个体慢卡或局部瓶颈，优先核对该卡链路与负载")
        else:
            out.append(f"判断：该成因波及约 {rc} 个 Rank，为全局性现象，"
                       "优先排查 Host 下发节奏、同步策略与通信对齐")
    return out


def comm_bn_insights(data, label=None):
    """通信瓶颈 Top N → 中文结论列表。"""
    items = data.get("items") or []
    lead = f"{label}" if label else ""
    if not items:
        return [f"{lead}未识别到显著通信瓶颈。" if lead else "未识别到显著通信瓶颈。"]
    out = []
    top = items[0]
    op = top.get("op", "-")
    dur = top.get("duration_ms", "-")
    slow, fast = top.get("slow_rank", "-"), top.get("fast_rank", "-")
    out.append(f"{'首要瓶颈' if not lead else lead + '首要瓶颈'}算子 {op}：慢卡 Rank {slow} "
               f"比快卡 Rank {fast} 平均多耗时 {dur} ms")
    reason = zh_text(top.get("reason", ""))
    if reason:
        out.append(f"工具推断原因：{reason}")
    # 算子集中度
    op_cnt = {}
    for it in items:
        op_cnt[it.get("op", "-")] = op_cnt.get(it.get("op", "-"), 0) + 1
    top_op, top_n = max(op_cnt.items(), key=lambda x: x[1])
    if top_n > 1:
        out.append(f"判断：瓶颈集中于算子 {top_op}（{top_n}/{len(items)} 条记录），"
                   "建议结合该算子的通信算法与切分策略排查")
    # 慢卡集中度
    slow_ranks = [str(it.get("slow_rank")) for it in items if it.get("slow_rank") not in (None, "-")]
    uniq = sorted(set(slow_ranks))
    if len(uniq) == 1:
        out.append(f"判断：慢卡集中于 Rank {uniq[0]}，疑似该卡链路或硬件异常，建议交叉核对该卡网络状态")
    elif len(uniq) > 1 and len(uniq) <= 3:
        out.append(f"判断：慢卡集中在少数 Rank（{'、'.join(uniq)}），为局部瓶颈")
    else:
        out.append(f"判断：慢卡分布较分散（{len(uniq)} 个 Rank），更可能是全局通信对齐问题而非单卡故障")
    return out


def time_cmp_insights(data):
    """双集群时间拆解对比 → 中文结论列表。"""
    overall = data.get("overall") or {}
    metrics = data.get("metrics") or []
    if not overall or not metrics:
        return ["对比数据为空，无法给出结论。"]
    out = []
    ranked = sorted(
        ((m, (overall.get(m) or {})) for m in metrics),
        key=lambda x: abs(float((x[1].get("diff_avg_ms") or 0))),
        reverse=True,
    )
    m0, o0 = ranked[0]
    d0 = o0.get("diff_avg_ms")
    out.append(f"「{m0}」均值差最大：基准 {o0.get('base_avg_ms', '-')} ms → 对比 "
               f"{o0.get('cur_avg_ms', '-')} ms（Δ {d0} ms）")
    pos = [m for m, o in ranked if float(o.get("diff_avg_ms") or 0) > 0]
    neg = [m for m, o in ranked if float(o.get("diff_avg_ms") or 0) < 0]
    if pos and not neg:
        out.append(f"判断：各分项（{'、'.join(pos)}）全面增加，整体呈劣化态势")
    elif neg and not pos:
        out.append(f"判断：各分项（{'、'.join(neg)}）全面减少，整体呈改善态势")
    else:
        out.append(f"判断：分项有增有减（增加：{'、'.join(pos) or '无'}；减少：{'、'.join(neg) or '无'}），"
                   "需以 Stage 净变化与贡献度为准")
    stage_m = next((m for m, _ in ranked if "stage" in m.lower()), None)
    if stage_m:
        sd = float(overall.get(stage_m, {}).get("diff_avg_ms") or 0)
        if abs(sd) > 1e-9:
            parts = []
            for m, o in ranked:
                if m == stage_m:
                    continue
                dm = float(o.get("diff_avg_ms") or 0)
                if dm:
                    parts.append(f"{m} 贡献 {round(dm / sd * 100, 1)}%")
            if parts:
                out.append("Stage 净变化归因：" + "，".join(parts))
    tops = data.get("top_ranks") or []
    if tops:
        t0 = tops[0]
        diff = None
        for m in metrics:
            v = t0.get(f"{m}_diff_ms")
            if v is not None:
                diff = v
                break
        if diff is not None:
            out.append(f"最差 Rank 为 {t0.get('rank', '-')}（{t0.get('step_count', '-')} 个 Step，"
                       f"差值均值 {diff} ms），建议优先核对该卡的链路状态与负载分布")
    return out


# ==================== A/B 对照综合判断 ====================

def free_compare(data_a, data_b):
    out = []
    ta = float(data_a.get("total_ms") or 0)
    tb = float(data_b.get("total_ms") or 0)
    ra = data_a.get("reasons") or []
    rb = data_b.get("reasons") or []
    if tb > ta and ta > 0:
        out.append(f"集群B 空闲总量较基准增加 {round(tb - ta, 2)} ms"
                   f"（+{round((tb - ta) / ta * 100, 1)}%），空闲劣化属实")
    elif tb < ta:
        out.append(f"集群B 空闲总量较基准减少 {round(ta - tb, 2)} ms，空闲改善")
    if ra and rb:
        ta_r, tb_r = zh_text(ra[0].get("reason", "")), zh_text(rb[0].get("reason", ""))
        if ta_r == tb_r:
            out.append(f"判断：两侧首要成因一致（{ta_r}），说明瓶颈类型未变、程度加深，"
                       "优先对比该成因的耗时与频次差异")
        else:
            out.append(f"判断：首要成因发生迁移（基准：{ta_r} → 对比：{tb_r}），"
                       "需重点排查对比集群新增成因的来源")
    return out


def comm_bn_compare(data_a, data_b):
    out = []
    ia = data_a.get("items") or []
    ib = data_b.get("items") or []
    if not ib:
        return ["对比集群未识别到显著通信瓶颈。"]
    if ia:
        ga = float(ia[0].get("duration_ms") or 0)
        gb = float(ib[0].get("duration_ms") or 0)
        if gb > ga and ga > 0:
            out.append(f"集群B 首要瓶颈差距 {gb} ms，大于基准 {ga} ms"
                       f"（+{round((gb - ga) / ga * 100, 1)}%），通信劣化加剧")
        elif gb < ga:
            out.append(f"集群B 首要瓶颈差距 {gb} ms，小于基准 {ga} ms，瓶颈差距收窄")
    ops_a = {it.get("op") for it in ia}
    only_b = [it.get("op") for it in ib if it.get("op") not in ops_a]
    if only_b:
        out.append(f"判断：对比集群出现基准不存在的瓶颈算子（{'、'.join(sorted(set(only_b))[:3])}），"
                   "重点排查其引入的通信量或切分变化")
    slow_b = {str(it.get("slow_rank")) for it in ib if it.get("slow_rank") not in (None, "-")}
    slow_a = {str(it.get("slow_rank")) for it in ia if it.get("slow_rank") not in (None, "-")}
    if slow_b and slow_b == slow_a and len(slow_b) == 1:
        out.append(f"判断：两侧慢卡均为 Rank {sorted(slow_b)[0]}，同一卡持续异常，"
                   "高度怀疑该卡硬件/链路问题")
    elif slow_a and slow_b and not (slow_a & slow_b):
        out.append("判断：两侧慢卡集合不重叠，排除固定单卡故障，更可能是负载或调度差异所致")
    return out


# ==================== 统一入口 ====================

def feature_insights(feat):
    """单 feature 中文分析判断: 优先内嵌 analysis 字段, 否则按数据规则计算。"""
    a = feat.get("analysis")
    if isinstance(a, list) and a:
        return [str(x) for x in a][:8]
    if feat.get("status") not in ("success", "cached"):
        return []
    data = feat.get("data") or {}
    name = feat.get("name", "")
    label = _label_of(feat)
    try:
        if name == FEAT_NAME_FREE:
            return free_insights(data, label)
        if name == FEAT_NAME_COMM_BN:
            return comm_bn_insights(data, label)
        if name == FEAT_NAME_TIME_CMP:
            return time_cmp_insights(data)
    except Exception:
        return []
    return []


def compare_insights(feat_a, feat_b):
    """双集群同 feature 综合判断（两侧均有成功数据时才有输出）。"""
    name = feat_a.get("name", "")
    ok = lambda f: f.get("status") in ("success", "cached") and (f.get("data") or {})
    if not (ok(feat_a) and ok(feat_b)):
        return []
    da, db = feat_a["data"], feat_b["data"]
    try:
        if name == FEAT_NAME_FREE:
            return free_compare(da, db)
        if name == FEAT_NAME_COMM_BN:
            return comm_bn_compare(da, db)
    except Exception:
        return []
    return []

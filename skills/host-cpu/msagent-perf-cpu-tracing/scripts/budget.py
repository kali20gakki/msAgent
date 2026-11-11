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

"""采集规模预算：卡 × 核集 × 事件类型 × 时长 -> 预估事件量 / 字节数。

硬约束（来自设计 §3.1.3）：
  - 采集单位 = 单卡核集（<=32 核）；全卡必须显式 all，逐卡串行
  - 事件类型：最小 4 / 标准 8 / 扩展 10
  - 单段时长：按矩阵，默认 30s
  - 单段事件量 <= 50 万；原始文本 <= 150MB；DB <= 200MB
超限按「事件集 -> 时长/分段 -> 核集 -> 拒绝」四步收窄。
"""

MAX_CPUS_PER_CARD = 32
MAX_EVENTS_PER_SEGMENT = 500_000
MAX_TEXT_BYTES = 150 * 1024 * 1024
MAX_DB_BYTES = 200 * 1024 * 1024
MAX_SEGMENTS_PER_CARD = 6
MAX_SESSION_SECONDS = 600

BYTES_PER_EVENT_DB = 150
BYTES_PER_EVENT_TEXT = 120

TIERS = {
    "min": ["sched_switch", "sched_wakeup", "irq_handler_entry", "irq_handler_exit"],
    "std": [
        "sched_switch", "sched_wakeup", "irq_handler_entry", "irq_handler_exit",
        "sched_wakeup_new", "sched_process_exit", "softirq_entry", "softirq_exit",
    ],
    "ext": [
        "sched_switch", "sched_wakeup", "irq_handler_entry", "irq_handler_exit",
        "sched_wakeup_new", "sched_process_exit", "softirq_entry", "softirq_exit",
        "sched_migrate_task", "page_fault_user",
    ],
}

MAX_EVENT_TYPES = 10

DURATION_MATRIX = {
    "min": [(8, 180), (16, 90), (24, 60), (32, 45)],
    "std": [(8, 120), (16, 60), (24, 45), (32, 30)],
    "ext": [(8, 60), (16, 30), (24, 20), (32, 15)],
}

REF_RATE_PER_CORE = 300.0


def events_for_tier(tier):
    return list(TIERS.get(tier) or TIERS["std"])


def duration_cap(cpus, tier):
    rows = DURATION_MATRIX.get(tier) or DURATION_MATRIX["std"]
    for limit, seconds in rows:
        if cpus <= limit:
            return seconds
    return rows[-1][1]


def estimate(cpus, tier, duration_s, rate_per_core=REF_RATE_PER_CORE, cards=1):
    events = int(rate_per_core * cpus * duration_s * max(1, cards))
    return {
        "cards": cards,
        "cpus": cpus,
        "tier": tier,
        "event_types": len(events_for_tier(tier)),
        "duration_s": duration_s,
        "rate_per_core": rate_per_core,
        "events": events,
        "db_bytes": events * BYTES_PER_EVENT_DB,
        "text_bytes": events * BYTES_PER_EVENT_TEXT,
    }


def _exceeds(budget):
    if budget["events"] > MAX_EVENTS_PER_SEGMENT:
        return True
    if budget["text_bytes"] > MAX_TEXT_BYTES:
        return True
    if budget["db_bytes"] > MAX_DB_BYTES:
        return True
    return False


def plan(cpus, tier="std", duration_s=30, cards=1, rate_per_core=REF_RATE_PER_CORE, verify=False):
    """返回 (ok, budget, adjustments)。

    verify 模式不可减核（否则失去验证意义）；单卡核集 >32 直接拒绝。
    自定义事件集保留 custom 语义（仅估算速率按 std 基线），且不做事件降档。
    """
    adjustments = []
    tier = tier if tier in TIERS else "custom"
    if cpus > MAX_CPUS_PER_CARD:
        return False, estimate(cpus, tier, duration_s, rate_per_core, cards), [
            {"step": "reject", "reason": "单卡核集超过 %d 核，非单卡核集" % MAX_CPUS_PER_CARD}
        ]
    # 多卡一律逐卡串行（每卡独立 instance 与 DB 段），不按卡数叠加单卡核集上限
    if cards > 1:
        adjustments.append({
            "step": "per-card",
            "reason": "全卡改为逐卡串行采集（每卡独立 instance 与 DB 段）",
            "cards_serial": cards,
        })

    cap = duration_cap(cpus, tier if tier in TIERS else "std")
    if duration_s > cap:
        adjustments.append({"step": "duration", "from": duration_s, "to": cap})
        duration_s = cap

    order = ["ext", "std", "min"]
    while True:
        budget = estimate(cpus, tier, duration_s, rate_per_core, 1)
        if not _exceeds(budget):
            break
        if tier in order:
            idx = order.index(tier)
            if idx + 1 < len(order):
                adjustments.append({"step": "events", "from": tier, "to": order[idx + 1]})
                tier = order[idx + 1]
                cap = duration_cap(cpus, tier)
                if duration_s > cap:
                    adjustments.append({"step": "duration", "from": duration_s, "to": cap})
                    duration_s = cap
                continue
        if duration_s > 10:
            new_duration = max(10, duration_s // 2)
            adjustments.append({"step": "duration", "from": duration_s, "to": new_duration})
            duration_s = new_duration
            budget = estimate(cpus, tier, duration_s, rate_per_core, 1)
            if not _exceeds(budget):
                break
        if not verify and cpus > 8:
            new_cpus = max(8, cpus // 2)
            adjustments.append({"step": "cpus", "from": cpus, "to": new_cpus})
            cpus = new_cpus
            continue
        budget["rejected"] = True
        adjustments.append({"step": "reject", "reason": "预算仍超限，建议改 verify 模式或缩小窗口"})
        return False, budget, adjustments

    budget = estimate(cpus, tier, duration_s, rate_per_core, 1)
    segments = 1
    if cards > 1:
        segments = cards
    budget["segments"] = segments
    budget["session_seconds"] = duration_s * segments
    budget["adjustments"] = adjustments
    ok = budget["session_seconds"] <= MAX_SESSION_SECONDS
    if not ok:
        adjustments.append({"step": "reject", "reason": "会话总时长超 10min，需分批执行"})
    return ok, budget, adjustments


def format_bytes(value):
    value = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return "%.1f%s" % (value, unit)
        value /= 1024.0
    return "%.1fGB" % value

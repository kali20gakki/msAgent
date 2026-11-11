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
"""baseline.py — hostcpu-baseline.json 归档与 diff 对比。

基线内容 = 8 域配置原始值（可回滚依据 + 复检对比）。只落 JSON，无 DB。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# 纳入基线的配置事实项（可回滚的配置项）
BASELINE_ITEMS = (
    "numa_balancing", "zone_reclaim_mode", "swappiness", "overcommit_memory",
    "thp_enabled", "thp_defrag", "hugepages_total", "hugepages_free",
    "scaling_governor", "epp", "cstates", "cmdline_boot_params",
    "cpuset_cpus_effective", "cpuset_mems_effective", "cpu_max",
    "cpu_stat_throttled", "cgroup_version",
)


def build_baseline(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从 facts 提取配置原始值生成基线。"""
    values: Dict[str, Any] = {}
    for fact in facts:
        if fact.get("item") in BASELINE_ITEMS:
            values[fact["item"]] = fact.get("value")
    return {
        "type": "hostcpu-baseline",
        "version": 1,
        "values": values,
    }


def save_baseline(baseline: Dict[str, Any], path: str) -> str:
    """基线落盘（路径由主入口生成的会话目录内，非用户任意路径）。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, ensure_ascii=False, indent=2)
    return path


def load_baseline(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# 每个配置项的优劣值域独立定义：同一取值在不同 key 下方向相反（如 numa_balancing
# 与 zone_reclaim_mode 的 0/1 语义相反），不能跨 key 共用一个值集合
_DIRECTION_RULES: Dict[str, tuple] = {
    "scaling_governor": ({"performance", "balance_performance"}, {"powersave", "balance_power"}),
    "thp_enabled": ({"madvise", "always"}, {"never"}),
    "thp_defrag": ({"madvise", "defer", "defer+madvise"}, {"always"}),
    "numa_balancing": ({"0"}, {"1"}),
    "zone_reclaim_mode": ({"0"}, {"1"}),
}


def _direction_value(key: str, value: Any) -> str:
    """scaling_governor 记录为 {governor: count}，取占比最高的 governor 作为方向依据。"""
    if key == "scaling_governor" and isinstance(value, dict) and value:
        return str(max(value.items(), key=lambda kv: kv[1])[0])
    return str(value)


def _direction(key: str, old: Any, new: Any) -> str:
    rule = _DIRECTION_RULES.get(key)
    if rule is None:
        return "changed"
    good_values, bad_values = rule
    old_value, new_value = _direction_value(key, old), _direction_value(key, new)
    if new_value in good_values and old_value in bad_values:
        return "improved"
    if new_value in bad_values and old_value in good_values:
        return "degraded"
    return "changed"


def diff_baseline(current: Dict[str, Any], previous: Dict[str, Any]) -> List[Dict[str, Any]]:
    """与上轮基线对比：改善 / 恶化 / 变化。

    判定规则：本轮值 vs 上轮值，按配置项自身的值域语义给方向。
    """
    diffs: List[Dict[str, Any]] = []
    prev_values = previous.get("values", {})
    curr_values = current.get("values", {})
    for key in sorted(set(prev_values) & set(curr_values)):
        old, new = prev_values[key], curr_values[key]
        if old == new:
            continue
        diffs.append({"item": key, "old": old, "new": new,
                      "direction": _direction(key, old, new)})
    return diffs

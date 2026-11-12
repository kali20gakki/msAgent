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
"""action_build.py — Action[] 装填与排序（8 字段统一契约）。

映射模板与排序权重来自 resources/action-templates.json（唯一来源，本脚本不内嵌文案）。

来源三分：
  1. config-audit 已产出的 Action（原样承接，补齐编排层字段）；
  2. 运行时 Finding 的 action_hint 映射（TASKSET_T3 / BIND_TUNE / IRQ_AFFINITY）；
  3. 等待维唤醒源映射（网络/设备 → IRQ/NIC 亲和；锁 → 绑核/并行度；IO/通信 → 建议 + 转出）。

排序：代价×收益（配置类优先）。绑核类动作标 bind_tune_handoff 并携带 suggested_cpuset，
由主会话转出 bind-tune 出 plan，本 skill 不产出精确绑核方案。
"""
from __future__ import annotations

import argparse
import json
import os

import workspace

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "action-templates.json")


def _load_templates(path=None):
    path = path or _TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Action[] 装填与排序")
    parser.add_argument("--session", required=True)
    parser.add_argument("--config-triage", default="")
    parser.add_argument("--trace-verdict", default="")
    parser.add_argument("--attribution", default="")
    parser.add_argument("--templates", default="", help="动作模板（默认 resources/action-templates.json）")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    config = workspace.load_json(
        args.config_triage or os.path.join(args.session, "00-config", "config_triage.json"),
        default={}) or {}
    trace = workspace.load_json(
        args.trace_verdict or os.path.join(args.session, "01-trace", "phase_verdict_1.json"),
        default={}) or {}
    attribution = workspace.load_json(
        args.attribution or os.path.join(args.session, "attribution.json"),
        default={}) or {}

    try:
        templates = _load_templates(args.templates)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": "动作模板不可用：%s" % exc},
                         ensure_ascii=False, indent=2))
        return 2
    if not isinstance(templates, dict):
        print(json.dumps({"ok": False, "error": "动作模板格式非法：需为 JSON 对象"},
                         ensure_ascii=False, indent=2))
        return 2

    actions = build_actions(config, trace, attribution, templates)
    out_path = args.out or os.path.join(args.session, "actions", "action_plan.json")
    workspace.write_json(out_path, {"actions": actions, "count": len(actions)})
    workspace.manifest_upsert(args.session, "actions", {
        "count": len(actions), "artifacts": [os.path.relpath(out_path, args.session)],
    })
    print(json.dumps({"count": len(actions),
                      "actions": [{"action_id": a["action_id"], "type": a["type"],
                                   "source": a["source"], "risk": a["risk"],
                                   "handoff": a["handoff"]} for a in actions]},
                     ensure_ascii=False, indent=2))
    return 0


def build_actions(config, trace, attribution, templates):
    actions = []
    actions.extend(_from_config(config))
    actions.extend(_from_trace(trace, templates.get("hint_templates") or {}))
    actions.extend(_from_wait(attribution, templates.get("wait_actions") or {}))
    return _sort_assign(actions, templates)


_REBOOT_LEVELS = ("L3",)


def _config_exec(item):
    """配置类动作的执行语义：优先取上游字段，缺省时按等级与回滚命令推导。

    L3 需运维落地（可能涉及重启），不得与 L1 一样一律标成运行时可回滚。
    """
    level = str(item.get("level") or "L1").upper()
    timing = item.get("timing") or ("reboot" if level in _REBOOT_LEVELS else "running")
    revertible = item.get("revertible")
    if revertible is None:
        revertible = bool(item.get("rollback"))
    restart = item.get("restart_required")
    if restart is None:
        restart = level in _REBOOT_LEVELS
    return timing, bool(revertible), bool(restart)


def _from_config(config):
    """承接 config-audit Action：补齐编排层字段（不改其命令与判定）。"""
    out = []
    for item in config.get("actions") or []:
        timing, revertible, restart = _config_exec(item)
        out.append({
            "type": item.get("type") or "SYSCTL",
            "scope": (item.get("target") or {}).get("scope") or "system_runtime",
            "timing": timing,
            "target": item.get("target") or {"scope": "system_runtime"},
            "params": item.get("params") or {},
            "commands": item.get("commands") or [],
            "rollback": item.get("rollback") or [],
            "expected_effect": item.get("expected_effect") or item.get("explain") or "",
            "verify": item.get("verify") or {},
            "risk": item.get("risk") or "low",
            "revertible": revertible,
            "restart_required": restart,
            "source": "config-audit",
            "source_id": item.get("action_id") or "",
            "handoff": "",
            "level": item.get("level") or "L1",
            "cost": "low",
            "benefit": "high" if item.get("risk") in ("high", "medium") else "medium",
            "explain": item.get("explain") or "",
        })
    return out


def _from_trace(trace, hint_templates):
    out = []
    for card in trace.get("cards") or []:
        if card.get("status") != "OBSERVED":
            continue
        for finding in card.get("findings") or []:
            hint = finding.get("action_hint") or ""
            template = hint_templates.get(hint)
            if not template or finding.get("verdict") != "FAIL":
                continue
            item = dict(template)
            item.update({
                "target": {"pid": None, "tids": _tids(card)},
                "params": {"suggested_cpuset": finding.get("suggested_cpuset") or ""},
                "commands": [],
                "rollback": [],
                "risk": "medium" if hint != "IRQ_AFFINITY" else "low",
                "revertible": True,
                "restart_required": hint == "BIND_TUNE",
                "source": "tracing",
                "source_id": finding.get("finding_id") or "",
                "card": card.get("card"),
                "explain": finding.get("problem") or "",
                "suggested_cpuset": finding.get("suggested_cpuset") or "",
                "level": "L0" if hint == "TASKSET_T3" else "L2",
            })
            out.append(item)
    return out


WAIT_ACTION_MIN_PCT = 5.0


def _from_wait(attribution, wait_actions):
    breakdown = attribution.get("wait_breakdown") or {}
    if not breakdown:
        return []
    cls, share = max(breakdown.items(), key=lambda kv: kv[1])
    template = wait_actions.get(cls)
    if not template:
        return []
    # breakdown 是占「睡眠时长」的内部份额；乘以整体等待占比才是占窗贡献，
    # 只看内部份额会把「等待本身很轻」的场景误判为值得动作为
    wait_pct = attribution.get("wait_pct")
    pct = round(float(share) * (100.0 if wait_pct is None else float(wait_pct)) / 100.0, 1)
    if pct < WAIT_ACTION_MIN_PCT:
        return []
    return [{
        "type": template["type"],
        "scope": template["scope"],
        "timing": template["timing"],
        "target": {"scope": template["scope"]},
        "params": {"wait_class": cls, "wait_pct": pct},
        "commands": [],
        "rollback": [],
        "expected_effect": "%s（等待占比 %.1f%%，唤醒源 %s）" % (
            template["expected_effect"], pct, cls),
        "verify": {"commands": [], "metrics": ["wait_pct", "wait_breakdown"],
                   "window": "复采同窗，对比等待占比与唤醒源分布"},
        "risk": "low" if template["type"] == "IRQ_AFFINITY" else "medium",
        "revertible": template["type"] != "CODE_SUGGESTION",
        "restart_required": template["timing"] == "next_start",
        "source": "perf-triage",
        "source_id": "wait-%s" % cls,
        "handoff": template["handoff"],
        "level": "L0" if template["type"] == "IRQ_AFFINITY" else "L4",
        "cost": template["cost"],
        "benefit": template["benefit"],
        "explain": template["hint"],
    }]


def _sort_assign(actions, templates):
    """代价×收益排序：配置类优先 → 收益降序 → 代价升序，之后按序重编号。"""
    benefit_rank = templates.get("benefit_rank") or {}
    cost_rank = templates.get("cost_rank") or {}

    def key(action):
        config_first = 1 if action.get("source") == "config-audit" else 0
        return (-config_first,
                -benefit_rank.get(action.get("benefit"), 0),
                -cost_rank.get(action.get("cost"), 0))

    ordered = sorted(actions, key=key)
    for index, action in enumerate(ordered, start=1):
        action["action_id"] = "A-%03d" % index
    return ordered


def _tids(card):
    victim = card.get("victim") or {}
    return [victim["tid"]] if victim.get("tid") is not None else []


if __name__ == "__main__":
    raise SystemExit(main())

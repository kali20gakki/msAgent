#!/usr/bin/python3
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

"""decompose.py —— 场景化拆解主引擎（阶段归因 + 单次执行边界 + 报告，一步到位）

三步流程（对齐 SKILL.md）：
1. 场景确认：CLI（--framework/--task-type）> 数据自动探测（STRING_IDS/MSTX 打点名）>
   兜底默认 + 告警（提示由编排层询问用户或重跑指定）。感知到的框架/任务类型写入
   stage.json，供下游 layer-decompose 通过 --framework 继承，无需重新探测。
2. 执行拆解：按场景资源文件 resources/scenarios/<scenario>/<framework>.json 的声明式
   规则执行——阶段归因（level=阶段）与单次执行边界（level=单次执行）。引擎框架无关，
   新框架 = 在场景目录新增一个 <framework>.json（支持 extends 继承），不改引擎。
3. 生成报告：按 references/report-template.md 呈现 stage/step 表格 + 守恒自检 +
   findings（8 字段），不再依赖独立 breakdown-report skill。

用法：
  python decompose.py --db <ascend_pytorch_profiler_*.db> \
                      [--framework vllm|sglang|verl|slime|megatron|mindspeed|fsdp] \
                      [--task-type inference|train|rl] \
                      [--anchor "forward"] [--inspect-index 0] [--skip-first-step] \
                      [--warmup-factor 3.0] [--min-steps 2] \
                      [--scenarios-dir ../resources/scenarios] \
                      [--output-dir .] [--prefix breakdown]

规则依据见 references/scenario-taxonomy.md、references/db-usage.md、
references/check-rules.md、references/report-template.md；场景规则见 resources/scenarios/。
"""

import argparse
import csv
import html
import json
import math
import os
import sqlite3
import statistics

from db_query import MSTX_TYPE, OP_TYPE, open_ro, query_mstx, query_pytorch_api

DEFAULT_SCENARIOS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "resources", "scenarios"))

# 框架 -> 任务类型兜底映射（场景资源文件存在时以其 scenario 字段为准；
# 新增框架只需在 resources/scenarios/<scenario>/ 放 <framework>.json，引擎自动识别）
FRAMEWORK_SCENARIO = {
    "vllm": "inference", "sglang": "inference",
    "megatron": "train", "mindspeed": "train", "fsdp": "train",
    "verl": "rl", "slime": "rl",
}

# verl worker 侧 MSTX 打点名（框架探测信号；见 references/frameworks/verl.md §2.1）
VERL_MARKERS = ("actor_update", "actor_compute_log_prob", "ref_compute_log_prob",
                "train_batch")

# 报告判定阈值（对齐 references/check-rules.md）
CONSERVATION_WARN_PCT = 0.05     # 守恒缺口 > 5% wall 告警
OUTLIER_K = 3.0                  # duration > P50 + k·σ 判 outlier
DRIFT_CV = 0.20                  # CV > 0.20 判漂移
DEFAULT_WARMUP_FACTOR = 3.0      # 热身阈值因子兜底（CLI > 场景资源 step_decompose.warmup_factor）

LEVEL_END_TO_END = "端到端"
LEVEL_STAGE = "阶段"
LEVEL_STEP = "单次执行"


# ---------------------------------------------------------------------------
# 场景确认：框架探测 + 场景资源加载
# ---------------------------------------------------------------------------

def detect_framework(cur):
    """框架探测（场景确认第 1 步，判断一次并写产物供下游 layer-decompose 继承）。

    规则：STRING_IDS.value 含 'vllm%' 前缀 → 'vllm'；否则存在 verl worker 侧
    MSTX 打点名 → 'verl'；否则 None。
    """
    for _ in cur.execute(
            "SELECT 1 FROM STRING_IDS WHERE value LIKE 'vllm%' LIMIT 1"):
        return "vllm"
    qmarks = ",".join("?" for _ in VERL_MARKERS)
    for _ in cur.execute(
            "SELECT 1 FROM STRING_IDS WHERE value IN (%s) LIMIT 1" % qmarks,
            list(VERL_MARKERS)):
        return "verl"
    return None


def load_scenario_registry(scenarios_dir):
    """扫描 resources/scenarios/<scenario>/*.json，返回 {framework: {scenario, data, path}}。

    新增框架 = 在对应场景目录放一个 <framework>.json，引擎自动识别，无需改代码。
    """
    reg = {}
    if not os.path.isdir(scenarios_dir):
        print("[WARN] 场景资源目录不存在：%s（用 --scenarios-dir 指定）" % scenarios_dir)
        return reg
    for scenario in sorted(os.listdir(scenarios_dir)):
        d = os.path.join(scenarios_dir, scenario)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(d, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print("[WARN] 场景资源解析失败 %s：%s" % (path, e))
                continue
            fw = data.get("framework") or os.path.splitext(fn)[0]
            reg[fw] = {"scenario": data.get("scenario", scenario),
                       "data": data, "path": path}
    return reg


def _merge_rules(base, child):
    """递归合并两套场景规则：child 未提供的键继承 base；stages/anchor 并集去重。

    对齐 frameworks/README.md extends 契约：
    - 子缺省（无该键 / 值为 None、[]、{}）→ 直接继承父值，不再静默丢弃；
    - 子提供 stages / anchor 列表 → 与父取并集去重；
    - 子提供 dict（如 stage_decompose / step_decompose）→ 递归深合并。
    """
    merged = json.loads(json.dumps(base)) if base else {}
    for k, v in (child or {}).items():
        if v in (None, [], {}):
            continue
        if k == "stages" and isinstance(v, list):
            names = {s["name"] for s in merged.get("stages", [])}
            for s in v:
                if s["name"] not in names:
                    merged.setdefault("stages", []).append(s)
        elif k == "anchor" and isinstance(v, list):
            merged["anchor"] = list(dict.fromkeys(
                list(merged.get("anchor", [])) + v))
        elif k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_rules(merged[k], v)
        else:
            merged[k] = v
    return merged


def merge_rules(base, child):
    """按 extends 契约合并两套场景规则（子缺省继承父；stages/anchor 并集去重）。

    用于 extends 继承：子框架在父规则上追加 scope / 覆盖字段，不改父文件。
    """
    return _merge_rules(base, child)


def get_scenario_rules(reg, fw, task):
    """按框架取场景规则（含 extends 继承），返回 (rules, source, path)。

    - 资源文件存在且 stages 已填充 → 直接采用（scenario-resource）；path 为该框架
      自身资源文件（extends 时仍指向子框架文件，便于人工按文件复核）。
    - 推理场景缺资源 / stages 未填充 → 回退 vllm 约定兜底并告警（scope 名可能
      不适用，需人工复核）；path 返回实际生效的 vllm.json，与规则内容同源。
    - 训练 / RL 场景缺资源或未填充 → 空规则占位（等待真实数据核验后回填，
      不回退推理规则）；path 指向框架自身资源文件（存在时），否则空串。
    """
    def _resolve(name, seen):
        seen = seen or set()
        if name in seen or name not in reg:
            return None
        seen.add(name)
        entry = reg[name]
        parent = entry["data"].get("extends")
        base = _resolve(parent, seen) if parent else None
        return merge_rules(base, entry["data"]) if base is not None else entry["data"]

    rules = _resolve(fw, set())
    if rules is not None and rules.get("stage_decompose", {}).get("stages"):
        return rules, "scenario-resource", reg[fw]["path"]
    if task == "inference":
        print("[WARN] 框架 %s 无场景资源或规则未填充（resources/scenarios/inference/%s.json"
              " 缺失或 stages 为空），按 vllm 约定兜底，scope 名可能不适用，请人工复核"
              % (fw, fw))
        base = _resolve("vllm", set())
        vllm_entry = reg.get("vllm", {})
        return base or {}, "fallback-vllm", vllm_entry.get("path", "")
    print("[WARN] 框架 %s 无场景资源或规则未填充（resources/scenarios/%s/%s.json），"
          "输出空阶段占位；待真实数据核验后回填" % (fw, task, fw))
    return {}, "placeholder", reg.get(fw, {}).get("path", "")


# ---------------------------------------------------------------------------
# DB 查询（场景拆解用，JOIN STRING_IDS 反查名；口径见 references/db-usage.md）
# ---------------------------------------------------------------------------

def query_scope_events(cur, names, backing):
    """按 backing 选择阶段/锚点事件来源，返回 [{start, end, name}, ...] 按 start 升序。

    - PYTORCH_API：推理场景 record_function 顶层 scope / mstx（JOIN name）。
    - MSTX_EVENTS：verl RL 场景低开销打点（JOIN message）。
    """
    if backing == "MSTX_EVENTS":
        return query_mstx(cur, list(names), "exact")
    return query_pytorch_api(cur, list(names), "exact", (OP_TYPE, MSTX_TYPE))


def query_scope_events_grouped(cur, names, backing):
    """按 backing 一次性取回多个 scope 的事件并按键分组（{name: [events]}）。

    同一 backing 表只扫一次（scope IN (...) + ORDER BY startNs），Python 侧按
    name 分组，避免逐 stage 重复全表扫描；各组内事件仍按 start 升序。
    """
    events = query_scope_events(cur, list(names), backing)
    groups = {}
    for e in events:
        groups.setdefault(e["name"], []).append(e)
    return groups


def session_span(cur):
    """SESSION_TIME_INFO -> (start, end)（ns），表缺失（OperationalError）返回 None。"""
    try:
        cur.execute("SELECT startTimeNs, endTimeNs FROM SESSION_TIME_INFO LIMIT 1")
        r = cur.fetchone()
        if r and r[0] is not None and r[1] is not None:
            return int(r[0]), int(r[1])
    except sqlite3.OperationalError:
        # 表缺失（no such table）为预期回退：会话跨度改由事件跨度推断；
        # 其余非预期错误（表损坏/权限/字段改名）向上抛出，避免口径被静默改变
        return None
    return None


def summarize(events):
    """给定 [{start,end}]（按 start 升序），返回 count/累计/均值/最大/最小/首个。"""
    durs = [e["end"] - e["start"] for e in events]
    if not durs:
        return {"count": 0, "total": 0, "mean": None, "max": None, "min": None,
                "first": None}
    return {
        "count": len(durs),
        "total": sum(durs),
        "mean": statistics.fmean(durs),
        "max": max(durs),
        "min": min(durs),
        "first": events[0],
    }


def pct(sorted_vals, q):
    """就近取分位（无插值，够用）。"""
    if not sorted_vals:
        return None
    k = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[k]


# ---------------------------------------------------------------------------
# 执行拆解：阶段归因（level=阶段）
# ---------------------------------------------------------------------------

def decompose_stages(cur, task, fw_name, rules, inspect_index, groups=None):
    """阶段归因。训练占位；推理/RL 按场景资源 stages 统计。

    同一 backing 的全部 stage scope 合并为一次查询（query_scope_events_grouped），
    避免逐 stage 全表扫描；groups 可由 main 预取，与 step 锚点共用。

    返回 (children, all_events)。all_events 供会话跨度回退与守恒自检。
    """
    if task == "train":
        print("[WARN] 训练阶段划分本轮未实现（LLM 训练逐 step 均质，不区分 stage；"
              "规则见 references/scenario-taxonomy.md）。输出空 children 占位。")
        return [], []

    sd = rules.get("stage_decompose", {})
    stages = sd.get("stages", [])
    backing = sd.get("backing", "PYTORCH_API")
    src_label = "MSTX 打点" if backing == "MSTX_EVENTS" else "record_function 顶层 scope"

    if groups is None:
        scope_names = [name for sg in stages for name in sg["scopes"]]
        groups = (query_scope_events_grouped(cur, scope_names, backing)
                  if scope_names else {})

    children, all_events = [], []
    for sg in stages:
        events = sorted((e for name in sg["scopes"] for e in groups.get(name, [])),
                        key=lambda e: e["start"])
        all_events += events
        st = summarize(events)
        occur = [{"index": i, "start": e["start"], "end": e["end"],
                  "wall": e["end"] - e["start"]} for i, e in enumerate(events)]
        node = {
            "level": LEVEL_STAGE,
            "name": sg["name"],
            "scope": sg["scopes"],
            "count": st["count"],
            "wall_time": st["total"],
            "self_time": st["total"],
            "wall_mean": st["mean"],
            "wall_max": st["max"],
            "wall_min": st["min"],
            "first": None,
            "confidence": {"method": "A", "level": "high",
                           "detail": "框架 %s 直接切（%s）" %
                                      (src_label, ", ".join(sg["scopes"]))},
        }
        if st["first"]:
            node["first"] = {"index": 0, "start": st["first"]["start"],
                             "end": st["first"]["end"],
                             "wall": st["first"]["end"] - st["first"]["start"]}
        if inspect_index is not None:
            node["inspect_index"] = inspect_index
            node["inspect"] = (occur[inspect_index]
                               if 0 <= inspect_index < len(occur) else None)
        children.append(node)

    if not stages:
        print("[WARN] 框架 %s 阶段规则为空（%s），输出空 children 占位" % (fw_name, rules and "placeholder"))
    return children, all_events


# ---------------------------------------------------------------------------
# 执行拆解：单次执行边界（level=单次执行）
# ---------------------------------------------------------------------------

def decompose_steps(cur, fw_name, rules, anchor_cli, warmup_factor,
                    skip_first_step, min_steps, groups=None, groups_backing=None):
    """单次执行边界：锚点 scope 切 step 序列 + 热身剔除 + 稳态统计。

    - 锚点优先级：--anchor 显式覆盖 > 场景资源 step_decompose.anchor。
    - 热身因子优先级：CLI --warmup-factor（显式指定）> 场景资源
      step_decompose.warmup_factor > 默认 DEFAULT_WARMUP_FACTOR。
    - groups / groups_backing：main 预取的 stage 事件分组（backing 一致时复用，
      锚点 scope 缺的再补查，避免重复全表扫描）。
    - 锚点为空（如 RL/训练待核验）→ 跳过 step 拆解，返回 None。
    返回单次执行分支节点 dict（无锚点或未命中时返回 None），并入统一拆解树。
    """
    sd = rules.get("step_decompose", {})
    if warmup_factor is None:
        warmup_factor = sd.get("warmup_factor", DEFAULT_WARMUP_FACTOR)
    if anchor_cli:
        anchors = set(a.strip() for a in anchor_cli.split(",") if a.strip())
    else:
        anchors = set(sd.get("anchor", []))
    if not anchors:
        print("[WARN] 框架 %s 无 step 边界锚点（step_decompose.anchor 为空），跳过 step 拆解"
              % fw_name)
        return None

    backing = sd.get("backing", "PYTORCH_API")
    if groups is not None and groups_backing == backing:
        # 复用 stage 查询结果：锚点已覆盖则直接用，缺失 scope 补查一次
        missing = [a for a in anchors if a not in groups]
        events = [e for a in anchors for e in groups.get(a, [])]
        if missing:
            events += [e for sub in query_scope_events_grouped(
                cur, missing, backing).values() for e in sub]
        raw = sorted(events, key=lambda e: e["start"])
    else:
        raw = query_scope_events(cur, anchors, backing)
    if not raw:
        print("[ERR] 未找到锚点 scope %s，请核对 --anchor/--framework"
              "（或框架 record_function/MSTX 打点未开启）" % sorted(anchors))
        return None

    # 热身判定：位置优先（前 position_warmup 步图捕获/编译），其余步阈值兜底；
    # position_warmup > 2 时中间位置步也带 reason="position"，避免显示为空
    pos_warmup = int(sd.get("position_warmup", 2))
    durs = sorted(s["end"] - s["start"] for s in raw)
    median = statistics.median(durs)

    steps = []
    for i, s in enumerate(raw):
        wall = s["end"] - s["start"]
        is_warmup = i < pos_warmup
        if i == 0:
            reason = "graph_capture"
        elif i == 1:
            reason = "compile"
        else:
            reason = "position" if is_warmup else None
        # 阈值兜底：wall 异常大时以 threshold 为准（覆盖位置语义）
        if wall > warmup_factor * median:
            is_warmup, reason = True, "threshold"
        steps.append({
            "level": LEVEL_STEP,
            "name": "step.%d" % i,
            "index": i,
            "start": s["start"],
            "end": s["end"],
            "wall_time": wall,
            "self_time": wall,      # 单次执行为叶节点，净自耗 = 墙钟
            "overlap": 0,
            "is_warmup": is_warmup,
            "warmup_reason": reason,
            "is_prefill": False,
        })

    # 首个 step（疑似 prefill/首次执行），标记后与热身一并排除出稳态
    if skip_first_step and steps:
        steps[0]["is_prefill"] = True

    steady = [s for s in steps if not s["is_warmup"] and not s["is_prefill"]]
    sdv = sorted(s["wall_time"] for s in steady)
    mean = statistics.fmean(sdv) if sdv else None
    variance = statistics.pvariance(sdv) if len(sdv) > 1 else (0.0 if sdv else None)
    # 单步稳态 variance=0.0 时显式令 cv=0.0，统一 HTML/JSON/控制台口径
    cv = (math.sqrt(variance) / mean) if (variance is not None and mean) else None

    out = {
        "level": LEVEL_STEP,
        "name": "单次执行边界 + per-step 序列",
        "framework": fw_name,
        "method": "A",
        "confidence": {
            "method": "A",
            "level": "high",
            "detail": "锚点 scope（%s）每次执行出现一次；热身按位置 + 阈值剔除" %
                      ", ".join(sorted(anchors)),
        },
        "anchor": sorted(anchors),
        "backing": backing,
        "num_steps": len(steps),
        "num_steady": len(steady),
        "warmup_indices": [s["index"] for s in steps if s["is_warmup"]],
        "prefill_indices": [s["index"] for s in steps if s["is_prefill"]],
        "skip_first_step": skip_first_step,
        "steps": steps,
        "stats": {
            "count": len(steady),
            "mean": mean,
            "p50": pct(sdv, 0.50),
            "p99": pct(sdv, 0.99),
            "min": sdv[0] if sdv else None,
            "max": sdv[-1] if sdv else None,
            "variance": variance,
            "cv": cv,
        },
    }
    if len(steps) < min_steps:
        print("[WARN] step 数 < %d，样本过小，统计不可靠" % min_steps)
    return out


# ---------------------------------------------------------------------------
# 生成报告：HTML + findings + 明细 CSV（模板见 references/report-template.md）
# ---------------------------------------------------------------------------

def _ms(ns):
    return ns / 1e6 if ns is not None else None


def _fmt_ms(ns, nd=2):
    v = _ms(ns)
    return ("%.*f" % (nd, v)) if v is not None else "-"


def _pct_of(part, whole):
    return (part / whole * 100.0) if whole else 0.0


def _csv_cell(v):
    """CSV 单元格安全化：转字符串；以 = + - @ 开头（Excel 公式注入向量）时加单引号前缀。"""
    s = str(v)
    return "'" + s if s.startswith(("=", "+", "-", "@")) else s


# 供 render_* 各章节小函数共用（等价于原 render_html 内局部 esc = html.escape）
_esc = html.escape


class _SectionCounter:
    """render_html 章节号计数器：只给实际渲染的章节递增编号（无数据章节不占号）。"""

    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n


def _derive_step_signals(steps):
    """从 step 产物收敛出稳态/最慢/异常/漂移信号（build_findings 与 render_html 共用）。

    统一口径：steady 过滤（剔除热身/prefill）、slowest_step、outlier 阈值
    （P50 + OUTLIER_K·sqrt(variance)，一次计算供筛选与展示复用）、drift 判定
    （CV > DRIFT_CV）。steps 为 None 或空时各字段取安全值。
    """
    step_list = steps.get("steps", []) if steps else []
    ss = steps.get("stats", {}) if steps else {}
    steady = [s for s in step_list
              if not s.get("is_warmup") and not s.get("is_prefill")]
    slowest_step = max(steady, key=lambda s: s.get("wall_time", 0)) if steady else None
    thr = None
    if ss.get("p50") is not None and ss.get("variance") is not None:
        thr = ss["p50"] + OUTLIER_K * math.sqrt(ss["variance"])
    outliers = [s for s in steady if thr is not None and s.get("wall_time", 0) > thr]
    return {
        "step_list": step_list,
        "ss": ss,
        "steady": steady,
        "slowest_step": slowest_step,
        "outlier_thr": thr,
        "outliers": outliers,
        "drift": ss.get("cv") is not None and ss.get("cv") > DRIFT_CV,
        "has_steps": bool(step_list),
        "num_steady": len(steady),
    }


def build_findings(stage, steps):
    """把两级产物派生判据收敛为统一 findings 数组（8 字段：问题/证据/影响/根因/
    优化动作/预期收益/验证路径/置信度），作为向工作流层 / 下游交接的结论最小单元。"""
    findings = []
    stages = stage.get("children", [])
    sess_wall = stage.get("wall_time")
    sess_gap = stage.get("conservation_gap")
    slowest_stage = max(stages, key=lambda s: s.get("self_time", 0)) if stages else None
    if slowest_stage is not None and slowest_stage.get("self_time", 0) <= 0:
        slowest_stage = None  # 阶段全为 0（无打点）时不产生「最慢阶段」结论

    sig = _derive_step_signals(steps)
    ss = sig["ss"]
    step_outliers = sig["outliers"]
    drift = sig["drift"]

    low_conf = []
    for s in stages:
        if s.get("confidence", {}).get("level") == "low":
            low_conf.append("阶段「%s」：%s" % (s["name"],
                            s.get("confidence", {}).get("detail", "")))
    if stage.get("task_type_source") == "default":
        low_conf.append("任务类型未自动判定，默认按推理处理，需人工确认场景")

    # F1 最耗时阶段（主线）
    if slowest_stage:
        findings.append({
            "问题": "最耗时阶段为「%s」" % slowest_stage["name"],
            "证据": "「%s」%s ms，占会话 %.1f%%；稳态单步 P50=%s ms（P99=%s ms）" % (
                slowest_stage["name"], _fmt_ms(slowest_stage["self_time"]),
                _pct_of(slowest_stage["self_time"], sess_wall),
                _fmt_ms(ss.get("p50")), _fmt_ms(ss.get("p99"))),
            "影响": "决定整体吞吐 / TTFT / TPOT 上限",
            "根因": "时间归因层面：最耗时阶段（未下钻算子级）",
            "优化动作": "沿 TOP-N 下钻链定位：最慢阶段 → 最慢 step →（待 layer-decompose 恢复后）单层/层内",
            "预期收益": "待 A/B 验证",
            "验证路径": "clean 跑 + compare_runs 对比 TTFT/TPOT/吞吐",
            "置信度": "high",
        })

    # F2 step outlier
    if step_outliers:
        ids = ",".join("step#%d" % s["index"] for s in step_outliers[:5])
        findings.append({
            "问题": "检出 %d 个 step outlier（>P50+%dσ）" % (len(step_outliers), OUTLIER_K),
            "证据": "%s 耗时超阈值（P50+%dσ=%s ms）" % (
                ids, OUTLIER_K,
                _fmt_ms(sig["outlier_thr"]) if sig["outlier_thr"] is not None else "-"),
            "影响": "拉高 P99 / 尾延迟",
            "根因": "待下钻（可能与样本长度/缓存/调度有关）",
            "优化动作": "下钻 outlier step 与样本时间/负载的对应关系",
            "预期收益": "稳定尾延迟",
            "验证路径": "定位后 clean 跑观察 P99 收敛",
            "置信度": "high",
        })

    # F3 漂移
    if drift:
        findings.append({
            "问题": "稳态 per-step 波动偏大（CV=%.2f>%.2f）" % (ss.get("cv"), DRIFT_CV),
            "证据": "稳态 per-step CV=%.3f" % ss.get("cv"),
            "影响": "吞吐不稳定，预测性差",
            "根因": "待查（负载/缓存/通信抖动）",
            "优化动作": "核查负载、KV cache、通信抖动来源",
            "预期收益": "降低 CV、稳定吞吐",
            "验证路径": "优化后对比 clean 跑 CV",
            "置信度": "high",
        })

    # F4 守恒缺口超阈值
    gap_pct = _pct_of(sess_gap, sess_wall) if sess_gap is not None else None
    if gap_pct is not None and gap_pct > CONSERVATION_WARN_PCT * 100:
        findings.append({
            "问题": "阶段归因守恒缺口 %.2f%% 超阈值（>5%%）" % gap_pct,
            "证据": "会话 wall=%.2f ms，Σ阶段 self_time=%.2f ms，缺口 %.2f ms" % (
                _ms(sess_wall), _ms(sess_wall - sess_gap) if sess_gap is not None else 0,
                _ms(sess_gap)),
            "影响": "归属不完整，阶段占比结论可信度下降",
            "根因": "未归因事件（重叠/自定义 scope/表缺失）",
            "优化动作": "核对打点是否开启（record_function/MSTX），或人工复核未覆盖 scope",
            "预期收益": "守恒自检通过，结论可信",
            "验证路径": "补采打点后重跑 decompose",
            "置信度": "medium",
        })

    # F5 低置信度项
    if low_conf:
        findings.append({
            "问题": "存在 %d 项低置信度归因" % len(low_conf),
            "证据": "；".join(low_conf),
            "影响": "相关结论不可直接采信，需人工复核",
            "根因": "无框架打点 / 场景未确认 / 规则回退兜底",
            "优化动作": "人工确认场景（--framework/--task-type）；补采打点",
            "预期收益": "提升归因可信度",
            "验证路径": "确认后重跑 decompose 复核置信度",
            "置信度": "low",
        })

    return findings


def _render_level_table(counter, stage, stages, ss):
    """逐级耗时表：端到端/阶段/稳态 step 的 wall/self/占比/置信度。"""
    sess_wall = stage.get("wall_time")
    sec = counter.next()
    h = ["<h2>%d. 逐级耗时</h2><table><tr><th>级别</th><th>对象</th>"
         "<th class='num'>wall_time</th><th class='num'>self_time</th>"
         "<th class='num'>占比</th><th>置信度</th></tr>" % sec]
    h.append("<tr><td>端到端</td><td>会话</td><td class='num'>%.1f ms</td><td class='num'>%.1f ms</td>"
             "<td class='num'>100%%</td><td>-</td></tr>" %
             (_ms(sess_wall), _ms(stage.get("self_time"))))
    for s in stages:
        if s.get("self_time", 0) <= 0:
            continue
        lv = s.get("confidence", {}).get("level", "high")
        cls = {"high": "hi", "medium": "med", "low": "lo"}.get(lv, "hi")
        h.append("<tr><td>阶段</td><td>%s</td><td class='num'>%.2f ms</td>"
                 "<td class='num'>%.2f ms</td><td class='num'>%.1f%%</td>"
                 "<td><span class='tag %s'>%s</span></td></tr>" %
                 (_esc(s["name"]), _ms(s["wall_time"]), _ms(s["self_time"]),
                  _pct_of(s["self_time"], sess_wall), cls, lv))
    if ss.get("p50") is not None:
        h.append("<tr><td>单次执行</td><td>稳态 step（P50）</td><td class='num'>%.2f ms</td>"
                 "<td class='num'>-</td><td class='num'>-</td><td>-</td></tr>" % _ms(ss["p50"]))
    h.append("</table>")
    return h


def _render_stage_stats(counter, stages):
    """阶段统计表（次数/累计/均值/最大/最小/首个）；无数据阶段不占章节号。"""
    stat_stages = [s for s in stages if s.get("count")]
    if not stat_stages:
        return []
    sec = counter.next()
    h = ["<h2>%d. 阶段统计</h2><table><tr><th>stage</th>"
         "<th class='num'>次数</th><th class='num'>累计</th><th class='num'>均值</th>"
         "<th class='num'>最大</th><th class='num'>最小</th><th class='num'>首个</th></tr>" % sec]
    for s in stat_stages:
        first_s = ("%.3f ms" % _ms(s["first"]["wall"])) if s.get("first") else "-"
        h.append("<tr><td>%s</td><td class='num'>%d</td><td class='num'>%.2f ms</td>"
                 "<td class='num'>%.3f ms</td><td class='num'>%.3f ms</td>"
                 "<td class='num'>%.3f ms</td><td class='num'>%s</td></tr>" %
                 (_esc(s["name"]), s["count"], _ms(s["wall_time"]),
                  _ms(s.get("wall_mean")), _ms(s.get("wall_max")),
                  _ms(s.get("wall_min")), first_s))
    h.append("</table>")
    return h


def _render_drilldown_chain(counter, slowest_stage, slowest_step):
    """TOP-N 下钻链：阶段 → 单次执行。"""
    sec = counter.next()
    h = ["<h2>%d. TOP-N 下钻链</h2><table><tr><th>级别</th><th>最慢对象</th><th>耗时</th></tr>" % sec]
    chain = [
        ("阶段", slowest_stage["name"] if slowest_stage else "-",
         _fmt_ms(slowest_stage["self_time"]) if slowest_stage else "-"),
        ("单次执行", ("step#%d" % slowest_step["index"]) if slowest_step else "-",
         _fmt_ms(slowest_step["wall_time"]) if slowest_step else "-"),
    ]
    for lv, obj, t in chain:
        h.append("<tr><td>%s</td><td>%s</td><td class='num'>%s</td></tr>" % (lv, _esc(obj), t))
    h.append("</table>")
    return h


def _render_step_detail(counter, sig):
    """per-step 明细表（wall_time/热身/原因/prefill）；无 step 数据不占章节号。"""
    if not sig["has_steps"]:
        return []
    sec = counter.next()
    h = [("<h2>%d. per-step 明细（%d 步，稳态 %d）</h2><table><tr><th>step</th>"
          "<th class='num'>wall_time</th><th>热身</th><th>原因</th><th>prefill</th></tr>")
         % (sec, len(sig["step_list"]), sig["num_steady"])]
    for s in sig["step_list"]:
        warm = ("<span class='warn'>是</span>" if s.get("is_warmup") else "否")
        pre = ("<span class='warn'>是</span>" if s.get("is_prefill") else "否")
        reason = s.get("warmup_reason") or "-"
        h.append("<tr><td>%s</td><td class='num'>%.3f ms</td><td>%s</td><td>%s</td><td>%s</td></tr>" %
                 (_esc(s["name"]), _ms(s["wall_time"]), warm, _esc(reason), pre))
    h.append("</table>")
    return h


def _render_steady_stats(counter, ss, has_steps):
    """稳态统计（剔除热身/prefill）；无 step 数据不占章节号。"""
    if not has_steps:
        return []
    sec = counter.next()
    h = ["<h2>%d. 稳态统计（剔除热身/prefill）</h2><table><tr><th class='num'>count</th>"
         "<th class='num'>mean</th><th class='num'>P50</th><th class='num'>P99</th>"
         "<th class='num'>min</th><th class='num'>max</th><th class='num'>CV</th></tr>" % sec]
    h.append("<tr><td class='num'>%d</td><td class='num'>%s</td><td class='num'>%s</td>"
             "<td class='num'>%s</td><td class='num'>%s</td><td class='num'>%s</td>"
             "<td class='num'>%s</td></tr>" %
             (ss.get("count", 0), _fmt_ms(ss.get("mean"), 3), _fmt_ms(ss.get("p50"), 3),
              _fmt_ms(ss.get("p99"), 3), _fmt_ms(ss.get("min"), 3),
              _fmt_ms(ss.get("max"), 3),
              ("%.3f" % ss["cv"]) if ss.get("cv") is not None else "-"))
    h.append("</table>")
    return h


def _render_anomalies(counter, sig):
    """异常 / 漂移清单。"""
    sec = counter.next()
    h = ["<h2>%d. 异常 / 漂移</h2><ul>" % sec]
    if sig["has_steps"]:
        h.append("<li>per-step：稳态 %d 步，CV=%.3f%s；outlier %d 步%s。</li>" %
                 (sig["num_steady"], sig["ss"].get("cv") or 0,
                  "（漂移）" if sig["drift"] else "",
                  len(sig["outliers"]),
                  "：" + ",".join("step#%d" % s["index"] for s in sig["outliers"][:5])
                  if sig["outliers"] else ""))
    else:
        h.append("<li>step 拆解未启用（无锚点），无 per-step 波动判定。</li>")
    h.append("</ul>")
    return h


def _render_low_confidence(counter, stages):
    """低置信度清单。"""
    low_conf = [s for s in stages if s.get("confidence", {}).get("level") == "low"]
    sec = counter.next()
    h = ["<h2>%d. 低置信度清单</h2>" % sec]
    if low_conf:
        h.append("<ul>")
        for s in low_conf:
            h.append("<li class='warn'>阶段「%s」：%s</li>" %
                     (_esc(s["name"]), _esc(s.get("confidence", {}).get("detail", ""))))
        h.append("</ul>")
    else:
        h.append("<p class='ok'>无低置信度项。</p>")
    return h


def _render_next_steps(counter, stage, sig):
    """建议下一步（不做根因判定）。"""
    sec = counter.next()
    h = ["<h2>%d. 建议下一步（不做根因判定）</h2><ul>" % sec]
    if stage.get("task_type_source") == "default" or stage.get("framework_source") == "default":
        h.append("<li class='warn'>框架/任务类型为兜底默认值，建议确认场景（--framework/--task-type）后重跑。</li>")
    if sig["outliers"]:
        h.append("<li>检出 %d 个 step outlier（>P50+%dσ），建议下钻其与样本时间的对应关系。</li>" %
                 (len(sig["outliers"]), OUTLIER_K))
    if sig["drift"]:
        h.append("<li>per-step CV=%.2f&gt;%.2f，存在波动，建议核查负载/缓存/通信抖动。</li>" %
                 (sig["ss"].get("cv"), DRIFT_CV))
    if not sig["has_steps"]:
        h.append("<li>本次未做 step 拆解（框架无边界锚点或未指定 --anchor），如需 per-step 序列请补资源规则。</li>")
    h.append("<li>层拆解（layer-decompose）暂缓：本报告仅覆盖阶段与单次执行两级；框架/任务类型已写入 stage.json，供其经 --framework 继承。</li>")
    h.append("</ul>")
    return h


def _render_findings(counter, findings):
    """结构化结论（findings，8 字段）。"""
    sec = counter.next()
    h = ["<h2>%d. 结构化结论（findings，8 字段）</h2>" % sec]
    if findings:
        h.append("<ol>")
        for fd in findings:
            h.append("<li><b>%s</b> <span class='tag %s'>%s</span><br>"
                     "证据：%s<br>根因：%s<br>优化动作：%s<br>验证路径：%s</li>" %
                     (_esc(fd["问题"]),
                      {"high": "hi", "medium": "med", "low": "lo"}.get(fd["置信度"], "hi"),
                      _esc(fd["置信度"]), _esc(fd["证据"]), _esc(fd["根因"]),
                      _esc(fd["优化动作"]), _esc(fd["验证路径"])))
        h.append("</ol>")
    else:
        h.append("<p class='ok'>无结构化结论。</p>")
    return h


def render_html(stage, steps, findings, args):
    """按 references/report-template.md 组织结论先行 HTML（自包含，无 JS 依赖）。

    各章节由 render_* 小函数拼装（章节号经 _SectionCounter 按实际渲染顺序递增），
    主函数只负责章节编排与一句话结论。
    """
    stages = stage.get("children", [])
    sess_wall = stage.get("wall_time")
    slowest_stage = max(stages, key=lambda s: s.get("self_time", 0)) if stages else None
    if slowest_stage is not None and slowest_stage.get("self_time", 0) <= 0:
        slowest_stage = None  # 阶段全为 0（无打点）时不产生「最慢阶段」结论

    sig = _derive_step_signals(steps)
    ss = sig["ss"]

    # 一句话结论
    parts = []
    if slowest_stage:
        parts.append("最耗时阶段为「%s」（%s ms，占会话 %.1f%%）" %
                     (slowest_stage["name"], _fmt_ms(slowest_stage["self_time"]),
                      _pct_of(slowest_stage["self_time"], sess_wall)))
    if ss.get("p50") is not None:
        parts.append("稳态单步 P50=%.2f ms（P99=%.2f ms）" % (_ms(ss["p50"]), _ms(ss.get("p99"))))
    conclusion = "；".join(parts) + "。" if parts else "未检出显著阶段/step 热点。"

    counter = _SectionCounter()
    h = []
    h.append("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    h.append("<title>性能拆解报告</title><style>")
    h.append("body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;max-width:1080px;margin:24px auto;"
             "padding:0 16px;color:#1a1a1a;line-height:1.5}")
    h.append("h1{font-size:22px;border-bottom:2px solid #2c6fbb;padding-bottom:8px}")
    h.append("h2{font-size:17px;margin-top:28px;color:#2c6fbb}")
    h.append(".concl{background:#eef5fd;border-left:4px solid #2c6fbb;padding:12px 16px;border-radius:4px;font-size:15px}")
    h.append("table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}")
    h.append("th,td{border:1px solid #d0d7de;padding:5px 8px;text-align:left}")
    h.append("th{background:#f6f8fa}")
    h.append(".num{text-align:right;font-variant-numeric:tabular-nums}")
    h.append(".warn{color:#b00020;font-weight:600}.ok{color:#0a7d32;font-weight:600}")
    h.append(".tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}")
    h.append(".hi{background:#d4edda}.med{background:#fff3cd}.lo{background:#f8d7da}")
    h.append("</style></head><body>")

    h.append("<h1>性能拆解报告</h1>")
    h.append("<div class='concl'><b>一句话结论：</b>%s</div>" % _esc(conclusion))

    h.extend(_render_level_table(counter, stage, stages, ss))
    h.extend(_render_stage_stats(counter, stages))
    h.extend(_render_drilldown_chain(counter, slowest_stage, sig["slowest_step"]))
    h.extend(_render_step_detail(counter, sig))
    h.extend(_render_steady_stats(counter, ss, sig["has_steps"]))
    h.extend(_render_anomalies(counter, sig))
    h.extend(_render_low_confidence(counter, stages))
    h.extend(_render_next_steps(counter, stage, sig))
    h.extend(_render_findings(counter, findings))

    h.append("<footer style='margin-top:32px;color:#888;font-size:12px'>"
             "由 msagent-profiler-breakdown(decompose.py) 生成 · 场景=%s/%s · 未启用层拆解</footer>"
             % (_esc(stage.get("task_type", "-")), _esc(stage.get("framework", "-"))))
    h.append("</body></html>")
    return "\n".join(h)


def write_outputs(args, tree, steps, findings, html_str):
    """写 breakdown.json（统一拆解树）/ stage.json（别名）/ 报告 / findings / 明细 CSV。

    tree：统一拆解树（顶层端到端 -> children 阶段 + 单次执行分支）。
    stage.json 与 breakdown.json 同一内容，作为兼容别名（场景信息供 layer-decompose
    经 --framework 继承）；steps.json 不再单独产出。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    tree_path = os.path.join(args.output_dir, "breakdown.json")
    stage_path = os.path.join(args.output_dir, "stage.json")
    html_path = os.path.join(args.output_dir, args.prefix + ".html")
    findings_path = os.path.join(args.output_dir, args.prefix + "_findings.json")
    detail_path = os.path.join(args.output_dir, args.prefix + "_detail.csv")
    handoff_path = os.path.join(args.output_dir, args.prefix + "_handoff.csv")

    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    with open(stage_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    with open(findings_path, "w", encoding="utf-8") as f:
        json.dump({"findings": findings}, f, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "start_ns", "end_ns", "wall_time_ms",
                    "is_warmup", "warmup_reason", "is_prefill"])
        for s in (steps.get("steps", []) if steps else []):
            w.writerow([s["index"], s["name"], s["start"], s["end"],
                        _csv_cell("%.3f" % (s["wall_time"] / 1e6)),
                        s["is_warmup"], _csv_cell(s.get("warmup_reason") or ""),
                        s["is_prefill"]])
    with open(handoff_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "start", "end", "device", "confidence", "note"])
        w.writerow(["-", "-", "-", "-", "-",
                    _csv_cell("未启用层拆解（layer-decompose 暂缓），框架/任务类型见 stage.json")])

    print("[OK] %s" % tree_path)
    print("[OK] %s（breakdown.json 别名）" % stage_path)
    print("[OK] %s" % html_path)
    print("[OK] %s（%d 条 findings）" % (findings_path, len(findings)))
    print("[OK] %s" % detail_path)
    print("[OK] %s" % handoff_path)
    return html_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="场景化拆解（阶段归因 + 单次执行边界 + 报告，一步到位）")
    p.add_argument("--db", required=True, help="ascend_pytorch_profiler_{rank_id}.db 路径")
    p.add_argument("--framework", default=None,
                   help="框架名（场景确认：显式指定 > 自动探测 > 默认 vllm）；感知结果写入 stage.json 供 layer-decompose 继承")
    p.add_argument("--task-type", default=None, choices=["inference", "train", "rl"],
                   help="任务类型（场景确认：显式指定 > 框架映射 > 默认推理 + 告警）")
    p.add_argument("--anchor", default=None,
                   help="覆盖 step 边界锚点 scope 名，逗号分隔（覆盖场景资源的默认锚点）")
    p.add_argument("--inspect-index", type=int, default=None,
                   help="额外输出第 N 次（0 起）stage 的起止/耗时")
    p.add_argument("--skip-first-step", action="store_true",
                   help="首个 step 标为疑似 prefill，剔除出稳态统计")
    p.add_argument("--warmup-factor", type=float, default=None,
                   help="位置判热身后，其余步 wall_time > factor×median 判为热身；"
                        "缺省取场景资源 step_decompose.warmup_factor，再缺省 3.0")
    p.add_argument("--min-steps", type=int, default=2, help="最少 step 数，否则告警（样本过小）")
    p.add_argument("--scenarios-dir", default=DEFAULT_SCENARIOS_DIR, help="场景资源目录")
    p.add_argument("--output-dir", default=".", help="输出目录（breakdown.json/stage.json/报告）")
    p.add_argument("--prefix", default="breakdown", help="输出文件名前缀")
    args = p.parse_args(argv)

    conn = open_ro(args.db)
    cur = conn.cursor()

    # ---------- 1. 场景确认 ----------
    detected = detect_framework(cur)
    fw_name = args.framework or detected or "vllm"
    fw_source = "cli" if args.framework else ("detected" if detected else "default")

    task = args.task_type
    task_source = "cli" if args.task_type else None
    if task is None:
        base = args.framework or detected
        mapped = FRAMEWORK_SCENARIO.get(base)
        if mapped:
            task = mapped
            task_source = "detected"
        else:
            # 无法自动判定：默认按推理处理，显式告警提示编排层询问用户限定场景
            task = "inference"
            task_source = "default"
            print("[WARN] 无法自动判定任务类型（训练/推理/RL）。默认按「推理」处理；"
                  "若为训练/RL 请用 --task-type 指定，或由编排层询问用户后重跑。")

    reg = load_scenario_registry(args.scenarios_dir)
    rules, rules_source, rules_path = get_scenario_rules(reg, fw_name, task)

    # ---------- 2. 执行拆解 ----------
    # 同一 backing 事件预取一次（阶段 scope 与 step 锚点共用），避免逐 stage 全表扫描
    stage_backing = None
    stage_groups = None
    if task != "train":
        sd = rules.get("stage_decompose", {})
        if sd.get("stages"):
            stage_backing = sd.get("backing", "PYTORCH_API")
            stage_groups = query_scope_events_grouped(
                cur, [name for sg in sd["stages"] for name in sg["scopes"]],
                stage_backing)

    stage_children, all_events = decompose_stages(
        cur, task, fw_name, rules, args.inspect_index, stage_groups)

    sess = session_span(cur)
    if sess:
        sess_start, sess_end = sess
    else:
        spans = [(e["start"], e["end"]) for e in all_events]
        sess_start = min(s for s, _ in spans) if spans else 0
        sess_end = max(e for _, e in spans) if spans else 0
    sess_wall = sess_end - sess_start
    total_self = sum(c["self_time"] for c in stage_children)

    stage = {
        "level": LEVEL_END_TO_END,
        "name": "端到端（会话）",
        "task_type": task,
        "task_type_source": task_source,
        "framework": fw_name,
        "framework_source": fw_source,
        "scenario_rules_source": rules_source,
        "scenario_rules_path": rules_path,
        "start": sess_start,
        "end": sess_end,
        "wall_time": sess_wall,
        "self_time": total_self,
        "overlap": max(sess_wall - total_self, 0),
        "conservation_gap": sess_wall - total_self,
        "children": stage_children,
    }

    steps = decompose_steps(
        cur, fw_name, rules, args.anchor, args.warmup_factor,
        args.skip_first_step, args.min_steps, stage_groups, stage_backing)
    conn.close()

    # ---------- 3. 生成报告 ----------
    findings = build_findings(stage, steps)
    html_str = render_html(stage, steps, findings, args)
    # 统一拆解树：顶层端到端 -> children（阶段... + 单次执行分支）
    tree = dict(stage)
    tree["children"] = (list(stage.get("children", []))
                        + ([steps] if steps is not None else []))
    html_path = write_outputs(args, tree, steps, findings, html_str)

    # ---------- 控制台摘要 ----------
    print("[OK] 阶段归因（任务类型=%s，来源=%s；框架=%s，来源=%s）" %
          (task, task_source, fw_name, fw_source))
    for c in stage_children:
        extra = ""
        if c.get("count"):
            extra = " count=%d mean=%.2fms max=%.2fms min=%.2fms" % (
                c["count"], c["wall_mean"] / 1e6,
                c["wall_max"] / 1e6, c["wall_min"] / 1e6)
            if c.get("first"):
                extra += " first=%.2fms" % (c["first"]["wall"] / 1e6)
        print("  %-14s total=%8.2fms%s" % (c["name"], c["wall_time"] / 1e6, extra))
    print("[INFO] 会话 wall=%.3fs, stage 累计=%.3fs, 守恒缺口=%.3fs" %
          (sess_wall / 1e9, total_self / 1e9, (sess_wall - total_self) / 1e9))
    if steps:
        print("[OK] %d steps（稳态 %d，热身 %d）-> breakdown.json" %
              (steps["num_steps"], steps["num_steady"],
               len(steps["steps"]) - steps["num_steady"]))
        if steps["stats"]["p50"] is not None:
            print("[INFO] 稳态 per-step P50=%.3fms P99=%.3fms CV=%.3f" %
                  (steps["stats"]["p50"] / 1e6, steps["stats"]["p99"] / 1e6,
                   steps["stats"]["cv"] or 0.0))
    print("[INFO] 报告：%s" % html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

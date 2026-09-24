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
"""提取结果指标校验器（verify_extracted_metrics.py）

用途:
  替代 Windows PowerShell 下易引号冲突的内联 `python -c`，对提取器产出的 JSON 做
  快速交叉校验。核心目标：识别"均值带宽下降但有效吞吐一致"的统计口径陷阱，
  避免把流量构成变化误判为链路劣化。

用法:
  python verify_extracted_metrics.py --data-a cluster_a.json
  python verify_extracted_metrics.py --data-a cluster_a.json --data-b cluster_b.json

输出:
  单集群: Step 均值(ms)、Rank 离散度（慢卡线索）、各带宽类型均值带宽与有效吞吐(GB/s)、
          算子列表 Total 汇总行残留检查。
  双集群: 上述指标并排对比 + 相对差 + 带宽口径判定提示。
"""
import argparse
import json


TOTAL_PREFIX = "total"


def to_ms(v, fmt):
    """DB 模式 step/rank 时间单位为 μs，TEXT 模式为 ms —— 统一转 ms。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if fmt == "text" else v / 1000.0


def get_eff_bw(entry):
    """有效吞吐 = 总传输量/总传输时间；旧 JSON 无 effective_bw 字段时按 avg_size/avg_time 现算。"""
    eff = entry.get("effective_bw")
    if eff is not None:
        try:
            return float(eff)
        except (TypeError, ValueError):
            pass
    size = float(entry.get("avg_size", 0) or 0)
    t = float(entry.get("avg_time", 0) or 0)
    return round(size / t, 4) if t > 1e-9 else 0.0


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def rel_diff(b, a):
    return (b - a) / a * 100 if a else 0.0


def op_name(op):
    return op.get("op_name") or op.get("hccl_op_name") or "?"


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    fmt = d.get("format", "?")
    ss = d.get("step_summary", {})
    rs = d.get("rank_summary", {})

    stages = [to_ms(s.get("avg_stage"), fmt) for s in ss.values()]
    info = {
        "path": path,
        "format": fmt,
        "step_count": len(ss),
        "rank_count": len(rs),
        "stage": mean(stages),
        "compute": mean([to_ms(s.get("avg_compute"), fmt) for s in ss.values()]),
        "comm": mean([to_ms(s.get("avg_comm"), fmt) for s in ss.values()]),
        "free": mean([to_ms(s.get("avg_free"), fmt) for s in ss.values()]),
        "overlapped": mean([to_ms(s.get("avg_overlapped"), fmt) for s in ss.values()]),
        "stage_min": min(stages) if stages else 0.0,
        "stage_max": max(stages) if stages else 0.0,
        "bw": {},
        "ops_total_rows": 0,
        "ops_count": 0,
        "top_ops": [],
    }
    for b in d.get("comm_bandwidth", []):
        t = b.get("transport_type", b.get("band_type", "?"))
        info["bw"][t] = {
            "avg_bw": float(b.get("avg_bw", 0) or 0),
            "eff_bw": get_eff_bw(b),
        }
    ops = d.get("comm_time_ops", [])
    info["ops_count"] = len(ops)
    info["ops_total_rows"] = sum(
        1 for op in ops if str(op_name(op)).strip().lower().startswith(TOTAL_PREFIX))

    # Rank 离散度（基于 rank_summary 的 stage 均值，慢卡线索）
    r_stages = sorted(to_ms(r.get("avg_stage"), fmt) for r in rs.values())
    info["rank_stage_spread_pct"] = (
        (r_stages[-1] - r_stages[0]) / r_stages[0] * 100 if r_stages and r_stages[0] > 1e-9 else 0.0)
    return info


def print_single(info):
    print(f"\n=== {info['path']} ===")
    print(f"格式: {info['format']} | Step 数: {info['step_count']} | Rank 数: {info['rank_count']}")
    print(f"Step 均值(ms): stage={info['stage']:.2f} compute={info['compute']:.2f} "
          f"comm={info['comm']:.2f} free={info['free']:.2f} overlapped={info['overlapped']:.2f}")
    print(f"Rank stage 极差: {info['stage_min']:.2f} ~ {info['stage_max']:.2f} ms "
          f"(离散度 {info['rank_stage_spread_pct']:.2f}%)")
    if info["rank_stage_spread_pct"] > 3:
        print("  [提示] Rank 离散度 > 3%，存在慢卡线索，建议进一步查看 rank_summary。")
    print("带宽（均值口径 vs 有效吞吐 GB/s）:")
    for t, v in sorted(info["bw"].items()):
        print(f"  {t:<12} 均值={v['avg_bw']:.1f}  有效吞吐={v['eff_bw']:.1f}")
    if info["ops_total_rows"]:
        print(f"  [警告] comm_time_ops 检出 {info['ops_total_rows']} 条 Total 汇总行残留，"
              f"算子级对比前需过滤。")
    else:
        print(f"算子列表: {info['ops_count']} 条，无 Total 汇总行残留")


def print_compare(a, b):
    print(f"\n=== 对比: A={a['path']}  vs  B={b['path']} ===")
    rows = [
        ("Stage 均值(ms)", a["stage"], b["stage"]),
        ("计算均值(ms)", a["compute"], b["compute"]),
        ("通信均值(ms)", a["comm"], b["comm"]),
        ("空闲均值(ms)", a["free"], b["free"]),
    ]
    print(f"{'指标':<16}{'集群A':>12}{'集群B':>12}{'相对差':>10}")
    for name, va, vb in rows:
        print(f"{name:<16}{va:>12.2f}{vb:>12.2f}{rel_diff(vb, va):>+9.2f}%")

    print(f"\n{'带宽类型':<14}{'均值A':>10}{'均值B':>10}{'吞吐A':>10}{'吞吐B':>10}{'均值差':>9}{'吞吐差':>9}")
    for t in sorted(set(a["bw"]) | set(b["bw"])):
        va, vb = a["bw"].get(t, {}), b["bw"].get(t, {})
        avg_a, avg_b = va.get("avg_bw", 0), vb.get("avg_bw", 0)
        eff_a, eff_b = va.get("eff_bw", 0), vb.get("eff_bw", 0)
        avg_d = rel_diff(avg_b, avg_a)
        eff_d = rel_diff(eff_b, eff_a)
        print(f"{t:<14}{avg_a:>10.1f}{avg_b:>10.1f}{eff_a:>10.1f}{eff_b:>10.1f}{avg_d:>+8.1f}%{eff_d:>+8.1f}%")

    # 口径陷阱判定（与报告 classify_bw_change 分级一致）
    warnings = 0
    for t in sorted(set(a["bw"]) & set(b["bw"])):
        avg_d = abs(rel_diff(b["bw"][t]["avg_bw"], a["bw"][t]["avg_bw"]))
        eff_d = abs(rel_diff(b["bw"][t]["eff_bw"], a["bw"][t]["eff_bw"]))
        if eff_d >= 15:
            print(f"  [P0] {t}: 有效吞吐下降 {eff_d:.1f}%，真实链路劣化，需排查硬件。")
            warnings += 1
        elif eff_d >= 5:
            print(f"  [P1] {t}: 有效吞吐下降 {eff_d:.1f}%，轻微吞吐劣化，建议关注。")
            warnings += 1
        elif avg_d >= 10:
            print(f"  [口径] {t}: 均值带宽降 {avg_d:.1f}% 但有效吞吐差仅 {eff_d:.1f}%，"
                  f"系流量构成（包大小分布）差异，非链路劣化。")
    if warnings == 0:
        print("结论提示: 未检出真实吞吐劣化。均值带宽波动请结合上表'吞吐差'列解读。")

    if a["ops_total_rows"] or b["ops_total_rows"]:
        print(f"  [警告] Total 汇总行残留: A={a['ops_total_rows']} 条, B={b['ops_total_rows']} 条")


def main():
    parser = argparse.ArgumentParser(description="集群提取结果指标校验器")
    parser.add_argument("--data-a", required=True, help="集群A提取结果 JSON")
    parser.add_argument("--data-b", default=None, help="集群B提取结果 JSON（可选，双集群对比模式）")
    args = parser.parse_args()

    a = load_summary(args.data_a)
    print_single(a)
    if args.data_b:
        b = load_summary(args.data_b)
        print_single(b)
        print_compare(a, b)


if __name__ == "__main__":
    main()

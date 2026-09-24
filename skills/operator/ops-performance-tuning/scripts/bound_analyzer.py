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
"""Bound 初筛工具 — 从 msOpProf PipeUtilization.csv 输出 Bound 候选。

用法:
  python3 scripts/bound_analyzer.py --csv PipeUtilization.csv --output bound_report.json

判定规则（最终结论仍需结合带宽、时间线和源码证据）:
  MTE2/CUBE/VEC/FIXP/MTE3/SCALAR BOUND、LATENCY/SYNC 可疑、OCCUPANCY 可疑、无 bound

注意（A5/dav-3510 实测教训）:
  - A5 上 vector kernel 的有效数据在 aiv_* 列，aic_* 列常为 NA；cube kernel 反之。
    本工具同时读两组列名并按各自均值/峰值取大者，不要再只看 aic_*。
  - 判定用 mean（核间平均），max 仅用于识别长尾；单核峰值不能代表全 kernel。
  - block 行数远小于 SoC 物理核数（A5 典型 28 AIC / 48~56 AIV）时优先查 occupancy，
    PipeUtilization 行数本身不能区分"核数不足"与"核间不均"，需结合 OpBasicInfo 的 Block Dim。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def parse_float(value: str) -> float:
    try:
        return float(value.strip().replace("%", ""))
    except (ValueError, AttributeError):
        return 0.0


def ratio_to_pct(value: float) -> float:
    """兼容 0~1 比例和 0~100 百分数两种版本字段。"""
    return value * 100 if 0 <= value <= 1.5 else value


def load_csv(csv_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# 每档 Bound 的候选列名（覆盖 aic_*/aiv_* 两组 A5 实测格式 + 旧版百分数格式）
CATEGORY_COLUMNS: dict[str, tuple[str, ...]] = {
    "MTE2": ("aic_mte2_ratio", "aiv_mte2_ratio", "Mte2Utilization(%)", "mte2_ratio", "MTE2(%)"),
    "MTE3": ("aic_mte3_ratio", "aiv_mte3_ratio", "Mte3Utilization(%)", "mte3_ratio", "MTE3(%)"),
    "CUBE": ("aic_cube_ratio", "aiv_cube_ratio", "CubeUtilization(%)", "cube_ratio", "CUBE(%)"),
    "VEC": ("aiv_vec_ratio", "aic_vec_ratio", "VectorUtilization(%)", "vec_ratio", "VECTOR(%)"),
    "FIXP": ("aic_fixpipe_ratio", "aiv_fixpipe_ratio", "FixpipeUtilization(%)", "fixp_ratio", "FIXP(%)"),
    "SCALAR": ("aic_scalar_ratio", "aiv_scalar_ratio", "ScalarUtilization(%)", "scalar_ratio", "SCALAR(%)"),
}

# A5 (dav-3510) 实测物理核数参考：occupancy 可疑判定的分母
A5_TYPICAL_CORES = 48


def category_stats(rows: list[dict[str, str]], columns: tuple[str, ...]) -> tuple[float, float]:
    """返回 (mean_pct, max_pct)：对每个 block 行取候选列中的最大值，再跨行统计。"""
    per_row: list[float] = []
    for row in rows:
        best = 0.0
        for col in columns:
            best = max(best, ratio_to_pct(parse_float(str(row.get(col, "")))))
        per_row.append(best)
    if not per_row:
        return 0.0, 0.0
    return sum(per_row) / len(per_row), max(per_row)


def determine_bound(rows: list[dict[str, str]]) -> dict[str, Any]:
    """从 PipeUtilization.csv 生成 Bound 候选。"""
    means: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for cat, cols in CATEGORY_COLUMNS.items():
        means[cat], maxs[cat] = category_stats(rows, cols)

    max_cat = max(means, key=means.get)  # type: ignore[arg-type]
    max_mean = means[max_cat]
    max_peak = maxs[max_cat]

    bound = "无 bound"
    reason = ""
    notes: list[str] = []

    if max_mean > 80:
        bound = f"{max_cat} BOUND"
        reason = f"{max_cat} busy 核间均值 {max_mean:.1f}%（峰值 {max_peak:.1f}%），大于 80%"
    elif max_mean > 70:
        bound = f"{max_cat} BOUND"
        reason = f"{max_cat} busy 核间均值 {max_mean:.1f}%（峰值 {max_peak:.1f}%），占比最大且大于 70%"
    elif max_mean < 50:
        # 全部 pipe 低占用：不是"无 bound"，是 latency/同步可疑（全 pipe 等依赖）
        bound = "LATENCY/SYNC 可疑"
        reason = (
            f"所有 pipe busy 均值均低于 50%（最高 {max_cat}={max_mean:.1f}%）："
            "排除 compute 与带宽饱和后，典型为流水未重叠、频繁同步或小包/轮询等待，"
            "按 diagnose-pipeline.md 判读规则核实用 SetFlag/WaitFlag、PipeBarrier 与跨核同步"
        )

    # 长尾提示：峰值显著高于均值说明核间不均
    if max_mean >= 50 and max_peak > max_mean * 1.25:
        notes.append(
            f"{max_cat} 峰值 {max_peak:.1f}% 显著高于均值 {max_mean:.1f}%（>{1.25}x），"
            "存在核间长尾嫌疑，按 diagnose-occupancy.md 核对各 block 行"
        )

    # occupancy 可疑：block 行数少
    if 0 < len(rows) < 16:
        notes.append(
            f"PipeUtilization 仅 {len(rows)} 个 block 行；若 OpBasicInfo 的 Block Dim "
            f"同样远小于 SoC 物理核数（A5 约 {A5_TYPICAL_CORES} AIV / 28 AIC），"
            "优先按 occupancy（核数不足/切核过少）排查，本工具的 pipe 占比仅统计已活跃核"
        )

    # 优化方向推荐
    recommendation_map = {
        "MTE2": ("数据搬运优化", [
            "references/optimize/optimize-data-copy.md",
            "references/optimize/optimize-memory-hierarchy.md",
        ]),
        "MTE3": ("数据搬出优化", [
            "references/optimize/optimize-data-copy.md",
        ]),
        "CUBE": ("Cube 利用率优化", [
            "references/optimize/optimize-ascendc-tiling.md",
            "references/optimize/optimize-catlass.md",
        ]),
        "VEC": ("Vector 利用率优化", [
            "references/optimize/optimize-api-usage.md",
            "references/optimize/optimize-pipeline.md",
            "references/optimize/optimize-memory-hierarchy.md",
        ]),
        "FIXP": ("Fixpipe 优化", [
            "references/optimize/optimize-pipeline.md",
        ]),
        "SCALAR": ("标量削减", [
            "references/optimize/optimize-api-usage.md",
        ]),
    }
    if bound == "LATENCY/SYNC 可疑":
        rec = ("流水/同步优化（latency bound）", [
            "references/diagnose/diagnose-pipeline.md",
            "references/optimize/optimize-pipeline.md",
            "references/optimize/optimize-data-copy.md",
        ])
    else:
        rec = recommendation_map.get(max_cat, ("进一步分析", [
            "references/profile/profile-msopprof.md",
        ]))

    return {
        "bound": bound,
        "reason": reason,
        "categories_mean": {k: round(v, 1) for k, v in means.items()},
        "categories_max": {k: round(v, 1) for k, v in maxs.items()},
        "notes": notes,
        "recommendation": {"direction": rec[0], "docs": rec[1]},
        "source_file": None,
        "row_count": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bound 自动初筛（mean/max 双口径，含 occupancy 提示）")
    parser.add_argument("--csv", default=None, help="msprof PipeUtilization.csv 路径")
    parser.add_argument("--msprof-dir", default=None, help="msprof 输出目录，自动搜索 **/PipeUtilization.csv")
    parser.add_argument("--output", required=True, help="bound_report.json 输出路径")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path and args.msprof_dir:
        # 自动搜索 msprof 目录下的 PipeUtilization.csv
        msprof_root = Path(args.msprof_dir)
        if not msprof_root.is_dir():
            print(f"ERROR: msprof 目录不存在: {args.msprof_dir}", file=sys.stderr)
            return 1
        candidates = list(msprof_root.rglob("PipeUtilization.csv"))
        if not candidates:
            print(f"ERROR: 在 {args.msprof_dir} 下未找到 PipeUtilization.csv", file=sys.stderr)
            print("提示: 确保 msprof op 采集时使用了 --aic-metrics=PipeUtilization", file=sys.stderr)
            return 1
        # 优先选 roofline 目录下的
        roofline = [c for c in candidates if "roofline" in str(c).lower()]
        chosen = roofline[0] if roofline else candidates[0]
        csv_path = str(chosen)
        print(f"自动选择: {csv_path}")
    if not csv_path:
        print("ERROR: 必须提供 --csv 或 --msprof-dir", file=sys.stderr)
        return 1
    if not Path(csv_path).is_file():
        print(f"ERROR: CSV 文件不存在: {csv_path}", file=sys.stderr)
        return 1

    rows = load_csv(csv_path)
    result = determine_bound(rows)
    result["source_file"] = str(Path(csv_path).resolve())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")

    print(f"Bound 判定: {result['bound']}")
    print(f"原因: {result['reason']}")
    for note in result.get("notes", []):
        print(f"提示: {note}")
    if "recommendation" in result:
        print(f"优化方向: {result['recommendation']['direction']}")
        print(f"参考文档: {', '.join(result['recommendation']['docs'])}")
    print(f"结果已保存: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

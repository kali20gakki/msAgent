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
"""性能对比报告 — 从 baseline+after json 生成 Δ% 对比表。

用法:
  python3 scripts/delta_report.py \
    --baseline results/baseline_board_xxx.json \
    --after results/after_board_xxx.json \
    --output delta_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# 硬口径：这些字段不一致则禁止比较（跨 shape/dtype/SoC/设备/计时法无意义）
HARD_KEYS = (
    "op", "soc", "mode", "timing_method", "baseline_kind", "cann_version", "device_id",
    "shape", "dtype", "format", "warmup", "repeat",
)
# 机制变量：分核/tiling 类优化本身就会改变它们，差异降级为 WARN 并写入报告
MECHANISM_KEYS = ("tiling_key", "block_dim")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成优化前后 Δ% 对比报告")
    parser.add_argument("--baseline", required=True, help="基线 result json")
    parser.add_argument("--after", required=True, help="优化后 result json")
    parser.add_argument("--output", required=True, help="输出 markdown 报告路径")
    parser.add_argument("--strict", action="store_true",
                        help="恢复旧行为：tiling_key/block_dim 差异也视为口径不一致")
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))

    hard_keys = tuple(HARD_KEYS) + (tuple(MECHANISM_KEYS) if args.strict else ())
    differences = []
    for key in hard_keys:
        before_value = baseline.get(key)
        after_value = after.get(key)
        if before_value is not None and after_value is not None and before_value != after_value:
            differences.append(f"{key}: {before_value!r} != {after_value!r}")
    if differences:
        print("ERROR: 基线与优化后结果口径不一致：", file=sys.stderr)
        for item in differences:
            print(f"  - {item}", file=sys.stderr)
        return 2

    mechanism_notes = []
    if not args.strict:
        for key in MECHANISM_KEYS:
            before_value = baseline.get(key)
            after_value = after.get(key)
            if before_value is not None and after_value is not None and before_value != after_value:
                mechanism_notes.append(f"{key}: {before_value!r} -> {after_value!r}")
        for item in mechanism_notes:
            print(f"WARN: 机制变量变化（分核/tiling 类优化属预期）: {item}", file=sys.stderr)

    if baseline.get("precision") != "pass" or after.get("precision") != "pass":
        print("ERROR: 只有精度均为 pass 的结果才能生成性能结论", file=sys.stderr)
        return 2

    b_us = baseline.get("kernel_avg_us", 0)
    a_us = after.get("kernel_avg_us", 0)
    speedup = b_us / a_us if a_us > 0 else 0
    delta_pct = ((a_us - b_us) / b_us * 100) if b_us > 0 else 0

    op = baseline.get("op", after.get("op", "unknown"))
    soc = baseline.get("soc_full", baseline.get("soc", "?"))

    lines = [
        f"# 性能对比报告：{op}",
        "",
        f"| 项目 | 基线 (before) | 优化后 (after) | Δ% | 加速比 |",
        f"|---|---|---|---|---|",
        f"| kernel 耗时 | {b_us:.2f} µs | {a_us:.2f} µs | {delta_pct:+.1f}% | {speedup:.2f}x |",
        f"| 精度 | {baseline.get('precision','?')} | {after.get('precision','?')} | — | — |",
        f"| 芯片 | {baseline.get('soc','?')} | {after.get('soc','?')} | — | — |",
        f"| 模式 | {baseline.get('mode','?')} | {after.get('mode','?')} | — | — |",
        f"| shape | {baseline.get('shape','?')} | {after.get('shape','?')} | — | — |",
        f"| dtype | {baseline.get('dtype','?')} | {after.get('dtype','?')} | — | — |",
        "",
        f"## 结论",
        "",
    ]
    if mechanism_notes:
        lines.append("机制变量变化（分核/tiling 类优化的预期变量，非口径问题）：")
        for item in mechanism_notes:
            lines.append(f"- {item}")
        lines.append("")
    if speedup > 1.05:
        lines.append(f"性能提升 {speedup:.2f}x（{delta_pct:+.1f}%），可进入稳定性复测。")
    elif speedup >= 0.95:
        lines.append(
            f"性能变化 {speedup:.2f}x（{delta_pct:+.1f}%）落在 ±5% 噪声带内："
            "按 SKILL Step 7 规则应视为无稳定收益，增加采样轮次确认或回滚，"
            "不要把噪声内变化当作优化收益。")
    else:
        lines.append(f"性能劣化 {speedup:.2f}x（{delta_pct:+.1f}%），应回滚该轮修改。")

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已生成: {args.output}")
    print(f"加速比: {speedup:.2f}x (Δ={delta_pct:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

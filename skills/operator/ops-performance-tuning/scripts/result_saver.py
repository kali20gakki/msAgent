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
"""结果文件落盘工具 — 生成符合 skill 契约的 <variant>_<board|sim>_<timestamp>.json。

用法:
  python3 scripts/result_saver.py \
    --variant baseline \
    --mode board \
    --soc a5 \
    --precision pass \
    --kernel-avg-us 336.6 \
    --output results/

生成: results/baseline_board_20260808T120000Z.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Save tuning result as structured JSON")
    parser.add_argument("--op", required=True, help="算子或性能用例名称")
    parser.add_argument("--variant", required=True, help="源码变体名 (e.g. baseline, round1, round2)")
    parser.add_argument("--mode", required=True, choices=("board", "sim"), help="运行模式")
    parser.add_argument("--soc", required=True, help="芯片代际 (a2/a3/a5)")
    parser.add_argument("--precision", required=True, choices=("pass", "fail"), help="精度校验结果")
    parser.add_argument("--mismatch", type=int, default=0, help="精度不匹配的数量")
    parser.add_argument("--max-abs-err", type=float, default=0.0, help="最大绝对误差")
    parser.add_argument("--max-rel-err", type=float, default=0.0, help="最大相对误差")
    parser.add_argument("--kernel-avg-us", type=float, required=True, help="kernel 平均耗时 (µs)")
    parser.add_argument("--timing-method", default="event",
                        choices=("event", "sim", "msprof", "msprof_task_duration", "profiler_summary"),
                        help="计时方法：event=设备 event 计时（首选）；msprof_task_duration=工程无 event 设施且 "
                             "kernel 为长耗时（ms 级）时用 OpBasicInfo Task Duration 作基线；"
                             "profiler_summary=工程自带 profiler 对比框架（如 torch_npu op_summary）")
    parser.add_argument("--status", default="ok", choices=("ok", "partial", "dry_run"),
                        help="结果状态：partial=采集/验证不完整，dry_run=未上板仅离线结论")
    parser.add_argument("--baseline-kind", default="self_before_after",
                        choices=("system", "self_before_after"), help="基线类型")
    parser.add_argument("--cann-version", default=None, help="CANN 完整版本")
    parser.add_argument("--repo-commit", default=None,
                        help="源码 commit；非 git 工程填 n/a 或 sha256:<源码hash>")
    parser.add_argument("--device-id", default=None, help="NPU 设备 ID")
    parser.add_argument("--shape", default=None, help="逻辑/物理 shape；多组时写用例 ID")
    parser.add_argument("--dtype", default=None, help="输入输出 dtype 摘要")
    parser.add_argument("--format", default=None, help="输入输出 format 摘要")
    parser.add_argument("--tiling-key", default=None, help="TilingKey")
    parser.add_argument("--block-dim", type=int, default=None, help="blockDim")
    parser.add_argument("--warmup", type=int, default=None, help="预热次数")
    parser.add_argument("--repeat", type=int, default=None, help="正式采样次数")
    parser.add_argument("--msprof-dir", default=None, help="msprof 采集目录路径（无则 null）")
    parser.add_argument("--exit-code", type=int, default=0, help="运行退出码")
    parser.add_argument("--output", required=True, help="结果输出目录")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="额外键值对 key=value")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{args.variant}_{args.mode}_{timestamp}.json"
    filepath = output_dir / filename

    payload = {
        "op": args.op,
        "variant": args.variant,
        "soc": args.soc,
        "mode": args.mode,
        "status": args.status,
        "precision": args.precision,
        "mismatch": args.mismatch,
        "maxAbsErr": args.max_abs_err,
        "maxRelErr": args.max_rel_err,
        "kernel_avg_us": args.kernel_avg_us,
        "timing_method": args.timing_method,
        "baseline_kind": args.baseline_kind,
        "cann_version": args.cann_version,
        "repo_commit": args.repo_commit,
        "device_id": args.device_id,
        "shape": args.shape,
        "dtype": args.dtype,
        "format": args.format,
        "tiling_key": args.tiling_key,
        "block_dim": args.block_dim,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "msprof_dir": args.msprof_dir,
        "exit_code": args.exit_code,
        "timestamp": timestamp,
    }
    for kv in args.extra:
        if "=" in kv:
            k, v = kv.split("=", 1)
            payload[k] = v

    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Result saved: {filepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

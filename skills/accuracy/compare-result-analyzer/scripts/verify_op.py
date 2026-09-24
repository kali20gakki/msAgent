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
"""单算子精度验证框架 — NPU vs CPU — CLI入口

用法:
    python verify_op.py <compare.csv> --op-list "op1,op2" -o verify.json
"""
import argparse
import json
import os
import sys
from typing import Optional, Dict, List

from _verify_core import parse_csv, group_ops, get_operator_fn, _parse_op_list_item
from _verify_engine import verify_operator, VerifyResult, print_results, output_json


def main():
    parser = argparse.ArgumentParser(description="单算子精度验证框架 — NPU vs CPU")
    parser.add_argument("csv", nargs="?", help="msProbe compare CSV")
    parser.add_argument("--op-list", help="逗号分隔算子列表")
    parser.add_argument("--output", "-o", help="输出 JSON 路径")
    args = parser.parse_args()

    if not args.csv:
        parser.print_help()
        sys.exit(1)

    print(f"  📂 解析 CSV: {args.csv}")
    tensors = parse_csv(args.csv)
    if not tensors:
        print("  ⚠ 未发现 API 算子")
        if args.output:
            _out = args.output or os.path.join(
                os.path.dirname(os.path.abspath(args.csv)),
                '.compare_result_analyzer',
                f'{os.path.splitext(os.path.basename(args.csv))[0]}_verify.json')
            os.makedirs(os.path.dirname(_out), exist_ok=True)
            output_json([], _out, atol=1e-4, rtol=1e-3)
        sys.exit(0)

    groups = group_ops(tensors)
    print(f"  📊 {len(tensors)} tensors, {len(groups)} 算子实例")

    op_directions: Dict[str, Optional[str]] = {}
    if args.op_list:
        raw_items = [n.strip() for n in args.op_list.split(",") if n.strip()]
        for item in raw_items:
            key, d = _parse_op_list_item(item)
            op_directions[key] = d
        groups = {k: v for k, v in groups.items() if k in op_directions}
        if not groups:
            print(f"  ❌ 未找到: {args.op_list}")
            sys.exit(1)
        print(f"  🔍 过滤后: {len(groups)} 算子实例")

    all_results = []
    for key, group in groups.items():
        try:
            results = verify_operator(group, 1e-4, 1e-3,
                                      direction=op_directions.get(key))
        except Exception as e:
            results = [VerifyResult(
                op_name=group.op_name, instance=group.instance,
                direction=op_directions.get(key, "forward"),
                tensor_name="N/A", shape_match=False,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0, passed=False,
                error=f"验证异常: {str(e)}", construct_l2norm_err=0)]
        all_results.extend(results)

    print_results(all_results)

    if args.output:
        output_json(all_results, args.output, atol=1e-4, rtol=1e-3)
    else:
        d = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                         '.compare_result_analyzer')
        os.makedirs(d, exist_ok=True)
        output_json(all_results,
                    os.path.join(d, f'{os.path.splitext(os.path.basename(args.csv))[0]}_verify.json'),
                    atol=1e-4, rtol=1e-3)

    failed = sum(1 for r in all_results if not r.passed)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()

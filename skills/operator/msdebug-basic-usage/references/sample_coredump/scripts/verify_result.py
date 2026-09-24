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

#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import argparse
import sys

import numpy as np

# 数值容差
_RTOL = 1e-4          # 相对误差上限
_ATOL = 1e-5          # 绝对误差上限
_ERROR_TOL = 1e-4     # 允许的最大错误比例
_REPORT_LIMIT = 101   # 差异条目打印上限


def _load_f32(path):
    """读取 float32 二进制文件，返回 1 维 ndarray。"""
    with open(path, "rb") as fh:
        return np.frombuffer(fh.read(), dtype=np.float32)


def _locate_mismatch(actual, expect):
    """返回 actual 与 expect 逐元素不匹配的下标（NaN 视作相等）。"""
    close = np.isclose(actual, expect, rtol=_RTOL, atol=_ATOL, equal_nan=True)
    return np.flatnonzero(np.logical_not(close))


def _print_one(idx, actual, expect):
    """打印某个下标处期望值/实际值与相对差。"""
    a = float(actual[idx])
    e = float(expect[idx])
    rd = abs(a - e) / e if e != 0.0 else float("nan")
    print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f"
          % (idx, e, a, rd))


def compare_files(actual_path, expect_path):
    """比对两个文件，返回是否通过及错误比例。"""
    actual = _load_f32(actual_path)
    expect = _load_f32(expect_path)
    bad = _locate_mismatch(actual, expect)

    shown = 0
    for idx in bad:
        if shown >= _REPORT_LIMIT:
            break
        _print_one(idx, actual, expect)
        shown += 1

    total = expect.size
    ratio = float(bad.size) / total if total else 0.0
    print("error ratio: %.4f, tolerance: %.4f" % (ratio, _ERROR_TOL))
    return ratio <= _ERROR_TOL, ratio


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="compare two float32 binary result files (actual vs golden)")
    parser.add_argument("actual", help="算子实际输出文件路径")
    parser.add_argument("golden", help="期望结果文件路径")
    args = parser.parse_args(argv)

    try:
        ok, _ = compare_files(args.actual, args.golden)
    except Exception as exc:  # noqa: BLE001 - 统一失败出口
        print("[ERROR] %s" % exc)
        return 1
    if ok:
        print("test pass!")
        return 0
    print("[ERROR] result error")
    return 1


if __name__ == "__main__":
    sys.exit(main())

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
# MERCHANTABILITY OR FIT FOR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""CPU list 解析唯一来源。

npu-smi 的 CPU Affinity 与 sysfs 的 cpulist 使用同一套格式，
统一在本模块解析，避免 npu_query / topology 两份实现口径漂移。
"""


def parse_cpulist(text):
    """'144-167' / '0-23,48-71' -> [144, ..., 167]。非法片段跳过。"""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                pass
        else:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out

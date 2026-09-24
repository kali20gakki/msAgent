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
"""
msMemScope Python 接口采集参考模板。

【使用说明】
1. 先注入环境变量（必须 source）：
     source msmemscope --load-api-env
2. 将本模板中的 train() 替换为用户实际训练/推理代码（或在采集范围处直接嵌入
   msmemscope.start()/stop()）。
3. 采集完成后清理环境变量：
     source msmemscope --unload-api-env

【注意】
- 本模板仅作对照参考，不要求用户脚本照抄；按实际诉求裁剪 config 参数。
- 分析功能（analysis）需 events 含 alloc,free。
- 采集结束后务必清理环境变量，避免污染后续操作。
"""

import msmemscope


def train():
    """用户代码：替换为实际训练/推理逻辑。"""
    # 示例：采集 Step 信息（可选）
    for step in range(10):
        do_train_step(step)
        msmemscope.step()


def do_train_step(step: int):
    """占位函数，替换为用户的训练逻辑。"""
    pass


def main() -> None:
    # ---- 按诉求选择配置组合（以下为可选参数的示例） ----
    msmemscope.config(
        events="alloc,free",          # 必须包含 alloc,free（分析功能依赖）
        level="op",                   # op 或 kernel
        device="npu",                 # npu / npu:{id} / cpu
        analysis="leaks,decompose",   # leaks / decompose / oom[:K] / none
        call_stack="c:10,python:5",   # 泄漏/OOM 归因必需
        data_format="csv",            # csv（默认，推荐）或 db（仅 MindStudio Insight 可视化用）
        output="/home/user/output",   # 输出路径
    )

    # OOM 分析示例（自动联动 alloc/free 采集）：
    # msmemscope.config(events="alloc,free", analysis="oom:50",
    #                   call_stack="python:10", output="/home/user/output")

    # 推理/RL 场景一键分析示例（vLLM-Ascend 11.0 快照）：
    # msmemscope.cleanup_framework_hooks()
    # msmemscope.init_framework_hooks("vllm_ascend", "11.0", "worker", "snapshot")

    msmemscope.start()
    train()
    msmemscope.stop()


if __name__ == "__main__":
    main()

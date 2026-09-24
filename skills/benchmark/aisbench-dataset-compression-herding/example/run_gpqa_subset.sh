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

#!/usr/bin/env bash
# ============================================================
# 将 GPQA 数据集 path 改为「部分子集」，并支持运行前后修改/还原
#  - 把 3 个配置文件中的 path 改为子集目录
# 用法: bash run_gpqa_subset.sh
# ============================================================
set -uo pipefail

# ais_bench 源码根目录（配置文件所在目录）
BASE_DIR="<AIS_BENCH_ROOT>"

# 本次目标数据集绝对路径（部分子集）
TARGET_PATH="<GPQA_PART_DATASET_PATH>"

# ais_bench 的 --work-dir
WORK_DIR="<WORK_DIR_FOR_PART_EVAL>"

# 被修改的配置文件
FILES=(
  "$BASE_DIR/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_str.py"
  "$BASE_DIR/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py"
  "$BASE_DIR/benchmark/configs/datasets/gpqa/gpqa_ppl_0_shot_str.py"
)

# 1. 备份原始 path 值，并改为部分子集（只替换引号内路径，保留缩进/逗号/注释）
declare -a ORIG_PATHS=()
for f in "${FILES[@]}"; do
  ORIG_PATHS+=("$(sed -n "s/.*path='\([^']*\)'.*/\1/p" "$f" | head -1)")
  sed -i "s#path='[^']*'#path='$TARGET_PATH'#" "$f"
done
echo "[修改完成] 3 个配置文件的 path 已改为部分子集：$TARGET_PATH"

# 2. 展示修改结果
echo "----- 修改后 path -----"
for f in "${FILES[@]}"; do
  sed -n 's/.*path=/path=/p' "$f"
done


### 执行aisbench命令
# --num-warmups 0 : 跳过 warmup（预热）阶段，直接进入推理
ais_bench --models vllm_api_general_chat --datasets gpqa_gen --work-dir "$WORK_DIR" --dump-eval-details --num-warmups 0
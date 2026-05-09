#!/bin/bash
# e2e_compile_run.sh - 编译并运行算子（基准版本）
# Usage: bash scripts/e2e_compile_run.sh <operator_dir> [batch_size] [num_class]
# 依赖：compile_run.sh 位于 operator_dir 下

set -e

OPERATOR_DIR="${1:?Error: operator_dir is required}"
BATCH_SIZE="${2:-128}"
NUM_CLASS="${3:-1024}"

cd "$OPERATOR_DIR"

echo "===== 步骤 1：编译算子 ====="
bash compile_run.sh "$BATCH_SIZE" "$NUM_CLASS"

echo ""
echo "===== 步骤 2：验证运行结果 ====="
echo "检查输出中是否包含 'precision pass'..."

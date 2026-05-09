#!/bin/bash
# e2e_profile_onboard.sh - 编译 + 上板运行 + msprof op 性能采集
# Usage: bash scripts/e2e_profile_onboard.sh <operator_dir> [batch_size] [num_class]
# 依赖：compile_run.sh 位于 operator_dir 下

set -e

OPERATOR_DIR="${1:?Error: operator_dir is required}"
BATCH_SIZE="${2:-128}"
NUM_CLASS="${3:-1024}"

cd "$OPERATOR_DIR"

echo "===== 步骤 1：清理旧的 msprof 输出 ====="
rm -rf OPPROF_*

echo "===== 步骤 2：编译算子 ====="
bash compile_run.sh "$BATCH_SIZE" "$NUM_CLASS"

echo ""
echo "===== 步骤 3：执行 msprof op 上板性能采集 ====="
echo "命令：msprof op ./run.fatbin $BATCH_SIZE $NUM_CLASS"
msprof op ./run.fatbin "$BATCH_SIZE" "$NUM_CLASS"

echo ""
echo "===== 步骤 4：查找生成的性能数据目录 ====="
PROFILE_DIR=$(ls -d OPPROF_* 2>/dev/null | head -1)
if [ -n "$PROFILE_DIR" ]; then
    echo "性能数据目录：$OPERATOR_DIR/$PROFILE_DIR"
    echo "关键文件列表："
    ls "$PROFILE_DIR"/
else
    echo "错误：未找到 OPPROF_* 目录，性能采集可能失败"
    exit 1
fi

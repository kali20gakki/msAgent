#!/bin/bash
# e2e_compare.sh - 对比优化前后的 msprof op 性能数据
# Usage: bash scripts/e2e_compare.sh <before_dir> <after_dir>
# 对比内容包括：总耗时、ArithmeticUtilization、PipeUtilization、L2Cache

set -e

BEFORE_DIR="${1:?Error: before profile data dir is required}"
AFTER_DIR="${2:?Error: after profile data dir is required}"

echo "================================================================="
echo "      基于 msprof op 的算子性能优化前后对比报告"
echo "================================================================="
echo ""
echo "优化前目录：$BEFORE_DIR"
echo "优化后目录：$AFTER_DIR"
echo ""

# 对比总耗时
echo "--- 1. 总耗时对比 ---"
if [ -f "$BEFORE_DIR/OpBasicInfo.csv" ] && [ -f "$AFTER_DIR/OpBasicInfo.csv" ]; then
    BEFORE_TIME=$(tail -1 "$BEFORE_DIR/OpBasicInfo.csv" | cut -d',' -f2)
    AFTER_TIME=$(tail -1 "$AFTER_DIR/OpBasicInfo.csv" | cut -d',' -f2)
    echo "优化前总耗时：$BEFORE_TIME us"
    echo "优化后总耗时：$AFTER_TIME us"
    if [ -n "$BEFORE_TIME" ] && [ -n "$AFTER_TIME" ] && [ "$BEFORE_TIME" != "0" ]; then
        SPEEDUP=$(echo "scale=2; ($BEFORE_TIME - $AFTER_TIME) / $BEFORE_TIME * 100" | bc)
        echo "加速比：${SPEEDUP}%"
    fi
else
    echo "OpBasicInfo.csv 不存在，跳过总耗时对比"
fi
echo ""

# 对比 ArithmeticUtilization
echo "--- 2. 计算单元利用率对比 ---"
if [ -f "$BEFORE_DIR/ArithmeticUtilization.csv" ] && [ -f "$AFTER_DIR/ArithmeticUtilization.csv" ]; then
    echo "指标                     | 优化前    | 优化后    | 改善"
    echo "-------------------------|-----------|-----------|------"
    
    # 读取指标（跳过表头）
    while IFS=',' read -r METRIC BEFORE_VAL _; do
        AFTER_VAL=$(grep "^$METRIC," "$AFTER_DIR/ArithmeticUtilization.csv" | cut -d',' -f2)
        if [ -n "$AFTER_VAL" ]; then
            DIFF=$(echo "scale=2; $AFTER_VAL - $BEFORE_VAL" | bc 2>/dev/null || echo "N/A")
            printf "%-24s | %-9s | %-9s | %s\n" "$METRIC" "$BEFORE_VAL" "$AFTER_VAL" "$DIFF"
        fi
    done < <(tail -n +2 "$BEFORE_DIR/ArithmeticUtilization.csv")
else
    echo "ArithmeticUtilization.csv 不存在，跳过"
fi
echo ""

# 对比 PipeUtilization
echo "--- 3. 流水线利用率对比 ---"
if [ -f "$BEFORE_DIR/PipeUtilization.csv" ] && [ -f "$AFTER_DIR/PipeUtilization.csv" ]; then
    echo "指标                     | 优化前    | 优化后    | 改善"
    echo "-------------------------|-----------|-----------|------"
    while IFS=',' read -r METRIC BEFORE_VAL _; do
        AFTER_VAL=$(grep "^$METRIC," "$AFTER_DIR/PipeUtilization.csv" | cut -d',' -f2)
        if [ -n "$AFTER_VAL" ]; then
            DIFF=$(echo "scale=2; $AFTER_VAL - $BEFORE_VAL" | bc 2>/dev/null || echo "N/A")
            printf "%-24s | %-9s | %-9s | %s\n" "$METRIC" "$BEFORE_VAL" "$AFTER_VAL" "$DIFF"
        fi
    done < <(tail -n +2 "$BEFORE_DIR/PipeUtilization.csv")
else
    echo "PipeUtilization.csv 不存在，跳过"
fi
echo ""

# 对比 L2Cache
echo "--- 4. L2 Cache 命中率对比 ---"
if [ -f "$BEFORE_DIR/L2Cache.csv" ] && [ -f "$AFTER_DIR/L2Cache.csv" ]; then
    echo "指标                     | 优化前    | 优化后    | 改善"
    echo "-------------------------|-----------|-----------|------"
    while IFS=',' read -r METRIC BEFORE_VAL _; do
        AFTER_VAL=$(grep "^$METRIC," "$AFTER_DIR/L2Cache.csv" | cut -d',' -f2)
        if [ -n "$AFTER_VAL" ]; then
            DIFF=$(echo "scale=2; $AFTER_VAL - $BEFORE_VAL" | bc 2>/dev/null || echo "N/A")
            printf "%-24s | %-9s | %-9s | %s\n" "$METRIC" "$BEFORE_VAL" "$AFTER_VAL" "$DIFF"
        fi
    done < <(tail -n +2 "$BEFORE_DIR/L2Cache.csv")
else
    echo "L2Cache.csv 不存在，跳过"
fi
echo ""

# 对比 Memory（如果有）
echo "--- 5. 内存带宽对比 ---"
if [ -f "$BEFORE_DIR/Memory.csv" ] && [ -f "$AFTER_DIR/Memory.csv" ]; then
    echo "指标                     | 优化前    | 优化后    | 改善"
    echo "-------------------------|-----------|-----------|------"
    while IFS=',' read -r METRIC BEFORE_VAL _; do
        AFTER_VAL=$(grep "^$METRIC," "$AFTER_DIR/Memory.csv" | cut -d',' -f2)
        if [ -n "$AFTER_VAL" ]; then
            DIFF=$(echo "scale=2; $AFTER_VAL - $BEFORE_VAL" | bc 2>/dev/null || echo "N/A")
            printf "%-24s | %-9s | %-9s | %s\n" "$METRIC" "$BEFORE_VAL" "$AFTER_VAL" "$DIFF"
        fi
    done < <(tail -n +2 "$BEFORE_DIR/Memory.csv")
else
    echo "Memory.csv 不存在，跳过"
fi
echo ""

echo "================================================================="
echo "                      对比报告结束"
echo "================================================================="

#!/usr/bin/env python3
"""
多卡溢出检测脚本 - 跨 rank 联合分析
找出通信算子输入中包含 NaN 的 rank（源卡）以及传播 NaN 的 rank
"""

import json
import os
import sys
from collections import defaultdict

OVERFLOW_LIST = ['nan', 'inf', '-inf', 'nan\t', 'inf\t', '-inf\t']
COMMUNICATION_KEYWORDS = {
    'send', 'recv', 'broadcast', 'all_reduce', 'reduce', 'all_gather', 'gather',
    'isend', 'irecv', 'scatter', 'reduce_scatter', 'all_to_all'
}


def is_communication_op(op_name):
    """判断是否为通信算子"""
    return op_name.startswith('Distributed.') and any(k in op_name for k in COMMUNICATION_KEYWORDS)


def check_nan(param):
    """检查参数是否包含 NaN"""
    items = []
    if isinstance(param, list):
        items = param
    elif isinstance(param, dict):
        items = param.values()

    for item in items:
        if not isinstance(item, dict):
            continue
        max_val = str(item.get('Max', '')).lower().strip()
        min_val = str(item.get('Min', '')).lower().strip()
        if max_val == 'nan' or min_val == 'nan':
            return True
    return False


def analyze_comm_ops(input_path):
    """
    分析所有 rank 的通信算子
    返回: {comm_op_name: {rank: {input_has_nan, output_has_nan, input_shape, input_max}}}
    """
    if not os.path.isdir(input_path):
        print(f"Error: {input_path} is not a directory")
        return None

    # 收集所有 rank
    ranks = []
    for item in os.listdir(input_path):
        if item.startswith('rank'):
            rank_str = item[4:]
            rank = int(rank_str) if rank_str.isdigit() else 0
            ranks.append((rank, os.path.join(input_path, item)))

    ranks.sort(key=lambda x: x[0])

    # 分析每个 rank 的通信算子
    comm_ops_data = {}

    for rank, rank_path in ranks:
        dump_path = os.path.join(rank_path, 'dump.json')
        if not os.path.exists(dump_path):
            continue

        try:
            with open(dump_path, 'r', encoding='utf-8') as f:
                dump_data = json.load(f)['data']
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Warning: Failed to load {dump_path}: {e}")
            continue

        for op_name, op_data in dump_data.items():
            if not is_communication_op(op_name):
                continue

            # 获取输入输出信息
            input_args = op_data.get('input_args', [])
            output = op_data.get('output', [])

            input_has_nan = check_nan(input_args)
            output_has_nan = check_nan(output)

            # 获取输入 shape 和 max
            input_shape = []
            input_max = None
            if input_args and isinstance(input_args[0], dict):
                input_shape = input_args[0].get('shape', [])
                input_max = input_args[0].get('Max', None)

            if op_name not in comm_ops_data:
                comm_ops_data[op_name] = {}

            comm_ops_data[op_name][rank] = {
                'input_has_nan': input_has_nan,
                'output_has_nan': output_has_nan,
                'input_shape': input_shape,
                'input_max': input_max
            }

    return comm_ops_data


def cross_rank_analysis(input_path):
    """
    跨 rank 联合分析
    返回: {op_name: {source_ranks, propagated_ranks, normal_ranks, total_ranks}}
    """
    comm_ops_data = analyze_comm_ops(input_path)
    if not comm_ops_data:
        return None

    results = {}
    for op_name, rank_data in comm_ops_data.items():
        source_ranks = []
        propagated_ranks = []
        normal_ranks = []

        for rank, status in rank_data.items():
            if status['input_has_nan']:
                source_ranks.append(rank)
            elif status['output_has_nan']:
                propagated_ranks.append(rank)
            else:
                normal_ranks.append(rank)

        if source_ranks or propagated_ranks:
            results[op_name] = {
                'source_ranks': sorted(source_ranks),
                'propagated_ranks': sorted(propagated_ranks),
                'normal_ranks': sorted(normal_ranks),
                'total_ranks': len(rank_data)
            }

    return results


def analyze(input_path, output_path=None):
    """分析入口"""
    print("=" * 60)
    print("Cross-rank Overflow Analysis")
    print("=" * 60)

    results = cross_rank_analysis(input_path)

    if not results:
        print("No communication operators with NaN detected")
        return

    # 打印结果
    for op_name, data in results.items():
        print(f"\n{op_name}:")
        print(f"  Source ranks (input has NaN): {data['source_ranks']}")
        print(f"  Propagated ranks (output has NaN): {len(data['propagated_ranks'])} ranks")
        if data['propagated_ranks']:
            print(f"    -> {data['propagated_ranks'][:10]}{'...' if len(data['propagated_ranks']) > 10 else ''}")
        print(f"  Normal ranks: {len(data['normal_ranks'])} ranks")

    # 保存结果
    if output_path:
        output_file = os.path.join(output_path, 'cross_rank_analysis.json')
        os.makedirs(output_path, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cross_rank_analyzer.py <input_path> [output_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    analyze(input_path, output_path)
#!/usr/bin/env python3
"""
单卡溢出追溯脚本 - 在已知源卡的情况下，追溯导致 NaN 的具体计算算子
"""

import json
import os
import sys
import re

OVERFLOW_LIST = ['nan', 'inf', '-inf', 'nan\t', 'inf\t', '-inf\t']

# 非计算算子模式
NON_COMPUTE_PATTERNS = [
    r'^Torch\.empty',
    r'^Torch\.full',
    r'^Torch\.zeros',
    r'^Torch\.ones',
    r'^Tensor\.to',
    r'^Tensor\.clone',
    r'^Tensor\.copy',
    r'^Tensor\.new',
    r'^Tensor\.detach',
    r'^NPU\..*_empty',
    r'^NPU\..*_full',
    r'^Module\..*\.parameters',
    r'^Module\..*\.buffers',
]

# 合法 -Infinity 算子
LEGITIMATE_INF_PATTERNS = [
    r'^Tensor\.masked_fill',
    r'^Torch\.masked_fill',
    r'^Tensor\.where',
    r'^Torch\.where',
    r'^Module\..*\.router\.',
    r'^Torch\.triu',
    r'^Torch\.tril',
    r'^Tensor\.triu',
    r'^Tensor\.tril',
]


def is_non_compute_op(op_name):
    """判断是否为非计算算子"""
    for pattern in NON_COMPUTE_PATTERNS:
        if re.match(pattern, op_name):
            return True
    return False


def is_legitimate_inf(op_name):
    """判断是否合法产生 -Infinity"""
    for pattern in LEGITIMATE_INF_PATTERNS:
        if re.match(pattern, op_name):
            return True
    return False


def check_anomaly_value(value):
    """检查单个值是否包含异常，返回异常类型"""
    if isinstance(value, dict):
        max_val = str(value.get('Max', '')).lower().strip()
        min_val = str(value.get('Min', '')).lower().strip()

        if max_val == 'nan' or min_val == 'nan':
            return 'nan'
        if max_val == 'inf':
            return 'inf'
        if min_val == '-inf':
            return '-inf'
    return None


def check_anomaly_recursive(param):
    """递归检查参数是否包含异常值，返回异常类型"""
    if param is None:
        return False, None

    # 如果是 dict，检查所有 values
    if isinstance(param, dict):
        for key, value in param.items():
            anomaly = check_anomaly_value(value)
            if anomaly:
                return True, anomaly
            # 递归检查嵌套的 dict
            if isinstance(value, (dict, list, tuple)):
                result = check_anomaly_recursive(value)
                if result[0]:
                    return result
        return False, None

    # 如果是 list 或 tuple，遍历所有元素
    if isinstance(param, (list, tuple)):
        for item in param:
            if isinstance(item, dict):
                anomaly = check_anomaly_value(item)
                if anomaly:
                    return True, anomaly
            # 递归检查嵌套结构
            if isinstance(item, (dict, list, tuple)):
                result = check_anomaly_recursive(item)
                if result[0]:
                    return result
        return False, None

    return False, None


def check_anomaly(param):
    """检查参数是否包含异常值，返回异常类型（兼容旧接口）"""
    return check_anomaly_recursive(param)


def is_anomaly(op_data, op_name=None):
    """
    判断是否为异常节点（输出有异常）
    对于单卡追溯，我们检测所有输出包含 NaN/Inf 的算子
    然后按执行顺序排序找到第一个
    """
    inputs = op_data.get('input', [])
    input_args = op_data.get('input_args', [])
    input_kwargs = op_data.get('input_kwargs', {})
    outputs = op_data.get('output', {})

    is_input_anomaly, _ = check_anomaly(inputs)
    if not is_input_anomaly:
        is_input_anomaly, _ = check_anomaly(input_args)
    if not is_input_anomaly:
        is_input_anomaly, _ = check_anomaly(input_kwargs)

    is_output_anomaly, output_type = check_anomaly(outputs)

    # 排除合法的 -Infinity
    if is_output_anomaly and output_type == '-inf' and op_name:
        if is_legitimate_inf(op_name):
            return False

    # 对于通信算子，如果输入有 NaN，不算产生 NaN（是传播）
    # 但仍然标记为异常节点（因为输出有 NaN）
    if op_name and op_name.startswith('Distributed.'):
        return is_output_anomaly

    # 对于计算算子：输入无异常，输出有异常 = 根因算子
    return (not is_input_anomaly) and is_output_anomaly


def has_output_nan(op_data):
    """检查输出是否有 NaN（用于通信算子等）"""
    outputs = op_data.get('output', {})
    return has_real_nan(outputs)


def get_construct_chain(op_name, construct_data):
    """追溯调用链"""
    chain = [op_name]
    seen = set()
    curr = op_name
    while curr in construct_data and curr not in seen:
        seen.add(curr)
        next_val = construct_data.get(curr)
        if next_val is None:
            break
        if isinstance(next_val, list) and len(next_val) > 0:
            curr = next_val[0] if isinstance(next_val[0], str) else None
        elif isinstance(next_val, str):
            curr = next_val
        else:
            break
        if curr:
            chain.append(curr)
    return chain


def check_nan_value(value):
    """检查单个值是否包含 NaN"""
    if isinstance(value, dict):
        max_val = str(value.get('Max', '')).lower().strip()
        min_val = str(value.get('Min', '')).lower().strip()
        if max_val == 'nan' or min_val == 'nan':
            return True
    return False


def has_real_nan_recursive(param):
    """递归检查参数是否包含真实的 NaN（不是 -inf）"""
    if param is None:
        return False

    # 如果是 dict，检查所有 values
    if isinstance(param, dict):
        for key, value in param.items():
            if check_nan_value(value):
                return True
            # 递归检查嵌套结构
            if isinstance(value, (dict, list, tuple)):
                if has_real_nan_recursive(value):
                    return True
        return False

    # 如果是 list 或 tuple，遍历所有元素
    if isinstance(param, (list, tuple)):
        for item in param:
            if isinstance(item, dict):
                if check_nan_value(item):
                    return True
            # 递归检查嵌套结构
            if isinstance(item, (dict, list, tuple)):
                if has_real_nan_recursive(item):
                    return True
        return False

    return False


def has_real_nan(param):
    """检查参数是否包含真实的 NaN（兼容旧接口）"""
    return has_real_nan_recursive(param)


def analyze_rank(rank_path):
    """
    分析单个 rank 的数据

    注意：dump.json 中算子的出现顺序即为执行顺序，无需根据算子名称中的数字排序
    """
    dump_path = os.path.join(rank_path, 'dump.json')
    construct_path = os.path.join(rank_path, 'construct.json')
    stack_path = os.path.join(rank_path, 'stack.json')

    if not os.path.exists(dump_path):
        print(f"Error: {dump_path} not found")
        return None

    with open(dump_path, 'r', encoding='utf-8') as f:
        dump_data = json.load(f)

    construct_data = {}
    if os.path.exists(construct_path):
        with open(construct_path, 'r', encoding='utf-8') as f:
            construct_data = json.load(f)

    stack_data = {}
    if os.path.exists(stack_path):
        with open(stack_path, 'r', encoding='utf-8') as f:
            stack_data = json.load(f)

    data = dump_data.get('data', {})

    # dump.json 中算子的出现顺序即为执行顺序
    # 使用列表保存 (index, op_name, op_data) 保持原始顺序
    op_list = list(data.items())

    # 收集所有异常算子，记录其在执行序列中的位置
    anomalies = []
    for idx, (op_name, op_data) in enumerate(op_list):
        # 跳过非计算算子
        if is_non_compute_op(op_name):
            continue

        if is_anomaly(op_data, op_name):
            # 获取输入信息
            input_args = op_data.get('input_args', [])
            input_info = []
            input_has_nan = False
            for arg in input_args:
                if isinstance(arg, dict):
                    input_has_nan = input_has_nan or has_real_nan([arg])
                    input_info.append({
                        'shape': arg.get('shape', []),
                        'Max': arg.get('Max', 'N/A'),
                        'dtype': arg.get('dtype', 'N/A')
                    })

            # 追溯调用链
            construct_chain = get_construct_chain(op_name, construct_data)

            # 获取堆栈信息
            stack_info = stack_data.get(op_name, [])

            # 检查输出是否包含真实 NaN
            output = op_data.get('output', {})
            output_has_nan = has_real_nan(output)


            anomalies.append({
                'op_name': op_name,
                'exec_order': idx,  # 使用执行顺序而非算子名称中的序号
                'output': op_data.get('output', {}),
                'output_has_nan': output_has_nan,
                'input_has_nan': input_has_nan,
                'input_info': input_info,
                'construct_chain': construct_chain,
                'stack_info': stack_info
            })

    # 按执行顺序排序（保持 dump.json 中的原始顺序）
    anomalies.sort(key=lambda x: x['exec_order'])
    # print(anomalies)

    return anomalies


def find_first_anomaly(anomalies):
    """
    找到第一个产生 NaN 的算子
    优先级：
    1. 输入正常，输出有 NaN（真正的根因）
    2. 输入有 NaN，输出有 NaN（传播点）
    """
    if not anomalies:
        return None

    # 先找输入正常但输出有 NaN 的算子
    for a in anomalies:
        if not a['input_has_nan'] and a['output_has_nan']:
            return a

    # 如果没有找到，返回第一个输出有 NaN 的算子
    for a in anomalies:
        if a['output_has_nan']:
            return a

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python single_rank_tracer.py <rank_path> [output_path]")
        print("  python single_rank_tracer.py <input_path> --rank <rank> [output_path]")
        sys.exit(1)

    output_path = None
    rank_path = sys.argv[1]

    # 检查是否指定了 rank
    if len(sys.argv) >= 3 and sys.argv[2] == '--rank':
        input_path = sys.argv[1]
        rank = int(sys.argv[3])
        rank_path = os.path.join(input_path, f'rank{rank}')
        output_path = sys.argv[4] if len(sys.argv) > 4 else None
    elif len(sys.argv) > 2:
        output_path = sys.argv[2]

    print("=" * 60)
    print(f"Single Rank Overflow Tracer: {rank_path}")
    print("=" * 60)

    anomalies = analyze_rank(rank_path)

    if not anomalies:
        print("No anomaly operators found")
        return

    print(f"\nFound {len(anomalies)} anomaly operators")

    # 找到第一个异常算子
    first = find_first_anomaly(anomalies)

    if first:
        print("\nFirst anomaly (earliest in execution):")
        print(f"  Op: {first['op_name']}")
        print(f"  Exec order: {first['exec_order']}")
        print(f"  Input has NaN: {first['input_has_nan']}")
        print(f"  Output has NaN: {first['output_has_nan']}")
        print(f"  Input: {first['input_info']}")
        print(f"  Construct chain: {first['construct_chain']}")
        print(f"  Stack: {first['stack_info']}")

        # 如果输入已经有 NaN，说明根因不在当前 dump 数据中
        if first['input_has_nan']:
            print("\n[WARNING] First anomaly's input already has NaN!")
            print("  This means the root cause is NOT captured in the dump data.")
            print("  The NaN may have originated from:")
            print("    - A computation before dump started (e.g., model initialization)")
            print("    - A computation operator not captured in this rank's dump")
            print("    - A previous training step")
    else:
        print("\nNo valid anomaly found")

    # 输出结果
    result = {
        'rank_path': rank_path,
        'anomaly_count': len(anomalies),
        'first_anomaly': first,
        'all_anomalies': anomalies
    }

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'single_rank_trace.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
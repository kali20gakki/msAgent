"""msprobe分析工具公共模块：排除规则、双列表格输出、API排除判断。"""

# 不参与md5比对的API参数
EXCLUDED_PARAMS = {
    'Distributed.recv':                 {'inputs': [0]},
    'Distributed.irecv':                {'inputs': [0, 'tensor']},
    'Distributed.isend':                {'outputs': [0]},
    'Distributed.all_gather':           {'inputs': [0]},
    'Distributed.gather':               {'inputs': [1]},
    'Distributed.scatter':              {'inputs': [0]},
    'Distributed.reduce_scatter':       {'inputs': [0]},
    'Distributed._reduce_scatter_base': {'inputs': [0]},
    'Distributed._all_gather_base':     {'inputs': [0]},
    'Distributed.all_to_all_single':    {'inputs': [0]},
    'Distributed.all_to_all':           {'inputs': [0]},
    'Distributed.all_gather_into_tensor': {'inputs': [0]},
    'Distributed.reduce_scatter_tensor':  {'inputs': [0]},
    'NPU.npu_fusion_attention':         {'outputs': [4, 5]},
}


def build_excluded_rules():
    """从EXCLUDED_PARAMS构建排除规则列表，格式: [(prefix, direction, set_of_str_indices)]"""
    rules = []
    for prefix, config in EXCLUDED_PARAMS.items():
        if 'inputs' in config:
            rules.append((prefix, 'input', set(str(i) for i in config['inputs'])))
        if 'outputs' in config:
            rules.append((prefix, 'output', set(str(i) for i in config['outputs'])))
    return rules


def is_key_excluded(api_name, io_type, index_str, rules):
    """判断(api_name, io_type, index_str)是否在排除规则中。"""
    for prefix, direction, indices in rules:
        if direction != io_type:
            continue
        if api_name.startswith(prefix):
            if index_str in indices:
                return True
    return False


def print_kv_table(rank, kv_rows, col_field=20, col_value=120):
    """打印单个rank的双列表格。value_str支持换行符自动分行。"""
    sep = '+' + '-' * (col_field + 2) + '+' + '-' * (col_value + 2) + '+'
    header = f"| {'Rank ' + str(rank):^{col_field + col_value + 3}} |"

    print()
    print(sep)
    print(header)
    print(sep)
    for field, value in kv_rows:
        value_lines = value.split('\n')
        first = True
        for vl in value_lines:
            f_display = field if first else ''
            print(f"| {f_display:<{col_field}} | {vl:<{col_value}} |")
            first = False
        print(sep)


def is_api_excluded(api_name, excluded_apis):
    """判断API是否被用户排除（前缀匹配）。"""
    if not excluded_apis:
        return False
    for prefix in excluded_apis:
        if api_name.startswith(prefix):
            return True
    return False

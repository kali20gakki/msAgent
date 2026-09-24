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

"""公共工具函数 — NPU Name 解析、安全数值转换、自然排序"""

import re
import sys

# 跟踪无法按 Module./API 约定解析的 NPU Name（命名格式不兼容时用于告警）
_UNPARSEABLE_SEEN = set()        # 去重成员（用于精确去重计数）
_UNPARSEABLE_SAMPLES = []        # 首次命中顺序的样例，cap 在 MAX
_UNPARSEABLE_WARNED = False
_UNPARSEABLE_MAX_SAMPLES = 5


def _record_unparseable(name):
    """记录无法解析的 NPU Name；首次命中即时告警，便于用户立刻发现命名格式不兼容。"""
    global _UNPARSEABLE_WARNED
    if not name or not name.strip():
        return
    if name in _UNPARSEABLE_SEEN:
        return
    _UNPARSEABLE_SEEN.add(name)
    if len(_UNPARSEABLE_SAMPLES) < _UNPARSEABLE_MAX_SAMPLES:
        _UNPARSEABLE_SAMPLES.append(name)
    if not _UNPARSEABLE_WARNED:
        _UNPARSEABLE_WARNED = True
        print("⚠ [命名格式告警] 检测到无法按 Module./API 约定解析的 NPU Name，"
              "可能来自不兼容的 msprobe compare 版本。首条样例: {!r}".format(name),
              file=sys.stderr)


def flush_unparseable_names():
    """汇总输出无法解析的 NPU Name 告警（分析末尾调用，无则静默）。"""
    n = len(_UNPARSEABLE_SEEN)
    if n == 0:
        return
    samples = _UNPARSEABLE_SAMPLES[:_UNPARSEABLE_MAX_SAMPLES]
    print("⚠ [命名格式告警] 共 {} 个无法解析的 NPU Name（已去重），"
          "相关节点方向被判为 'unknown'，传播跳变分析可能缺数据。样例:".format(n),
          file=sys.stderr)
    for s in samples:
        print("    - {!r}".format(s), file=sys.stderr)
    if n > len(samples):
        print("    ... 其余 {} 个已省略".format(n - len(samples)),
              file=sys.stderr)



def safe_float(val):
    """将字符串值安全转为 float。空值 / N/A / nan / inf / unsupported 返回 None。"""
    if val is None or val.strip() in ('', 'unsupported', 'N/A', 'nan', 'inf', '-inf'):
        return None
    v = val.strip().rstrip('%')
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _nat_key(k):
    """Natural sort key: sort strings containing numbers numerically.
    e.g. 'output.2' < 'output.10' instead of lexicographic 'output.10' < 'output.2'.
    """
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', k)]


def extract_op_prefix(name):
    """从 NPU Name 中提取 (prefix, direction)。

    NPU Name 命名格式:
      - Module: Module.{path}.{direction}.{call_idx}.{param_type}.{idx}
      - API:    {api_type}.{api_name}.{call_idx}.{direction}.{param_type}.{idx}
      - parameters_grad: Module.{path}.parameters_grad.{idx}.{weight/bias}

    返回:
        prefix:    算子前缀（含 direction 和 call_idx）
                   Module → Module.{path}.{direction}.{call_idx}
                   API   → {api_type}.{api_name}.{call_idx}.{direction}
                   parameters_grad → Module.{path}.backward.{idx}
                   即 parameters_grad.{idx} 替换为 backward.{idx}
        direction: 'forward' | 'backward'

    ⚠️ parameters_grad 方向统一归为 backward。
    """
    # 1. parameters_grad → backward，prefix 中替换为 backward.{idx}
    marker = '.parameters_grad.'
    idx = name.find(marker)
    if idx >= 0:
        prefix_base = name[:idx]  # Module.{path}
        after_marker = name[idx + len(marker):]  # "{idx}.{weight/bias}"
        next_dot = after_marker.find('.')
        call_idx = after_marker[:next_dot] if next_dot > 0 else ''
        if call_idx:
            prefix = prefix_base + '.backward.' + call_idx
        else:
            prefix = prefix_base + '.backward'
        return (prefix, 'backward')

    # 2. forward / backward
    for kw in ('forward', 'backward'):
        marker = '.' + kw + '.'
        idx = name.find(marker)
        if idx >= 0:
            prefix_base = name[:idx]
            # after_marker = `${direction}.` 之后的部分（不含 trailing dot）
            after_marker = name[idx + len(marker):]

            if name.startswith('Module.'):
                # Module 格式: {path}.{direction}.{call_idx}.{param}.{idx}
                # prefix → Module.{path}.{direction}.{call_idx}
                next_dot = after_marker.find('.')
                if next_dot > 0:
                    call_idx = after_marker[:next_dot]
                    prefix = prefix_base + '.' + kw + '.' + call_idx
                else:
                    prefix = prefix_base + '.' + kw
                return (prefix, kw)
            else:
                # API 格式: {api_type}.{api_name}.{call_idx}.{direction}.{param}.{idx}
                # prefix → {api_type}.{api_name}.{call_idx}.{direction}
                prefix = prefix_base + '.' + kw
                return (prefix, kw)

    # 3. 无法识别
    _record_unparseable(name)
    return (name, 'unknown')


def get_param_key(name, prefix, direction):
    """从 NPU Name 中提取 param_key。

    新的 prefix 已包含 direction（和 Module 的 call_idx），
    param_key 就是 name 中去掉 "prefix." 后的剩余部分。

    示例:
      - Module 行:   name="Module.A.B.forward.0.input.0",  prefix="Module.A.B.forward.0"
                    → param_key = "input.0"
      - API 行:      name="torch.add.0.forward.output.0",   prefix="torch.add.0.forward"
                    → param_key = "output.0"
      - parameters_grad 行: name="Module.A.parameters_grad.0.weight",
                          prefix="Module.A.backward.0"
                    → param_key = "weight" (parameters_grad.{idx}. 之后的部分)
    """
    # parameters_grad 行：name 中的 "parameters_grad.{idx}." 与 prefix 中的 "backward.{idx}." 不匹配，
    # 所以需要特殊处理——取 "parameters_grad.{idx}." 之后的部分作为 param_key
    if '.parameters_grad.' in name:
        pg_marker = '.parameters_grad.'
        pg_idx = name.find(pg_marker)
        after_pg = name[pg_idx + len(pg_marker):]  # "{idx}.{weight/bias}"
        next_dot = after_pg.find('.')
        if next_dot > 0 and next_dot + 1 < len(after_pg):
            return after_pg[next_dot + 1:]  # 返回 "weight" 等（idx 之后的部分）
        else:
            return after_pg
    if name.startswith(prefix + '.'):
        return name[len(prefix) + 1:]
    return ''


def get_param_type(param_key):
    """根据 param_key 的内容判定 param_type。

    param_key 格式示例:
        "0.input.0"           → input
        "0.output.1"          → output
        "0.kwargs.0"          → input
        "parameters.0"        → input
        "parameters_grad.0.weight"  → output (由 flatten_rows 根据 is_pg 标记)
        "parameters_grad.0.bias"    → output (同上)
        "output.0"            → output

    规则:
        - 如果 param_key 的任一段完全等于 'parameters_grad' → output
        - 否则看 param_key 中是否包含 'output' 段 → output
        - 否则 → input

    ⚠️ parameters_grad 行 (param_key="weight"/"bias") 在 flatten_rows 中直接标记为 output，
        get_param_type 不处理此类行，此处仅保留 parameters_grad 段检测以供兼容。
    """
    parts = param_key.split('.')
    if 'parameters_grad' in parts:
        return 'output'
    if 'output' in parts:
        return 'output'
    # 单 token 且非 input/kwargs/parameters 段 → output（parameters_grad 反向梯度）
    if len(parts) == 1 and parts[0] not in ('input', 'kwargs', 'parameters'):
        return 'output'
    return 'input'

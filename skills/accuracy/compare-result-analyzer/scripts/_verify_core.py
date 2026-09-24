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

# _verify_core.py — 数据模型 + CSV解析 + 算子注册 + Tensor构造

#!/usr/bin/env python3
"""
=========================================================================
单算子精度验证框架 — NPU vs CPU
=========================================================================

用途:
    从 msProbe compare 输出的 CSV 中读取算子信息，根据 NPU 侧的统计值
    (shape / dtype / mean / l2norm / min / max) 构造输入数据，
    分别在 NPU 和 CPU 上跑该算子的 forward + backward，比对输出是否一致。

数据来源:
    CSV 中所有取值都来自 "NPU xxx" 列:
      - NPU Tensor Shape      →  tensor 的 shape
      - NPU Dtype             →  数据类型 (torch.float32 等)
      - NPU Requires_grad     →  是否需要梯度
      - NPU max / NPU min     →  值域范围 (用于 clamp)
      - NPU mean              →  均值 (用于平移)
      - NPU l2norm            →  L2 范数 (用于缩放)

    取 NPU 侧而非 Bench 侧，是因为 NPU 侧数据反映的是实际跑在 NPU 上的
    输入分布，我们要验证的是 "NPU 和 CPU 在相同输入下是否算得一致"。

验证流程:
    1. 解析 CSV → 按算子名 + 实例号分组 → 得到 OpGroup
    2. 从 OpGroup 中读取前向的输入/输出、反向的输入/输出
    3. 用 NPU 统计值构造 tensor (randn → 缩放到目标 l2norm → 平移到目标 mean → 裁剪)
    4. 同一份数据分别创建 CPU 和 NPU 两个 leaf tensor
    5. 调用注册的 torch 函数，分别在 CPU 和 NPU 上前向计算
    6. 比对 NPU 和 CPU 的输出: max_diff / l2norm_diff / rel_err
    7. 有反向数据时: 用构造的 grad_output 分别 backward，比对梯度
    8. 输出报告

=========================================================================
"""

import ast
import csv
import re
import math
import json
import os
import argparse
import sys
import functools
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Dict, List, Any, Set, Tuple

import torch
import numpy as np


# ============================================================
#  数据模型
# ============================================================
#  TensorInfo: 从 CSV 一行解析出的完整 tensor 信息
#  OpGroup:    按 (算子名, 实例号) 分组后的所有 tensor

@dataclass
class TensorInfo:
    """从 CSV 行中提取的 tensor 信息"""
    name: str                     # 完整名称，如 Tensor.__truediv__.3.forward.input.0
    op_name: str                  # 算子名，如 Tensor.__truediv__
    instance: str                 # 实例编号，如 "3"
    direction: str                # forward / backward
    io_type: str                  # input / output
    idx: int                      # 序号

    shape: List[int]              # 取自 CSV "NPU Tensor Shape" 列
    npu_dtype: str                # 取自 CSV "NPU Dtype" 列，如 "torch.float32"
    requires_grad: bool           # 取自 CSV "NPU Requires_grad" 列

    # 取自 CSV "NPU max/min/mean/l2norm" 列 — 用于构造输入 tensor
    npu_max: float
    npu_min: float
    npu_mean: float
    npu_l2norm: float

    # 是否为 None tensor (CSV 中 Shape 为 None)
    is_none: bool = False

    @property
    def is_forward(self) -> bool:
        return self.direction == "forward"

    @property
    def is_backward(self) -> bool:
        return self.direction == "backward"

    @property
    def is_output(self) -> bool:
        return self.io_type == "output"

    @property
    def is_input(self) -> bool:
        return self.io_type == "input"

    @property
    def torch_dtype(self):
        """将字符串 "torch.float32" 转为 torch.float32"""
        return getattr(torch, self.npu_dtype.split(".")[-1], torch.float32)


@dataclass
class OpGroup:
    """一个算子实例的一组 tensor 信息

    例如 Tensor.__truediv__.3 会包含:
      - fwd_inputs : [input.0 (x),  input.1 (y)]
      - fwd_outputs: [output.0 (z)]
      - bwd_inputs : [input.0 (grad_output)]
      - bwd_outputs: [output.0 (grad_x), output.1 (grad_y=None)]

    注意: backward.output 的 idx 不一定和 forward.input 的 idx 对应。
    具体配对由 _verify_backward 中的 shape 匹配算法确定。
    """
    op_name: str
    instance: str

    # 前向
    fwd_inputs: List[TensorInfo] = field(default_factory=list)
    fwd_outputs: List[TensorInfo] = field(default_factory=list)

    # 反向
    bwd_inputs: List[TensorInfo] = field(default_factory=list)
    bwd_outputs: List[TensorInfo] = field(default_factory=list)


@dataclass
class ConstructQuality:
    """单个 input tensor 的构造质量"""
    l2norm_err: float = 0.0       # l2norm 相对偏差
    clamp_ratio: float = 0.0      # 被 clamp 的元素比例
    degraded: bool = False        # 是否降级


# ============================================================
#  CSV 解析
# ============================================================
#  解析 msProbe compare 输出的 CSV 文件，提取算子信息。
#  CSV 列名中的 "NPU xxx" 对应 NPU 侧的数据，"Bench xxx" 对应 GPU 侧。
#  本框架只用 NPU 侧数据（shape, dtype, requires_grad, 统计值）来构造输入。

#  文件名格式: {算子名}.{实例号}.{forward/backward}.{input/output}.{序号}
#  正则分组:     op        instance  direction     io_type    idx
PATTERN = re.compile(
    r"^(?P<op>.+?)\.(?P<instance>\d+)\.(?P<direction>forward|backward)"
    r"\.(?P<io_type>input|output)\.(?P<idx>\d+)$"
)

# 非 torch.* 前缀、但算子真正需要的数值标量参数 dtype。
# msProbe 把 Python 标量参数（如 x * 4.0 中的 4.0）记作 <class 'float'>，
# 统计值列存有真实数值，可构造 0 维 tensor 传给算子，因此不应跳过。
_NUMERIC_SCALAR_DTYPES = {
    "<class 'float'>", "<class 'int'>", "<class 'bool'>", "<class 'complex'>",
    "float", "int", "bool", "complex",
}


def parse_csv(filepath: str) -> List[TensorInfo]:
    """解析 msProbe compare CSV，返回 TensorInfo 列表

    取值来源说明:
      - NPU Tensor Shape      → shape (tensor 维度)
      - NPU Dtype             → npu_dtype (数据类型)
      - NPU Requires_grad     → requires_grad (是否参与梯度计算)
      - NPU max / min / mean / l2norm → 统计值 (用于后续构造 tensor)
    """
    tensors = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            name = row.get("NPU Name", "").strip()
            if not name:
                continue

            m = PATTERN.match(name)
            if not m:
                # 不匹配命名规范的行跳过
                continue

            # Task 2.2: Keep _rankN suffix in op_name for grouping purposes.
            # Stripping _rankN was causing cross-rank merging of OpGroups
            # (e.g., 16 inputs → "48 arguments" error in multi-rank scenarios).
            # The wildcard matching in get_operator_fn() handles _rankN suffixes
            # via casefold substring matching. Registry lookup is unaffected.
            op_name = m.group('op')

            # 检查是否为 None tensor:
            #   - Tensor Shape 列为 None / 空
            #   - 非 tensor 输入（NPU Dtype 非 torch.* 前缀）:
            #       input: 数值标量（float/int/bool）是算子真正需要的参数，保留构造；
            #              其余（slice/str/ellipsis 等）不可构造，跳过
            #       output: 非 tensor 输出无法与 tensor 比对，统一按 None 处理
            shape_raw = row.get("NPU Tensor Shape", "").strip()
            dtype_raw = row.get("NPU Dtype", "").strip()
            is_none = shape_raw.upper() == "NONE" or shape_raw == ""
            if not is_none and not dtype_raw.startswith("torch."):
                if m.group("io_type") == "input":
                    is_none = dtype_raw not in _NUMERIC_SCALAR_DTYPES
                else:
                    is_none = True

            shape = []
            if not is_none and shape_raw:
                try:
                    shape = json.loads(shape_raw.replace("'", "\""))
                except (json.JSONDecodeError, ValueError):
                    # 兼容 '(3,)' 等 Python tuple 字面量（如 dtype=slice 的切片索引）
                    try:
                        shape = list(ast.literal_eval(shape_raw))
                    except (ValueError, SyntaxError):
                        shape = []

            rg = row.get("NPU Requires_grad", "").strip().upper()
            requires_grad = rg == "TRUE"

            def _float(key):
                v = row.get(key, "0").strip()
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return 0.0

            ti = TensorInfo(
                name=name,
                op_name=op_name,
                instance=m.group("instance"),
                direction=m.group("direction"),
                io_type=m.group("io_type"),
                idx=int(m.group("idx")),
                shape=shape,
                npu_dtype=row.get("NPU Dtype", "").strip(),
                requires_grad=requires_grad,
                # ↓ 从 NPU 侧统计值列取值，用于构造 tensor
                npu_max=_float("NPU max"),
                npu_min=_float("NPU min"),
                npu_mean=_float("NPU mean"),
                npu_l2norm=_float("NPU l2norm"),
                is_none=is_none,
            )
            tensors.append(ti)

    return tensors


def group_ops(tensors: List[TensorInfo]) -> Dict[str, OpGroup]:
    """按 (op_name, instance) 分组

    将同一个算子的同一次调用（例如 Tensor.__truediv__.3）的所有 tensor
    (forward input.0/1, forward output.0, backward input.0, ...)
    归入一个 OpGroup，便于后续统一验证。
    """
    groups: Dict[str, OpGroup] = {}
    for t in tensors:
        key = f"{t.op_name}.{t.instance}"
        if key not in groups:
            groups[key] = OpGroup(op_name=t.op_name, instance=t.instance)
        g = groups[key]
        if t.is_forward:
            if t.is_input:
                g.fwd_inputs.append(t)
            else:
                g.fwd_outputs.append(t)
        else:
            if t.is_input:
                g.bwd_inputs.append(t)
            else:
                g.bwd_outputs.append(t)

    # 按 idx 排序，保证输入/输出顺序
    for g in groups.values():
        g.fwd_inputs.sort(key=lambda x: x.idx)
        g.fwd_outputs.sort(key=lambda x: x.idx)
        g.bwd_inputs.sort(key=lambda x: x.idx)
        g.bwd_outputs.sort(key=lambda x: x.idx)

    return groups


# ============================================================
#  算子注册表
# ============================================================
#  定义算子名到实际 torch 函数的映射。
#  框架内置常用算子，未内置的通过 @register_op 装饰器注册。
#
#  forward_fn: (*inputs) → output | Tuple[outputs]
#  其中 inputs 就是按 CSV 中 input.0/1/... 顺序传入的参数。

OpFn = Callable[..., Any]
OpRegistry = Dict[str, OpFn]

OP_REGISTRY: OpRegistry = {}


def register_op(name: str):
    """装饰器: 注册一个算子"""
    def wrapper(fn: OpFn):
        OP_REGISTRY[name] = fn
        return fn
    return wrapper


def _resolve_by_getattr(root: Any, dotted_path: str) -> Optional[Callable]:
    """逐层 getattr 解析点号路径

    e.g. _resolve_by_getattr(torch, "nn.functional.relu") → torch.nn.functional.relu
    """
    parts = dotted_path.split(".")
    obj = root
    for part in parts:
        if not part or not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if callable(obj) else None


def _wrap_tensor_method(method: Callable) -> Callable:
    """将 tensor 方法包装为普通函数 fn(x, *args, **kwargs)

    torch.Tensor.__xxx__ 等 slot wrapper 可直接调用，但统一包装以防
    边缘情况（@property / @staticmethod 等 descriptor 行为不一致）。
    """
    @functools.wraps(method)
    def wrapper(x, *args, **kwargs):
        return method(x, *args, **kwargs)
    return wrapper


# ============================================================
#  PREFIX_MAP: msProbe dump 前缀 → Python 模块映射
# ============================================================
#  msProbe dump 产出遵循 support_wrap_ops.yaml 的命名规范，前缀与
#  PyTorch 实际模块名不同（如 dump 用 Torch 而 PyTorch 用 torch）。
#  本映射表覆盖全部 9 种 dump 前缀，支持多候选（如 Distributed 同时
#  映射到 torch.distributed 和 torch_npu.distributed，按优先级尝试）。
#
#  每个条目: {dump_prefix}: [(module_path, is_tensor_method), ...]
#    - module_path: Python 模块的 getattr 路径
#    - is_tensor_method: True 时走 getattr(torch.Tensor, rest) + _wrap_tensor_method

PREFIX_MAP: Dict[str, List[Tuple[str, bool]]] = {
    "Tensor":      [("torch.Tensor", True)],
    "Functional":  [("torch.nn.functional", False)],
    "Torch":       [("torch", False)],
    "NPU":         [("torch_npu", False)],
    "Aten":        [("torch.aten", False)],
    "VF":          [("torch._VF", False)],
    "Distributed": [("torch.distributed", False),
                    ("torch_npu.distributed", False)],
    "MindSpeed":   [("mindspeed", False)],
}


def _resolve_by_prefix_map(op_name: str) -> Optional[Callable]:
    """通过 PREFIX_MAP 映射表解析算子函数

    1. 提取 op_name 第一个 '.' 前的前缀
    2. 查 PREFIX_MAP 获取候选 module 列表
    3. 逐个尝试：
       - tensor method: getattr(torch.Tensor, rest) + _wrap_tensor_method
       - 普通函数: 逐层 getattr 获取模块，再解析算子
    4. 返回第一个成功的可调用对象，全部失败返回 None
    """
    dot_idx = op_name.find(".")
    if dot_idx == -1:
        return None
    prefix = op_name[:dot_idx]
    rest = op_name[dot_idx + 1:]

    candidates = PREFIX_MAP.get(prefix)
    if candidates is None:
        return None

    for module_path, is_tensor_method in candidates:
        if is_tensor_method:
            fn = getattr(torch.Tensor, rest, None)
            if fn is not None:
                return _wrap_tensor_method(fn)
        else:
            # 获取根模块对象 (如 "torch.distributed" → torch.distributed)
            parts = module_path.split(".")
            try:
                root_obj = __import__(parts[0])
            except ImportError:
                continue
            for part in parts[1:]:
                root_obj = getattr(root_obj, part, None)
                if root_obj is None:
                    break
            if root_obj is None:
                continue
            # 在目标模块上解析算子
            fn = _resolve_by_getattr(root_obj, rest)
            if fn is not None:
                return fn
    return None


def _try_resolve_op(op_name: str) -> Optional[Callable]:
    """尝试从算子名推断 PyTorch 函数（三层递进策略）

    Layer 1: PREFIX_MAP 表驱动映射
      覆盖 msProbe dump 全部 9 种前缀 → Python 模块映射:
      Tensor → torch.Tensor, Functional → torch.nn.functional,
      Torch → torch, NPU → torch_npu, Aten → torch.aten,
      VF → torch._VF, Distributed → [torch.distributed, torch_npu.distributed],
      MindSpeed → mindspeed

    Layer 2: 逐层 getattr (torch.* 路径，大小写不敏感)
      torch.nn.functional.linear → torch → .nn → .functional → .linear

    Returns:
        可调用对象，或 None（推断失败）
    """
    # Layer 1: PREFIX_MAP 表驱动映射
    fn = _resolve_by_prefix_map(op_name)
    if fn is not None:
        return fn

    # Layer 2: 逐层 getattr (大小写不敏感，兼容 Torch./torch. 等变体)
    if op_name[:6].lower() == "torch.":
        fn = _resolve_by_getattr(torch, op_name[len("torch."):])
        if fn is not None:
            return fn

    return None


_DIRECTION_PATTERN = re.compile(r"\.(forward|backward)$")


def _parse_op_list_item(item: str) -> Tuple[str, Optional[str]]:
    """解析 --op-list 条目，提取分组 key 和方向限定。

    Args:
        item: 如 "Tensor.__truediv__.3.backward" 或 "Tensor.__truediv__.3"

    Returns:
        (group_key, direction)
          - group_key: 去掉方向后缀的算子 key，用于匹配 groups dict
          - direction: "forward" / "backward" / None（None 表示全验）

    Examples:
        "Tensor.__truediv__.3.backward" → ("Tensor.__truediv__.3", "backward")
        "Tensor.__truediv__.3"          → ("Tensor.__truediv__.3", None)
        "torch.matmul.1.forward"        → ("torch.matmul.1", "forward")
    """
    m = _DIRECTION_PATTERN.search(item)
    if m:
        direction = m.group(1)
        group_key = item[:m.start()]
        return group_key, direction
    return item, None


def get_operator_fn(op_name: str, auto_register: bool = True) -> Optional[OpFn]:
    """获取算子对应的 forward 函数

    Args:
        op_name: 算子名
        auto_register: 未找到时是否尝试自动推断注册

    查找优先级:
      1. OP_REGISTRY 精确匹配
      2. OP_REGISTRY 大小写不敏感的精确匹配 (op_name.casefold() == key.casefold())
      3. 自动推断注册（auto_register=True 时）
    """
    # 1. 精确匹配
    if op_name in OP_REGISTRY:
        return OP_REGISTRY[op_name]
    # 2. 大小写不敏感的精确匹配
    #    仅做整串相等比较（不再用子串包含），避免 "torch.matmul" 误匹配
    #    "torch.matmul_backward" / "torch.matmul_out" 等长名算子。
    #    op_name 来自 NPU Name 按正则截取，反向由 direction 字段区分，
    #    不存在 "_backward" 后缀；多卡场景的 _rankN 后缀在当前命名规范下
    #    也不出现，故无需剥后缀。
    op_lower = op_name.casefold()
    for key, fn in OP_REGISTRY.items():
        if op_lower == key.casefold():
            return fn
    # 3. 自动推断注册
    if auto_register:
        fn = _try_resolve_op(op_name)
        if fn is not None:
            OP_REGISTRY[op_name] = fn
            return fn
    return None


# ---------- 内置算子 ----------
# 每个注册的算子只需实现前向逻辑，反向由 torch autograd 自动完成。

@register_op("Tensor.__truediv__")
def _truediv(x, y):
    return x / y


@register_op("Tensor.__add__")
def _add(x, y):
    return x + y


@register_op("Tensor.__mul__")
def _mul(x, y):
    return x * y


@register_op("Tensor.__sub__")
def _sub(x, y):
    return x - y


@register_op("torch.matmul")
def _matmul(x, y):
    return torch.matmul(x, y)


@register_op("torch.bmm")
def _bmm(x, y):
    return torch.bmm(x, y)


@register_op("torch.nn.functional.linear")
def _linear(x, weight, bias=None):
    if bias is not None:
        return torch.nn.functional.linear(x, weight, bias)
    return torch.nn.functional.linear(x, weight)


@register_op("torch.nn.functional.softmax")
def _softmax(x, dim=None):
    if dim is None:
        dim = -1
    return torch.nn.functional.softmax(x, dim=dim)


@register_op("torch.nn.functional.layer_norm")
def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return torch.nn.functional.layer_norm(
        x, normalized_shape, weight, bias, eps
    )


# ============================================================
#  Tensor 构造
# ============================================================
#  核心思路: CSV 文件存的是 NPU 侧统计值 (mean/l2norm/min/max)，
#  不是原始数据。我们通过以下步骤还原一个"分布相似"的 tensor:
#
#  策略自适应 (auto 模式):
#    - 区间宽松 (max - min >= 4σ): randn → scale(l2norm) → shift(mean) → clamp
#    - 区间紧   (max - min <  4σ): truncated normal → scale/shift → clamp
#
#  构造后质量校验: 回算 l2norm 偏差率和 clamp 比例，超标时降级标记。
#
#  重要: 随机数据只生成一次，然后创建两个 leaf tensor (CPU 和 NPU)，
#  确保两边输入完全一致，比对结果只反映计算差异。

# 构造参数（可通过 CLI 覆盖）
CONSTRUCT_STRATEGY = "auto"       # auto | randn | truncnorm
CONSTRUCT_L2NORM_RTOL = 0.05      # l2norm 相对偏差容忍度
CONSTRUCT_CLAMP_RATIO = 0.10      # clamp 比例容忍度
_CONSTRUCT_SEED = None             # 固定随机种子（None = 不固定）
_CONSTRUCT_SEED_COUNTER = 0        # 种子计数器

# dtype → (atol, rtol) 映射，基于各 dtype 精度边界
# float32: atol≈840×eps，matmul K=1024 理论累加误差 ≈1.2e-4
# float16: eps≈9.77e-4，atol≈1×eps 能覆盖元素级操作的硬件差异
# bfloat16: eps≈7.81e-3，atol≈0.6×eps
# 归约操作（matmul 等）由 rtol 兜底
_DTYPE_TOLERANCE = {
    'torch.float64': (1e-9, 1e-7),
    'torch.float32': (1e-4, 1e-3),
    'torch.float16': (1e-3, 1e-3),
    'torch.bfloat16': (5e-3, 5e-3),
}
_DEFAULT_ATOL = 1e-4
_DEFAULT_RTOL = 1e-3


def _effective_tolerance(output_dtype, atol, rtol):
    """根据输出 dtype 返回有效 (atol, rtol)。

    若用户通过 CLI 显式传了 atol/rtol（与默认值不同），优先使用用户值；
    否则使用 dtype 对应的默认容差。
    """
    if atol != _DEFAULT_ATOL or rtol != _DEFAULT_RTOL:
        return atol, rtol
    dtype_str = str(output_dtype)
    # 处理 torch.float32 → 'torch.float32'
    for key in _DTYPE_TOLERANCE:
        if key in dtype_str or dtype_str in key:
            return _DTYPE_TOLERANCE[key]
    return atol, rtol


def _get_construct_rng_state():
    """获取本次构造的随机种子状态"""
    global _CONSTRUCT_SEED, _CONSTRUCT_SEED_COUNTER
    if _CONSTRUCT_SEED is not None:
        seed = _CONSTRUCT_SEED + _CONSTRUCT_SEED_COUNTER
        _CONSTRUCT_SEED_COUNTER += 1
        return torch.Generator().manual_seed(seed)
    return None


def _estimate_range_tightness(stats, n_elements: int) -> bool:
    """判断 [min, max] 区间是否紧（需用 truncated normal）

    Args:
        stats: 含 max/min/l2norm 的字典
        n_elements: tensor 元素总数

    Returns:
        True = 区间紧，应使用 truncated normal
        False = 区间宽松，randn 即可
    """
    if n_elements <= 0:
        return False
    span = stats["max"] - stats["min"]
    if span <= 0:
        return True  # 区间退化，必须 truncated normal
    # σ ≈ l2norm / √n
    std_est = stats["l2norm"] / math.sqrt(n_elements)
    if std_est < 1e-30:
        return False  # l2norm 极小的零向量，无需判断区间
    # 4σ 覆盖 randn 约 95% 样本，span < 4σ 则 clamp 会大量截断
    return span < 4.0 * std_est


def _make_truncnorm(shape, dtype, stats, generator=None):
    """逆 CDF 法 truncated normal 构造 tensor

    算法: U ~ Uniform(Φ(a), Φ(b))，X = Φ⁻¹(U) × σ + μ

    Args:
        shape: tensor 形状
        dtype: 数据类型
        stats: 含 max/min/mean/l2norm 的字典
        generator: 随机数生成器（用于种子控制）

    Returns:
        CPU tensor，在 [min, max] 范围内，统计上接近目标
    """
    n_elements = 1
    for d in shape:
        n_elements *= d
    sigma = stats["l2norm"] / math.sqrt(max(n_elements, 1))
    if sigma < 1e-30:
        # l2norm 极小，直接返回常量
        val = max(stats["min"], min(stats["max"], stats["mean"]))
        return torch.full(shape, val, dtype=dtype)

    mu = stats["mean"]
    a, b = stats["min"], stats["max"]

    # 标准正态 CDF: α = Φ((a-μ)/σ), β = Φ((b-μ)/σ)
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    alpha = 0.5 * (1.0 + math.erf(a_std / math.sqrt(2.0)))
    beta = 0.5 * (1.0 + math.erf(b_std / math.sqrt(2.0)))

    if beta - alpha <= 1e-15:
        # 区间极窄退化 → 均匀分布
        t = torch.rand(shape, dtype=dtype, generator=generator)
        t = t * (b - a) + a
    else:
        # 逆 CDF 法
        u = torch.rand(shape, dtype=dtype, generator=generator)
        u = alpha + (beta - alpha) * u
        # 保护 u 不越界（数值精度）
        u = u.clamp(alpha, beta)
        # Φ⁻¹(u) = √2 × erfinv(2u - 1)
        t = mu + sigma * math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        # 保护性 clamp（数值精度兜底）
        t = t.clamp(a, b)

    # 微调: scale + shift 逼近 l2norm 和 mean
    if t.dim() > 0:
        t_norm = torch.linalg.vector_norm(t)
        if t_norm > 1e-30:
            t = t * (stats["l2norm"] / t_norm)
        t = t - t.mean() + stats["mean"]
        # 再次保护性 clamp
        t = t.clamp(a, b)

    return t


def _validate_construction(tensor, stats):
    """构造后质量校验

    Args:
        tensor: 构造完成的 tensor
        stats: 目标统计值字典

    Returns:
        ConstructQuality 对象
    """
    if tensor.dim() == 0:
        return ConstructQuality()  # 标量不校验

    n_elements = tensor.numel()

    # l2norm 相对偏差 (整数/布尔 tensor 不支持 vector_norm，跳过此项校验)
    if tensor.is_floating_point():
        actual_l2norm = torch.linalg.vector_norm(tensor).item()
        target_l2norm = max(abs(stats["l2norm"]), 1e-12)
        l2norm_err = abs(actual_l2norm - stats["l2norm"]) / target_l2norm
    else:
        l2norm_err = 0.0   # 整数不参与 l2norm 校验

    # clamp 比例
    clamped = ((tensor <= stats["min"]) | (tensor >= stats["max"])).sum().item()
    clamp_ratio = clamped / max(n_elements, 1)

    # 降级判定
    degraded = (
        l2norm_err >= CONSTRUCT_L2NORM_RTOL or
        clamp_ratio >= CONSTRUCT_CLAMP_RATIO
    )
    return ConstructQuality(
        l2norm_err=l2norm_err,
        clamp_ratio=clamp_ratio,
        degraded=degraded,
    )


def make_tensor_from_stats(shape, dtype, stats, strategy=None, generator=None):
    """从统计值构造 tensor（策略自适应）

    Args:
        shape: tensor 形状
        dtype: 数据类型 (torch.float32 等)
        stats: 包含 max/min/mean/l2norm 的字典
        strategy: "auto" | "randn" | "truncnorm"，None 使用全局 CONSTRUCT_STRATEGY
        generator: 随机数生成器

    Returns:
        CPU tensor (无 grad)，统计上与目标分布匹配

    数据源: CSV "NPU max / NPU min / NPU mean / NPU l2norm" 列
    """
    if strategy is None:
        strategy = CONSTRUCT_STRATEGY

    # 标量: 直接用均值 clamp 到 [min, max]
    if len(shape) == 0:
        val = max(stats["min"], min(stats["max"], stats["mean"]))
        return torch.tensor(val, dtype=dtype)

    # l2norm 极小的零向量: 直接返回全零（或接近零）
    if abs(stats["l2norm"]) < 1e-30:
        val = max(stats["min"], min(stats["max"], stats["mean"]))
        return torch.full(shape, val, dtype=dtype)

    # 计算元素数
    n_elements = 1
    for d in shape:
        n_elements *= d

    # P0-B#3: Integer dtypes cannot use randn/truncnorm — use torch.randint instead
    integer_dtypes = {torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,
                      torch.long, torch.short, torch.bool}
    if dtype in integer_dtypes:
        if dtype == torch.bool:
            # bool 无算数语义，直接用 randint(0,2) 生成 0/1
            return torch.randint(0, 2, shape, dtype=dtype, generator=generator)
        lo = max(-2**31, int(stats["min"]) - 1) if stats["min"] > -2**30 else int(stats["min"])
        hi = min(2**31 - 1, int(stats["max"]) + 1) if stats["max"] < 2**30 else int(stats["max"])
        if lo >= hi:
            hi = lo + 1
        return torch.randint(lo, hi, shape, dtype=dtype, generator=generator)

    # 选择策略
    use_truncnorm = False
    if strategy == "truncnorm":
        use_truncnorm = True
    elif strategy == "randn":
        use_truncnorm = False
    else:  # auto
        use_truncnorm = _estimate_range_tightness(stats, n_elements)

    if use_truncnorm:
        return _make_truncnorm(shape, dtype, stats, generator)
    else:
        # randn 策略
        t = torch.randn(shape, dtype=dtype, generator=generator)
        if t.dim() > 0:
            t_norm = torch.linalg.vector_norm(t)
            if t_norm > 1e-30:
                t = t * (stats["l2norm"] / t_norm)
            t = t - t.mean() + stats["mean"]
        t = t.clamp(stats["min"], stats["max"])
        return t


def extract_stats(ti: TensorInfo) -> dict:
    """从 TensorInfo 提取 NPU 侧统计值"""
    return {
        "max": ti.npu_max,
        "min": ti.npu_min,
        "mean": ti.npu_mean,
        "l2norm": ti.npu_l2norm,
    }


def _to_native_scalar(dtype_str: str, val: float):
    """把 CSV 中的数值标量按其原生 Python 类型构造（不当 tensor）。

    msProbe 把算子的 Python 标量参数（如 x * 4.0 中的 4.0）记作
    <class 'float'> 等形式，它本就不是 tensor，而是 Python 标量。
    这里按原生类型还原：
      float  → float,  int → int,  bool → bool,  complex → complex(实部, 0)
    complex 的虚部信息在 CSV 统计值（mean/min/max/l2norm 均为实数）中缺失，
    无法忠实还原，调用方应据此标记降级。
    未识别的 dtype 返回 None（正常流程不应到达，parse_csv 已过滤）。
    """
    if dtype_str in ("<class 'float'>", "float"):
        return float(val)
    if dtype_str in ("<class 'int'>", "int"):
        return int(val)
    if dtype_str in ("<class 'bool'>", "bool"):
        return bool(val)
    if dtype_str in ("<class 'complex'>", "complex"):
        return complex(float(val), 0.0)
    return None


def construct_tensor_pair(ti: TensorInfo) -> tuple:
    """从同一份数据构造 (cpu_tensor, npu_tensor) 对 + 构造质量

    关键设计: 随机数据只生成一次，用同一份 numpy 数组创建两个 leaf tensor。
    这样 NPU 和 CPU 的输入完全相同，比对出的 diff 只反映计算差异。

    Returns:
        (cpu_input, npu_input, ConstructQuality)
        - cpu_input / npu_input: 该 input 的构造值。None 表示该 input 为 None；
          torch.* dtype → leaf tensor；非 torch 数值标量（<class 'float'> 等）
          → 原生 Python 标量（设备无关，cpu/npu 同值）
        - ConstructQuality: 构造质量指标

    注意:
      - 用 torch.tensor(data, device="npu") 直接创建 NPU tensor，
        而不是先建 CPU tensor 再 .npu() — 后者可能产生非 leaf tensor，
        导致 backward 时 .grad 为 None。
      - 用 device="cpu" 和 device="npu" 直接创建，确保 leaf 属性。
    """
    if ti.is_none or ti.shape is None:
        return (None, None, ConstructQuality())

    # 非 torch 前缀的数值标量（<class 'float'> 等）：它本就是 Python 标量
    # 参数而非 tensor，按原生类型构造后直接传给算子；标量设备无关，
    # cpu/npu 用同一值即可，不进 tensor 路径、不调用 torch_dtype。
    if not ti.npu_dtype.startswith("torch."):
        val = max(ti.npu_min, min(ti.npu_max, ti.npu_mean))
        py_val = _to_native_scalar(ti.npu_dtype, val)
        if py_val is None:
            return (None, None, ConstructQuality())
        quality = ConstructQuality()
        if ti.npu_dtype in ("<class 'complex'>", "complex"):
            # CSV 统计值只有实数，虚部信息缺失，无法忠实还原 complex 标量 → 降级
            quality.degraded = True
        return (py_val, py_val, quality)

    stats = extract_stats(ti)
    gen = _get_construct_rng_state()
    # 生成一次随机数据，转为 numpy
    # P0-B#1: bfloat16 has no numpy dtype — convert via float32 intermediate
    is_bf16 = (ti.torch_dtype == torch.bfloat16)
    construct_dtype = ti.torch_dtype if not is_bf16 else torch.float32
    tensor_data = make_tensor_from_stats(ti.shape, construct_dtype, stats, generator=gen)
    data = tensor_data.numpy()
    # 从同一份数据创建 CPU / NPU 两个 leaf tensor
    cpu_t = torch.tensor(data, dtype=ti.torch_dtype, device="cpu",
                          requires_grad=ti.requires_grad)
    npu_t = torch.tensor(data, dtype=ti.torch_dtype, device="npu",
                          requires_grad=ti.requires_grad)
    # 构造质量校验
    quality = _validate_construction(cpu_t, stats)
    return (cpu_t, npu_t, quality)

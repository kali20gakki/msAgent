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

import json
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Set

from _verify_core import (OpGroup, TensorInfo, ConstructQuality, get_operator_fn,
                          _DTYPE_TOLERANCE, construct_tensor_pair,
                          _effective_tolerance, CONSTRUCT_STRATEGY,
                          CONSTRUCT_L2NORM_RTOL, CONSTRUCT_CLAMP_RATIO)


def _merge_construct_quality(qualities: List[ConstructQuality]) -> ConstructQuality:
    """合并多个 input 的构造质量，取最差值"""
    if not qualities:
        return ConstructQuality()
    return ConstructQuality(
        l2norm_err=max(q.l2norm_err for q in qualities),
        clamp_ratio=max(q.clamp_ratio for q in qualities),
        degraded=any(q.degraded for q in qualities),
    )


@dataclass
class VerifyResult:
    """一个 output tensor 的验证结果"""
    op_name: str
    instance: str
    direction: str           # forward / backward
    tensor_name: str         # 如 output.0
    shape_match: bool
    max_diff: float
    l2norm_diff: float
    mean_diff: float
    max_rel_err: float       # 百分比
    mean_rel_err: float      # 百分比
    passed: bool
    error: str = ""
    note: str = ""           # 配对信息，如 "matched input.1 (transposed)"
    # 构造质量（input 侧最差值）
    construct_l2norm_err: float = 0.0
    construct_clamp_ratio: float = 0.0
    construct_degraded: bool = False


def verify_operator(op_group: OpGroup,
                    atol: float = 1e-4,
                    rtol: float = 1e-3,
                    direction: Optional[str] = None) -> List[VerifyResult]:
    """验证一个算子的 forward + backward（可按方向限定）。

    流程:
      1. 查注册表找到该算子对应的 torch 函数
      2. 从 OpGroup 中读取前向输入的信息 (shape / dtype / stats)
      3. 构造输入 → 分别在 CPU/NPU 上前向计算 → 比对
      4. 如果 CSV 里有反向数据，构造 grad_output → 分别 backward → 比对梯度

    Args:
        op_group: 一个算子实例的分组信息
        atol: 绝对误差阈值 (max_diff < atol)
        rtol: 相对误差阈值 (max_rel_err < rtol × 100%)
        direction: "forward" / "backward" / None
          - "forward" → 只输出前向比对结果，跳过反向
          - "backward" → 前向仍执行（构造计算图），但只输出反向比对结果
          - None → 现有行为不变（前向+反向全输出）

    Returns:
        每个 output tensor 的验证结果列表
    """
    results: List[VerifyResult] = []

    # 查找对应 torch 函数
    fn = get_operator_fn(op_group.op_name)
    if fn is None:
        err_dir = direction if direction else "forward"
        results.append(VerifyResult(
            op_name=f"{op_group.op_name}.{op_group.instance}.{err_dir}",
            instance=op_group.instance,
            direction=err_dir,
            tensor_name="N/A",
            shape_match=True,
            max_diff=0, l2norm_diff=0, mean_diff=0,
            max_rel_err=0, mean_rel_err=0,
            passed=False,
            error=f"算子 '{op_group.op_name}' 未注册（未在注册表中找到且自动注册失败），请在 _verify_core.py 中通过 @register_op 装饰器手动注册"
        ))
        return results

    # ---- 反向验证守卫：当前组无前向数据时直接报错，不尝试空参数调用 ----
    if direction == "backward" and not op_group.fwd_inputs:
        results.append(VerifyResult(
            op_name=f"{op_group.op_name}.{op_group.instance}.backward",
            instance=op_group.instance,
            direction="backward",
            tensor_name="N/A",
            shape_match=False,
            max_diff=0, l2norm_diff=0, mean_diff=0,
            max_rel_err=0, mean_rel_err=0,
            passed=False,
            error=f"未找到同调用序号({op_group.instance})的前向数据，无法验证反向"
        ))
        return results

    # ---- 前向验证 (direction="backward" 时仍需执行以构造计算图，但不在最终结果中暴露) ----
    _verify_forward(op_group, fn, atol, rtol, results)

    # ---- 反向验证 (仅当 direction 未限定为 forward、有反向数据时) ----
    skip_backward = (direction == "forward")
    if not skip_backward:
        fwd_results = [r for r in results if r.direction == "forward"]
        fwd_pass = all(r.passed for r in fwd_results) if fwd_results else True
        if fwd_pass and op_group.bwd_inputs:
            _verify_backward(op_group, fn, atol, rtol, results)
        elif not fwd_pass and op_group.bwd_outputs:
            # Task 2.3: Forward failed — do NOT silently discard backward candidates.
            # Record each backward candidate with an explicit error entry so the
            # summary does not falsely claim "✅ 全部通过".
            for ti in op_group.bwd_outputs:
                if not ti.is_none:
                    results.append(VerifyResult(
                        op_name=op_group.op_name, instance=op_group.instance,
                        direction="backward",
                        tensor_name=f"output.{ti.idx}",
                        shape_match=False,
                        max_diff=0, l2norm_diff=0, mean_diff=0,
                        max_rel_err=0, mean_rel_err=0,
                        passed=False,
                        error="前向验证失败，反向无法验证",
                    ))

    # ---- 方向过滤 ----
    if direction is not None:
        results = [r for r in results if r.direction == direction]

    # ---- 统一 op_name 格式: {op}.{instance}.{direction}，与分析报告 NPU Name 对齐 ----
    for r in results:
        r.op_name = f"{op_group.op_name}.{op_group.instance}.{r.direction}"

    return results


def _to_comparison_dtype(t: torch.Tensor) -> torch.Tensor:
    """整数/布尔 tensor 转 float32 以便逐元素比对（减法/均值/范数仅支持浮点）。

    布尔转 1.0/0.0，整数原值转 float32；浮点/复数保持原样，
    不改变既有浮点验证行为（float 输入下返回同一对象）。
    """
    if t.is_floating_point() or t.is_complex():
        return t
    return t.to(torch.float32)


def _verify_forward(op_group, fn, atol, rtol, results):
    """验证前向传播

    步骤:
      1. 遍历 op_group.fwd_inputs，对每个 input tensor:
         - 读取其 NPU 统计值 (shape/dtype/mean/l2norm/min/max)
         - 调用 construct_tensor_pair 生成随机数据 + 构造质量
         - 生成一次数据 → 创建 cpu_t 和 npu_t 两个 leaf tensor
      2. 把构造好的输入传给 fn，分别在 CPU 和 NPU 上计算
      3. 遍历 op_group.fwd_outputs，比对每个输出的 NPU vs CPU 结果

    dtype 一致性说明: 不再做"混合 dtype 预检测/跳过"。torch 算子带 bool
    mask、int64 索引与不同 dtype 数据共存（where/masked_fill/index/gather）
    是正常签名，类型提升规则与设备无关，NPU/CPU 一致；真正不兼容的组合
    会在 fn(*inputs) 执行时报错，由下方 try/except 捕获并标注"dtype不匹配"，
    不需要也无法在构造前精确预判。
    """
    # 构造前向输入 (一次数据，CPU/NPU 对) + 收集构造质量
    cpu_inputs, npu_inputs = [], []
    fwd_qualities: List[ConstructQuality] = []
    for ti in op_group.fwd_inputs:
        cpu_t, npu_t, quality = construct_tensor_pair(ti)
        cpu_inputs.append(cpu_t)
        npu_inputs.append(npu_t)
        fwd_qualities.append(quality)

    fwd_quality = _merge_construct_quality(fwd_qualities)

    # 忽略 None 输入
    cpu_inputs_valid = [t for t in cpu_inputs if t is not None]
    npu_inputs_valid = [t for t in npu_inputs if t is not None]

    try:
        cpu_out = fn(*cpu_inputs_valid)
        npu_out = fn(*npu_inputs_valid)

        # 统一为 tuple 方便按 idx 索引
        if not isinstance(cpu_out, tuple):
            cpu_out = (cpu_out,)
            npu_out = (npu_out,)
    except Exception as e:
        # Task 2.1: Detect dtype mismatch and report it specifically,
        # not as a generic FAIL. Mixed-dtype (e.g., fp32 input + bf16 weight)
        # is a reportable conclusion, not a tool error.
        # 只按 'dtype' 关键词判定：避免 'type' 子串在普通 TypeError 消息
        # （如 "unsupported operand type(s) ... 'float'"）中误命中。
        error_msg = str(e)
        is_dtype_error = 'dtype' in error_msg.lower()
        if is_dtype_error:
            error_msg = f"dtype不匹配: {error_msg}"
        else:
            error_msg = f"前向执行异常: {error_msg}"
        for ti in op_group.fwd_outputs:
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="forward",
                tensor_name=f"output.{ti.idx}",
                shape_match=False,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=False,
                error=error_msg,
                construct_l2norm_err=fwd_quality.l2norm_err,
                construct_clamp_ratio=fwd_quality.clamp_ratio,
                construct_degraded=fwd_quality.degraded,
            ))
        return False

    all_passed = True
    for ti in op_group.fwd_outputs:
        idx = ti.idx
        if ti.is_none:
            # CSV 中标记为 None → 预期输出为 None
            actual_is_none = (idx >= len(cpu_out)) or (cpu_out[idx] is None)
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="forward",
                tensor_name=f"output.{idx}",
                shape_match=True,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=actual_is_none,
                error="" if actual_is_none else "预期 None 但得到非 None",
                construct_l2norm_err=fwd_quality.l2norm_err,
                construct_clamp_ratio=fwd_quality.clamp_ratio,
                construct_degraded=fwd_quality.degraded,
            ))
            if not actual_is_none:
                all_passed = False
            continue

        if idx >= len(cpu_out) or idx >= len(npu_out):
            # CSV 列出该输出，但算子实际未产生对应序号的输出
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="forward",
                tensor_name=f"output.{idx}",
                shape_match=False,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=False,
                error=f"算子未产生第 {idx} 个输出 (实际 CPU={len(cpu_out)}/NPU={len(npu_out)} 个)，无法验证",
                construct_l2norm_err=fwd_quality.l2norm_err,
                construct_clamp_ratio=fwd_quality.clamp_ratio,
                construct_degraded=fwd_quality.degraded,
            ))
            all_passed = False
            continue

        cpu_val = cpu_out[idx]
        npu_val = npu_out[idx]

        # 整数/布尔输出转 float32 后比对（减法/均值/范数仅支持浮点）
        cpu_cmp = _to_comparison_dtype(cpu_val)
        npu_cmp = _to_comparison_dtype(npu_val)

        if cpu_cmp.numel() == 0 or npu_cmp.numel() == 0:
            # 空 tensor: 无元素可比，仅靠 shape 判定
            max_diff = mean_diff = l2norm_diff = 0.0
            max_rel_err = mean_rel_err = 0.0
        else:
            # 逐元素比对: diff = |NPU - CPU|
            diff = (npu_cmp.cpu() - cpu_cmp).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            l2norm_diff = torch.linalg.vector_norm(npu_cmp.cpu() - cpu_cmp).item()

            # 相对误差 (百分比): |NPU - CPU| / |CPU| × 100%
            cpu_abs = cpu_cmp.abs()
            cpu_abs_clamped = cpu_abs.clamp(min=1e-12)  # 避免除零
            rel_err = diff / cpu_abs_clamped
            max_rel_err = rel_err.max().item() * 100
            mean_rel_err = rel_err.mean().item() * 100

        # shape 检查
        shape_match = list(npu_val.shape) == list(cpu_val.shape) == list(ti.shape)
        # 通过判定: shape 一致 + 绝对/相对误差在 dtype 对应阈值内
        _eff_atol, _eff_rtol = _effective_tolerance(npu_val.dtype, atol, rtol)
        passed = shape_match and (max_diff < _eff_atol) and (max_rel_err < _eff_rtol * 100)

        if not passed:
            all_passed = False

        results.append(VerifyResult(
            op_name=op_group.op_name, instance=op_group.instance,
            direction="forward",
            tensor_name=f"output.{idx}",
            shape_match=shape_match,
            max_diff=max_diff,
            l2norm_diff=l2norm_diff,
            mean_diff=mean_diff,
            max_rel_err=max_rel_err,
            mean_rel_err=mean_rel_err,
            passed=passed,
            construct_l2norm_err=fwd_quality.l2norm_err,
            construct_clamp_ratio=fwd_quality.clamp_ratio,
            construct_degraded=fwd_quality.degraded,
        ))

    return all_passed


def _match_tensor_by_shape(
    ti: TensorInfo,
    candidates: List[Tuple[int, TensorInfo]],
    allow_transpose: bool = True
) -> Tuple[Optional[int], str]:
    """核心匹配：按 shape 找最佳候选（精确 > 转置）

    Args:
        ti: 待匹配的 tensor info
        candidates: [(index, TensorInfo), ...] 候选列表
        allow_transpose: 是否允许转置匹配 (dim >= 2 时)

    Returns:
        (matched_index, match_type)
        match_type: "exact" | "transposed" | "ambiguous" | "no_match"
    """
    # Phase 1: 精确 shape 匹配
    exact = [(i, c) for i, c in candidates if c.shape == ti.shape]
    if len(exact) == 1:
        return (exact[0][0], "exact")
    if len(exact) > 1:
        return (None, "ambiguous")

    # Phase 2: 转置匹配 (dim >= 2)
    if allow_transpose:
        transposed = []
        for i, c in candidates:
            if len(c.shape) >= 2 and len(ti.shape) >= 2 and c.shape[::-1] == ti.shape:
                transposed.append((i, c))
        if len(transposed) == 1:
            return (transposed[0][0], "transposed")
        if len(transposed) > 1:
            return (None, "ambiguous")

    return (None, "no_match")


def _match_bwd_output_to_fwd_input(
    ti: TensorInfo,
    fwd_inputs: List[TensorInfo],
    already_matched: Set[int]
) -> Tuple[Optional[int], str]:
    """backward.output → forward.input：按 shape 匹配

    对每个 backward output tensor，找出它对应哪个 forward input。
    匹配依据：dtype 一致 + shape 一致（允许转置）。
    已配对的 forward input 不重复匹配。

    Returns:
        (fwd_idx, match_type)
        fwd_idx: forward.input 的原始 index，未匹配到则返回 None
    """
    candidates = []
    for i, fi in enumerate(fwd_inputs):
        if i in already_matched or fi.is_none or fi.npu_dtype != ti.npu_dtype:
            continue
        candidates.append((i, fi))
    return _match_tensor_by_shape(ti, candidates)


def _match_bwd_input_to_fwd_output(
    ti: TensorInfo,
    fwd_outputs: List[TensorInfo]
) -> Tuple[Optional[int], str]:
    """backward.input → forward.output：按 shape 匹配

    对 backward input（即 grad_output），找出它流向哪个前向输出。

    Returns:
        (fwd_out_idx, match_type)
    """
    candidates = []
    for i, fo in enumerate(fwd_outputs):
        if fo.is_none or fo.npu_dtype != ti.npu_dtype:
            continue
        candidates.append((i, fo))
    return _match_tensor_by_shape(ti, candidates)


def _verify_backward(op_group, fn, atol, rtol, results):
    """验证反向传播

    与 _verify_forward 不同，反向的输出（梯度）和前向的输入不是按 index
    一一对应的。这里的策略是用 shape 匹配（允许转置）来确定每个
    backward.output 对应哪个 forward.input，从而拿到正确的 .grad 来比对。

    匹配步骤:
      1. backward.output → shape 匹配 → forward.input (梯度来源)
      2. backward.input  → shape 匹配 → forward.output (grad_output 流向)

    这解决了两个问题:
      - CSV 中 backward 方向的输入/输出顺序不固定，不能按 idx 硬对齐
      - 部分算子的梯度 shape 是前向输入 shape 的转置 (如 linear weight)
    """
    # ---- 构建索引映射: 原始 forward input 索引 → valid 列表位置 ----
    # 有些 forward input 是 None (如 bias=None)，construct_tensor_pair
    # 返回 None，传入函数前需过滤。这导致 valid 列表的索引和原始
    # fwd_inputs 的索引不一致，需要一个映射。
    cpu_inputs, npu_inputs = [], []
    bwd_fwd_qualities: List[ConstructQuality] = []
    for ti in op_group.fwd_inputs:
        cpu_t, npu_t, quality = construct_tensor_pair(ti)
        cpu_inputs.append(cpu_t)
        npu_inputs.append(npu_t)
        bwd_fwd_qualities.append(quality)

    idx_map = {}
    cpu_inputs_valid = []
    npu_inputs_valid = []
    for i, (c, n) in enumerate(zip(cpu_inputs, npu_inputs)):
        if c is not None:
            idx_map[i] = len(cpu_inputs_valid)
            cpu_inputs_valid.append(c)
            npu_inputs_valid.append(n)

    if not cpu_inputs_valid:
        return  # 无有效输入，无法验证反向

    # ---- 重建前向计算图 ----
    try:
        cpu_out = fn(*cpu_inputs_valid)
        npu_out = fn(*npu_inputs_valid)
        if not isinstance(cpu_out, tuple):
            cpu_out = (cpu_out,)
            npu_out = (npu_out,)
    except Exception as e:
        # Task 2.1: Detect dtype mismatch for backward reconstruction too
        # 只按 'dtype' 关键词判定：避免 'type' 子串在普通 TypeError 消息
        # （如 "unsupported operand type(s) ... 'float'"）中误命中。
        error_msg = str(e)
        is_dtype_error = 'dtype' in error_msg.lower()
        if is_dtype_error:
            error_msg = f"dtype不匹配 (反向重建): {error_msg}"
        else:
            error_msg = f"反向前向重建异常: {error_msg}"
        # bwd_quality is defined after this block; use zero defaults
        for ti in op_group.bwd_outputs:
            if not ti.is_none:
                results.append(VerifyResult(
                    op_name=op_group.op_name, instance=op_group.instance,
                    direction="backward",
                    tensor_name=f"output.{ti.idx}",
                    shape_match=False,
                    max_diff=0, l2norm_diff=0, mean_diff=0,
                    max_rel_err=0, mean_rel_err=0,
                    passed=False,
                    error=error_msg,
                ))
        return

    # ---- 构造 grad_output ----
    # 用 shape 匹配找到 backward.input 对应的 forward.output
    matched_bwd_in = None
    for ti in op_group.bwd_inputs:
        if ti.is_none:
            continue
        fwd_out_idx, match_type = _match_bwd_input_to_fwd_output(ti, op_group.fwd_outputs)
        if fwd_out_idx is not None:
            matched_bwd_in = (fwd_out_idx, ti)
            break

    if matched_bwd_in is None:
        # 没找到匹配 → 用第一个非 None backward input 兜底
        for ti in op_group.bwd_inputs:
            if not ti.is_none:
                matched_bwd_in = (0, ti)  # 默认 output.0
                break

    if matched_bwd_in is None:
        return  # 无 grad_output 数据

    bwd_out_idx, grad_ti = matched_bwd_in
    grad_cpu, grad_npu, grad_quality = construct_tensor_pair(grad_ti)
    if grad_cpu is None:
        return

    # 合并构造质量: 前向输入 + grad_output
    bwd_fwd_qualities.append(grad_quality)
    bwd_quality = _merge_construct_quality(bwd_fwd_qualities)

    # ---- 执行反向传播 ----
    # 两侧匹配输出都必须有效：cpu_out/npu_out 来自同一次 fn 调用，结构通常一致，
    # 但 NPU 算子实现可能与 CPU 不等价（某侧输出为 None）。任一侧无效都应走 else
    # 记录"匹配的 forward output 无效"，而非让 .backward() 抛泛化异常。
    if (bwd_out_idx < len(cpu_out) and cpu_out[bwd_out_idx] is not None
            and bwd_out_idx < len(npu_out) and npu_out[bwd_out_idx] is not None):
        try:
            cpu_out[bwd_out_idx].backward(grad_cpu)
            npu_out[bwd_out_idx].backward(grad_npu)
        except Exception as e:
            for ti in op_group.bwd_outputs:
                if not ti.is_none:
                    results.append(VerifyResult(
                        op_name=op_group.op_name, instance=op_group.instance,
                        direction="backward",
                        tensor_name=f"output.{ti.idx}",
                        shape_match=False,
                        max_diff=0, l2norm_diff=0, mean_diff=0,
                        max_rel_err=0, mean_rel_err=0,
                        passed=False,
                        error=f"反向执行异常: {e}",
                        construct_l2norm_err=bwd_quality.l2norm_err,
                        construct_clamp_ratio=bwd_quality.clamp_ratio,
                        construct_degraded=bwd_quality.degraded,
                    ))
            return
    else:
        # 匹配到的 forward output 无效（越界或为 None），无法执行反向：
        # 不能兜底用 output.0，否则 grad_output 会流向错误的 forward output，
        # 得到的梯度比对结果无意义且会静默产生误导性结论。显式记错误并返回。
        error_msg = (f"匹配的 forward output.{bwd_out_idx} 无效"
                     f"（cpu_out 长度={len(cpu_out)}, npu_out 长度={len(npu_out)}），"
                     f"无法执行反向")
        for ti in op_group.bwd_outputs:
            if not ti.is_none:
                results.append(VerifyResult(
                    op_name=op_group.op_name, instance=op_group.instance,
                    direction="backward",
                    tensor_name=f"output.{ti.idx}",
                    shape_match=False,
                    max_diff=0, l2norm_diff=0, mean_diff=0,
                    max_rel_err=0, mean_rel_err=0,
                    passed=False,
                    error=error_msg,
                    construct_l2norm_err=bwd_quality.l2norm_err,
                    construct_clamp_ratio=bwd_quality.clamp_ratio,
                    construct_degraded=bwd_quality.degraded,
                ))
        return

    # ---- 比对梯度 ----
    # 用 shape 匹配来确定每个 backward.output 对应哪个 forward.input
    already_matched: Set[int] = set()

    for ti in op_group.bwd_outputs:
        if ti.is_none:
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="backward",
                tensor_name=f"output.{ti.idx}",
                shape_match=True,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=True,
                note="CSV marked as None",
                construct_l2norm_err=bwd_quality.l2norm_err,
                construct_clamp_ratio=bwd_quality.clamp_ratio,
                construct_degraded=bwd_quality.degraded,
            ))
            continue

        # shape 匹配：backward.output → forward.input
        fwd_idx, match_type = _match_bwd_output_to_fwd_input(
            ti, op_group.fwd_inputs, already_matched
        )

        if fwd_idx is None:
            note = match_type
            if match_type == "ambiguous":
                note = "⚠️ 无法判定（工具限制）: multiple forward inputs match"
            elif match_type == "no_match":
                note = "no matching forward input by shape (check dtype/dims)"
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="backward",
                tensor_name=f"output.{ti.idx}",
                shape_match=False,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=False,
                error=note,
                note=note,
                construct_l2norm_err=bwd_quality.l2norm_err,
                construct_clamp_ratio=bwd_quality.clamp_ratio,
                construct_degraded=bwd_quality.degraded,
            ))
            continue

        # 标记已匹配，防止重复配对
        already_matched.add(fwd_idx)

        # 原始 index → valid 列表位置
        if fwd_idx not in idx_map:
            # 对应的 forward input 是 None → 不会有梯度
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="backward",
                tensor_name=f"output.{ti.idx}",
                shape_match=True,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=True,
                note=f"matched input.{fwd_idx} (input is None, no grad expected)",
                construct_l2norm_err=bwd_quality.l2norm_err,
                construct_clamp_ratio=bwd_quality.clamp_ratio,
                construct_degraded=bwd_quality.degraded,
            ))
            continue

        valid_pos = idx_map[fwd_idx]
        cpu_in = cpu_inputs_valid[valid_pos]
        npu_in = npu_inputs_valid[valid_pos]
        # Python 标量参数（<class 'float'> 等）非 leaf tensor，不累积梯度 → 跳过比对
        if not isinstance(cpu_in, torch.Tensor) or not isinstance(npu_in, torch.Tensor):
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="backward",
                tensor_name=f"output.{ti.idx}",
                shape_match=True,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=True,
                note=f"matched input.{fwd_idx} (scalar param, no grad)",
                construct_l2norm_err=bwd_quality.l2norm_err,
                construct_clamp_ratio=bwd_quality.clamp_ratio,
                construct_degraded=bwd_quality.degraded,
            ))
            continue
        cpu_grad = cpu_in.grad
        npu_grad = npu_in.grad

        # 两边梯度均为 None → 一致 (requires_grad=False)
        if cpu_grad is None and npu_grad is None:
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="backward",
                tensor_name=f"output.{ti.idx}",
                shape_match=True,
                max_diff=0, l2norm_diff=0, mean_diff=0,
                max_rel_err=0, mean_rel_err=0,
                passed=True,
                note=f"matched input.{fwd_idx} (no grad, requires_grad=False)",
                construct_l2norm_err=bwd_quality.l2norm_err,
                construct_clamp_ratio=bwd_quality.clamp_ratio,
                construct_degraded=bwd_quality.degraded,
            ))
            continue

        # 一边 None 一边非 None → 不一致
        if cpu_grad is None or npu_grad is None:
            results.append(VerifyResult(
                op_name=op_group.op_name, instance=op_group.instance,
                direction="backward",
                tensor_name=f"output.{ti.idx}",
                shape_match=False,
                max_diff=float('inf'), l2norm_diff=float('inf'),
                mean_diff=float('inf'),
                max_rel_err=float('inf'), mean_rel_err=float('inf'),
                passed=False,
                error=f"梯度不一致: CPU={cpu_grad is not None}, NPU={npu_grad is not None}",
                note=f"matched input.{fwd_idx}",
                construct_l2norm_err=bwd_quality.l2norm_err,
                construct_clamp_ratio=bwd_quality.clamp_ratio,
                construct_degraded=bwd_quality.degraded,
            ))
            continue

        # 都非 None → 逐一比对（非浮点梯度转 float32，防 bool/int 减法崩溃）
        cpu_cmp = _to_comparison_dtype(cpu_grad)
        npu_cmp = _to_comparison_dtype(npu_grad)
        if cpu_cmp.numel() == 0 or npu_cmp.numel() == 0:
            # 空 tensor: 仅靠 shape 判定
            max_diff = mean_diff = l2norm_diff = 0.0
            max_rel_err = mean_rel_err = 0.0
        else:
            diff = (npu_cmp.cpu() - cpu_cmp).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            l2norm_diff = torch.linalg.vector_norm(npu_cmp.cpu() - cpu_cmp).item()

            cpu_abs = cpu_cmp.abs().clamp(min=1e-12)
            rel_err = diff / cpu_abs
            max_rel_err = rel_err.max().item() * 100
            mean_rel_err = rel_err.mean().item() * 100

        shape_match = list(npu_grad.shape) == list(ti.shape)
        _eff_atol, _eff_rtol = _effective_tolerance(npu_grad.dtype, atol, rtol)
        passed = shape_match and (max_diff < _eff_atol) and (max_rel_err < _eff_rtol * 100)

        if match_type == "transposed":
            note = f"matched input.{fwd_idx} (transposed)"
        else:
            note = f"matched input.{fwd_idx}"

        results.append(VerifyResult(
            op_name=op_group.op_name, instance=op_group.instance,
            direction="backward",
            tensor_name=f"output.{ti.idx}",
            shape_match=shape_match,
            max_diff=max_diff,
            l2norm_diff=l2norm_diff,
            mean_diff=mean_diff,
            max_rel_err=max_rel_err,
            mean_rel_err=mean_rel_err,
            passed=passed,
            note=note,
            construct_l2norm_err=bwd_quality.l2norm_err,
            construct_clamp_ratio=bwd_quality.clamp_ratio,
            construct_degraded=bwd_quality.degraded,
        ))


def print_header(text: str, w: int = 70):
    print(f"\n{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}")


def print_results(results: List[VerifyResult]):
    """打印验证结果到终端"""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    degraded_count = sum(1 for r in results if r.construct_degraded)

    print_header("单算子验证报告")

    for direction in ["forward", "backward"]:
        dir_results = [r for r in results if r.direction == direction]
        if not dir_results:
            continue
        print(f"\n  [{direction.upper()}]")
        for r in dir_results:
            err_str = f"  ⚠ {r.error}" if r.error else ""
            note_str = f"  [{r.note}]" if r.note else ""
            quality_str = ""
            if r.construct_degraded:
                quality_str = (f"  ⚠构造质量差(l2norm偏差={r.construct_l2norm_err:.1%},"
                             f"clamp比例={r.construct_clamp_ratio:.1%})")
            if r.passed:
                status_str = "✅ PASS"
            else:
                status_str = "❌ FAIL"
            print(f"    {status_str} | {r.tensor_name:20s} | "
                  f"max_diff={r.max_diff:.2e} | l2norm_diff={r.l2norm_diff:.2e} | "
                  f"rel_err={r.max_rel_err:.4e}%{note_str}{err_str}{quality_str}")

        dir_passed = sum(1 for r in dir_results if r.passed)
        dir_total = len(dir_results)
        print(f"    → [{dir_passed}/{dir_total} passed]")

    unregistered = sum(1 for r in results if r.error and '未注册' in str(r.error))
    unable_to_determine = sum(1 for r in results if r.note and '无法判定' in str(r.note))
    actual_failed = failed - unregistered - unable_to_determine

    print(f"\n  ──────────────────────────────────")
    parts = [f"总计: {total}", f"通过: {passed}", f"失败: {actual_failed}"]
    if unregistered > 0:
        parts.append(f"未注册: {unregistered}")
    if unable_to_determine > 0:
        parts.append(f"无法判定: {unable_to_determine}")
    print(f"  {'  |  '.join(parts)}")
    if degraded_count > 0:
        print(f"  ⚠ 构造质量偏差: {degraded_count} 条")
    if failed == 0 and unregistered == 0 and unable_to_determine == 0:
        print(f"  ✅ 全部通过!")
    elif actual_failed == 0 and unregistered == 0:
        print(f"  ⚠️  {passed} 通过，{unable_to_determine} 个无法判定（工具限制）")
    elif unregistered > 0 and actual_failed == 0:
        print(f"  ⚠️ 存在 {unregistered} 个未注册算子，{passed} 个已注册算子通过")
    else:
        print(f"  ❌ 存在 {actual_failed} 个不通过项" + (f"，{unregistered} 个未注册" if unregistered else ""))


def print_verify_params(atol, rtol):
    """打印验证参数到终端"""
    print_header("验证参数")
    print(f"  atol (float32 fallback) = {atol}")
    print(f"  rtol (float32 fallback) = {rtol} (max_rel_err < {rtol * 100:.1e}%)")
    print(f"  dtype-aware tolerance:")
    for dtype_key, (a, r) in _DTYPE_TOLERANCE.items():
        dtype_short = dtype_key.split('.')[-1]
        print(f"    {dtype_short:>8s}  atol={a:.0e}, rtol={r:.0e} (max_rel_err < {r * 100:.1e}%)")
    print(f"  construct-strategy     = {CONSTRUCT_STRATEGY}")
    print(f"  construct-l2norm-rtol  = {CONSTRUCT_L2NORM_RTOL}")
    print(f"  construct-clamp-ratio  = {CONSTRUCT_CLAMP_RATIO}")
    if _CONSTRUCT_SEED is not None:
        print(f"  construct-seed         = {_CONSTRUCT_SEED}")


def output_json(results: List[VerifyResult], filepath: str, atol: float = None, rtol: float = None):
    """输出 JSON 格式的报告"""
    data = {
        "verify_params": {
            "atol": atol,
            "rtol": rtol,
            "note": "atol/rtol are float32 fallbacks; effective tolerance is dtype-aware (see dtype_tolerance field)",
            "dtype_tolerance": {k.split('.')[-1]: {"atol": v[0], "rtol": v[1]} for k, v in _DTYPE_TOLERANCE.items()},
            "construct_strategy": CONSTRUCT_STRATEGY,
            "construct_l2norm_rtol": CONSTRUCT_L2NORM_RTOL,
            "construct_clamp_ratio": CONSTRUCT_CLAMP_RATIO,
        },
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "construct_degraded": sum(1 for r in results if r.construct_degraded),
        },
        "results": [asdict(r) for r in results],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 报告已保存至: {filepath}")

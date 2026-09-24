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
import math
import os
import sys
from collections import defaultdict
from _common import extract_op_prefix

# dtype 精度边界 — 数学常数，非场景特定值。
# 含义：当 bench_l2norm 低于此值时，NRE 的分母趋近 0，即使最小差异
#       也会导致 NRE 无条件膨胀。此时 NRE 不反映真实精度问题。
# 仅浮点类 dtype 参与过滤，理由见 FILTERABLE_DTYPES 注释。
DTYPE_EPS_BOUNDARY = {
    'torch.float32':     1.2e-7,   # ε ≈ 1.19e-7，23bit 尾数
    'torch.float':       1.2e-7,   # torch.float 是 float32 的别名
    'torch.float16':     1e-3,     # 10bit 尾数，ε ≈ 9.77e-4
    'torch.half':        1e-3,     # torch.half 是 float16 的别名
    'torch.bfloat16':    1e-2,     # 7bit 尾数，ε ≈ 7.81e-3
    'torch.bfloat':      1e-2,     # torch.bfloat 是 bfloat16 的别名
    'torch.float64':     1e-15,    # ε ≈ 2.2e-16，精度足够，极少需要处理
    'torch.double':      1e-15,    # torch.double 是 float64 的别名
    'torch.complex64':   1.2e-7,   # 等价 float32（实部精度）
    'torch.complex128':  1e-15,    # 等价 float64
}

# 参与近零噪声过滤的 dtype 集合（仅在浮点 dtype 有限精度时才有"噪声"概念）。
FILTERABLE_DTYPES = set(DTYPE_EPS_BOUNDARY.keys())

# 输出目录名
OUTPUT_SUBDIR = '.compare_result_analyzer'



def _ensure_output_dir(csv_path):
    """创建 .compare_result_analyzer/ 目录（在 CSV 文件同级），不存在时自动创建。"""
    csv_dir = os.path.dirname(os.path.abspath(csv_path)) if csv_path else os.getcwd()
    output_dir = os.path.join(csv_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def _default_output_path(csv_path):
    """返回默认 JSON 输出路径: <csv_dir>/.compare_result_analyzer/<csv_stem>_result.json"""
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(csv_dir, OUTPUT_SUBDIR, f'{csv_stem}_result.json')



def find_gap_cutoff(l2norms):
    """三守卫自适应断层检测：从 l2norm 分布中寻找天然断层作为噪声分界线。

    对同一 dtype 的所有 l2norm 排序后，计算相邻值之比。取最大比值处
    为候选分界线，但需三个守卫条件同时满足才启用：

      守卫 1 (min_samples >= 10)：至少 10 个正 l2norm 值，分布才有统计意义。
          稀疏数据（如 3~5 个点）的相邻比天然大，但不代表有意义的断层。
      守卫 2a (max_ratio >= 100)：最大相邻比的绝对倍数。
          防止密集数据（所有比值 ≈ 1）中 ratio_signal 因分母过小虚高。
      守卫 2b (ratio_signal >= 5)：最大比值 / 中位数比值 >= 5。
          最大跳变须显著大于正常波动，防止均匀长尾分布中偶然出现的
          大比值被误判为断层。

    三个守卫同时满足 → 启用 gap_cutoff；任一不满足 → 返回 None，
    回退到 dtype 精度边界兜底（design principle: 宁可漏过滤，不可误杀）。

    返回 gap_cutoff (float|None)，None 表示未找到显著断层。
    """
    MIN_SAMPLES = 10     # 守卫 1
    MIN_RATIO = 100      # 守卫 2a
    MIN_SIGNAL = 5       # 守卫 2b

    sorted_v = sorted({v for v in l2norms if v and v > 0})

    if len(sorted_v) < MIN_SAMPLES:
        return None

    ratios = [sorted_v[i + 1] / sorted_v[i] for i in range(len(sorted_v) - 1)]
    max_ratio = max(ratios)

    # 计算 ratios 的中位数（用作 ratio_signal 分母）
    sorted_ratios = sorted(ratios)
    n = len(sorted_ratios)
    if n % 2 == 0:
        median_ratio = (sorted_ratios[n // 2 - 1] + sorted_ratios[n // 2]) / 2
    else:
        median_ratio = sorted_ratios[n // 2]

    if median_ratio <= 0:
        return None

    ratio_signal = max_ratio / median_ratio

    if max_ratio >= MIN_RATIO and ratio_signal >= MIN_SIGNAL:
        return sorted_v[ratios.index(max_ratio)]

    return None



def _shape_str_to_tuple(shape_str):
    """解析 CSV shape 字符串（如 '[1, 192]'、'[]'、'None'）为 tuple 或 None。"""
    if not shape_str:
        return None
    s = shape_str.strip()
    if s.upper() == 'NONE':
        return None
    try:
        v = json.loads(s.replace("'", '"'))
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return None
    except (json.JSONDecodeError, ValueError):
        return None



def _is_divergence_legitimate(node):
    """divergence_signal 前置一致性/合法性检查。

    返回 False = 该节点是"不可比对"的对齐伪影/非法行，不应判为发散信号：
      1. bench_l2norm < 0 → 非法行（L2 范数恒 ≥ 0）
      2. bench shape 为标量 [] → 对齐伪影（bench 侧根本不是可比对的 tensor）
      3. 两侧 shape 不一致 → 对齐伪影（rank 本地分片 vs 全局序列未对齐）
    返回 True = 合法近零行，可正常走发散信号判定。
    """
    bl2 = node.get('bench_l2norm')
    if bl2 is not None and bl2 < 0:
        return False
    row = node.get('row') or {}
    bench_shape = _shape_str_to_tuple(row.get('Bench Tensor Shape'))
    npu_shape = _shape_str_to_tuple(node.get('shape'))
    if bench_shape is not None and len(bench_shape) == 0:
        return False  # bench 侧标量行
    if bench_shape is not None and npu_shape is not None and bench_shape != npu_shape:
        return False  # 两侧 shape 不一致
    return True





def _check_nre_relative(out_val, inp_val, relative_tolerance=0.001):
    """相对比例法判断 output 与 input 的误差是否一致。

    公式: |out - in| / max(|in|, EPS) <= relative_tolerance
    """
    if out_val is None or inp_val is None:
        return False
    EPS = 1e-12
    denom = max(abs(inp_val), EPS)
    return abs(out_val - inp_val) / denom <= relative_tolerance



def _is_problem_node(nre, mean_bias, threshold):
    """统一判断：节点误差是否"不可忽略"。

    双层判断：
      1. NRE 做主：NRE >= threshold → True（主信号，覆盖绝大多数场景）
      2. MeanBias 补充：NRE < threshold 但 MeanBias >= α × threshold（α = 1.2）
         → True（捕获整体均值偏移：MeanBias = |mean_diff| / bench_l2norm）

    返回 (is_problem: bool, trigger: str|None)
    """
    if nre is None:
        return False, None
    if nre >= threshold:
        return True, 'NRE'
    if mean_bias is not None:
        # MeanBias 为原始比率，threshold 为百分比，需统一量纲
        if mean_bias >= 1.2 * threshold / 100:
            return True, 'MeanBias'
    return False, None



def _check_tensor_consistency(out_nre, out_mean_re, out_max_re, out_dtype,
                               inp_nre, inp_mean_re, inp_max_re, inp_dtype):
    """检查 output 与 input 是否为同一份数据（用于判断误差继承/下游吸收）。

    NRE / MeanRE / MaxRE 使用相对比例法（`_check_nre_relative`）判断，
    避免对同量级误差在大数值区误判。
    MaxRE 作为第三维一致性验证，用于捕获"范数相近但逐元素误差分布不同"的
    方向性差异场景（仅 NRE+MeanRE 一致仍有误判风险）。
    """
    # NRE 相对比例判断
    if not _check_nre_relative(out_nre, inp_nre):
        return False
    # Dtype 一致（若非空且非 N/A）
    if out_dtype and inp_dtype and out_dtype != 'N/A' and inp_dtype != 'N/A':
        if out_dtype != inp_dtype:
            return False
    # MeanRE 相对比例判断（若两者均有值）
    if out_mean_re is not None and inp_mean_re is not None:
        if not _check_nre_relative(out_mean_re, inp_mean_re):
            return False
    # MaxRE 相对比例判断（若两者均有值）
    if out_max_re is not None and inp_max_re is not None:
        if not _check_nre_relative(out_max_re, inp_max_re):
            return False
    return True
def classify_near_zero_noise(nodes, threshold, grad_norm_spike=False):
    """近零噪声过滤：标记因 bench_l2norm 趋近 dtype 下限导致的 NRE 虚高节点。

    两段自适应方案：
      1. 数据驱动——对每种过滤 dtype 的 l2norm 分布执行三守卫断层检测，
         若发现显著断层，以断层位置为分界线
      2. 回退——无显著断层时，以该 dtype 的精度边界兜底
      最终分界线 = max(gap_cutoff, DTYPE_EPS_BOUNDARY[dtype])

    被标记为噪声的节点，其误差指标全部清零（NRE/MeanRE/MeanBias/MaxRE/MinRE → None），
    清零前原始值保存到 _orig_* 字段（供调试追溯用）。
    下游分析函数（_is_problem_node / first_problem_point / propagation_analysis）
    自然跳过噪声节点，无需修改任何下游逻辑。

    Task 4.2: When grad_norm_spike is True, backward parameters_grad nodes
    (bench_l2norm near zero) are NOT zeroed — near-zero bench is a real
    GPU gradient underflow signal, not noise.

    参数:
        nodes: collect_stat_nodes() 输出的节点列表（含 input/output）
        threshold: 用户指定的 NRE 阈值（百分比）
        format: 'text' 或 'json'
        grad_norm_spike: 是否处于 grad_norm_spike 场景（True 时保留梯度消失信号）

    返回:
        noise_nodes: 被标记的噪声节点列表
        cutoff_info: dict{ dtype: {cutoff, source, gap_found, total, noise_count} }
    """
    # Task 9.3: Universal near-zero noise floor (l2 < 1e-5).
    # Independent of dtype-specific cutoff — any tensor with l2 < 1e-5
    # and inflated NRE (>= threshold) is near-zero noise.
    NEAR_ZERO_L2_FLOOR = 1e-5

    # 双向近零判定比值：npu_l2norm > bench_l2norm × DIVERGENCE_RATIO 时，
    # 即使绝对值都小，NPU 侧仍有真实信号而 bench 侧≈0 = 梯度消失/发散信号。
    DIVERGENCE_RATIO = 10

    # Step 1: 按 dtype 汇总 l2norm 分布
    dtype_l2norms = defaultdict(list)
    dtype_nodes_map = defaultdict(list)
    for n in nodes:
        d = n.get('dtype', '').strip().lower()
        bl2 = n.get('bench_l2norm')
        if bl2 is not None and d in FILTERABLE_DTYPES:
            dtype_l2norms[d].append(bl2)
            dtype_nodes_map[d].append(n)

    # Step 2: 每种 dtype 独立计算 cutoff
    cutoff_info = {}
    for d, l2s in dtype_l2norms.items():
        gap = find_gap_cutoff(l2s) if len(l2s) >= 10 else None
        boundary = DTYPE_EPS_BOUNDARY.get(d, 0.0)
        final_cutoff = max(gap or 0.0, boundary)

        noise_count = 0
        for n in dtype_nodes_map[d]:
            nre = n.get('nre')
            bl2 = n.get('bench_l2norm')
            nl2 = n.get('npu_l2norm')
            if bl2 is not None and nre is not None and bl2 < final_cutoff and nre >= threshold:
                # 双向近零判定：仅当 bench 与 NPU 两侧都处于精度下限才是分母效应噪声。
                # 单侧近零(bench≈0 而 NPU 有信号)= 梯度消失/发散信号, 保留并标记。
                if _is_divergence_legitimate(n) and \
                   ((nl2 is not None and nl2 >= final_cutoff) or \
                    (nl2 is not None and bl2 is not None and nl2 > bl2 * DIVERGENCE_RATIO)):
                    n['divergence_signal'] = True
                    continue
                # Task 4.2: grad_norm_spike 场景下 backward parameters_grad 节点
                # bench 近零是真实 GPU 梯度下溢信号，不应被噪声过滤清零
                if grad_norm_spike:
                    name = n.get('name', '')
                    if '.parameters_grad.' in name and '.backward' in name:
                        continue
                n['is_noise'] = True
                # 清零前保存原始值（供 all_bad_nodes_detail 使用原始指标）
                n['_orig_nre'] = nre
                n['_orig_mean_re'] = n.get('mean_re')
                n['_orig_max_re'] = n.get('max_re')
                n['_orig_min_re'] = n.get('min_re')
                n['_orig_mean_bias'] = n.get('mean_bias')
                # 清零 — 下游分析自动跳过
                n['nre'] = None
                n['mean_re'] = None
                n['max_re'] = None
                n['min_re'] = None
                n['mean_bias'] = None
                noise_count += 1

        cutoff_info[d] = {
            'cutoff': final_cutoff,
            'source': 'gap' if gap is not None else 'dtype_boundary',
            'gap_found': gap is not None,
            'total': len(dtype_nodes_map[d]),
            'noise_count': noise_count,
        }

    # Task 9.3: Universal near-zero noise by scale (l2 < 1e-5).
    # Applied to ALL nodes regardless of dtype, including non-filterable types.
    # Only affects nodes not already marked as noise by dtype-specific filter.
    universal_noise_count = 0
    for n in nodes:
        if n.get('is_noise'):
            continue
        bl2 = n.get('bench_l2norm')
        nre = n.get('nre')
        nl2 = n.get('npu_l2norm')
        if bl2 is not None and nre is not None and bl2 < NEAR_ZERO_L2_FLOOR and nre >= threshold:
            # 双向近零判定 (通用下限循环)
            if _is_divergence_legitimate(n) and \
               ((nl2 is not None and nl2 >= NEAR_ZERO_L2_FLOOR) or \
                (nl2 is not None and bl2 is not None and nl2 > bl2 * DIVERGENCE_RATIO)):
                n['divergence_signal'] = True
                continue
            # grad_norm_spike 场景下 backward parameters_grad 保留
            if grad_norm_spike:
                name = n.get('name', '')
                if '.parameters_grad.' in name and '.backward' in name:
                    continue
            n['is_noise'] = True
            n['_orig_nre'] = nre
            n['_orig_mean_re'] = n.get('mean_re')
            n['_orig_max_re'] = n.get('max_re')
            n['_orig_min_re'] = n.get('min_re')
            n['_orig_mean_bias'] = n.get('mean_bias')
            n['nre'] = None
            n['mean_re'] = None
            n['max_re'] = None
            n['min_re'] = None
            n['mean_bias'] = None
            universal_noise_count += 1

    noise_nodes = [n for n in nodes if n.get('is_noise')]
    divergence_signal_nodes = [n for n in nodes if n.get('divergence_signal')]

    # Build structured result
    # P0-2: divergence_signal_details — 保留被过滤节点的可查记录（节点名/方向/原始NRE）
    divergence_signal_details = [
        {
            "node_name": n.get('name', ''),
            "direction": extract_op_prefix(n.get('name', ''))[1],
            "original_nre": n.get('_orig_nre') or n.get('nre'),
            "bench_l2": n.get('bench_l2norm'),
            "npu_l2": n.get('npu_l2norm'),
            "max_diff": n.get('max_diff')
        }
        for n in divergence_signal_nodes
    ]

    result = {
        "total_noise_nodes": len(noise_nodes),
        "divergence_signal_count": len(divergence_signal_nodes),
        "divergence_signal_details": divergence_signal_details,
        "gradient_vanishing_preserved": grad_norm_spike,
        "by_dtype": {},
        "noise_filtered_nodes": [
            {
                "name": n.get('name', ''),
                "row_index": n.get('idx', 0),
                "_orig_nre": n.get('_orig_nre'),
                "_orig_mean_re": n.get('_orig_mean_re'),
                "_orig_mean_bias": n.get('_orig_mean_bias'),
                "bench_l2norm": n.get('bench_l2norm'),
                "npu_l2norm": n.get('npu_l2norm'),
                "dtype": n.get('dtype', '')
            }
            for n in noise_nodes
        ]
    }
    for d, info in sorted(cutoff_info.items()):
        result["by_dtype"][d] = {
            "cutoff": info['cutoff'],
            "source": info['source'],
            "total": info['total'],
            "noise_count": info['noise_count']
        }

    # C1: Per-dtype self-check log to stderr
    for d, info in sorted(cutoff_info.items()):
        source_label = 'adaptive_gap' if info['source'] == 'gap' else 'dtype_boundary'
        print("[noise_filter] dtype={} cutoff={:g} source={} total={} noise={}".format(
            d, info['cutoff'], source_label, info['total'], info['noise_count']),
            file=sys.stderr)

    return noise_nodes, cutoff_info, result

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

from _common import safe_float, extract_op_prefix
from _propagation import _nre_close

def dedup_root_causes_by_nre_l2(root_cause_list):
    """Task 9.2: 按 (NRE, l2) 聚组去重。

    同一张量在多个身份（backward op output / parameters_grad / foreach_norm）
    中重复出现时，只保留一个代表行参与候选池排序。
    去重依据为 round(NRE, 4) 和 round(l2, 4) 近似匹配。

    参数:
        root_cause_list: propagation_analysis() 返回的 ROOT_CAUSE 列表
                        (tuples with at least: prefix, direction, input_nre, output_nre, jump)

    返回:
        去重后的 ROOT_CAUSE 列表
    """
    if not root_cause_list:
        return root_cause_list

    seen = set()
    deduped = []
    for rc in root_cause_list:
        out_nre = rc[3] if rc[3] is not None else 0.0
        # Extract l2 from the tuple (we store bench_l2norm in the row data,
        # but in root_cause tuples we only have NRE. Use output_nre as key.
        # For more precise matching, we round output_nre to 4 decimal places.
        nre_key = round(out_nre, 4)
        # Also use jump for additional discrimination
        jump = rc[4] if rc[4] is not None else 0.0
        jump_key = round(abs(jump), 4)
        dedup_key = (nre_key, jump_key)
        if dedup_key not in seen:
            seen.add(dedup_key)
            deduped.append(rc)
    return deduped

SHAPE_TRANSFORM_OPS = [
    'reshape', 'view', 'permute', 'transpose', 't', 'expand', 'repeat',
    'flatten', 'unsqueeze', 'squeeze', 'contiguous', 'swapaxes', 'swapdims',
    'movedim', 'moveaxis', 'ravel', 'narrow',
]

def _is_shape_transform_op(prefix):
    """检查算子名是否为纯 shape 变换算子。"""
    prefix_lower = prefix.lower()
    for sop in SHAPE_TRANSFORM_OPS:
        if sop.lower() in prefix_lower:
            return True
    return False



def trace_execution_chain(first_point, rows, threshold, max_range=500):
    """按执行顺序链追溯误差源头 (CSV 行号递减)。

    Dump 中算子的执行顺序天然反映数据流——前一个算子的 output
    通常是后一个算子的 input。按行号递减回溯, 跳过纯 shape 变换算子。

    参数:
        first_point: 首问题点 dict
        rows: 所有 CSV 行
        threshold: NRE 阈值
        max_range: 溯范围 (行数), 默认 ±500

    返回:
        {"trace_path": [...], "trace_method": "execution_order",
         "skipped_shape_ops": [...], "trace_boundary_reached": bool}
    """
    if first_point is None:
        return {"trace_path": [], "trace_method": "execution_order",
                "skipped_shape_ops": [], "trace_boundary_reached": False}

    fp_prefix = first_point.get('prefix', '')
    fp_row = first_point.get('row_index', 0)
    fp_name = first_point.get('name', '')

    # P0-A#1: Read input_nres from confirmed dict (list of dicts)
    fp_input_nres = first_point.get('input_nres', [])
    max_input_nre = max((x.get('nre') or 0 for x in fp_input_nres), default=0) if fp_input_nres else 0
    if max_input_nre < threshold:
        # 首问题点 input 干净, 误差由本算子引入, 无需追溯
        return {
            "trace_path": [{"prefix": fp_prefix, "row": fp_row,
                           "nre": first_point.get('nre', 0),
                           "role": "first_point", "note": "input 干净, 误差由本算子引入"}],
            "trace_method": "execution_order",
            "skipped_shape_ops": [],
            "trace_boundary_reached": False
        }

    # 构建行号→行数据映射 (仅 output 行)
    output_rows = {}
    for r in rows:
        name = r.get('NPU Name', '')
        if '.output.' in name or '.parameters_grad.' in name:
            from _common import safe_float
            nre = safe_float(r.get('NormRelativeErr', ''))
            row_idx = r.get('RowIndex', 0)
            prefix, direction = extract_op_prefix(name)
            if prefix and direction != 'unknown':
                output_rows[row_idx] = {
                    'name': name, 'prefix': prefix, 'direction': direction,
                    'nre': nre if nre is not None else 0, 'row': row_idx
                }

    if not output_rows:
        return {"trace_path": [], "trace_method": "execution_order",
                "skipped_shape_ops": [], "trace_boundary_reached": False}

    # P2-10: 检测首问题点方向，backward pass 需要反转行号方向
    fp_direction = extract_op_prefix(fp_name)[1] if fp_name else 'forward'
    is_backward_trace = (fp_direction == 'backward'
                         or '.backward' in fp_prefix.lower())

    # P2-10: backward 节点的"上游"取更大行号 (更接近网络深层)
    # forward 节点的"上游"取更小行号
    if is_backward_trace:
        step = 1   # 向行号增大方向追溯
    else:
        step = -1  # 向行号减小方向追溯

    # 按行号回溯 (方向感知)
    trace_path = [{
        "prefix": fp_prefix, "row": fp_row,
        "nre": first_point.get('nre', 0),
        "role": "first_point"
    }]
    skipped_shape_ops = []
    skipped_inconsistent = []
    boundary_reached = False
    trace_boundary_reason = None  # P2: reason for trace boundary

    current_row = fp_row
    visited_prefixes = {fp_prefix}

    for _ in range(max_range):
        current_row += step
        if current_row <= 0 or current_row > max(r.get('RowIndex', 0) for r in rows):
            boundary_reached = True
            trace_boundary_reason = 'out_of_range'
            break

        if current_row not in output_rows:
            continue

        prev_op = output_rows[current_row]
        prev_prefix = prev_op['prefix']

        # 避免循环
        if prev_prefix in visited_prefixes:
            continue
        visited_prefixes.add(prev_prefix)

        # 跳过 shape 变换算子
        if _is_shape_transform_op(prev_prefix):
            skipped_shape_ops.append({
                "prefix": prev_prefix, "row": current_row,
                "nre": prev_op['nre']
            })
            continue

        # NRE 一致性校验：与目标 input 误差不一致的节点不是真实生产者
        if not _nre_close(prev_op['nre'], max_input_nre):
            skipped_inconsistent.append({
                "prefix": prev_prefix, "row": current_row,
                "nre": prev_op['nre']
            })
            continue

        # 找到候选溯源节点
        trace_path.append({
            "prefix": prev_prefix, "row": current_row,
            "nre": prev_op['nre'],
            "role": "upstream_source"
        })

        # 检查该节点是否本身引入了误差 (ROOT_CAUSE 特征)
        # 如果 NRE < threshold, 继续向上追溯
        if prev_op['nre'] < threshold:
            continue
        else:
            # 找到可能的误差源, 但继续追溯确保没有更早的源
            pass

        # 如果已经向上追溯了足够远且找到了 NRE >= threshold 的节点, 可停止
        if len(trace_path) >= 3 and prev_op['nre'] >= threshold:
            break

    if len(trace_path) <= 1 and not boundary_reached:
        boundary_reached = True
        if not trace_boundary_reason:
            trace_boundary_reason = 'no_match'

    if len(trace_path) <= 1 and skipped_inconsistent and trace_boundary_reason == 'out_of_range':
        trace_boundary_reason = 'no_match'

    trace_note = None
    if skipped_inconsistent:
        trace_note = ("上游存在 {} 个 NRE 与目标 input 不一致的节点, "
                      "真实生产者可能位于数据覆盖缺口内或结构错位区").format(len(skipped_inconsistent))

    return {
        "trace_path": trace_path,
        "trace_method": "execution_order",
        "skipped_shape_ops": skipped_shape_ops,
        "skipped_inconsistent": skipped_inconsistent,
        "trace_note": trace_note,
        "trace_boundary_reached": boundary_reached,
        "trace_boundary_reason": trace_boundary_reason
    }


def detect_data_coverage_gaps(rows, threshold, noise_level=0.01):
    """检测数据覆盖缺口: 干净output→脏input之间无可比对节点的区域。

    遍历执行顺序链，若相邻两个可比对节点 A→B 满足:
      1. A.output NRE <= noise_level (干净输出)
      2. B.input NRE > threshold (脏输入)
      3. A→B 之间不存在其他可比对节点
    则标记为 data_coverage_gap。

    Args:
        rows: CSV 行数据
        threshold: NRE 阈值
        noise_level: 干净输出 NRE 上限 (默认 0.01%)

    Returns:
        list of dict: [{from_op, to_op, from_row, to_row, gap_size, from_output_nre, to_input_nre}]
    """
    # 收集所有可比对的 output/input 行
    comparable = []  # (row_idx, prefix, direction, nre, io_type)
    for r in rows:
        name = r.get('NPU Name', '')
        nre = safe_float(r.get('NormRelativeErr', ''))
        if nre is None:
            continue
        row_idx = r.get('RowIndex', 0)
        prefix, direction = extract_op_prefix(name)
        if direction == 'unknown':
            continue
        if '.output.' in name or '.parameters_grad.' in name:
            comparable.append((row_idx, prefix, direction, nre, 'output'))
        elif '.input.' in name:
            comparable.append((row_idx, prefix, direction, nre, 'input'))

    comparable.sort(key=lambda x: x[0])

    gaps = []
    gap_seen = set()
    for i in range(len(comparable) - 1):
        a = comparable[i]
        b = comparable[i + 1]
        # A must be an output, B must be an input
        if a[4] != 'output' or b[4] != 'input':
            continue
        # A.output NRE <= noise_level (clean)
        if a[3] > noise_level:
            continue
        # B.input NRE > threshold (dirty)
        if b[3] <= threshold:
            continue
        # No other comparable node between them (already ensured by adjacent in sorted list)
        gap_size = b[0] - a[0]
        if gap_size <= 1:
            continue
        # Dedup by (from_op, to_op)
        gap_key = (a[1], b[1])
        if gap_key in gap_seen:
            continue
        gap_seen.add(gap_key)
        gaps.append({
            "from_op": a[1],
            "to_op": b[1],
            "from_row": a[0],
            "to_row": b[0],
            "gap_size": gap_size,
            "from_output_nre": round(a[3], 4),
            "to_input_nre": round(b[3], 4)
        })

    return gaps

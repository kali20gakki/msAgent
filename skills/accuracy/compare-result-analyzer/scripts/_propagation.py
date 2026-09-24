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

from collections import defaultdict
from _common import safe_float, extract_op_prefix, get_param_key, get_param_type, _nat_key
from _data_io import get_input_nres_for_op
from _noise_filter import (_check_nre_relative, _is_problem_node, _check_tensor_consistency)



def first_problem_point(output_nodes, threshold, all_rows=None, analysis_range=500):
    """首问题点：按执行顺序第一个 NRE >= threshold 的 output 节点。

    关键约束：
    1. INPUT_PROPAGATION：output 误差继承自 input（NRE + dtype + MeanRE + MaxRE 一致）。
    2. DOWNSTREAM_ABSORBED：output 误差被下游算子吸收（同上一致性检查 + 下游 output NRE < 阈值）。
       - 下游吸收检查限定 +analysis_range 行范围（传播总是按序，无需全量扫描）。
    """
    DOWNSTREAM_ROW_RANGE = analysis_range  # 下游吸收检查行数范围

    first = None
    first_trigger = None
    skipped = []
    for n in output_nodes:
        is_prob, trigger = _is_problem_node(n['nre'], n.get('mean_bias'), threshold)
        if is_prob:
            out_nre = n['nre']
            out_mean_re = n.get('mean_re')
            out_max_re = n.get('max_re')
            out_mean_bias = n.get('mean_bias')
            out_shape = n.get('shape', '')
            out_dtype = n.get('dtype', '')

            # === 检查 1: INPUT_PROPAGATION ===
            if all_rows:
                prefix, direction = extract_op_prefix(n['name'])
                if direction != 'unknown':
                    inputs = get_input_nres_for_op(all_rows, prefix, direction)
                    inherited = False
                    inherited_key = None
                    for inp_key, inp_nre, inp_mean_re, inp_max_re, inp_shape, inp_dtype, _ in inputs:
                        if _check_tensor_consistency(out_nre, out_mean_re, out_max_re,
                                                     out_dtype,
                                                     inp_nre, inp_mean_re, inp_max_re,
                                                     inp_dtype):
                            inherited = True
                            inherited_key = inp_key
                            break
                    if inherited:
                        skipped.append((n, inherited_key, 'INPUT_PROPAGATION'))
                        continue

            # === 检查 2: DOWNSTREAM_ABSORBED ===
            # Task 3: Group all output rows of each downstream operator by
            # (prefix, direction), then check max NRE across all output rows.
            # An operator is "absorbed" only if ALL its output rows have NRE < threshold.
            # Any single dirty output means the error is still propagating downstream.
            if all_rows:
                candidate_idx = n['idx']
                absorbed_downstream = False
                absorbed_by = None
                # Build a dict grouping downstream outputs by (prefix, direction)
                dn_op_outputs = {}
                dn_op_order = []  # preserve order
                for dn in output_nodes:
                    if dn['idx'] <= candidate_idx:
                        continue
                    if dn['idx'] > candidate_idx + DOWNSTREAM_ROW_RANGE:
                        break
                    dn_prefix, dn_dir = extract_op_prefix(dn['name'])
                    if dn_dir == 'unknown':
                        continue
                    key = (dn_prefix, dn_dir)
                    if key not in dn_op_outputs:
                        dn_op_outputs[key] = []
                        dn_op_order.append(key)
                    dn_op_outputs[key].append(dn)
                for key in dn_op_order:
                    dn_nodes = dn_op_outputs[key]
                    dn_prefix, dn_dir = key
                    dn_inputs = get_input_nres_for_op(all_rows, dn_prefix, dn_dir)
                    # Check if candidate's output matches any downstream input
                    matches_candidate_output = False
                    for inp_key, inp_nre, inp_mean_re, inp_max_re, inp_shape, inp_dtype, _ in dn_inputs:
                        if _check_tensor_consistency(out_nre, out_mean_re, out_max_re,
                                                     out_dtype,
                                                     inp_nre, inp_mean_re, inp_max_re,
                                                     inp_dtype):
                            matches_candidate_output = True
                            break
                    if not matches_candidate_output:
                        continue
                    # Candidate's output feeds into this downstream operator.
                    # Now check ALL outputs of this downstream operator:
                    # only absorbed if max(all output NREs) < threshold.
                    dn_output_nres = [d.get('nre') for d in dn_nodes if d.get('nre') is not None]
                    if dn_output_nres and max(dn_output_nres) < threshold:
                        absorbed_downstream = True
                        absorbed_by = dn_prefix
                        break
                if absorbed_downstream:
                    skipped.append((n, absorbed_by, 'DOWNSTREAM_ABSORBED'))
                    continue

            first = n
            first_trigger = trigger
            break

    if first:
        first['_trigger'] = first_trigger

    # Build structured result
    discovery_chain = []
    for node, reason, category in skipped:
        discovery_chain.append({
            "node": node['name'],
            "row_index": node['idx'],
            "result": category,
            "reason": reason
        })

    confirmed = None
    if first:
        # P0-A#1: Collect per-input NRE details for trace_execution_chain
        input_nres = []
        if all_rows:
            prefix, direction = extract_op_prefix(first['name'])
            if direction != 'unknown':
                inputs = get_input_nres_for_op(all_rows, prefix, direction)
                input_nres = [
                    {
                        "param_key": inp_key,
                        "nre": inp_nre,
                        "dtype": inp_dtype,
                        "mean_re": inp_mean_re,
                        "max_re": inp_max_re,
                    }
                    for inp_key, inp_nre, inp_mean_re, inp_max_re, inp_shape, inp_dtype, _ in inputs
                ]
        confirmed = {
            "prefix": extract_op_prefix(first['name'])[0],
            "direction": extract_op_prefix(first['name'])[1],
            "name": first['name'],
            "row_index": first['idx'],
            "nre": first['nre'],
            "mean_re": first.get('mean_re'),
            "mean_bias": first.get('mean_bias'),
            "max_re": first.get('max_re'),
            "min_re": first.get('min_re'),
            "trigger": first_trigger,
            "dtype": first.get('dtype', ''),
            "shape": first.get('shape', ''),
            "result": first.get('result', ''),
            "input_nres": input_nres
        }
        discovery_chain.append({
            "node": first['name'],
            "row_index": first['idx'],
            "result": "CONFIRMED"
        })

    result = {
        "confirmed": confirmed,
        "discovery_chain": discovery_chain
    }

    return first, skipped, result


def _nre_close(a, b, tol=0.5):
    """检查两个 NRE 值是否在相对容差内一致（相对差 ≤50%）。"""
    if a is None or b is None:
        return False
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= tol



def propagation_analysis(rows, threshold, noise_node_names=None):
    """传播跳变分析

    NPU Name 命名格式：
      - API:    {api_type}.{api_name}.{call_idx}.{forward/backward}.{io}.{idx}
      - Module: Module.{path}.{forward/backward}.{call_idx}.{io}.{idx}
      - parameters_grad: Module.{path}.parameters_grad.{idx}.{weight/bias}
        (parameters_grad 归到该算子的 backward 方向，作为反向输出)

    分类规则：
    - INPUT_PROPAGATION：某 output 误差继承自某 input
    - ROOT CAUSE：某 input < 阈值，某 output >= 阈值
    - ABSORBED：某 input >= 阈值，某 output < 阈值
    - PROPAGATION：某 input >= 阈值，某 output > 该 input（放大）
    - PASS_THROUGH：某 input >= 阈值，某 output >= 阈值且 output < 该 input（输出误差缩小）
    """

    # NRE/MeanRE 一致性判断使用相对比例法，不再基于 threshold 派生绝对差值

    op_prefixes = {}
    param_grad_keys = {}
    for r in rows:
        name = r['NPU Name']
        # Task 4.1: Skip noise-zeroed nodes — propagation uses same filtered
        # data as first_problem_point discovery
        if noise_node_names and name in noise_node_names:
            continue
        prefix, direction = extract_op_prefix(name)
        if direction == 'unknown':
            continue
        # 提取 param（parameters_grad: prefix 之后; 普通行: prefix.direction. 之后）
        param = get_param_key(name, prefix, direction)
        if '.parameters_grad.' in name:
            param_grad_keys.setdefault((prefix, direction), set()).add(param)
        if prefix not in op_prefixes:
            op_prefixes[prefix] = {}
        if direction not in op_prefixes[prefix]:
            op_prefixes[prefix][direction] = {}
        op_prefixes[prefix][direction][param] = r

    root_cause = []
    absorbed = []
    propagation = []
    input_propagation = []
    pass_through = []

    root_cause_param_grad = set()  # P0-A#3: entries from param-grad outputs (backward parameters_grad)

    for prefix, directions in op_prefixes.items():
        for direction, params in directions.items():
            # 收集所有 input 的 NRE 值
            input_nres = []
            for key in sorted(params.keys(), key=_nat_key):
                if get_param_type(key) == 'input':
                    nre = safe_float(params[key].get('NormRelativeErr', ''))
                    mean_re = safe_float(params[key].get('MeanRelativeErr', ''))
                    max_re = safe_float(params[key].get('MaxRelativeErr', ''))
                    mean_diff = safe_float(params[key].get('Mean diff', ''))
                    bench_l2norm = safe_float(params[key].get('Bench l2norm', ''))
                    mean_bias = abs(mean_diff) / bench_l2norm if (mean_diff is not None and bench_l2norm is not None and bench_l2norm > 1e-12) else None
                    shape = params[key].get('NPU Tensor Shape', '').strip()
                    dtype = params[key].get('NPU Dtype', '').strip()
                    if nre is not None:
                        input_nres.append((nre, mean_re, max_re, mean_bias, shape, dtype, key))

            # 收集所有 output 的 NRE 值
            output_nres = []
            for key in sorted(params.keys(), key=_nat_key):
                if get_param_type(key) == 'output':
                    nre = safe_float(params[key].get('NormRelativeErr', ''))
                    mean_re = safe_float(params[key].get('MeanRelativeErr', ''))
                    max_re = safe_float(params[key].get('MaxRelativeErr', ''))
                    mean_diff = safe_float(params[key].get('Mean diff', ''))
                    bench_l2norm = safe_float(params[key].get('Bench l2norm', ''))
                    mean_bias = abs(mean_diff) / bench_l2norm if (mean_diff is not None and bench_l2norm is not None and bench_l2norm > 1e-12) else None
                    shape = params[key].get('NPU Tensor Shape', '').strip()
                    dtype = params[key].get('NPU Dtype', '').strip()
                    if nre is not None:
                        output_nres.append((nre, mean_re, max_re, mean_bias, shape, dtype, key))

            if not output_nres:
                continue

            # P0-A#3: For backward direction, split outputs into propagation-chain
            # (grad_input) and param-grad (parameters_grad: grad_weight/bias).
            # Param-grad outputs are independently evaluated — their error is produced
            # by the backward kernel itself, not propagated through the gradient chain.
            if direction == 'backward':
                _pg_keys = param_grad_keys.get((prefix, direction), set())
                grad_input_output_nres = []
                param_grad_output_nres = []
                for out_nre, out_mean, out_max_re, out_mean_bias, out_shape, out_dtype, out_key in output_nres:
                    if out_key in _pg_keys:
                        param_grad_output_nres.append((out_nre, out_mean, out_max_re, out_mean_bias, out_shape, out_dtype, out_key))
                    else:
                        grad_input_output_nres.append((out_nre, out_mean, out_max_re, out_mean_bias, out_shape, out_dtype, out_key))

                # Independently evaluate param-grad outputs (C-ANALYSIS-021)
                # 继承性对照三档 (2026-08-07)：按本算子 backward 输入 max_input_nre 判定
                #   - 输入已脏(max_in>=threshold) 且 out <= max_in×2  → 继承：误差来自输入，降级入 input_propagation
                #   - 输入已脏 且 out > max_in×2                        → 放大：backward kernel 放大/引入，ROOT_CAUSE
                #   - 输入干净(max_in<threshold) 仍超阈值                → 实现问题：纯 backward kernel 差异，ROOT_CAUSE
                #   聚焦子集（input_nres 空，--keep-only parameters_grad）无输入可对照 → 保持独立评估 ROOT_CAUSE
                _max_input_nre = (max((inp[0] for inp in input_nres if inp[0] is not None), default=0)
                                  if input_nres else None)
                for out_nre, out_mean, out_max_re, out_mean_bias, out_shape, out_dtype, out_key in param_grad_output_nres:
                    out_prob, out_trig = _is_problem_node(out_nre, out_mean_bias, threshold)
                    if not out_prob:
                        continue
                    # 继承性对照三档（仅 NRE 触发且全量分析可判）
                    inheritance = None
                    if out_trig == 'NRE' and _max_input_nre is not None and threshold is not None:
                        if _max_input_nre >= threshold:
                            inheritance = 'inherited' if out_nre <= _max_input_nre * 2 else 'amplified'
                        else:
                            inheritance = 'impl_only'
                    if inheritance == 'inherited':
                        # 继承：误差继承自本算子输入（grad_output/saved tensors），根因在输入来源，
                        # 降级为传播（非根因），并指向脏输入来源供溯源
                        _src_key = max(input_nres, key=lambda inp: inp[0])[6]
                        input_propagation.append((prefix, direction, _max_input_nre, out_nre, 0.0,
                                                  out_mean, out_mean, 'PARAM_GRAD_INHERITED', _src_key))
                        continue
                    # ROOT_CAUSE：放大 / 实现问题 / 聚焦子集 / MeanBias 触发（保持独立评估）
                    if inheritance in ('amplified', 'impl_only'):
                        _cat = 'ROOT_CAUSE (param_grad: {})'.format(inheritance)
                    elif out_trig == 'MeanBias':
                        _cat = 'ROOT_CAUSE (MeanBias)'
                    else:
                        _cat = 'ROOT_CAUSE'
                    if out_trig == 'NRE':
                        rc_entry = (prefix, direction, None, out_nre,
                                    out_nre, out_mean, out_mean,
                                    _cat, 'NRE', inheritance)
                    else:
                        rc_entry = (prefix, direction, None, out_mean_bias,
                                    out_mean_bias, out_mean, out_mean,
                                    _cat, 'MeanBias', inheritance)
                    root_cause.append(rc_entry)
                    root_cause_param_grad.add(rc_entry)

                # Use only grad_input (propagation-chain) outputs for propagation analysis
                output_nres = grad_input_output_nres
                if not output_nres:
                    continue

            # 无 input 则跳过（无法判定传播跳变）
            if not input_nres:
                continue

            # ⚠️ 检查所有 output（而非仅第一个），取最差情况，避免漏排
            # Step 1: 对每个 output，检查是否继承自某个 input（INPUT_PROPAGATION）
            inherited_outputs = []
            non_inherited_outputs = []
            for out_nre, out_mean, out_max_re, out_mean_bias, out_shape, out_dtype, out_key in output_nres:
                out_inherited = False
                out_prob, _ = _is_problem_node(out_nre, out_mean_bias, threshold)
                if out_prob:
                    for inp_nre, inp_mean, inp_max_re, _, inp_shape, inp_dtype, inp_key in input_nres:
                        if inp_nre is not None and _check_tensor_consistency(
                                out_nre, out_mean, out_max_re,
                                out_dtype,
                                inp_nre, inp_mean, inp_max_re,
                                inp_dtype):
                            inherited_outputs.append((out_nre, out_mean, out_max_re, out_mean_bias, out_key,
                                                      inp_nre, inp_mean, inp_max_re, inp_key))
                            out_inherited = True
                            break
                if not out_inherited:
                    non_inherited_outputs.append((out_nre, out_mean, out_max_re, out_mean_bias, out_key))

            # Step 2: 继承的 output 始终归入 INPUT_PROPAGATION
            for out_nre, out_mean, out_max_re, out_mean_bias, out_key, inp_nre, inp_mean, inp_max_re, inp_key in inherited_outputs:
                input_propagation.append((prefix, direction, inp_nre, out_nre, 0.0,
                                          inp_mean, out_mean, 'INPUT_PROPAGATION', inp_key))

            # Step 3: 对所有 input/output 组合逐一检查分类条件
            if not input_nres:
                input_nres_for_check = [(0.0, None, None, None, '', '', 'virtual')]
            else:
                input_nres_for_check = input_nres

            all_outputs_for_check = non_inherited_outputs + [
                (out_nre, out_mean, out_max_re, out_mean_bias, out_key)
                for out_nre, out_mean, out_max_re, out_mean_bias, out_key, _, _, _, _ in inherited_outputs
            ]

            # P0-A#1: Compute max data-input NRE (exclude parameters.* inputs)
            # and check if any data input is already dirty.
            data_input_nres = [inp_nre for inp_nre, _, _, _, _, _, key in input_nres
                               if not key.startswith('parameters.') and inp_nre is not None]
            max_data_in_nre = max(data_input_nres) if data_input_nres else 0.0
            has_dirty_data_input = max_data_in_nre >= threshold

            for inp_nre, inp_mean, _, inp_mean_bias, _, _, inp_key in input_nres_for_check:
                for out_nre, out_mean, _, out_mean_bias, _ in all_outputs_for_check:

                    inp_prob, inp_trig = _is_problem_node(inp_nre, inp_mean_bias, threshold)
                    out_prob, out_trig = _is_problem_node(out_nre, out_mean_bias, threshold)

                    # ===== ROOT CAUSE =====
                    if not inp_prob and out_prob:
                        # P0-A#1: Data-input priority check.
                        # If any data input is already dirty (NRE >= threshold),
                        # only classify as ROOT_CAUSE if the output genuinely amplifies
                        # the error beyond the max data input NRE.
                        # P5#9: Skip non-tensor inputs (NRE is None/N/A) — they never
                        # participate in clean→dirty path determination.
                        if has_dirty_data_input:
                            if out_nre is not None and out_nre > max_data_in_nre * 1.1:
                                if out_trig == 'NRE':
                                    root_cause.append((prefix, direction, inp_nre, out_nre,
                                                       out_nre - (inp_nre or 0), inp_mean, out_mean,
                                                       'ROOT_CAUSE', 'NRE'))
                                else:
                                    root_cause.append((prefix, direction, inp_mean_bias, out_mean_bias,
                                                       (out_mean_bias or 0) - (inp_mean_bias or 0),
                                                       inp_mean, out_mean,
                                                       'ROOT_CAUSE (MeanBias)', 'MeanBias'))
                        else:
                            # All data inputs clean → original rule applies
                            if out_trig == 'NRE':
                                root_cause.append((prefix, direction, inp_nre, out_nre,
                                                   out_nre - (inp_nre or 0), inp_mean, out_mean,
                                                   'ROOT_CAUSE', 'NRE'))
                            else:
                                root_cause.append((prefix, direction, inp_mean_bias, out_mean_bias,
                                                   (out_mean_bias or 0) - (inp_mean_bias or 0),
                                                   inp_mean, out_mean,
                                                   'ROOT_CAUSE (MeanBias)', 'MeanBias'))

                    # ===== ABSORBED =====
                    if inp_prob and not out_prob:
                        if inp_trig == 'NRE':
                            absorbed.append((prefix, direction, inp_nre, out_nre,
                                             inp_nre - (out_nre or 0), inp_mean, out_mean,
                                             'ABSORBED', 'NRE'))
                        else:
                            absorbed.append((prefix, direction, inp_mean_bias, out_mean_bias,
                                             (inp_mean_bias or 0) - (out_mean_bias or 0),
                                             inp_mean, out_mean,
                                             'ABSORBED (MeanBias)', 'MeanBias'))

                    # ===== PROPAGATION / PASS_THROUGH =====
                    if inp_prob and out_prob:
                        nre_propagation = False
                        nre_pass_through = False
                        if inp_trig == 'NRE' and out_trig == 'NRE':
                            if not _check_nre_relative(out_nre, inp_nre) and out_nre > inp_nre:
                                nre_propagation = True
                            elif not _check_nre_relative(out_nre, inp_nre) and out_nre < inp_nre:
                                nre_pass_through = True

                        if nre_propagation:
                            propagation.append((prefix, direction, inp_nre, out_nre,
                                                out_nre - inp_nre, inp_mean, out_mean,
                                                'PROPAGATION', 'NRE'))
                        if nre_pass_through:
                            pass_through.append((prefix, direction, inp_nre, out_nre,
                                                 out_nre - inp_nre, inp_mean, out_mean,
                                                 'PASS_THROUGH', 'NRE'))

                        if (inp_mean_bias is not None and out_mean_bias is not None
                                and not nre_propagation and not nre_pass_through
                                and inp_trig == 'MeanBias' and out_trig == 'MeanBias'):
                            mb_amplifies = (not _check_nre_relative(out_mean_bias, inp_mean_bias)
                                            and out_mean_bias > inp_mean_bias)
                            mb_shrinks = (not _check_nre_relative(out_mean_bias, inp_mean_bias)
                                          and out_mean_bias < inp_mean_bias)
                            if mb_amplifies:
                                propagation.append((prefix, direction, inp_mean_bias, out_mean_bias,
                                                    out_mean_bias - inp_mean_bias, inp_mean, out_mean,
                                                    'PROPAGATION (MeanBias)', 'MeanBias'))
                            elif mb_shrinks:
                                pass_through.append((prefix, direction, inp_mean_bias, out_mean_bias,
                                                     out_mean_bias - inp_mean_bias, inp_mean, out_mean,
                                                     'PASS_THROUGH (MeanBias)', 'MeanBias'))

    # Build appearance-order lookup
    _op_order = {}
    _op_row_range = {}
    for _prefix, _directions in op_prefixes.items():
        for _dir, _params in _directions.items():
            all_rows_list = [_params[_k].get('RowIndex', 0) for _k in _params]
            min_row = min(all_rows_list, default=0)
            max_row = max(all_rows_list, default=0)
            if min_row:
                _op_order[(_prefix, _dir)] = min_row
                _op_row_range[(_prefix, _dir)] = (min_row, max_row)

    # ==== 复合优先级后处理: 消除同一 (prefix, direction) 的双标签 ====
    # 规则: 若存在至少一条 ROOT_CAUSE 路径 → 移除该 prefix+dir 的 PASS_THROUGH 条目
    #      ROOT_CAUSE > PROPAGATION > PASS_THROUGH
    rc_keys = set((rc[0], rc[1]) for rc in root_cause)
    if rc_keys:
        pass_through = [pt for pt in pass_through if (pt[0], pt[1]) not in rc_keys]

    # ==== 参数误差节点标记: weight/bias 节点 = 持久化误差, 排查优先级高 ====
    # P0-A#2: Also mark param_grad_output entries (backward parameters_grad) for priority promotion.
    # Task 5.2: Add param_grad_output and no_downstream_consumer flags.
    # param-grad rc 为 10 元组（原 9 + inheritance），标志位插到 9-11，inheritance 落到 index 12
    for i in range(len(root_cause)):
        rc = root_cause[i]
        prefix = rc[0]
        is_param_grad = rc in root_cause_param_grad
        is_param = is_param_grad or '.weight' in prefix or '.bias' in prefix
        # Tuple: original 9 + (is_param, param_grad_output, no_downstream_consumer[, inheritance])
        if len(rc) == 10:
            root_cause[i] = rc[:9] + (is_param, is_param_grad, is_param_grad, rc[9])
        else:
            root_cause[i] = rc + (is_param, is_param_grad, is_param_grad)
    # Same for propagation and pass_through (for consistency, 9→11)
    for lst in [propagation, pass_through, absorbed, input_propagation]:
        for i in range(len(lst)):
            prefix = lst[i][0]
            is_param = '.weight' in prefix or '.bias' in prefix
            lst[i] = lst[i] + (is_param, False, False)

    root_cause.sort(key=lambda x: _op_order.get((x[0], x[1]), 999999))
    absorbed.sort(key=lambda x: _op_order.get((x[0], x[1]), 999999))
    propagation.sort(key=lambda x: _op_order.get((x[0], x[1]), 999999))
    input_propagation.sort(key=lambda x: _op_order.get((x[0], x[1]), 999999))
    pass_through.sort(key=lambda x: _op_order.get((x[0], x[1]), 999999))

    # === 算子分组中间结果 ===
    # P1-4: 收集多输入算子的脏输入路径信息
    output_idx = []  # [(nre, prefix, name, row_index), ...]
    for r in rows:
        name = r.get('NPU Name', '')
        if '.output.' in name or '.parameters_grad.' in name:
            other_nre = safe_float(r.get('NormRelativeErr', ''))
            other_prefix, _ = extract_op_prefix(name)
            if other_nre is not None and other_prefix:
                output_idx.append((other_nre, other_prefix, name, r.get('RowIndex', 0)))
    dirty_inputs_map = {}
    for prefix, directions in op_prefixes.items():
        for direction, params_dict in directions.items():
            dirty_inputs = []
            for key in sorted(params_dict.keys(), key=_nat_key):
                if get_param_type(key) != 'input':
                    continue
                p = params_dict[key]
                nre = safe_float(p.get('NormRelativeErr', ''))
                inp_row = p.get('RowIndex', 0)
                if nre is not None and nre >= threshold:
                    # Find upstream source for this input
                    upstream = None
                    trace_boundary_reason = None  # P2: reason when upstream is None
                    has_upstream_outputs = False
                    for _other_nre, _other_prefix, _other_name, _other_row in output_idx:
                        has_upstream_outputs = True
                        # forward 上游 = 更小行号, backward 上游 = 更大行号
                        if direction == 'backward':
                            if _other_row <= inp_row:
                                continue
                        else:
                            if _other_row >= inp_row:
                                continue
                        if (_other_nre is not None and _other_nre >= threshold
                                and _other_prefix and _other_prefix != prefix):
                            other_shape = r.get('NPU Tensor Shape', '').strip()
                            # NRE 一致性校验：直接生产者 output 与目标 input 的误差应基本一致
                            if not _nre_close(_other_nre, nre):
                                continue
                            upstream = _other_prefix
                            break
                    # P2: Determine trace_boundary_reason when upstream not found
                    if upstream is None:
                        if not has_upstream_outputs:
                            trace_boundary_reason = 'data_gap'
                        else:
                            trace_boundary_reason = 'no_match'
                    dirty_inputs.append({
                        "param_key": key,
                        "nre": round(nre, 4),
                        "upstream_source": upstream,
                        "trace_boundary_reason": trace_boundary_reason
                    })
            if dirty_inputs:
                dirty_inputs_map[(prefix, direction)] = dirty_inputs

    # Build structured op_groups
    op_groups_list = []
    for (prefix, direction), _ in sorted(_op_order.items(), key=lambda x: x[1]):
        rr = _op_row_range.get((prefix, direction), (0, 0))
        params = op_prefixes[prefix][direction]
        n_inputs = sum(1 for k in params if get_param_type(k) == 'input')
        n_outputs = sum(1 for k in params if get_param_type(k) == 'output')
        worst_in = max((safe_float(params[k].get('NormRelativeErr', '')) or 0) for k in params
                       if get_param_type(k) == 'input') if n_inputs > 0 else None
        worst_out = max((safe_float(params[k].get('NormRelativeErr', '')) or 0) for k in params
                        if get_param_type(k) == 'output') if n_outputs > 0 else None
        # P1-4: 确定子分类
        all_dirty_inputs = dirty_inputs_map.get((prefix, direction), [])
        all_inputs = [k for k in params if get_param_type(k) == 'input']
        if all_dirty_inputs and len(all_dirty_inputs) < len(all_inputs):
            input_subtype = 'INPUT_PARTIALLY_DIRTY'
        elif all_dirty_inputs:
            input_subtype = 'INPUT_ALL_DIRTY'
        else:
            input_subtype = 'INPUT_ALL_CLEAN'
        op_groups_list.append({
            "prefix": prefix,
            "direction": direction,
            "row_range": [rr[0], rr[1]],
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "worst_input_nre": worst_in,
            "worst_output_nre": worst_out,
            # P1-4: 多输入脏路径
            "dirty_inputs": all_dirty_inputs,
            "input_subtype": input_subtype
        })

    return root_cause, absorbed, propagation, input_propagation, pass_through, op_groups_list

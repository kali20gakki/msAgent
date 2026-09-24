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

#!/usr/bin/env python3
"""
Phase 3 — 激活过程差异追溯: 四组对照逐层对比。

四组: A(异常) B(异常标杆) C(邻近正常) D(邻近标杆)
异常度 = (A/B) / (C/D)

对照原则:
  - 跨设备（A/B, C/D）: 锚点必须两侧同一算子 key 且都有有效数值，
    NaN/None/缺 key/shape 不一致 → 标「未对齐」，不算比值、不参与判定
  - 同设备（A/C, B/D）: shape 不一致的组对不计算比值

用法:
  python phase3_trace_analyzer.py \
    --dump-a <异常dump目录> --dump-b <异常标杆dump目录> \
    [--dump-c <邻近正常dump目录> --dump-d <邻近标杆dump目录>] \
    [-o p3.json]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict


def load_dump_ops(dump_dir):
    dump_file = os.path.join(dump_dir, 'dump.json')
    if not os.path.exists(dump_file):
        return None
    with open(dump_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('data', {})


def extract_stats(entry, fields=('output',)):
    """提取算子条目的 Max/Min/Mean/Norm (取绝对值最大者)。"""
    s = {'Max': None, 'Min': None, 'Mean': None, 'Norm': None, 'shape': None}

    def _e(obj):
        if isinstance(obj, dict):
            if 'Max' in obj and isinstance(obj['Max'], (int, float)):
                for k in ('Max', 'Min', 'Mean', 'Norm'):
                    v = obj.get(k)
                    if isinstance(v, (int, float)) and v == v:  # not NaN
                        if s[k] is None or abs(v) > abs(s[k]):
                            s[k] = v
                if s['shape'] is None and obj.get('shape'):
                    s['shape'] = list(obj['shape'])
            else:
                for v in obj.values():
                    _e(v)
        elif isinstance(obj, list):
            for item in obj:
                _e(item)

    for f in fields:
        if f in entry:
            _e(entry[f])
    return {k: v for k, v in s.items() if v is not None}


def get_grad_norm(entry):
    """从 parameters_grad 条目提取梯度 Norm。"""
    s = extract_stats(entry)
    return s.get('Norm')


# 层前缀自动检测: 支持 layers./blocks./encoder. 等任意层命名
# detect_layer_prefix() 在加载 dump 后调用，存入全局
LAYER_PREFIX = 'layers.'


def detect_layer_prefix(ops):
    """
    从算子名自动学习层前缀。
    统计形如 `{prefix}{N}.` 的段，出现最多的前缀即为层标识。
    返回: 前缀字符串（如 'layers.', 'blocks.'），找不到返回 None
    """
    candidates = Counter()
    for key in ops:
        for m in re.finditer(r'([a-zA-Z_]+\.)(\d+)\.', key):
            prefix, num = m.group(1), m.group(2)
            # 数字段后必须跟 '.' 或行尾，且数字不是算子序号（如 forward.0）
            # 层号通常较大且后面跟模块名；启发式: 前缀后数字 >= 0 且该段多次出现
            candidates[(prefix, num)] += 1

    # 取出现次数最多的 (prefix, num) 组合的前缀
    if not candidates:
        return None
    # 按前缀聚合: 同一前缀不同层号的出现总次数
    prefix_counts = Counter()
    for (prefix, num), cnt in candidates.items():
        prefix_counts[prefix] += cnt
    best_prefix = prefix_counts.most_common(1)[0][0]
    return best_prefix


def set_layer_prefix(prefix):
    """设置层前缀（加载 dump 后调用）。"""
    global LAYER_PREFIX
    LAYER_PREFIX = prefix or 'layers.'


def get_layer_from_op(op_name):
    """从算子名提取 layer index，非 layer 算子返回 None。"""
    m = re.search(re.escape(LAYER_PREFIX) + r'(\d+)\.', op_name)
    return int(m.group(1)) if m else None


# 不参与锚点代表算子的模块（position/rotary 频率表等非激活算子）
NON_ANCHOR_SUBSTRINGS = ('rotary', 'inv_freq', 'position_id', 'position_ids')


def _op_stats(ops, key):
    """按 key 取算子统计 (含 shape); 缺 key 或无数值 (None/NaN) 返回 None。"""
    entry = ops.get(key)
    if entry is None:
        return None
    s = extract_stats(entry)
    return s or None


def _shape_compat(a, b):
    """shape 可比: 元素总数一致（允许设备间 flatten/多维表示差异，如 MoE grouped linear）。"""
    if a is None or b is None:
        return False
    if not a or not b:
        return False  # 空 shape（标量）元素总数不可比
    pa = pb = 1
    for d in a:
        pa *= d
    for d in b:
        pb *= d
    return pa == pb


def normalize_op_key(key):
    """跨设备算子名归一化: 去掉设备实现标记（TE 前缀、Flash/Fused 差异）和
    forward/backward 的 recompute 序号（.0/.1）。"""
    norm = re.sub(r'\.TE(?=[A-Z])', '.', key)
    norm = re.sub(r'(Flash|Fused)Attention', 'Attention', norm)
    norm = re.sub(r'\.(forward|backward)\.\d+', r'.\1', norm)
    return norm


def _op_path_depth(key, anchor_kw):
    """anchor 段之后的子路径段数（module 输出算子 = 1 段，如 self_attention.MLASelfAttention）。"""
    rest = key.split(anchor_kw, 1)[-1].strip('.')
    return len([s for s in rest.split('.') if s])


def find_anchor_op_pair(ops_a, ops_b, layer_idx, anchor_kw, direction):
    """
    锚点代表算子选择: 锚点是 module，每层选其**输出算子**（anchor 后子路径段数最少，
    如 self_attention.MLASelfAttention / mlp.MoELayer / input_layernorm.RMSNorm）——
    输出算子携带 module 计算结果，只比 Norm 最大算子会漏掉 attention 内部产生的分叉。
    输出算子不可对齐时 fallback 到 Norm 最大可对齐算子。
    每个算子要求: 两侧同一 key（或归一化 key）+ 有有效数值 + shape 元素总数兼容。
    返回: (key_a, key_b, stats_a, stats_b) 或 (None, None, None, None)。
    """
    if not ops_a or not ops_b:
        return None, None, None, None
    layer_marker = f'{LAYER_PREFIX}{layer_idx}.'
    cands = []
    for key, entry in ops_a.items():
        if layer_marker not in key or anchor_kw not in key:
            continue
        if any(t in key.lower() for t in NON_ANCHOR_SUBSTRINGS):
            continue
        if direction == 'forward' and ('.backward' in key or 'parameters_grad' in key):
            continue
        if direction == 'backward' and '.backward' not in key:
            continue
        s = extract_stats(entry)
        if not s or s.get('Norm') is None:
            continue
        cands.append((key, s))
    # 输出算子优先（深度 1），其次 Norm 最大
    cands.sort(key=lambda x: (_op_path_depth(x[0], anchor_kw), -x[1]['Norm']))

    # 1) 精确同 key; 2) 归一化 key（跨设备命名差异，如 TE 前缀/Flash vs Fused/序号）。
    norm_index = {normalize_op_key(k): (k, _op_stats(ops_b, k)) for k in ops_b}
    for key, s_a in cands:
        s_b = _op_stats(ops_b, key)
        if s_b is not None and _shape_compat(s_a.get('shape'), s_b.get('shape')):
            return key, key, s_a, s_b
        hit = norm_index.get(normalize_op_key(key))
        if (hit is not None and hit[1] is not None
                and _shape_compat(s_a.get('shape'), hit[1].get('shape'))):
            return key, hit[0], s_a, hit[1]
    return None, None, None, None


def find_anchor_grad(ops, layer_idx, anchor_kw):
    """找 parameters_grad 条目，返回梯度 Norm。"""
    layer_marker = f'{LAYER_PREFIX}{layer_idx}.'
    for key, entry in ops.items():
        if layer_marker not in key:
            continue
        if anchor_kw not in key or 'parameters_grad' not in key:
            continue
        n = get_grad_norm(entry)
        if n is not None:
            return key, n
    return None, None


def ratio(a, b):
    """a/b，不合法（None/0/NaN/inf）返回 None。"""
    if a is None or b is None or b == 0:
        return None
    try:
        r = a / b
    except (TypeError, ZeroDivisionError):
        return None
    if isinstance(r, float) and not math.isfinite(r):
        return None
    return r


def auto_discover_anchors(dump_a, dump_b):
    """
    自动发现锚点 module: 在 A/B 两侧都出现、跨层重复、有前向+反向的模块。
    优先选主干模块（input_layernorm / self_attention / mlp 等）。
    返回: [(anchor_name, anchor_kw)] 最多 3 个
    """
    def candidate_ops(ops):
        """提取 {prefix}{N}.{sub_module} 的第一段模块名作为候选"""
        cands = Counter()
        for key in ops:
            m = re.search(re.escape(LAYER_PREFIX) + r'\d+\.([a-zA-Z_]+)', key)
            if m:
                sub = m.group(1)
                cands[sub] += 1
        return cands

    cands_a = candidate_ops(dump_a)
    cands_b = candidate_ops(dump_b)

    # 两侧都出现，跨层出现（>= 2 次）
    common = set(cands_a) & set(cands_b)
    candidates = []
    for name in common:
        cnt = min(cands_a[name], cands_b[name])
        if cnt < 2:
            continue
        # 主干模块优先: input_layernorm / self_attention / mlp / pre_mlp_layernorm
        priority = {'input_layernorm': 0, 'self_attention': 1, 'mlp': 2,
                    'pre_mlp_layernorm': 3, 'post_attention_layernorm': 3}
        rank = priority.get(name, 9)
        candidates.append((rank, -cnt, name))

    candidates.sort()
    # 取前 3 个主干模块
    return [(name, name) for _, _, name in candidates[:3]]


def run_four_group(dump_a, dump_b, dump_c, dump_d, anchors=None):
    """执行四组对照逐层对比, 返回结构化结果。"""
    result = {'anchors': []}
    has_cd = bool(dump_c and dump_d)

    for anchor_name, anchor_kw in anchors:
        anchor_result = {'name': anchor_name, 'layers': []}

        # 确定最大 layer 数
        max_layer = 0
        for ops in (dump_a, dump_b, dump_c, dump_d):
            if not ops:
                continue
            for key in ops:
                l = get_layer_from_op(key)
                if l is not None:
                    max_layer = max(max_layer, l)

        for layer_idx in range(max_layer + 1):
            # A/B 与 C/D 分别做对齐，每层取 module 输出代表算子
            (fwd_key_a, fwd_key_b, fwd_ab_a, fwd_ab_b) = find_anchor_op_pair(
                dump_a, dump_b, layer_idx, anchor_kw, 'forward')
            (fwd_cd_key, _, fwd_cd_c, fwd_cd_d) = (
                find_anchor_op_pair(dump_c, dump_d, layer_idx, anchor_kw, 'forward')
                if has_cd else (None, None, None, None))
            (bwd_key_a, bwd_key_b, bwd_ab_a, bwd_ab_b) = find_anchor_op_pair(
                dump_a, dump_b, layer_idx, anchor_kw, 'backward')
            (bwd_cd_key, _, bwd_cd_c, bwd_cd_d) = (
                find_anchor_op_pair(dump_c, dump_d, layer_idx, anchor_kw, 'backward')
                if has_cd else (None, None, None, None))

            fwd = {'A': fwd_ab_a, 'B': fwd_ab_b, 'C': fwd_cd_c, 'D': fwd_cd_d}
            bwd = {'A': bwd_ab_a, 'B': bwd_ab_b, 'C': bwd_cd_c, 'D': bwd_cd_d}
            grads = {}
            for label, ops in (('A', dump_a), ('B', dump_b), ('C', dump_c), ('D', dump_d)):
                if not ops:
                    grads[label] = None
                    continue
                _, grads[label] = find_anchor_grad(ops, layer_idx, anchor_kw)

            # 对齐状态: 同 key 双侧有值且元素总数一致才算可比对，否则「未对齐」
            f_ab_ok = (fwd_key_a is not None
                       and _shape_compat(fwd_ab_a.get('shape'), fwd_ab_b.get('shape')))
            f_cd_ok = (fwd_cd_key is not None
                       and _shape_compat(fwd_cd_c.get('shape'), fwd_cd_d.get('shape')))
            b_ab_ok = (bwd_key_a is not None
                       and _shape_compat(bwd_ab_a.get('shape'), bwd_ab_b.get('shape')))
            b_cd_ok = (bwd_cd_key is not None
                       and _shape_compat(bwd_cd_c.get('shape'), bwd_cd_d.get('shape')))

            def metric_of(d, label, metric):
                return d[label][metric] if d[label] else None

            # 每个指标（Max/Min/Mean/Norm）独立计算 A/B, C/D，净异常度
            # 单指标找到的分叉点可能不是真起点，需在其他指标上交叉验证是否更早
            METRICS = ('Max', 'Min', 'Mean', 'Norm')
            layer_entry = {
                'layer': layer_idx,
                'grad': {lbl: grads[lbl] for lbl in 'ABCD'},
                'metrics': {},
                'alignment': {
                    'fwd_key_a': fwd_key_a,
                    'fwd_key_b': fwd_key_b,
                    'bwd_key_a': bwd_key_a,
                    'bwd_key_b': bwd_key_b,
                    'f_ab': f_ab_ok, 'f_cd': f_cd_ok,
                    'b_ab': b_ab_ok, 'b_cd': b_cd_ok,
                },
            }
            for m in METRICS:
                f_ab = ratio(metric_of(fwd, 'A', m), metric_of(fwd, 'B', m)) if f_ab_ok else None
                f_cd = ratio(metric_of(fwd, 'C', m), metric_of(fwd, 'D', m)) if f_cd_ok else None
                b_ab = ratio(metric_of(bwd, 'A', m), metric_of(bwd, 'B', m)) if b_ab_ok else None
                b_cd = ratio(metric_of(bwd, 'C', m), metric_of(bwd, 'D', m)) if b_cd_ok else None

                me = {
                    'fwd': {lbl: metric_of(fwd, lbl, m) for lbl in 'ABCD'},
                    'bwd': {lbl: metric_of(bwd, lbl, m) for lbl in 'ABCD'},
                    'f_ab': f_ab, 'f_cd': f_cd,
                    'b_ab': b_ab, 'b_cd': b_cd,
                }
                # 净异常度
                if f_ab is not None and f_cd not in (None, 0):
                    me['f_anomaly'] = f_ab / f_cd
                else:
                    me['f_anomaly'] = None
                if b_ab is not None and b_cd not in (None, 0):
                    me['b_anomaly'] = b_ab / b_cd
                else:
                    me['b_anomaly'] = None
                layer_entry['metrics'][m] = me

            anchor_result['layers'].append(layer_entry)

        # 计算该 anchor 的结构化结论（供报告渲染）
        anchor_result['conclusion'] = _compute_anchor_conclusion(anchor_result)

        result['anchors'].append(anchor_result)

    return result


def _compute_anchor_conclusion(anchor_result):
    """计算 anchor 的结构化结论: 异常层、分叉段起点、各指标起点、最早分叉点。"""
    layers = anchor_result['layers']
    thresholds = compute_anomaly_thresholds({'layers': layers})

    # 每指标异常层
    metric_layers = {}
    for m in ('Max', 'Min', 'Mean', 'Norm'):
        f_thresh, b_thresh = thresholds[m]
        fwd_layers = []
        bwd_layers = []
        for le in layers:
            me = le['metrics'][m]
            if (f_thresh is not None and me.get('f_anomaly') is not None
                    and me['f_anomaly'] > f_thresh):
                fwd_layers.append(le['layer'])
            if (b_thresh is not None and me.get('b_anomaly') is not None
                    and me['b_anomaly'] > b_thresh):
                bwd_layers.append(le['layer'])
        metric_layers[m] = {'fwd': fwd_layers, 'bwd': bwd_layers}

    # 各指标主分叉段起点
    metric_start = {}
    for m in ('Max', 'Min', 'Mean', 'Norm'):
        fwd_m = metric_layers[m]['fwd']
        bwd_m = metric_layers[m]['bwd']
        fwd_seg = max(_split_segments(fwd_m), key=len) if fwd_m else None
        bwd_seg = max(_split_segments(bwd_m), key=len) if bwd_m else None
        # 前向起点取段首层（浅→深第一个异常），反向起点取段末层（深→浅第一个异常）
        fwd_start = fwd_seg[0] if fwd_seg else None
        bwd_start = max(bwd_seg) if bwd_seg else None
        metric_start[m] = {'fwd_start': fwd_start, 'bwd_start': bwd_start}

    # 最早分叉点（跨指标）
    all_fwd = [m['fwd_start'] for m in metric_start.values() if m['fwd_start'] is not None]
    all_bwd = [m['bwd_start'] for m in metric_start.values() if m['bwd_start'] is not None]
    earliest_fwd = min(all_fwd) if all_fwd else None
    earliest_bwd = max(all_bwd) if all_bwd else None

    # 未对齐层: 该方向有数据但 f_anomaly 无法计算（NA: None/NaN/缺 key/shape 不一致）。
    # 这些层≠正常，是「未比对」——异常层列表为空不代表该方向一致。
    uncompared = {'fwd': [], 'bwd': []}
    for le in layers:
        for direc, anom_key in (('fwd', 'f_anomaly'), ('bwd', 'b_anomaly')):
            vals = le['metrics']['Norm'][direc]
            if (any(v is not None for v in vals.values())
                    and le['metrics']['Norm'].get(anom_key) is None):
                uncompared[direc].append(le['layer'])

    return {
        'metric_layers': metric_layers,
        'metric_starts': metric_start,
        'earliest_fwd_divergence': earliest_fwd,
        'earliest_bwd_divergence': earliest_bwd,
        'uncompared_layers': uncompared,
    }


def _split_segments(layers):
    """把异常层列表拆成连续段（相邻层间隔 <= 1）。"""
    if not layers:
        return []
    segments = []
    seg = [layers[0]]
    for i in range(1, len(layers)):
        if layers[i] - layers[i - 1] <= 1:
            seg.append(layers[i])
        else:
            segments.append(seg)
            seg = [layers[i]]
    segments.append(seg)
    return segments


def compute_anomaly_thresholds(anchor):
    """
    从对照组 C/D 的分布学出异常判定阈值，每个指标独立。
    对照组 C/D 反映「正常对比」的噪声水平；A/B 偏离 C/D 分布超过 3*IQR（或 epsilon 下限）才算异常。
    不使用硬编码阈值。
    返回: {'Norm': (f_thresh, b_thresh), 'Max': (...), 'Min': (...), 'Mean': (...)}
    """
    METRICS = ('Max', 'Min', 'Mean', 'Norm')
    thresholds = {}

    def _threshold(cds):
        if not cds:
            return None
        s = sorted(cds)
        n = len(s)
        q1, q3 = s[n // 4], s[(3 * n) // 4]
        iqr = q3 - q1
        med = s[n // 2]
        # 对照组自身噪声 3*IQR + epsilon 下限 0.03（统计精度）
        return med + max(3 * iqr, 0.03)

    for m in METRICS:
        f_cds = [le['metrics'][m]['f_cd'] for le in anchor['layers']
                 if le['metrics'][m].get('f_cd') is not None]
        b_cds = [le['metrics'][m]['b_cd'] for le in anchor['layers']
                 if le['metrics'][m].get('b_cd') is not None]
        thresholds[m] = (_threshold(f_cds), _threshold(b_cds))

    return thresholds


def main():
    parser = argparse.ArgumentParser(description='Phase 3 — 四组对照逐层追溯')
    parser.add_argument('--dump-a', required=True, help='异常侧 dump 目录')
    parser.add_argument('--dump-b', required=True, help='异常标杆 dump 目录')
    parser.add_argument('--dump-c', help='邻近正常 dump 目录 (可选, 缺省两组对比)')
    parser.add_argument('--dump-d', help='邻近标杆 dump 目录 (可选)')
    parser.add_argument('--anchors', help='手动指定锚点 module，逗号分隔（如 input_layernorm,self_attention）')
    parser.add_argument('--output', '-o', help='输出 JSON')
    args = parser.parse_args()

    dump_a = load_dump_ops(args.dump_a)
    dump_b = load_dump_ops(args.dump_b)
    if not dump_a or not dump_b:
        print("Error: dump_a/dump_b 加载失败", file=sys.stderr)
        sys.exit(1)

    dump_c = load_dump_ops(args.dump_c) if args.dump_c else None
    dump_d = load_dump_ops(args.dump_d) if args.dump_d else None

    # 自动检测层前缀（layers./blocks./encoder. 等）
    detected = detect_layer_prefix(dump_a)
    set_layer_prefix(detected)

    # 锚点: 自动发现 或 用户 --anchors 指定
    if args.anchors:
        anchors = [(a.strip(), a.strip()) for a in args.anchors.split(',') if a.strip()]
    else:
        anchors = auto_discover_anchors(dump_a, dump_b)

    result = run_four_group(dump_a, dump_b, dump_c, dump_d, anchors=anchors)
    result['phase'] = 3
    result['layer_prefix'] = LAYER_PREFIX
    result['anchors_used'] = [a[0] for a in anchors]
    result['dirs'] = {'A': args.dump_a, 'B': args.dump_b,
                      'C': args.dump_c, 'D': args.dump_d}

    # 锚点 <3 时警告: 单锚点/双锚点判定可能不完整
    if len(anchors) < 3:
        msg = (f'仅发现 {len(anchors)} 个可对齐锚点 (<3), '
               f'分叉判定可能不完整, 建议补充或检查 dump 算子命名')
        print(f'WARNING: {msg}', file=sys.stderr)
        result['anchor_warning'] = {'anchors_found': len(anchors), 'message': msg}

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()

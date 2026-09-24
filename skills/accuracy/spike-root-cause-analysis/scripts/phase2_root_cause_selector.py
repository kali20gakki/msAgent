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
Phase 2 — 根因坐标选定: 从 Phase 1 候选坐标中选出最可能是根因的一个。

逻辑:
  1. 确定每 optimizer_step 的关注 target（final_norm 最大的）
  2. 跨 step 全 rank 对比（相邻 opt_step 同 target 梯度变化 ≥2x → 根因在前一步）
  3. 坐标选定: 无标杆取 max delta/norm; 有标杆从最异常开始逐个对比

自动适配: micro_step 累积数据 / step 级数据 / dump 数据

用法:
  python phase2_root_cause_selector.py --phase1 p1.json [-o result.json]
  python phase2_root_cause_selector.py --npu <p1.json> --gpu <p1.json> [-o result.json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import median


def load_json(path):
    if not path or not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_top_target_per_step(anomalies):
    """
    对每个 optimizer step，确定关注 target。
    取 Phase 1 anomalies 中最终 norm 最大的 target:
      - micro_step 数据: suspect_final_norm
      - step 级 / dump 数据: norm 本身
    """
    step_targets = defaultdict(lambda: defaultdict(float))

    for a in anomalies:
        os_step = a.get('optimizer_step', 0)
        tname = a.get('target_name', '')
        fn = a.get('suspect_final_norm') if 'suspect_final_norm' in a else a.get('norm', 0)
        if fn > step_targets[os_step][tname]:
            step_targets[os_step][tname] = fn

    result = {}
    for os_step in sorted(step_targets.keys()):
        top_target = max(step_targets[os_step], key=step_targets[os_step].get)
        result[os_step] = {
            'target_name': top_target,
            'max_final_norm': step_targets[os_step][top_target]
        }
    return result


def is_dump_data(anomalies):
    """判断 Phase 1 数据来源是否为 dump_statistic（无 delta 字段）"""
    if not anomalies:
        return False
    # dump 数据的 trigger 为 dump_abs_norm
    if all(a.get('trigger') in ('dump_abs_norm',) for a in anomalies[:3]):
        return True
    return False


def is_micro_step_data(anomalies):
    """判断是否为 micro_step 累积数据（有 delta 字段）"""
    if not anomalies:
        return False
    return any('delta' in a for a in anomalies[:3])


def cross_step_root_cause(p1_data, top_targets):
    """
    比较相邻 optimizer step 的梯度变化，判定根因 opt_step。

    使用 Phase 1 输出的 target_rank_norms (全 rank 最终 norm) 进行 pairwise 比较。
    仅 micro_step 累积数据有 target_rank_norms; step 级/dump 数据无此字段, 跳过。
    """
    rank_norms = p1_data.get('target_rank_norms')
    if not rank_norms:
        return None, "该数据类型无跨 step 判定依据 (step 级/dump)，跳过"
    sorted_steps = sorted(int(k) for k in rank_norms.keys())
    reasoning_parts = []

    root_opt_step = None

    # 自适应阈值: 从所有相邻 step 对的 rank 增长分布学出「显著增长」门槛
    # 收集所有（rank 增长比），正常训练增长集中在低位，尖刺传导导致的高增长是尾部
    all_growth_ratios = []
    for i in range(1, len(sorted_steps)):
        prev_data = rank_norms.get(str(sorted_steps[i - 1]), {})
        curr_data = rank_norms.get(str(sorted_steps[i]), {})
        prev_norms = prev_data.get('ranks', {})
        curr_norms = curr_data.get('ranks', {})
        for rank in set(prev_norms.keys()) & set(curr_norms.keys()):
            pn = prev_norms.get(str(rank), prev_norms.get(rank, 0))
            cn = curr_norms.get(str(rank), curr_norms.get(rank, 0))
            if pn > 0 and cn > 0:
                all_growth_ratios.append(cn / pn)

    # 显著增长阈值: 从相邻 step 增长分布学。
    # 需要至少 4 个相邻 step 对 (5 个 opt_step) 分布才可靠;
    # 样本不足时分布可能全被尖刺传导撑高, 退回默认 2.0x。
    n_step_pairs = len(sorted_steps) - 1
    if n_step_pairs >= 4:
        growth_med = median(all_growth_ratios)
        growth_threshold = max(growth_med * 2.0, 1.5)
    else:
        growth_threshold = 2.0

    # rank 占比阈值: 过半即「大多数」（0.5），与 2x 门槛配合
    ratio_threshold = 0.5

    reasoning_parts.append(f"自适应增长阈值: {growth_threshold:.2f}x ({n_step_pairs} 个相邻 step 对"
                           f"{', 分布中位数×2' if n_step_pairs >= 4 else ', 样本不足用默认'})")

    for i in range(1, len(sorted_steps)):
        prev_step = sorted_steps[i - 1]
        curr_step = sorted_steps[i]

        prev_data = rank_norms.get(str(prev_step), {})
        curr_data = rank_norms.get(str(curr_step), {})

        prev_norms = prev_data.get('ranks', {})
        curr_norms = curr_data.get('ranks', {})

        # 找重叠 rank
        common_ranks = (set(int(k) for k in prev_norms.keys())
                        & set(int(k) for k in curr_norms.keys()))
        if len(common_ranks) < 3:
            reasoning_parts.append(
                f"opt_step {prev_step} → {curr_step}: 重叠 rank 不足 ({len(common_ranks)})"
            )
            continue

        higher_count = 0
        total = len(common_ranks)
        for rank in common_ranks:
            pn = prev_norms.get(str(rank), prev_norms.get(rank, 0))
            cn = curr_norms.get(str(rank), curr_norms.get(rank, 0))
            if pn > 0 and cn >= growth_threshold * pn:
                higher_count += 1

        ratio = higher_count / total if total > 0 else 0
        prev_tname = prev_data.get('target_name', '').split('.')[-1]
        reasoning_parts.append(
            f"opt_step {prev_step} → {curr_step} ({prev_tname}): "
            f"{higher_count}/{total} ranks ({ratio:.0%}) 的最终 norm >= {growth_threshold:.2f}x 增长"
        )

        if ratio >= ratio_threshold:
            root_opt_step = prev_step
            reasoning_parts.append(
                f"  → 大多数 rank 梯度显著增长，根因在 opt_step {prev_step}"
            )
            break

    if root_opt_step is None:
        anomalies = p1_data.get('anomalies', [])
        max_dev = 0
        for a in anomalies:
            if a.get('deviation_ratio', 0) > max_dev:
                max_dev = a['deviation_ratio']
                root_opt_step = a.get('optimizer_step', 0)
        reasoning_parts.append(
            f"跨 step 梯度无明显跳跃增长，取异常偏离最大的 opt_step {root_opt_step}"
        )

    return root_opt_step, '; '.join(reasoning_parts)


def pick_root_coordinate(anomalies, root_opt_step, gpu_anomalies=None):
    """
    选定根因坐标。自动适配 dump / micro_step 数据类型。
    dump: 按 norm 排序, 按 target_name 匹配
    micro_step: 按 delta 排序, 按 (target_name, micro_step) 匹配
    """
    if not anomalies:
        return None, "无候选坐标"

    is_dump = is_dump_data(anomalies)
    is_ms = is_micro_step_data(anomalies)
    candidates = anomalies if (is_dump or root_opt_step is None) else (
        [a for a in anomalies if a.get('optimizer_step') == root_opt_step])

    def sort_key(a):
        return a.get('norm', 0) if (is_dump or not is_ms) else a.get('delta', 0)

    candidates = sorted(candidates, key=sort_key, reverse=True)

    if not gpu_anomalies:
        best = candidates[0]
        if is_dump:
            return build_coord(best), f"无标杆，取最大 norm (rank={best['rank']})"
        elif is_ms:
            return build_coord(best), f"无标杆，取最大 delta (rank={best['rank']} ms={best['micro_step']})"
        else:
            return build_coord(best), f"无标杆，取最大 norm (rank={best['rank']} step={best['step']})"

    # 有标杆: 按 target_name 匹配（dump / step 级，用 norm 对比）或（target_name, micro_step）匹配
    if is_dump or not is_ms:
        def match_key(a):
            return (a['target_name'],)
    else:
        def match_key(a):
            return (a['target_name'], a.get('micro_step', 0))
    val_key = 'norm' if (is_dump or not is_ms) else 'delta'

    gpu_index = {}
    for a in gpu_anomalies:
        key = match_key(a)
        v = a.get(val_key, 0)
        if key not in gpu_index or v > gpu_index[key].get(val_key, 0):
            gpu_index[key] = a

    # 自适应设备特异性阈值: 从异常侧/标杆侧同坐标的比值分布学出
    # 正常对比的比值集中在低位, 设备特异性差异在尾部
    all_ratios = []
    for a in candidates:
        npu_val = a.get(val_key, 0)
        key = match_key(a)
        gpu_match = gpu_index.get(key)
        if gpu_match:
            gpu_val = gpu_match.get(val_key, 0)
            if gpu_val > 0:
                all_ratios.append(npu_val / gpu_val)

    if all_ratios:
        device_threshold = max(median(all_ratios) * 2.0, 1.5)
    else:
        device_threshold = 2.0

    for a in candidates:
        npu_val = a.get(val_key, 0)
        key = match_key(a)
        gpu_match = gpu_index.get(key)

        if gpu_match:
            gpu_val = gpu_match.get(val_key, 0)
            ratio = npu_val / gpu_val if gpu_val > 0 else float('inf')
            if ratio >= device_threshold:
                coord = build_coord_dump(a) if is_dump else build_coord(a)
                return coord, (
                    f"NPU {val_key}={npu_val:.4g} (rank={a['rank']}), "
                    f"GPU {val_key}={gpu_val:.4g} (rank={gpu_match['rank']}), "
                    f"NPU/GPU={ratio:.1f}x ≥ {device_threshold:.1f}x → NPU 设备特异性异常")
            continue
        coord = build_coord_dump(a) if is_dump else build_coord(a)
        return coord, (
            f"NPU {val_key}={npu_val:.4g} (rank={a['rank']}), GPU 同参数无异常 → NPU 独有")

    best = candidates[0]
    if is_dump:
        return build_coord_dump(best), (
            f"所有异常参数 GPU 侧接近。最大 norm: rank={best['rank']} "
            f"(NPU {best['norm']:.4g})")
    return build_coord(best), (
        f"所有坐标 GPU 侧接近。最大 delta: rank={best['rank']} "
        f"(NPU {best['delta']:.4g})")


def build_coord(a):
    result = {
        'rank': a['rank'],
        'target_name': a['target_name'],
        'norm': a['norm'],
        'deviation_ratio': a['deviation_ratio'],
    }
    if 'optimizer_step' in a:
        result['optimizer_step'] = a['optimizer_step']
    if 'micro_step' in a:
        result['micro_step'] = a['micro_step']
    if 'delta' in a:
        result['delta'] = a['delta']
    if 'suspect_final_norm' in a:
        result['suspect_final_norm'] = a['suspect_final_norm']
    if 'step' in a:
        result['step'] = a['step']
    return result


def build_coord_dump(a):
    return {
        'rank': a['rank'],
        'target_name': a['target_name'],
        'norm': a['norm'],
        'deviation_ratio': a['deviation_ratio']
    }


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 — 根因坐标选定')
    parser.add_argument('--phase1', help='Phase 1 结果 JSON (单设备)')
    parser.add_argument('--npu', help='NPU Phase 1 JSON (跨设备)')
    parser.add_argument('--gpu', help='GPU Phase 1 JSON (跨设备)')
    parser.add_argument('--output', '-o', help='输出 JSON 文件路径')
    args = parser.parse_args()

    p1 = None
    p1_gpu = None
    anomalies = []

    if args.npu and args.gpu:
        p1 = load_json(args.npu)
        p1_gpu = load_json(args.gpu)
        if not p1 or not p1_gpu:
            sys.exit(1)
        anomalies = p1.get('anomalies', [])
        tag = "NPU (+GPU 标杆)"
    elif args.phase1:
        p1 = load_json(args.phase1)
        if not p1:
            sys.exit(1)
        anomalies = p1.get('anomalies', [])
        tag = "单设备"
    else:
        parser.print_help()
        sys.exit(1)

    if not anomalies:
        print("No anomalies found in Phase 1 output")
        sys.exit(0)

    is_dump = is_dump_data(anomalies)

    if not is_dump:
        # Step 2.1: 每 step 关注 target
        top_targets = get_top_target_per_step(anomalies)
        print("=== 每 step 关注 target ===")
        for os_step in sorted(top_targets.keys()):
            t = top_targets[os_step]
            print(f"  opt_step {os_step}: "
                  f"{t['target_name'].split('.')[-1]} (max final={t['max_final_norm']:.4g})")

        # Step 2.2: 跨 step 根因判定
        root_opt_step, reasoning = cross_step_root_cause(p1, top_targets)
        print(f"\n=== 跨 step 判定 ===")
        print(f"  {reasoning}")
        print(f"  根因 opt_step: {root_opt_step}")
    else:
        root_opt_step = None
        reasoning = "dump 数据（单点快照），跳过跨 step 判定"
        print(f"=== dump 数据 ===\n  {reasoning}")

    # Step 2.3: 选定坐标
    gpu_anomalies = p1_gpu.get('anomalies', []) if p1_gpu else None
    coord, pick_reason = pick_root_coordinate(anomalies, root_opt_step, gpu_anomalies)
    print(f"\n=== 根因坐标 ===")
    if coord:
        if is_dump_data(anomalies):
            print(f"  rank={coord['rank']} norm={coord['norm']:.4g} "
                  f"target={coord['target_name'].split('.')[-1]}")
        elif is_micro_step_data(anomalies):
            print(f"  opt_step={coord.get('optimizer_step','?')} rank={coord['rank']} "
                  f"ms={coord.get('micro_step','?')} delta={coord.get('delta',0):.4g} "
                  f"target={coord['target_name'].split('.')[-1]}")
        else:
            print(f"  rank={coord['rank']} step={coord.get('step','?')} norm={coord['norm']:.4g} "
                  f"target={coord['target_name'].split('.')[-1]}")
    print(f"  选定理由: {pick_reason}")

    # 跨设备结论
    cross_note = pick_reason if p1_gpu else "无标杆数据"

    # 输出
    result = {
        'phase': 2,
        'mode': tag,
        'root_opt_step': root_opt_step,
        'reasoning': reasoning,
        'root_cause_coordinate': coord,
        'cross_device_verification': cross_note
    }
    if not is_dump:
        result['top_targets_per_step'] = {str(k): v for k, v in top_targets.items()}

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nOutput written to {args.output}")
    else:
        print(f"\n{json.dumps(result, indent=2, ensure_ascii=False, default=str)}")


if __name__ == '__main__':
    main()

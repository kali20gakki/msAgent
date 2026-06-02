#!/usr/bin/env python3
"""分析msprobe mix级别db文件，寻找首个输入一致输出不一致的API。"""

import argparse
import sqlite3
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from msprobe_utils import build_excluded_rules, is_key_excluded, print_kv_table, is_api_excluded


def format_hierarchy(hierarchy):
    """格式化层级路径，返回格式化的行列表。"""
    lines = []
    dl = 0
    for node in hierarchy:
        nt = node.get('node_type', '')
        if nt not in ('0', '1'):
            continue
        tl = {'0': 'Module', '1': 'API'}.get(nt, '?')
        name = node.get('node_name', '?')
        prefix = '→ ' if dl > 0 else ''
        lines.append(f"{'  ' * dl}{prefix}{name} [{tl}]")
        dl += 1
    return lines


def extract_md5_values(data_json, node_name=None, excluded_rules=None):
    md5_map = {}
    if not data_json:
        return md5_map
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        return md5_map
    for key, value in data.items():
        if not (isinstance(value, dict) and 'md5' in value):
            continue
        if node_name and excluded_rules:
            excluded = False
            for io_type in ('input', 'output'):
                marker = f'.{io_type}.'
                if marker in key:
                    idx = key.rsplit('.', 1)[1]
                    if is_key_excluded(node_name, io_type, idx, excluded_rules):
                        excluded = True
                    break
            if excluded:
                continue
        md5_map[key] = value['md5']
    return md5_map


def compare_md5(npu_md5_map, bench_md5_map):
    """比较NPU和Bench的md5，返回 (是否一致, 差异列表)"""
    all_keys = sorted(set(list(npu_md5_map.keys()) + list(bench_md5_map.keys())))
    is_match = True
    diffs = []
    for key in all_keys:
        npu_m = npu_md5_map.get(key, "<MISSING>")
        bench_m = bench_md5_map.get(key, "<MISSING>")
        if npu_m != bench_m:
            is_match = False
            diffs.append(f"{key}: NPU={npu_m} vs Bench={bench_m}")
    return is_match, diffs, all_keys


def build_hierarchy(cursor, node_id, data_source, step, rank):
    """通过追踪up_node构建从root到指定节点的层级路径。返回 (路径列表, 子节点列表)"""
    path = []
    current_id = node_id
    max_depth = 50
    for _ in range(max_depth):
        cursor.execute(
            "SELECT id, node_name, node_type, up_node, sub_nodes FROM tb_nodes WHERE id=?",
            (current_id,)
        )
        row = cursor.fetchone()
        if not row:
            break
        nid, nname, ntype, up_node, sub_nodes = row
        children = []
        if sub_nodes:
            try:
                children = json.loads(sub_nodes)
            except (json.JSONDecodeError, TypeError):
                pass
        path.append({'node_id': nid, 'node_name': nname, 'node_type': ntype, 'sub_nodes': children})
        if not up_node or up_node == 'None':
            break
        cursor.execute(
            "SELECT id FROM tb_nodes WHERE data_source=? AND step=? AND rank=? AND node_name=?",
            (data_source, step, rank, up_node)
        )
        parent_row = cursor.fetchone()
        if not parent_row:
            break
        current_id = parent_row[0]
    else:
        path.append({'node_name': f'... (超过{max_depth}层)', 'node_type': '?'})
    path.reverse()
    return path


def analyze_rank(db_path, step, rank, excluded_apis=None):
    """分析单个rank，返回该rank的分析结果"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    excluded_rules = build_excluded_rules()

    # ── API级分析 ──
    cursor.execute(
        "SELECT id, node_name, node_order, node_type, input_data, output_data"
        " FROM tb_nodes WHERE data_source='NPU' AND precision_index=1"
        " AND step=? AND rank=? ORDER BY node_order",
        (step, rank)
    )
    npu_error_nodes = cursor.fetchall()
    api_result = None
    if npu_error_nodes:
        for npu_id, node_name, node_order, node_type, npu_input, npu_output in npu_error_nodes:
            if is_api_excluded(node_name, excluded_apis):
                continue
            bench_id = npu_id.replace("NPU_", "Bench_", 1)
            cursor.execute("SELECT input_data, output_data FROM tb_nodes WHERE id=?", (bench_id,))
            bench_row = cursor.fetchone()
            if not bench_row:
                continue
            bench_input, bench_output = bench_row
            input_match, _, all_input_keys = compare_md5(
                extract_md5_values(npu_input, node_name, excluded_rules),
                extract_md5_values(bench_input, node_name, excluded_rules)
            )
            output_match, output_diffs, _ = compare_md5(
                extract_md5_values(npu_output, node_name, excluded_rules),
                extract_md5_values(bench_output, node_name, excluded_rules)
            )
            if not input_match:
                continue
            if not output_match:
                hierarchy = build_hierarchy(cursor, npu_id, 'NPU', step, rank)
                type_label = 'API' if node_type == '1' else ('Module' if node_type == '0' else f'Type_{node_type}')
                api_result = dict(node_name=node_name, node_type=type_label, node_order=node_order,
                                  input_keys=all_input_keys, output_diffs=output_diffs, hierarchy=hierarchy)
                break

    # ── Module级分析 ──
    if api_result is not None:
        conn.close()
        return dict(step=step, rank=rank, found=True, **api_result)

    cursor.execute(
        "SELECT id, node_name, node_order, node_type, input_data, output_data"
        " FROM tb_nodes WHERE data_source='NPU' AND node_type='0'"
        " AND precision_index=1 AND step=? AND rank=? ORDER BY node_order",
        (step, rank)
    )
    npu_module_nodes = cursor.fetchall()
    module_result = None
    for npu_id, node_name, node_order, node_type, npu_input, npu_output in npu_module_nodes:
        if excluded_apis and node_name in excluded_apis:
            continue
        bench_id = npu_id.replace("NPU_", "Bench_", 1)
        cursor.execute("SELECT input_data, output_data FROM tb_nodes WHERE id=?", (bench_id,))
        bench_row = cursor.fetchone()
        if not bench_row:
            continue
        bench_input, bench_output = bench_row
        if not extract_md5_values(npu_input, node_name, excluded_rules) and not extract_md5_values(npu_output, node_name, excluded_rules):
            continue
        input_match, _, _ = compare_md5(
            extract_md5_values(npu_input, node_name, excluded_rules),
            extract_md5_values(bench_input, node_name, excluded_rules)
        )
        output_match, output_diffs, _ = compare_md5(
            extract_md5_values(npu_output, node_name, excluded_rules),
            extract_md5_values(bench_output, node_name, excluded_rules)
        )
        if not input_match:
            continue
        if not output_match:
            module_result = dict(module_name=node_name, node_order=node_order,
                                 output_diffs=output_diffs, hierarchy=build_hierarchy(cursor, npu_id, 'NPU', step, rank))
            break

    conn.close()
    result = dict(step=step, rank=rank, found=False, reason='未找到输入一致输出不一致的API')
    if module_result:
        result['module_analysis'] = module_result
    return result


def analyze_db(db_path, excluded_apis=None):
    if not os.path.exists(db_path):
        print(f"错误: 文件不存在: {db_path}")
        return

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT step, rank FROM tb_nodes ORDER BY step, rank")
    step_ranks = cursor.fetchall()
    cursor.execute("SELECT DISTINCT step FROM tb_nodes ORDER BY step")
    steps = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not step_ranks:
        print("db文件中没有数据")
        return

    print(f"共发现 {len(step_ranks)} 个 (step, rank) 组合: {step_ranks}")
    print(f"共 {len(steps)} 个step: {steps}")
    print("=" * 80)

    for step in steps:
        ranks_in_step = [sr[1] for sr in step_ranks if sr[0] == step]
        print(f"\n分析 step={step}, ranks={ranks_in_step} (并行分析):")
        print("-" * 60)
        with ThreadPoolExecutor(max_workers=min(len(ranks_in_step), 32)) as executor:
            future_to_rank = {executor.submit(analyze_rank, db_path, step, rank, excluded_apis): rank for rank in ranks_in_step}
            step_results = []
            for future in as_completed(future_to_rank):
                rank = future_to_rank[future]
                try:
                    step_results.append(future.result())
                except Exception as e:
                    step_results.append({'step': step, 'rank': rank, 'found': False, 'reason': f"分析异常: {e}"})

        step_results.sort(key=lambda r: r['rank'])

        # ── 每个rank一张双列表 ──
        for r in step_results:
            kv_rows = []
            if r['found']:
                hier = '\n'.join(format_hierarchy(r.get('hierarchy', []))) or '-'
                items = ['Input MD5 (全部一致):'] + [f'  {k}' for k in r.get('input_keys', [])]
                items += ['Output MD5 (不一致):'] + [f'  {d}' for d in r.get('output_diffs', [])]
                kv_rows = [('首个问题API', r['node_name']), ('API所在Module层级', hier),
                           ('API分析依据', '\n'.join(items)),
                           ('首个问题Module', '已找到首个问题API，不分析'), ('Module分析依据', '-')]
            else:
                kv_rows.extend([('首个问题API', '无'), ('API所在Module层级', '-'),
                                ('API分析依据', r.get('reason', '未找到首个输入一致输出不一致的API'))])
                ma = r.get('module_analysis')
                if ma:
                    kv_rows.append(('首个问题Module', ma['module_name']))
                    items = ['Module Output MD5 (不一致):'] + [f'  {d}' for d in ma.get('output_diffs', [])]
                    kv_rows.append(('Module分析依据', '\n'.join(items)))
                else:
                    kv_rows.extend([('首个问题Module', '无'),
                                    ('Module分析依据', '未找到首个输入一致输出不一致的Module')])
            print_kv_table(r['rank'], kv_rows)
            if not r['found'] and r.get('module_analysis'):
                ma = r['module_analysis']
                print("  可能原因: 该Module内可能有被msprobe漏采的API，导致Module整体输出不一致。")
                if ma.get('hierarchy'):
                    print("  请检查以下层级的子节点列表，确认是否有API未被采集:")
                    for line in format_hierarchy(ma['hierarchy']):
                        print(f"    {line}")

        if any(r['found'] for r in step_results):
            if not excluded_apis:
                print(f"\n提示: 如果不认为上述API是问题根因，可输入API名称排除后重新分析。支持前缀匹配，多个API以空格分隔。")
            if step < max(steps):
                print(f"\nstep={step} 已找到结果，跳过后续step。")
            break


def main():
    parser = argparse.ArgumentParser(description='分析msprobe mix级别db文件，寻找首个输入一致输出不一致的API。')
    parser.add_argument('db_path', help='mix级别比对结果db文件（.vis.db）')
    parser.add_argument('--exclude-api', nargs='+', default=[], metavar='NAME',
                        help='要排除的API名称（支持前缀匹配，多个以空格分隔）')
    args = parser.parse_args()

    excluded = set(args.exclude_api)
    if excluded:
        print(f"排除以下API前缀: {sorted(excluded)}")
    analyze_db(args.db_path, excluded)


if __name__ == "__main__":
    main()

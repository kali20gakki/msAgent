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

"""共享数据加载: trend.db / monitor CSV / dump_statistic 三种格式的梯度监控数据加载。"""

import csv
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict


def load_trend_db(db_path):
    """加载 trend.db，返回结构化数据。"""
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found", file=sys.stderr)
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        required = {'trend_data', 'monitoring_targets', 'monitoring_metrics',
                    'monitoring_tags', 'tag_target_mapping', 'global_stats'}
        if not required.issubset(set(tables)):
            print(f"Error: {db_path} missing tables: {required - set(tables)}", file=sys.stderr)
            return None

        targets = {}
        for r in conn.execute(
                "SELECT target_id, target_name, vpp_stage, micro_step FROM monitoring_targets"):
            targets[r['target_id']] = {
                'name': r['target_name'],
                'vpp_stage': r['vpp_stage'],
                'micro_step': r['micro_step']
            }

        metrics = {}
        for r in conn.execute("SELECT metric_id, metric_name FROM monitoring_metrics"):
            metrics[r['metric_id']] = r['metric_name']

        tags = {}
        for r in conn.execute("SELECT tag_id, tag_name, category, metric_id FROM monitoring_tags"):
            tags[r['tag_id']] = {
                'name': r['tag_name'],
                'category': r['category'],
                'metric_id': r['metric_id']
            }

        tag_target = defaultdict(set)
        for r in conn.execute("SELECT tag_id, target_id FROM tag_target_mapping"):
            tag_target[r['tag_id']].add(r['target_id'])

        gs = conn.execute(
            "SELECT min_step, max_step, max_rank, grad_unreduced, grad_reduced FROM global_stats").fetchone()
        global_stats = dict(gs) if gs else {}

        trend_rows = []
        cursor = conn.execute(
            "SELECT rank, step, target_id, metric_id, norm, min, max, mean "
            "FROM trend_data ORDER BY rank, step, target_id, metric_id"
        )
        for r in cursor:
            trend_rows.append({
                'rank': r['rank'],
                'step': r['step'],
                'target_id': r['target_id'],
                'metric_id': r['metric_id'],
                'norm': r['norm'],
                'min': r['min'],
                'max': r['max'],
                'mean': r['mean']
            })
    finally:
        conn.close()

    return {
        'targets': targets,
        'metrics': metrics,
        'tags': tags,
        'tag_target': {k: list(v) for k, v in tag_target.items()},
        'global_stats': global_stats,
        'trend_rows': trend_rows
    }


def load_dump_statistic(dump_dir):
    """
    加载 dump_statistic 目录中的 parameters_grad 数据。
    自动检测目录结构:
      - 单 step: dump_dir/rank{N}/dump.json
      - 多 step: dump_dir/step{N}/rank{M}/dump.json

    从每个 dump.json 中提取所有 parameters_grad.* 条目的梯度 norm。
    """
    if not os.path.isdir(dump_dir):
        print(f"Error: {dump_dir} is not a directory", file=sys.stderr)
        return None

    targets = {}
    rows = []
    name_to_id = {}
    step = 0  # 默认 step

    subdirs = [d for d in os.listdir(dump_dir) if os.path.isdir(os.path.join(dump_dir, d))]
    has_steps = any(d.startswith('step') for d in subdirs)

    if has_steps:
        step_dirs = sorted([d for d in subdirs if d.startswith('step')])
        for step_dir in step_dirs:
            step_num = int(step_dir.replace('step', ''))
            step_path = os.path.join(dump_dir, step_dir)
            _load_dump_ranks(step_path, step_num, name_to_id, targets, rows)
    else:
        _load_dump_ranks(dump_dir, step, name_to_id, targets, rows)

    if not rows:
        print("Error: no parameters_grad entries found", file=sys.stderr)
        return None

    return {
        'targets': targets,
        'metrics': {1: 'grad_norm'},
        'trend_rows': rows
    }


def _load_dump_ranks(base_path, step, name_to_id, targets, rows):
    """加载一个 step 下所有 rank 的 dump.json"""
    tid = len(targets)
    for entry in sorted(os.listdir(base_path)):
        rank_dir = os.path.join(base_path, entry)
        if not os.path.isdir(rank_dir) or not entry.startswith('rank'):
            continue

        rank = int(entry.replace('rank', ''))
        dump_file = os.path.join(rank_dir, 'dump.json')
        if not os.path.exists(dump_file):
            continue

        with open(dump_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ops = data.get('data', {})
        for op_name, op_data in ops.items():
            if 'parameters_grad' not in op_name:
                continue

            param_name = op_name.rsplit('.parameters_grad', 1)[0]

            for wkey, wval in op_data.items():
                if isinstance(wval, list) and wval and isinstance(wval[0], dict) and 'Norm' in wval[0]:
                    norm = wval[0].get('Norm', 0)
                    mx = wval[0].get('Max', 0)
                    mn = wval[0].get('Min', 0)
                    mean_val = wval[0].get('Mean', 0)
                    full_name = f"{param_name}.{wkey}"

                    if full_name not in name_to_id:
                        name_to_id[full_name] = tid
                        targets[tid] = {'name': full_name, 'vpp_stage': 0, 'micro_step': 0}
                        tid += 1

                    rows.append({
                        'rank': rank,
                        'step': step,
                        'target_id': name_to_id[full_name],
                        'norm': norm if isinstance(norm, (int, float)) else 0,
                        'min': mn if isinstance(mn, (int, float)) else 0,
                        'max': mx if isinstance(mx, (int, float)) else 0,
                        'mean': mean_val if isinstance(mean_val, (int, float)) else 0,
                        'metric_id': 1
                    })


def load_monitor_csv(path):
    """
    加载 monitor CSV 数据。支持单文件或目录 (批量加载多 rank × 多 step)。

    CSV 与 trend.db 是同源数据 (同一训练运行的两种导出形态),
    因此加载后必须走与 trend.db 相同的累积窗口检测流程。
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        return None

    csv_files = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith('.csv') and ('grad_unreduced' in f or 'grad_reduced' in f):
                    csv_files.append(os.path.join(root, f))
        if not csv_files:
            print(f"Error: no monitor CSV found in {path}", file=sys.stderr)
            return None
    else:
        csv_files = [path]

    rows = []
    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        if 'grad_reduced' in fname:
            default_metric = 2
        else:
            default_metric = 1

        default_rank = 0
        rank_match = re.search(r'rank(\d+)', csv_path)
        if rank_match:
            default_rank = int(rank_match.group(1))

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if 'name' not in r or not r['name'] or 'step' not in r:
                    continue
                rows.append({
                    'rank': default_rank,
                    'vpp_stage': int(r.get('vpp_stage') or 0),
                    'name': r['name'],
                    'step': int(r.get('step') or 0),
                    'micro_step': int(r.get('micro_step') or 0),
                    'min': float(r.get('min') or 0),
                    'max': float(r.get('max') or 0),
                    'mean': float(r.get('mean') or 0),
                    'norm': float(r.get('norm') or 0),
                    'metric_id': default_metric,
                    'shape': r.get('shape', ''),
                    'dtype': r.get('dtype', ''),
                })

    if not rows:
        print(f"Error: no data rows loaded from {path}", file=sys.stderr)
        return None

    name_micro_steps = defaultdict(set)
    name_vpp_stages = defaultdict(set)
    for r in rows:
        name_micro_steps[r['name']].add(r['micro_step'])
        name_vpp_stages[r['name']].add(r['vpp_stage'])

    names = sorted(name_micro_steps.keys())
    targets = {
        i: {'name': n,
            'vpp_stage': min(name_vpp_stages[n]) if name_vpp_stages[n] else 0,
            'micro_step': sorted(name_micro_steps[n])[0] if name_micro_steps[n] else 0}
        for i, n in enumerate(names)
    }
    name_to_id = {n: i for i, n in enumerate(names)}

    for r in rows:
        r['target_id'] = name_to_id[r['name']]

    return {
        'targets': targets,
        'metrics': {1: 'grad_unreduced', 2: 'grad_reduced'},
        'trend_rows': rows
    }


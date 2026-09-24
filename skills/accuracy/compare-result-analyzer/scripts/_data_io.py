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

import csv
import os
import sys
from datetime import datetime
import openpyxl
from _common import safe_float, extract_op_prefix
from _noise_filter import OUTPUT_SUBDIR

def _setup_encoding():
    """配置 stdout/stderr 为 UTF-8，使用 replace 策略优雅降级非可编码字符。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            # Python 3.7+
            stream.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            # Python 3.6 及以下：尝试用 TextIOWrapper 替换
            try:
                import io
                if hasattr(stream, 'buffer'):
                    setattr(sys, stream.name if hasattr(stream, 'name') else
                            ('stdout' if stream is sys.stdout else 'stderr'),
                            io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace'))
            except Exception:
                pass  # 最终回退：无法重配置则保持原样，依赖调用方 try/except
_setup_encoding()



def detect_data_mode(headers):
    """检测数据模式：Max diff 字段 → 统计量模式。

    Cosine 等指标仅在真实数据比对中出现，统计量模式不支持。
    若检测到 Cosine 字段，标记为 'real' 并给出警告。
    """
    if 'Max diff' in headers:
        return 'stat'
    if 'Cosine' in headers:
        return 'real'
    return 'unknown'



def load_rows(path):
    """加载 CSV 或 XLSX 文件，同时记录原始行号。

    支持格式：
      - .csv  — 使用 csv.DictReader，UTF-8 编码
      - .xlsx — 使用 openpyxl，第一行为表头，数据从第 2 行开始

    row 字典中额外存储 'RowIndex' 字段（1-based，header=第1行，数据从第2行开始）
    """
    ext = os.path.splitext(path)[1].lower()
    rows = []

    if ext == '.csv':
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for lineno, row in enumerate(reader, 2):
                row['RowIndex'] = lineno
                rows.append(row)
    elif ext in ('.xlsx', '.xls'):
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        headers = None
        for rowno, row_cells in enumerate(ws.iter_rows(values_only=True), 1):
            if headers is None:
                headers = [str(c) if c is not None else '' for c in row_cells]
                continue
            values = [str(c) if c is not None else '' for c in row_cells]
            row = dict(zip(headers, values))
            row['RowIndex'] = rowno
            rows.append(row)
        wb.close()
    else:
        print("ERROR: Unsupported file format '{}'. Please provide a .csv or .xlsx file.".format(ext),
              file=sys.stderr)
        sys.exit(1)

    return rows



def filter_na_rows(rows):
    filtered = []
    for r in rows:
        npu_name = r.get('NPU Name', '').strip()
        bench_name = r.get('Bench Name', '').strip()
        if npu_name not in ('', 'N/A') and bench_name not in ('', 'N/A'):
            filtered.append(r)
    return filtered



def collect_stat_nodes(rows):
    nodes = []
    for r in rows:
        name = r['NPU Name']
        nre = safe_float(r.get('NormRelativeErr', ''))
        mean_re = safe_float(r.get('MeanRelativeErr', ''))
        max_re = safe_float(r.get('MaxRelativeErr', ''))
        min_re = safe_float(r.get('MinRelativeErr', ''))
        max_diff = safe_float(r.get('Max diff', ''))
        mean_diff = safe_float(r.get('Mean diff', ''))
        l2norm_diff = safe_float(r.get('L2norm diff', ''))
        bench_l2norm = safe_float(r.get('Bench l2norm', ''))
        npu_l2norm = safe_float(r.get('NPU l2norm', ''))

        if nre is None and mean_re is None and max_re is None:
            continue

        # MeanBias = mean_diff / bench_l2norm，整体均值偏移占参考能量的比例
        # MeanBias 补充阈值 = α × 用户指定阈值（α = 1.2）
        if mean_diff is not None and bench_l2norm is not None and bench_l2norm > 1e-12:
            mean_bias = abs(mean_diff) / bench_l2norm  # 原始比率
        else:
            mean_bias = None

        result = r.get('Result', '')
        dtype = r.get('NPU Dtype', '')
        shape = r.get('NPU Tensor Shape', '')
        bench_name = r.get('Bench Name', '')

        # Use the actual CSV line number (RowIndex) stored during load_rows
        row_index = r.get('RowIndex', 0)

        nodes.append({
            'idx': row_index, 'name': name, 'bench_name': bench_name,
            'nre': nre, 'mean_re': mean_re, 'max_re': max_re, 'min_re': min_re,
            'max_diff': max_diff, 'mean_diff': mean_diff, 'l2norm_diff': l2norm_diff,
            'mean_bias': mean_bias, 'bench_l2norm': bench_l2norm,
            'npu_l2norm': npu_l2norm,
            'result': result, 'dtype': dtype, 'shape': shape, 'row': r
        })
    return nodes



def na_summary(rows, filtered_rows, threshold, csv_path, analysis_range=500):
    total = len(rows)
    filtered = len(filtered_rows)
    na_count = total - filtered
    na_ratio = round(na_count / max(total, 1), 6)

    return {
        "file_path": os.path.abspath(csv_path),
        "threshold": threshold,
        "analysis_time": datetime.now().isoformat(),
        "total_rows": total,
        "valid_rows": filtered,
        "na_count": na_count,
        "na_ratio": na_ratio,
        "analysis_range": {"upstream": analysis_range, "downstream": analysis_range}
    }



def meta_errors(rows):
    dtype_mm = []
    shape_mm = []
    rg_mismatch = []
    for r in rows:
        nd = r.get('NPU Dtype', '').strip()
        bd = r.get('Bench Dtype', '').strip()
        if nd and bd and nd != 'N/A' and bd != 'N/A' and nd != bd:
            dtype_mm.append((r.get('RowIndex', 0), r['NPU Name'], nd, bd))
        ns = r.get('NPU Tensor Shape', '').strip()
        bs = r.get('Bench Tensor Shape', '').strip()
        if ns and bs and ns != 'N/A' and bs != 'N/A' and ns != bs:
            shape_mm.append((r.get('RowIndex', 0), r['NPU Name'], ns, bs))
            r['shape_inconsistent'] = True
        rg_val = r.get('Requires_grad Consistent', '').strip().lower()
        if rg_val in ('false', 'f', 'no', 'n', '0', 'inconsistent'):
            rg_mismatch.append((r.get('RowIndex', 0), r['NPU Name'], rg_val))

    # P2-7: shape 不匹配分级告警
    total_valid = len(rows)
    shape_mm_count = len(shape_mm)
    shape_mismatch_ratio = shape_mm_count / max(total_valid, 1)
    if shape_mismatch_ratio > 0.5:
        shape_mismatch_level = 'critical'
    elif shape_mismatch_ratio > 0.1:
        shape_mismatch_level = 'warning'
    else:
        shape_mismatch_level = 'normal'

    return {
        "dtype_mismatch": [
            {"row": d[0], "name": d[1], "npu_dtype": d[2], "bench_dtype": d[3]}
            for d in dtype_mm
        ],
        "shape_mismatch": [
            {"row": s[0], "name": s[1], "npu_shape": s[2], "bench_shape": s[3]}
            for s in shape_mm
        ],
        # P2-7: shape 分级告警
        "shape_mismatch_level": shape_mismatch_level,
        "shape_mismatch_ratio": round(shape_mismatch_ratio, 4),
        "requires_grad_mismatch": [
            {"row": r[0], "name": r[1], "value": r[2]}
            for r in rg_mismatch
        ]
    }


# Task 7.1: Row index cache for O(1) lookups by (prefix, direction).
# Built once at analysis start, used by get_input_nres_for_op to avoid
# O(N) full-table scans on every call (especially in DOWNSTREAM_ABSORBED).
_row_index_cache = {}
# 记录构建 cache 的数据源 id(rows)，用于校验调用方传入的 rows 是否与 cache 一致。
# 不一致时回退全扫描，避免 cache 被静默套用到错误数据源（防 footgun）。
_cache_source_id = None


def _build_row_index(rows):
    """Build an index: dict[(prefix, direction)] -> list of row dicts.

    Called once before propagation analysis. Eliminates O(N²) behavior
    in get_input_nres_for_op which previously scanned all rows per call.
    """
    global _cache_source_id
    _row_index_cache.clear()
    _cache_source_id = id(rows)
    for r in rows:
        name = r.get('NPU Name', '')
        if not name:
            continue
        prefix, direction = extract_op_prefix(name)
        if direction == 'unknown':
            continue
        key = (prefix, direction)
        if key not in _row_index_cache:
            _row_index_cache[key] = []
        _row_index_cache[key].append(r)


def get_input_nres_for_op(rows, prefix, direction='forward'):
    """获取算子的所有输入行的指标值（NRE、MeanRE、MeanBias、shape、dtype）。

    Task 7.1: Uses row index cache for O(1) lookup when available.
    Falls back to full O(N) scan if cache is empty or rows 与 cache 数据源
    不一致（防 footgun：cache 是按某一 rows 构建的，传入不同 rows 时
    必须降级全扫描，否则会静默返回错结果）。

    返回 [(param_key, nre, mean_re, max_re, shape, dtype, mean_bias), ...]，按 param 排序。
    """
    # Use cache only when data source matches (Task 7.1)
    if _row_index_cache and id(rows) == _cache_source_id:
        full_prefix = prefix + '.'
        input_nres = []
        cached_rows = _row_index_cache.get((prefix, direction), [])
        for r in cached_rows:
            name = r.get('NPU Name', '')
            if not name.startswith(full_prefix):
                continue
            param_suffix = name[len(full_prefix):]
            param_type = param_suffix.split('.')[0] if param_suffix else ''
            if param_type not in ('input', 'kwargs', 'parameters'):
                continue
            nre = safe_float(r.get('NormRelativeErr', ''))
            mean_re = safe_float(r.get('MeanRelativeErr', ''))
            max_re = safe_float(r.get('MaxRelativeErr', ''))
            mean_diff = safe_float(r.get('Mean diff', ''))
            bench_l2norm = safe_float(r.get('Bench l2norm', ''))
            mean_bias = abs(mean_diff) / bench_l2norm if (mean_diff is not None and bench_l2norm is not None and bench_l2norm > 1e-12) else None
            shape = r.get('NPU Tensor Shape', '').strip()
            dtype = r.get('NPU Dtype', '').strip()
            input_nres.append((param_suffix, nre, mean_re, max_re, shape, dtype, mean_bias))
        input_nres.sort(key=lambda x: x[0])
        return input_nres

    # Fallback: full scan (backward compatible)
    full_prefix = prefix + '.'
    input_nres = []
    for r in rows:
        name = r.get('NPU Name', '')
        if not name:
            continue
        if not name.startswith(full_prefix):
            continue
        # 提取 param 类型（full_prefix 之后的第一段）
        param_suffix = name[len(full_prefix):]
        param_type = param_suffix.split('.')[0] if param_suffix else ''
        # 输入: input, kwargs, parameters
        if param_type not in ('input', 'kwargs', 'parameters'):
            continue
        nre = safe_float(r.get('NormRelativeErr', ''))
        mean_re = safe_float(r.get('MeanRelativeErr', ''))
        max_re = safe_float(r.get('MaxRelativeErr', ''))
        mean_diff = safe_float(r.get('Mean diff', ''))
        bench_l2norm = safe_float(r.get('Bench l2norm', ''))
        mean_bias = abs(mean_diff) / bench_l2norm if (mean_diff is not None and bench_l2norm is not None and bench_l2norm > 1e-12) else None
        shape = r.get('NPU Tensor Shape', '').strip()
        dtype = r.get('NPU Dtype', '').strip()
        input_nres.append((param_suffix, nre, mean_re, max_re, shape, dtype, mean_bias))
    # 按 param key 排序（input.0, input.1, kwargs.0, parameters.0, ...）
    input_nres.sort(key=lambda x: x[0])
    return input_nres



def output_nodes_detail(nodes):
    """所有 output 节点明细（含 parameters_grad 反向梯度输出）"""
    output_nodes = [n for n in nodes if '.output.' in n['name'] or '.parameters_grad.' in n['name']]

    # Build structured output
    result = []
    for n in output_nodes:
        result.append({
            "row_index": n['idx'],
            "name": n['name'],
            "nre": n['nre'],
            "mean_re": n.get('mean_re'),
            "max_re": n.get('max_re'),
            "min_re": n.get('min_re'),
            "dtype": n.get('dtype', ''),
            "shape": n.get('shape', ''),
            "result": n.get('result', '')
        })

    return output_nodes, result



def all_bad_nodes_detail(nodes, threshold):
    """所有 >= 阈值的节点明细

    近零噪声节点（is_noise=True）不在此处展示，仅在 §7 汇总中体现。
    """
    bad = []
    for n in nodes:
        if n.get('is_noise'):
            continue
        nre_val = n.get('nre')
        if nre_val is not None and nre_val >= threshold:
            bad.append(n)
    bad.sort(key=lambda x: x['idx'])

    # Build structured output
    result = []
    for n in bad:
        result.append({
            "row_index": n['idx'],
            "name": n['name'],
            "nre": n['nre'],
            "mean_re": n.get('mean_re'),
            "max_re": n.get('max_re'),
            "min_re": n.get('min_re'),
            "dtype": n.get('dtype', ''),
            "shape": n.get('shape', ''),
            "result": n.get('result', '')
        })

    return bad, result

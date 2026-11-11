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
"""Label-preserving numeric matrix summaries, without diagnostic rules."""
import argparse
import json
import math
import statistics
from run_manager import ANALYSIS_DIR_HELP, RunManager, window


def relative(before, after):
    return (after-before)/abs(before) if abs(before) > 1e-12 else None


def summarize(points):
    if not points:
        return None
    ts, values = zip(*points)
    avg_t, avg = statistics.mean(ts), statistics.mean(values)
    denominator = sum((t-avg_t)**2 for t in ts)
    ordered = sorted(values)
    position = (len(values)-1)*0.95
    lo, hi = math.floor(position), math.ceil(position)
    return dict(mean=avg, median=statistics.median(values), min=min(values), max=max(values), start=values[0], end=values[-1], delta=values[-1]-values[0], relative_change=relative(values[0], values[-1]), slope=sum((t-avg_t)*(v-avg) for t, v in points)/denominator if denominator else None, peak_timestamp=ts[values.index(max(values))], sample_p95=ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo))


def process(raw, request, trend_window=60):
    start, end, step = window(request['start'], request['end'], request['step'])
    if not math.isfinite(trend_window) or trend_window <= 0:
        raise ValueError('trend_window must be positive finite seconds')
    if raw.get('status') != 'success':
        raise ValueError('Cannot process failed API response')
    data = raw['data']
    if data.get('resultType') != 'matrix':
        raise ValueError('Expected query_range matrix')
    expected = int(math.floor((end-start)/step + 1e-9)) + 1
    output = dict(schema_version=1, query_id=request['query_id'], promql=request['promql'], window=dict(start=start, end=end, step=step), status='no_data', series=[], warnings=raw.get('warnings', []), infos=raw.get('infos', []))
    for series in data['result']:
        samples, duplicates, invalid, off_grid = {}, 0, 0, 0
        for pair in series.get('values', []):
            t, v = float(pair[0]), float(pair[1])
            if not math.isfinite(t) or t < start or t > end or abs((t-start)/step-round((t-start)/step)) > 1e-6:
                off_grid += 1
                continue
            if t in samples:
                duplicates += 1
            samples[t] = v if math.isfinite(v) else None
        points = sorted((t, v) for t, v in samples.items() if v is not None)
        invalid = sum(v is None for v in samples.values())
        trend = []
        buckets = {}
        for t, v in points:
            key = int((t-start)//trend_window)
            buckets.setdefault(key, []).append(v)
        for key, values in sorted(buckets.items()):
            a = start+key*trend_window
            trend.append(dict(start_time=a, end_time=min(a+trend_window, end), mean=statistics.mean(values), min=min(values), max=max(values), sample_count=len(values)))
        labels = dict(series['metric'])
        quality = dict(expected_samples=expected, actual_samples=len(samples), valid_samples=len(points), missing_ratio=1-len(points)/expected, first_timestamp=points[0][0] if points else None, last_timestamp=points[-1][0] if points else None, time_coverage=(points[-1][0]-points[0][0])/(end-start) if points and end > start else (1 if points else 0), duplicate_timestamps=duplicates, nonfinite_samples=invalid, off_grid_samples=off_grid, native_histogram_samples=len(series.get('histograms', [])))
        output['series'].append(dict(metric=labels.get('__name__'), labels=labels, unit=request.get('unit'), window=output['window'], status='ok' if points else 'no_data', samples=[dict(timestamp=t, value=v) for t, v in sorted(samples.items())], data_quality=quality, statistics=summarize(points), trend=trend))
    if any(s['status'] == 'ok' for s in output['series']):
        output['status'] = 'ok'
    output['samples_available_in_window'] = output['status'] == 'ok'
    return output


def compare(current, reference):
    if current['promql'] != reference['promql']:
        raise ValueError('Reference must use identical PromQL')
    key = lambda s: json.dumps(s['labels'], sort_keys=True)
    left, right = {key(s): s for s in current['series']}, {key(s): s for s in reference['series']}
    result = []
    for identity in sorted(set(left) | set(right)):
        a, b = left.get(identity), right.get(identity)
        sa, sb = a['statistics'] if a else None, b['statistics'] if b else None
        changes = {k: dict(delta=sa[k]-sb[k], relative_change=relative(sb[k], sa[k])) for k in ('mean', 'median', 'min', 'max', 'sample_p95') if sa and sb}
        result.append(dict(labels=json.loads(identity), current=sa, reference=sb, changes=changes, status='matched' if sa and sb else 'unmatched_or_no_data'))
    return dict(current_window=current['window'], reference_window=reference['window'], series=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', required=True, help=ANALYSIS_DIR_HELP)
    parser.add_argument('--query-id', required=True)
    parser.add_argument('--reference-query-id')
    parser.add_argument('--trend-window', type=float, default=60, help='Trend window in seconds (default: 60)')
    args = parser.parse_args()
    manager = RunManager(args.analysis_dir)
    history = {r['request']['query_id']: r for r in manager.load_query_results()}
    record = history[args.query_id]
    result = process(record['raw_response'], record['request'], args.trend_window)
    if args.reference_query_id:
        ref = history[args.reference_query_id]
        result['reference_comparison'] = compare(result, process(ref['raw_response'], ref['request'], args.trend_window))
    manager.save_processed_result(args.query_id, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()

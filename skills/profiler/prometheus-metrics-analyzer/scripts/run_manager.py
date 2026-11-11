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
"""Create, inspect and finalize an Analysis Workspace."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

ANALYSIS_DIR_HELP = 'Path to an existing Analysis Workspace directory.'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_write(path, data, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if compressed:
        content = gzip.compress(content)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise ValueError('RFC3339 time requires a timezone')
        result = dt.timestamp()
    if not math.isfinite(result):
        raise ValueError('Time must be finite')
    return result


def window(start, end, step):
    start, end, step = timestamp(start), timestamp(end), float(step)
    if not math.isfinite(step) or step <= 0 or end < start:
        raise ValueError('Require end >= start and finite step > 0 (seconds)')
    return start, end, step


def safe_config(request):
    # Explicit allowlist: credentials must be supplied through the environment.
    allowed = ('prometheus_url', 'target', 'start', 'end', 'step', 'symptom',
               'reference_window', 'slo', 'engine', 'budgets', 'schema_version')
    clean = {key: request[key] for key in allowed if key in request}
    if not {'prometheus_url', 'target', 'start', 'end'} <= clean.keys():
        raise ValueError('Run requires prometheus_url, target, start and end')
    from urllib.parse import urlsplit
    url = urlsplit(clean['prometheus_url'])
    if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use HTTP(S) base URL without credentials/query/fragment')
    target = clean.get('target')
    if not isinstance(target, dict) or not any(target.values()) or not all(
            re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', k) and isinstance(v, str)
            for k, v in target.items()):
        raise ValueError('target requires exact label/value pairs')
    start, end, _ = window(clean['start'], clean['end'], clean.get('step', 15))
    limits = clean.get('budgets', {})
    if not isinstance(limits, dict):
        raise ValueError('budgets must be a JSON object')
    budget = {'timeout': 30, **limits}
    if set(budget) - {'timeout', 'max_span'} or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 for value in budget.values()):
        raise ValueError('budgets supports positive timeout and optional max_span in seconds')
    max_span = budget.get('max_span')
    if max_span is not None and end - start > max_span:
        raise ValueError('Analysis window exceeds max_span')
    reference = clean.get('reference_window')
    if reference:
        a, b, _ = window(reference['start'], reference['end'], clean.get('step', 15))
        if max_span is not None and b - a > max_span:
            raise ValueError('Reference exceeds max_span')
    clean['budgets'] = budget
    return clean


def context_fingerprint(config):
    keys = ('prometheus_url', 'target', 'start', 'end')
    identity = json.dumps({key: config[key] for key in keys}, sort_keys=True)
    return hashlib.sha256(identity.encode()).hexdigest()


class RunManager:
    def __init__(self, path):
        self.path = Path(path)
        self.config = safe_config(read_json(self.path / 'request.json'))
        self.manifest = read_json(self.path / 'manifest.json')
        if self.manifest.get('context_fingerprint') != context_fingerprint(self.config):
            raise ValueError('Run context changed; create a new Run for a different source, target or window')
        # Reconcile durable orphan pairs after a crash between raw and manifest writes.
        known = {q['query_id'] for q in self.manifest['queries']}
        for path in sorted((self.path / 'queries').glob('round_*/query_*/request.json')):
            request = read_json(path)
            raw_path = path.with_name('raw_response.json.gz')
            if request['query_id'] in known or not raw_path.exists():
                continue
            response = json.loads(gzip.decompress(raw_path.read_bytes()))
            processed = path.with_name('processed.json')
            self.manifest['queries'].append(dict(query_id=request['query_id'], round_id=request['round_id'], status=response.get('status', 'error'), request_path=path.relative_to(self.path).as_posix(), raw_path=raw_path.relative_to(self.path).as_posix(), processed_path=processed.relative_to(self.path).as_posix() if processed.exists() else None))
            self.manifest['rounds'] = sorted(set(self.manifest['rounds'] + [request['round_id']]))
            if response.get('status') != 'success':
                self.manifest['errors'].append(dict(query_id=request['query_id'], error=response.get('error', 'query failed')))
            known.add(request['query_id'])

    @classmethod
    def create_run(cls, output, request):
        request = safe_config(request)
        run_id = 'analysis_{}_{}'.format(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), uuid.uuid4().hex[:8])
        path = Path(output).resolve() / 'metrics-analysis' / run_id
        path.mkdir(parents=True)
        (path / 'analysis').mkdir()
        (path / 'queries').mkdir()
        atomic_write(path / 'request.json', request)
        atomic_write(path / 'manifest.json', dict(schema_version=1, run_id=run_id, status='created', request='request.json', context_fingerprint=context_fingerprint(request), metric_catalog_path=None, rounds=[], queries=[], artifact_paths={}, errors=[]))
        return cls(path)

    def update_manifest(self, **changes):
        allowed = {'status', 'artifact_paths', 'errors'}
        if set(changes) - allowed:
            raise ValueError('Only status, artifact_paths, errors may be updated')
        self.manifest.update(changes)
        atomic_write(self.path / 'manifest.json', self.manifest)

    def save_metadata(self, catalog):
        atomic_write(self.path / 'metric_catalog.json', catalog)
        self.manifest['metric_catalog_path'] = 'metric_catalog.json'
        self.update_manifest(status='discovered')

    def prepare_query(self, round_id, promql, start, end, step, unit=None):
        config = self.config
        a, b, s = window(start, end, step)
        budget = config['budgets']
        if not isinstance(round_id, int) or round_id < 1:
            raise ValueError('Round must be a positive integer')
        if not isinstance(promql, str) or not promql.strip():
            raise ValueError('PromQL is required')
        max_span = budget.get('max_span')
        if max_span is not None and b - a > max_span:
            raise ValueError('Query window exceeds max_span')
        for item in self.manifest['queries']:
            if item.get('status') != 'success':
                continue
            previous = read_json(self.path / item['request_path'])
            if (previous['promql'], previous['start'], previous['end'], previous['step']) == (promql, a, b, s):
                raise ValueError('Duplicate query; reuse saved result')
        query_id = 'query_{:03d}'.format(len(self.manifest['queries']) + 1)
        return dict(schema_version=1, run_id=self.manifest['run_id'], round_id=round_id, query_id=query_id, prometheus_url=config['prometheus_url'], target=dict(config['target']), promql=promql, start=a, end=b, step=s, unit=unit, queried_at=now())

    def save_query_result(self, request, response):
        relative = Path('queries') / ('round_{:02d}'.format(request['round_id'])) / request['query_id']
        atomic_write(self.path / relative / 'request.json', request)
        atomic_write(self.path / relative / 'raw_response.json.gz', response, compressed=True)
        entry = dict(query_id=request['query_id'], round_id=request['round_id'], status=response.get('status', 'error'), request_path=(relative / 'request.json').as_posix(), raw_path=(relative / 'raw_response.json.gz').as_posix(), processed_path=None)
        self.manifest['queries'].append(entry)
        self.manifest['rounds'] = sorted(set(self.manifest['rounds'] + [request['round_id']]))
        if response.get('status') != 'success':
            self.manifest['errors'].append(dict(query_id=request['query_id'], error=response.get('error', 'query failed')))
        self.update_manifest(status='collecting')
        return entry

    def save_processed_result(self, query_id, result):
        entry = next(q for q in self.manifest['queries'] if q['query_id'] == query_id)
        path = str(Path(entry['request_path']).with_name('processed.json'))
        atomic_write(self.path / path, result)
        entry['processed_path'] = Path(path).as_posix()
        self.update_manifest(status='analyzing')

    def load_query_results(self):
        # Scan durable request/raw pairs as well: recover after a manifest-write failure.
        results = []
        for path in sorted((self.path / 'queries').glob('round_*/query_*/request.json')):
            raw_path = path.with_name('raw_response.json.gz')
            if raw_path.exists():
                raw = json.loads(gzip.decompress(raw_path.read_bytes()))
                processed = path.with_name('processed.json')
                results.append(dict(request=read_json(path), raw_response=raw, processed=read_json(processed) if processed.exists() else None))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create', help='Create a workspace and print its absolute path')
    create.add_argument('--request', required=True, help='Analysis input JSON file')
    create.add_argument('--output', default='outputs', help='Workspace output root (default: outputs)')
    load = sub.add_parser('load', help='Read saved queries and results')
    load.add_argument('--analysis-dir', required=True, help=ANALYSIS_DIR_HELP)
    finish = sub.add_parser('finalize', help='Archive evidence and report and update the manifest')
    finish.add_argument('--analysis-dir', required=True, help=ANALYSIS_DIR_HELP)
    finish.add_argument('--evidence', required=True, help='Agent-authored evidence JSON file')
    finish.add_argument('--report', required=True, help='Agent-authored Markdown report')
    args = parser.parse_args()
    if args.command == 'create':
        print(RunManager.create_run(args.output, read_json(args.request)).path)
    elif args.command == 'load':
        print(json.dumps(RunManager(args.analysis_dir).load_query_results(), ensure_ascii=False, allow_nan=False))
    else:
        manager = RunManager(args.analysis_dir)
        evidence_source = Path(args.evidence)
        report_source = Path(args.report)
        evidence = read_json(evidence_source)
        report = report_source.read_text(encoding='utf-8')
        evidence_dest = manager.path / 'analysis/evidence.json'
        report_dest = manager.path / 'analysis/report.md'
        if evidence_source.resolve() != evidence_dest.resolve():
            atomic_write(evidence_dest, evidence)
        if report_source.resolve() != report_dest.resolve():
            fd, temp = tempfile.mkstemp(dir=str(report_dest.parent))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    stream.write(report)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, str(report_dest))
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        manager.update_manifest(status='completed', artifact_paths=dict(evidence='analysis/evidence.json', report='analysis/report.md'))


if __name__ == '__main__':
    main()


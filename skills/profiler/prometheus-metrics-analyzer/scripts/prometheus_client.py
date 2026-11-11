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
"""Prometheus discovery and range queries; never chooses metrics or diagnoses."""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from run_manager import ANALYSIS_DIR_HELP, RunManager, now, read_json, safe_config, timestamp


class PrometheusError(RuntimeError):
    def __init__(self, payload):
        self.payload = payload
        super().__init__(payload.get('error', 'Prometheus request failed'))


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        before = urllib.parse.urlsplit(req.full_url)
        after = urllib.parse.urlsplit(newurl)
        if (before.scheme, before.netloc) != (after.scheme, after.netloc):
            raise urllib.error.HTTPError(req.full_url, code, 'Cross-origin redirect refused', headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PrometheusClient:
    def __init__(self, url, timeout=30):
        # Reject URL credentials before network access or persistence.
        safe_config(dict(prometheus_url=url, target={'job': '_'}, start=0, end=0))
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.token = os.environ.get('PROMETHEUS_BEARER_TOKEN')
        self.opener = urllib.request.build_opener(SameOriginRedirect())

    def _redact(self, value):
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        if isinstance(value, str):
            if self.token:
                value = value.replace(self.token, '[REDACTED]')
            # Scrape URLs can carry basic-auth userinfo and sensitive query params.
            value = re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', value)
            value = re.sub(r'([?&](?:token|password|access_token|api_key)=)[^&\s]*', r'\1[REDACTED]', value, flags=re.I)
        return value

    def get(self, path, params=None, text=False):
        url = self.url + path
        if params:
            url += '?' + urllib.parse.urlencode(params, doseq=True)
        headers = {'Authorization': 'Bearer ' + self.token} if self.token else {}
        try:
            with self.opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout) as response:
                body = response.read().decode('utf-8')
            payload = {'status': 'success', 'data': body} if text else json.loads(body)
            if not isinstance(payload, dict) or payload.get('status') != 'success':
                raise PrometheusError(self._redact(payload if isinstance(payload, dict) else dict(status='error', error='Invalid API envelope', response=payload)))
            return self._redact(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', 'replace')
            try:
                payload = json.loads(body)
            except ValueError:
                payload = dict(status='error', error=body)
            if not isinstance(payload, dict):
                payload = dict(status='error', response=payload)
            payload.update(status='error', http_status=exc.code)
            payload.setdefault('error', 'HTTP {}'.format(exc.code))
            raise PrometheusError(self._redact(payload)) from None
        except (OSError, ValueError) as exc:
            raise PrometheusError(self._redact(dict(status='error', errorType=type(exc).__name__, error=str(exc)))) from None

    @staticmethod
    def selector(target):
        return '{' + ','.join(k + '=' + json.dumps(v) for k, v in sorted(target.items())) + '}'

    def targets(self):
        active = self.get('/api/v1/targets', {'state': 'active'})['data']['activeTargets']
        fields = ('labels', 'scrapeUrl', 'health', 'lastScrape', 'lastError', 'scrapeInterval')
        return [{key: row.get(key) for key in fields} for row in active]

    def discover(self, target, start=None, end=None):
        """Collect current observations and historical series candidates separately."""
        def matches(labels):
            return all(labels.get(key) == value for key, value in target.items())
        checks = {}
        for name, path, plain in [('buildinfo', '/api/v1/status/buildinfo', False), ('ready', '/-/ready', True)]:
            try:
                checks[name] = self.get(path, text=plain)
            except PrometheusError as exc:
                checks[name] = exc.payload
        if checks['ready']['status'] != 'success':
            try:
                checks['healthy'] = self.get('/-/healthy', text=True)
            except PrometheusError as exc:
                checks['healthy'] = exc.payload
        selector = self.selector(target)
        warnings, records, candidates = [], [], []
        targets_available = True
        try:
            candidates = self.targets()
        except PrometheusError as exc:
            targets_available = False
            warnings.append('Current target listing unavailable: ' + str(exc))
        targets = [row for row in candidates if matches(row.get('labels') or {})]

        source = 'targets/metadata'
        metadata_error = None
        try:
            records = self.get('/api/v1/targets/metadata', {'match_target': selector})['data']
            records = [row for row in records if matches(row.get('target') or {})]
        except PrometheusError as exc:
            metadata_error = exc
            warnings.append('Target metadata unavailable: ' + str(exc))
        metadata_available = bool(records)

        historical = None
        analysis_window = None
        if start is not None and end is not None:
            analysis_window = dict(start=timestamp(start), end=timestamp(end))
            limit = 1000
            try:
                rows = self.get('/api/v1/series', dict(analysis_window, **{'match[]': selector, 'limit': limit}))['data']
                if len(rows) >= limit:
                    warnings.append('Historical series reached the discovery limit; catalog may be incomplete.')
                historical = [labels for labels in rows[:limit] if matches(labels)]
            except PrometheusError as exc:
                warnings.append('Historical series discovery unavailable: ' + str(exc))

        # Global Metadata provides type hints when scoped Metadata is unavailable.
        if metadata_error and metadata_error.payload.get('http_status') in (404, 405, 501):
            try:
                if historical is not None:
                    names = {labels.get('__name__') for labels in historical}
                else:
                    params = dict(analysis_window or {}, **{'match[]': selector})
                    names = set(self.get('/api/v1/label/__name__/values', params)['data'])
                metadata = self.get('/api/v1/metadata')['data']
                for name in names & metadata.keys():
                    for entry in metadata[name]:
                        records.append(dict(entry, metric=name, target=target, type='unknown',
                                            type_hint=entry.get('type'), attribution='unverified_global'))
                if records:
                    source = 'metadata+scoped_names'
                    warnings.append('Global metadata type/HELP attribution is unverified.')
            except PrometheusError as exc:
                warnings.append('Global metadata fallback unavailable: ' + str(exc))

        known_names = {record['metric'] for record in records}
        historical_names = {labels['__name__'] for labels in historical or [] if '__name__' in labels}
        for name in sorted(historical_names - known_names):
            records.append(dict(metric=name, type='unknown', help='', unit='', target=target,
                                attribution='historical_series'))
        if not metadata_available and source == 'targets/metadata' and historical_names:
            source = 'historical_series'

        historical_identities = {(labels.get('job'), labels.get('instance'))
                                 for labels in historical or [] if 'instance' in labels}
        ambiguous = len(targets) > 1 or len(historical_identities) > 1
        if ambiguous:
            warnings.append('Target selector matches multiple services; refine labels and create a new Run.')
        currently_up = None if not targets_available else (
            bool(targets) and all(row['health'] == 'up' for row in targets))
        unique = {}
        for record in records:
            item = {k: record.get(k, '') for k in ('metric', 'type', 'help', 'unit', 'target')}
            item['type'] = item['type'] or 'unknown'
            item['attribution'] = record.get('attribution', 'target_metadata')
            if 'type_hint' in record:
                item['type_hint'] = record['type_hint']
            unique[json.dumps(item, sort_keys=True)] = item
        return dict(
            schema_version=1, discovered_at=now(), prometheus_url=self.url, target=target,
            analysis_window=analysis_window, checks=checks, candidate_targets=candidates,
            targets=targets, target_currently_up=currently_up,
            target_ambiguous=ambiguous, metadata_available=metadata_available,
            historical_series_discovered=bool(historical) if historical is not None else None,
            historical_series=historical, samples_available_in_window=None,
            source=source, metrics=list(unique.values()), warnings=warnings,
        )

    def query_range(self, manager, round_id, promql, start, end, step, unit=None):
        if self.url != manager.config['prometheus_url'].rstrip('/'):
            raise ValueError('Prometheus URL must match the Run context')
        request = manager.prepare_query(round_id, promql, start, end, step, unit)
        try:
            result = self.get('/api/v1/query_range', dict(query=promql, start=request['start'], end=request['end'], step=request['step'], timeout=str(self.timeout) + 's'))
        except PrometheusError as exc:
            manager.save_query_result(request, exc.payload)
            raise
        manager.save_query_result(request, result)
        return request, result


    def query_ranges(self, manager, round_id, queries, start, end, step, unit=None):
        """Run an explicitly supplied batch serially; persist each API outcome."""
        if not isinstance(queries, list) or not queries:
            raise ValueError('queries-file must contain a nonempty JSON array')
        prepared, identities = [], set()
        for item in queries:
            if not isinstance(item, dict) or set(item) - {'promql', 'start', 'end', 'step', 'unit'}:
                raise ValueError('Each query requires an object with promql and optional start/end/step/unit')
            item_unit = item.get('unit', unit)
            if item_unit is not None and not isinstance(item_unit, str):
                raise ValueError('unit must be a string or null')
            request = manager.prepare_query(round_id, item.get('promql'), item.get('start', start), item.get('end', end), item.get('step', step), item_unit)
            identity = (request['promql'], request['start'], request['end'], request['step'])
            if identity in identities:
                raise ValueError('Duplicate query within batch')
            identities.add(identity)
            prepared.append(request)
        # All local validation completes before the first HTTP request. API failures
        # remain independent and do not discard or prevent the other query results.
        results = []
        for index, request in enumerate(prepared, 1):
            try:
                actual, _ = self.query_range(manager, round_id, request['promql'], request['start'], request['end'], request['step'], request['unit'])
                results.append(dict(index=index, query_id=actual['query_id'], status='success'))
            except PrometheusError as exc:
                results.append(dict(index=index, query_id=manager.manifest['queries'][-1]['query_id'], status='error', error=exc.payload.get('error')))
        failures = sum(r['status'] == 'error' for r in results)
        status = 'success' if not failures else 'error' if failures == len(results) else 'partial_failure'
        return dict(status=status, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['targets', 'discover', 'query'])
    parser.add_argument('--analysis-dir', help=ANALYSIS_DIR_HELP)
    parser.add_argument('--url', help='Prometheus URL for targets only')
    parser.add_argument('--round', type=int, default=1, help='Positive query batch number (default: 1)')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--promql')
    source.add_argument('--queries-file', help='JSON array of PromQL query objects')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--step', type=float)
    parser.add_argument('--unit')
    args = parser.parse_args()
    if args.command == 'query' and args.promql is None and args.queries_file is None:
        parser.error('query requires --promql or --queries-file')
    if args.command != 'query' and (args.promql is not None or args.queries_file is not None):
        parser.error('Query expressions are only valid with query')
    if args.command == 'targets':
        if not args.url or args.analysis_dir:
            parser.error('targets requires --url without --analysis-dir')
        client = PrometheusClient(args.url)
        print(json.dumps(dict(prometheus_url=client.url, targets=client.targets()), ensure_ascii=False))
        return
    if not args.analysis_dir or args.url is not None:
        parser.error('discover and query require --analysis-dir; the URL comes from the workspace')
    manager = RunManager(args.analysis_dir)
    config = manager.config
    client = PrometheusClient(config['prometheus_url'], config['budgets']['timeout'])
    if args.command == 'discover':
        catalog = client.discover(config['target'], config['start'], config['end'])
        manager.save_metadata(catalog)
        print(json.dumps(catalog, ensure_ascii=False))
        readiness = catalog['checks']['ready']
        if readiness['status'] != 'success':
            print('Prometheus is not ready: ' + readiness.get('error', 'readiness check failed'), file=sys.stderr)
            raise SystemExit(1)
        if catalog['target_ambiguous']:
            print('Target matches multiple services; confirm exact labels and create a new workspace.', file=sys.stderr)
            raise SystemExit(1)
    else:
        catalog_path = manager.path / 'metric_catalog.json'
        if catalog_path.exists() and read_json(catalog_path).get('target_ambiguous'):
            parser.error('Run target is ambiguous; refine labels and create a new Run')
        start = args.start if args.start is not None else config['start']
        end = args.end if args.end is not None else config['end']
        step = args.step if args.step is not None else config.get('step', 15)
        if args.queries_file is not None:
            try:
                result = client.query_ranges(manager, args.round, read_json(args.queries_file), start, end, step, args.unit)
            except (ValueError, TypeError, OSError) as exc:
                parser.error(str(exc))
            print(json.dumps(result, ensure_ascii=False))
            if result['status'] != 'success':
                raise SystemExit(1)
        else:
            try:
                request, _ = client.query_range(manager, args.round, args.promql, start, end, step, args.unit)
            except ValueError as exc:
                parser.error(str(exc))
            print(request['query_id'])


if __name__ == '__main__':
    main()

"""Regression checks for the local Critical-only scanner output."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[3] / 'pre-commit' / 'skill_scanner_critical.py'
SPEC = importlib.util.spec_from_file_location('skill_scanner_critical', SCRIPT)
scanner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scanner)


def finding(severity, title):
    return {
        'severity': severity,
        'title': title,
        'rule_id': 'TEST_RULE',
        'file_path': 'scripts/example.py',
        'line_number': 10,
        'description': title,
    }


class CriticalOutputTest(unittest.TestCase):
    def run_report(self, report, returncode=0):
        output, errors = io.StringIO(), io.StringIO()
        completed = subprocess.CompletedProcess([], returncode, json.dumps(report))
        with patch.object(scanner.subprocess, 'run', return_value=completed):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                status = scanner.main(['scan-all', 'skills', '--fail-on-severity=critical'])
        return status, output.getvalue(), errors.getvalue()

    def test_only_critical_including_cross_skill_findings_are_shown(self):
        report = {
            'summary': {'total_skills_scanned': 1},
            'results': [
                {
                    'skill_name': 'example',
                    'findings': [
                        finding(level, f'{level} finding') for level in ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')
                    ],
                }
            ],
            'cross_skill_findings': [finding('CRITICAL', 'cross finding'), finding('LOW', 'cross low')],
        }
        status, output, _ = self.run_report(report, returncode=1)
        self.assertEqual(status, 1)
        self.assertIn('2 CRITICAL findings', output)
        self.assertIn('cross finding', output)
        self.assertIn('scripts/example.py:10', output)
        for text in ('HIGH finding', 'MEDIUM finding', 'LOW finding', 'INFO finding', 'cross low'):
            self.assertNotIn(text, output)

    def test_lower_severity_findings_do_not_fail(self):
        report = {
            'summary': {'total_skills_scanned': 1},
            'results': [{'skill_name': 'example', 'findings': [finding('HIGH', 'hidden finding')]}],
            'cross_skill_findings': [],
        }
        status, output, _ = self.run_report(report)
        self.assertEqual(status, 0)
        self.assertIn('0 CRITICAL findings', output)
        self.assertNotIn('hidden finding', output)

    def test_scanner_failure_is_preserved(self):
        status, _, errors = self.run_report({}, returncode=2)
        self.assertEqual(status, 2)
        self.assertIn('failed with exit code 2', errors)

    def test_single_skill_report_is_filtered(self):
        report = {
            'skill_name': 'new-skill',
            'findings': [finding('CRITICAL', 'critical issue'), finding('HIGH', 'hidden issue')],
        }
        status, output, _ = self.run_report(report, returncode=1)
        self.assertEqual(status, 1)
        self.assertIn('1 skills scanned; 1 CRITICAL findings', output)
        self.assertIn('new-skill', output)
        self.assertNotIn('hidden issue', output)

    def test_invalid_report_is_not_a_success(self):
        status, _, errors = self.run_report({})
        self.assertEqual(status, 2)
        self.assertIn('invalid JSON report', errors)

    def test_nonzero_without_critical_is_not_a_success(self):
        report = {'summary': {'total_skills_scanned': 0}, 'results': [], 'cross_skill_findings': []}
        status, _, errors = self.run_report(report, returncode=1)
        self.assertEqual(status, 1)
        self.assertIn('failure without Critical findings', errors)


if __name__ == '__main__':
    unittest.main()

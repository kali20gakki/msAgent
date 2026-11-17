"""Display only Critical findings from the local Skill Scanner JSON report."""

import json
import subprocess
import sys


def critical_findings(report: dict) -> list[tuple[str, dict]]:
    """Keep Critical findings from both individual skills and overlap analysis."""
    if 'skill_name' in report:
        return [(report['skill_name'], finding) for finding in report['findings'] if finding['severity'] == 'CRITICAL']
    findings = []
    for result in report['results']:
        findings.extend(
            (result['skill_name'], finding) for finding in result['findings'] if finding['severity'] == 'CRITICAL'
        )
    findings.extend(
        ('cross-skill', finding) for finding in report['cross_skill_findings'] if finding['severity'] == 'CRITICAL'
    )
    return findings


def main(argv: list[str] | None = None) -> int:
    """Preserve scanner failures and print only Critical security findings."""
    args = sys.argv[1:] if argv is None else argv
    try:
        result = subprocess.run(
            ['skill-scanner', *args, '--format', 'json'],
            stdout=subprocess.PIPE,
            encoding='utf-8',
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'Skill Scanner could not complete: {exc}', file=sys.stderr)
        return 2
    if result.returncode not in (0, 1):
        print(f'Skill Scanner failed with exit code {result.returncode}.', file=sys.stderr)
        return 2
    try:
        report = json.loads(result.stdout)
        findings = critical_findings(report)
        total_skills = 1 if 'skill_name' in report else report['summary']['total_skills_scanned']
    except (ValueError, KeyError, TypeError) as exc:
        print(f'Skill Scanner returned an invalid JSON report: {exc}', file=sys.stderr)
        return 2

    print(f'Skill Scanner: {total_skills} skills scanned; {len(findings)} CRITICAL findings.')
    for skill, finding in findings:
        location = finding.get('file_path') or '(cross-skill analysis)'
        if finding.get('line_number') is not None:
            location = f"{location}:{finding['line_number']}"
        print(f"\n[CRITICAL] {skill} — {finding['title']}")
        print(f"  Rule: {finding['rule_id']} | Location: {location}")
        print(f"  {finding['description']}")
        if finding.get('remediation'):
            print(f"  Remediation: {finding['remediation']}")
    # A nonzero scan without Critical findings may indicate a scan failure.
    if result.returncode and not findings:
        print('Skill Scanner returned a failure without Critical findings.', file=sys.stderr)
    return 1 if findings else result.returncode


if __name__ == '__main__':
    sys.exit(main())

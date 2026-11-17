"""Run local checks only for Skills affected by staged file changes."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def changed_skill_files(paths: list[str]) -> list[str]:
    """Map staged additions, modifications and rename targets to owning Skills."""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '--find-renames', '-z', '--', 'skills/'],
        stdout=subprocess.PIPE,
        check=True,
        timeout=30,
    )
    staged = {os.fsdecode(path) for path in result.stdout.split(b'\0') if path}
    manifests = set()
    for filename in paths:
        path = Path(filename)
        if path.as_posix() not in staged or not path.as_posix().startswith('skills/'):
            continue
        for directory in path.parents:
            if directory == Path('skills'):
                break
            manifest = directory / 'SKILL.md'
            if manifest.is_file():
                manifests.add(manifest.as_posix())
                break
    return sorted(manifests)


def main(argv: list[str] | None = None) -> int:
    """Review affected manifests or scan their containing Skill directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tool', choices=('reviewer', 'scanner'))
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args(argv)
    try:
        paths = changed_skill_files(args.paths)
        if not paths:
            print('No staged changes to Skills in the selected files; skipping Skill checks.')
            return 0

        if args.tool == 'reviewer':
            commands = [['skill-reviewer', '--config', 'pre-commit/skill-reviewer-policy.yaml', *paths]]
        else:
            wrapper = Path(__file__).with_name('skill_scanner_critical.py')
            commands = [
                [
                    sys.executable,
                    str(wrapper),
                    'scan',
                    str(Path(path).parent),
                    '--use-behavioral',
                    '--fail-on-severity=critical',
                    '--policy',
                    'pre-commit/skill-scanner-policy.yaml',
                ]
                for path in paths
            ]
        status = 0
        for command in commands:
            result = subprocess.run(command, check=False, timeout=660)
            if result.returncode:
                status = max(status, result.returncode if result.returncode > 0 else 2)
        return status
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'Could not check changed Skills: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

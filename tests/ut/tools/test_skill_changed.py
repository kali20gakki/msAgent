"""Check staged Skill selection and local hook dispatch."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[3] / 'pre-commit' / 'skill_changed.py'
SPEC = importlib.util.spec_from_file_location('skill_changed', SCRIPT)
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


class StagedSkillTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.mkdtemp(prefix='msagent-changed-skill-')
        self.addCleanup(shutil.rmtree, directory)
        previous = Path.cwd()
        os.chdir(directory)
        self.addCleanup(os.chdir, previous)
        self.git('init', '-q')
        self.write('skills/test/existing/SKILL.md', 'existing skill\n')
        self.git('add', '.')
        self.git(
            '-c',
            'user.name=Test',
            '-c',
            'user.email=test@example.invalid',
            '-c',
            'core.hooksPath=/dev/null',
            'commit',
            '-qm',
            'baseline',
        )

    @staticmethod
    def git(*args):
        return subprocess.run(['git', *args], check=True, capture_output=True)

    @staticmethod
    def write(path, text):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')

    def test_new_and_modified_skills_are_selected_and_deduplicated(self):
        existing = 'skills/test/existing/SKILL.md'
        new = 'skills/test/new/SKILL.md'
        unstaged = 'skills/test/unstaged/SKILL.md'
        reference = 'skills/test/existing/references/new.md'
        self.write(existing, 'changed existing skill\n')
        self.write(new, 'new skill\n')
        self.write(reference, 'new reference\n')
        self.git('add', '.')
        self.write(unstaged, 'not staged\n')
        self.assertEqual(hook.changed_skill_files([existing, new, unstaged, reference]), [existing, new])
        self.assertEqual(hook.changed_skill_files([existing]), [existing])
        self.assertEqual(hook.changed_skill_files([reference]), [existing])

    def test_renamed_skill_is_selected(self):
        destination = 'skills/test/renamed/SKILL.md'
        Path(destination).parent.mkdir()
        self.git('mv', 'skills/test/existing/SKILL.md', destination)
        self.assertEqual(hook.changed_skill_files([destination]), [destination])

    def test_modified_script_selects_only_its_owning_skill(self):
        script = 'skills/test/existing/scripts/check.py'
        unrelated = 'skills/test/untouched/SKILL.md'
        self.write(script, 'original\n')
        self.write(unrelated, 'untouched\n')
        self.git('add', '.')
        self.git(
            '-c',
            'user.name=Test',
            '-c',
            'user.email=test@example.invalid',
            '-c',
            'core.hooksPath=/dev/null',
            'commit',
            '-qm',
            'script baseline',
        )
        self.write(script, 'modified\n')
        self.git('add', script)
        self.assertEqual(hook.changed_skill_files([script, unrelated]), ['skills/test/existing/SKILL.md'])

    def test_unstaged_or_unowned_changes_do_not_select_skills(self):
        existing = 'skills/test/existing/SKILL.md'
        unowned = 'skills/README.md'
        self.write(existing, 'unstaged change\n')
        self.write(unowned, 'new root documentation\n')
        self.git('add', unowned)
        self.assertEqual(hook.changed_skill_files([existing, unowned]), [])

    def test_html_csv_svg_changes_select_their_skill(self):
        paths = [f'skills/test/existing/references/example.{suffix}' for suffix in ('html', 'csv', 'svg')]
        for path in paths:
            self.write(path, 'fixture\n')
        self.git('add', '.')
        self.assertEqual(hook.changed_skill_files(paths), ['skills/test/existing/SKILL.md'])

    def test_initial_commit_and_spaces_in_paths(self):
        directory = Path('initial-repo')
        directory.mkdir()
        os.chdir(directory)
        self.git('init', '-q')
        new = 'skills/category with spaces/new/SKILL.md'
        self.write(new, 'new skill\n')
        self.git('add', '.')
        self.assertEqual(hook.changed_skill_files([new]), [new])


class DispatchTest(unittest.TestCase):
    def test_unchanged_skill_does_not_invoke_tools(self):
        with patch.object(hook, 'changed_skill_files', return_value=[]):
            with patch.object(hook.subprocess, 'run') as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(hook.main(['scanner', 'skills/existing/SKILL.md']), 0)
                run.assert_not_called()

    def test_reviewer_receives_only_selected_manifests(self):
        paths = ['skills/test/new/SKILL.md']
        with patch.object(hook, 'changed_skill_files', return_value=paths):
            with patch.object(hook.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)) as run:
                self.assertEqual(hook.main(['reviewer', *paths]), 1)
                self.assertEqual(
                    run.call_args.args[0],
                    [
                        'skill-reviewer',
                        '--config',
                        'pre-commit/skill-reviewer-policy.yaml',
                        *paths,
                    ],
                )

    def test_scanner_checks_each_affected_directory_and_preserves_failure(self):
        paths = ['skills/test/one/SKILL.md', 'skills/test/two/SKILL.md']
        results = [subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]
        with patch.object(hook, 'changed_skill_files', return_value=paths):
            with patch.object(hook.subprocess, 'run', side_effect=results) as run:
                self.assertEqual(hook.main(['scanner', *paths]), 1)
                self.assertEqual(run.call_count, 2)
                for call, path in zip(run.call_args_list, paths):
                    command = call.args[0]
                    self.assertEqual(command[2:4], ['scan', str(Path(path).parent)])
                    self.assertIn('--fail-on-severity=critical', command)
                    self.assertNotIn('scan-all', command)

    def test_git_failure_does_not_silently_skip_checks(self):
        with patch.object(hook, 'changed_skill_files', side_effect=subprocess.CalledProcessError(1, ['git'])):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(hook.main(['scanner']), 2)


if __name__ == '__main__':
    unittest.main()

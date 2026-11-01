"""Local signing migration and real Git autostash recovery; no Apple SDK required."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROJECT = 'prototypes/ios27-coreai/CoreAIProbe.xcodeproj/project.pbxproj'
PRIVATE = 'prototypes/ios27-coreai/Signing.local.xcconfig'
TEAM = 'ABC1234567'
ENV = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
           GIT_TERMINAL_PROMPT='0')


def git(cwd, *args, check=True):
    return subprocess.run(['git', *args], cwd=cwd, env=ENV, text=True,
                          capture_output=True, check=check)


def identity(cwd):
    git(cwd, 'config', 'user.name', 'Fixture')
    git(cwd, 'config', 'user.email', 'fixture@example.invalid')


class CoreAISigningTests(unittest.TestCase):
    def configure(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'scripts/configure_coreai_signing.py'),
                               *map(str, args)], text=True, capture_output=True)

    def test_saved_project_migrates_one_team_without_editing_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'project with spaces.pbxproj'
            text = f'DEVELOPMENT_TEAM = "{TEAM}";\nDevelopmentTeam = {TEAM};\nDEVELOPMENT_TEAM = "";\n'
            source.write_text(text);out = root / 'Signing.local.xcconfig'
            result = self.configure('--from-project', source, '--out', out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('DEVELOPMENT_TEAM = ' + TEAM, out.read_text())
            self.assertEqual(source.read_text(), text)
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)

    def test_invalid_or_ambiguous_team_does_not_overwrite_existing_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'project.pbxproj';out = root / 'Signing.local.xcconfig'
            out.write_text('preserve existing settings\n')
            source.write_text(f'DEVELOPMENT_TEAM = {TEAM};\nDEVELOPMENT_TEAM = ZZZ9876543;\n')
            for args in [('--from-project', source), ('--team', 'bad\nINJECTED = YES')]:
                with self.subTest(args=args):
                    result = self.configure(*args, '--out', out)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(out.read_text(), 'preserve existing settings\n')

    def fixture(self, root, *, extra_conflict=False, team_present=True):
        remote = root / 'remote.git';seed = root / 'seed';local = root / 'user checkout'
        git(root, 'init', '--bare', '--initial-branch=main', str(remote))
        seed.mkdir();git(seed, 'init', '--initial-branch=main');identity(seed)
        p = seed / PROJECT;p.parent.mkdir(parents=True)
        p.write_text('// original project\nDEVELOPMENT_TEAM = "";\n')
        (seed / 'notes.txt').write_text('base notes\n')
        git(seed, 'add', '.');git(seed, 'commit', '-m', 'base')
        git(seed, 'remote', 'add', 'origin', str(remote));git(seed, 'push', 'origin', 'main')
        git(root, 'clone', str(remote), str(local));identity(local)
        # Simulate Xcode saving its own formatting and local signing account.
        local_team = TEAM if team_present else ''
        (local / PROJECT).write_text(f'// Xcode local project\nDEVELOPMENT_TEAM = "{local_team}";\n')
        (local / 'notes.txt').write_text('keep local notes\n')
        p.write_text('// incoming model resources\nDEVELOPMENT_TEAM = "";\n')
        if extra_conflict:
            (seed / 'notes.txt').write_text('different upstream notes\n')
        git(seed, 'add', '.');git(seed, 'commit', '-m', 'model resources')
        git(seed, 'push', 'origin', 'main')
        git(local, 'pull', '--rebase', '--autostash', check=False)
        self.assertIn(PROJECT, git(local, 'diff', '--name-only', '--diff-filter=U').stdout)
        # Publish the fix *after* the user's conflicted pull, as in this incident.
        p.write_text('// shared project uses Signing.xcconfig\n')
        for name in ['configure_coreai_signing.py', 'recover_coreai_autostash.sh']:
            target = seed / 'scripts' / name;target.parent.mkdir(exist_ok=True)
            target.write_bytes((ROOT / 'scripts' / name).read_bytes())
        (seed / '.gitignore').write_text('/' + PRIVATE + '\n')
        git(seed, 'add', '.');git(seed, 'commit', '-m', 'private signing fix')
        git(seed, 'push', 'origin', 'main');git(local, 'fetch', 'origin', 'main')
        return local, git(seed, 'rev-parse', 'HEAD').stdout.strip()

    def recover(self, local):
        return subprocess.run(['bash', str(ROOT / 'scripts/recover_coreai_autostash.sh')],
                              cwd=local, env=ENV, text=True, capture_output=True, timeout=30)

    def test_real_autostash_conflict_is_recovered_with_team_notes_and_stash_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);local, latest = self.fixture(root)
            stash = git(local, 'rev-parse', 'refs/stash').stdout
            result = self.recover(local)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git(local, 'rev-parse', 'HEAD').stdout.strip(), latest)
            self.assertEqual(git(local, 'diff', '--name-only', '--diff-filter=U').stdout, '')
            self.assertEqual(git(local, 'diff', '--', PROJECT).stdout, '')
            self.assertIn(TEAM, (local / PRIVATE).read_text())
            self.assertEqual(git(local, 'check-ignore', PRIVATE).returncode, 0)
            self.assertEqual((local / 'notes.txt').read_text(), 'keep local notes\n')
            self.assertEqual(git(local, 'rev-parse', 'refs/stash').stdout, stash)
            backup = next(root.glob('stems-coreai-backup.*'))
            self.assertTrue((backup / 'coreai-project.tar.gz').is_file())
            self.assertIn(TEAM, (backup / 'project-local.pbxproj').read_text())

    def test_other_conflicted_files_stop_recovery_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);local, _ = self.fixture(root, extra_conflict=True)
            index = git(local, 'ls-files', '-u').stdout
            head = git(local, 'rev-parse', 'HEAD').stdout
            result = self.recover(local)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('expected only the CoreAI project conflict', result.stdout)
            self.assertEqual(git(local, 'ls-files', '-u').stdout, index)
            self.assertEqual(git(local, 'rev-parse', 'HEAD').stdout, head)
            self.assertFalse((local / PRIVATE).exists())

    def test_missing_team_stops_after_backup_before_restoring_the_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);local, _ = self.fixture(root, team_present=False)
            index = git(local, 'ls-files', '-u').stdout
            result = self.recover(local)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Expected exactly one signing Team', result.stderr)
            self.assertEqual(git(local, 'ls-files', '-u').stdout, index)
            self.assertTrue(list(root.glob('stems-coreai-backup.*/coreai-project.tar.gz')))
            self.assertFalse((local / PRIVATE).exists())


if __name__ == '__main__':
    unittest.main()

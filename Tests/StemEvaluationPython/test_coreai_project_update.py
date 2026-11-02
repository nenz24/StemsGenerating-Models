"""Real local Git fast-forward recovery; no models or Apple tooling involved."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = 'prototypes/ios27-coreai/CoreAIProbe.xcodeproj/project.pbxproj'
SIGNING = 'prototypes/ios27-coreai/Signing.local.xcconfig'


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


class ProjectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='coreai update tests ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin = self.root / 'origin.git'
        self.seed = self.root / 'seed'
        self.local = self.root / 'local'
        subprocess.run(['git', 'init', '--bare', str(self.origin)], check=True, capture_output=True)
        subprocess.run(['git', 'init', '-b', 'main', str(self.seed)], check=True, capture_output=True)
        self.identity(self.seed)
        self.write(self.seed, PROJECT, 'base project\n')
        self.write(self.seed, 'prototypes/ios27-coreai/Signing.xcconfig', '#include? "Signing.local.xcconfig"\n')
        self.write(self.seed, '.gitignore', SIGNING + '\nmodels/\n.runtime-cache/\n')
        self.write(self.seed, 'notes.txt', 'base notes\n')
        git(self.seed, 'add', '.');git(self.seed, 'commit', '-m', 'base')
        git(self.seed, 'remote', 'add', 'origin', str(self.origin))
        git(self.seed, 'push', 'origin', 'main')
        subprocess.run(['git', 'clone', '-b', 'main', str(self.origin), str(self.local)], check=True, capture_output=True)
        self.identity(self.local)
        self.write(self.seed, PROJECT, 'new upstream resource build phase\n')
        git(self.seed, 'add', '.');git(self.seed, 'commit', '-m', 'resource update')
        git(self.seed, 'push', 'origin', 'main');git(self.local, 'fetch', 'origin', 'main')

    def identity(self, root):
        git(root, 'config', 'user.name', 'Fixture');git(root, 'config', 'user.email', 'fixture@example.test')

    def write(self, root, name, text):
        path = root / name;path.parent.mkdir(parents=True, exist_ok=True);path.write_text(text)
        return path

    def run_update(self):
        # Script outside checkout reproduces git-show -> temporary-file bootstrap.
        script = self.root / 'updater.py'
        script.write_bytes((ROOT / 'scripts/update_coreai_project.py').read_bytes())
        return subprocess.run([sys.executable, str(script)], cwd=self.local, capture_output=True, text=True)

    def test_dirty_project_backed_up_and_team_migrated_without_copying_models(self):
        original = 'local edits; DEVELOPMENT_TEAM = ABCDE12345;\n'
        self.write(self.local, PROJECT, original)
        self.write(self.local, 'models/never-copy-marker', 'large model placeholder')
        self.write(self.local, '.runtime-cache/cache-marker', 'cache placeholder')
        self.write(self.local, 'notes.txt', 'keep local notes\n')
        git(self.local, 'add', PROJECT, 'notes.txt')
        self.write(self.local, PROJECT, original + 'unstaged setting\n')
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(git(self.local, 'rev-parse', 'HEAD'), git(self.local, 'rev-parse', 'origin/main'))
        self.assertEqual((self.local / PROJECT).read_text(), 'new upstream resource build phase\n')
        self.assertIn('DEVELOPMENT_TEAM = ABCDE12345', (self.local / SIGNING).read_text())
        backups = list(self.root.glob('stems-coreai-project-backup-*'))
        self.assertEqual(len(backups), 1)
        backup = backups[0]
        self.assertEqual((backup / 'project-local.pbxproj').read_text(), original + 'unstaged setting\n')
        self.assertEqual((backup / 'project-staged.pbxproj').read_text(), original)
        self.assertFalse(any(p.name in ('models', 'ModelCandidates', '.runtime-cache') for p in backup.rglob('*')))
        self.assertEqual((self.local / 'models/never-copy-marker').read_text(), 'large model placeholder')
        self.assertEqual((self.local / '.runtime-cache/cache-marker').read_text(), 'cache placeholder')
        self.assertEqual((self.local / 'notes.txt').read_text(), 'keep local notes\n')
        self.assertEqual(git(self.local, 'diff', '--cached', '--name-only').strip(), b'notes.txt')
        self.assertEqual(git(self.local, 'stash', 'list'), b'')

    def test_existing_include_custom_settings_and_original_bytes_preserved(self):
        self.write(self.local, PROJECT, 'DEVELOPMENT_TEAM = ABCDE12345;\n')
        original = 'DEVELOPMENT_TEAM = OTHER12345\nCUSTOM_SETTING = keep\n'
        self.write(self.local, SIGNING, original)
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.local / SIGNING).read_text().startswith(original))
        self.assertTrue((self.local / SIGNING).read_text().endswith('DEVELOPMENT_TEAM = ABCDE12345\n'))
        backup = next(self.root.glob('stems-coreai-project-backup-*'))
        self.assertEqual((backup / 'Signing.local.xcconfig').read_text(), original)

    def test_no_explicit_project_team_keeps_existing_include_byte_for_byte(self):
        original = 'DEVELOPMENT_TEAM = ABCDE12345\nCUSTOM_SETTING = keep\n'
        self.write(self.local, SIGNING, original)
        self.write(self.local, PROJECT, 'formatting only\n')
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.local / SIGNING).read_text(), original)

    def test_no_team_no_include_still_updates_without_inventing_signing(self):
        self.write(self.local, PROJECT, 'formatting only\n')
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.local / SIGNING).exists())

    def test_ambiguous_signing_and_divergence_stop_without_modifying_project(self):
        original = 'DEVELOPMENT_TEAM = ABCDE12345; DEVELOPMENT_TEAM = ZYXWV98765;\n'
        self.write(self.local, PROJECT, original)
        result = self.run_update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('multiple project signing Teams', result.stderr)
        self.assertEqual((self.local / PROJECT).read_text(), original)
        self.write(self.local, PROJECT, 'committed local edit\n')
        git(self.local, 'add', PROJECT);git(self.local, 'commit', '-m', 'local branch divergence')
        result = self.run_update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot fast-forward', result.stderr)
        self.assertEqual((self.local / PROJECT).read_text(), 'committed local edit\n')
        self.assertEqual(list(self.root.glob('stems-coreai-project-backup-*')), [])

    def test_other_incoming_edited_file_blocks_before_project_restore(self):
        self.write(self.seed, 'notes.txt', 'new upstream notes\n')
        git(self.seed, 'add', '.');git(self.seed, 'commit', '-m', 'notes update')
        git(self.seed, 'push', 'origin', 'main');git(self.local, 'fetch', 'origin', 'main')
        original = 'DEVELOPMENT_TEAM = ABCDE12345;\n'
        self.write(self.local, PROJECT, original)
        self.write(self.local, 'notes.txt', 'local notes\n')
        result = self.run_update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('other edited files', result.stderr)
        self.assertEqual((self.local / PROJECT).read_text(), original)
        self.assertEqual((self.local / 'notes.txt').read_text(), 'local notes\n')


if __name__ == '__main__':
    unittest.main()

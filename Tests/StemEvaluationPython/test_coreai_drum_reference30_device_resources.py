"""Component-only build selection/transport. Asset fixtures are NONEXECUTABLE."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import coreai_drum_reference30_contract as C
import activate_coreai_drum_reference30 as A
import stage_coreai_app_models as S

REVIEW = ROOT / 'reviews/coreai-drums-reference30-export-completed/reference.json'


class Drum30DeviceResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates = self.root / 'repo/ModelCandidates'
        self.candidates.mkdir(parents=True)
        self.raw = REVIEW.read_bytes()
        self.reference = C.validate_reference(self.raw)

    def component_fixture(self):
        folder = self.candidates / C.BUNDLE
        folder.mkdir()
        (folder / 'reference.json').write_bytes(self.raw)
        for name in C.KEEP:
            path = folder / name; path.parent.mkdir(parents=True, exist_ok=True)
            if name in C.FIXTURES:
                with path.open('wb') as stream: stream.truncate(math.prod(C.FIXTURES[name]) * 4)
            else:
                path.write_bytes(b'NONEXECUTABLE resource transport fixture')
        (self.candidates / 'active-preview.json').write_text(json.dumps(
            {'schema_version': 1, 'resource_profile': C.PROFILE, 'audio_seconds': 30.}))
        return folder

    def test_actual_reviewed_report_contract_and_complete_build_whitelist(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), C.REFERENCE_SHA256)
        self.assertEqual(len(C.KEEP), 60)
        self.assertEqual(len(C.FIXTURES), 11)
        self.assertEqual(len(C.ASSETS), 16)
        self.assertEqual(len(C.DECODER_KEEP), 6)
        self.assertTrue(C.DECODER_KEEP.issubset(C.KEEP))
        self.assertEqual(len(self.reference['host_checks']), 6)
        self.assertTrue(all(m['parity_passed'] for m in self.reference['host_checks'].values()))
        self.assertNotIn('capture/source48.f32le', C.KEEP)
        self.assertEqual(self.reference['full_trajectory_parity'], 'NOT EVALUATED')
        with self.assertRaisesRegex(ValueError, 'differs'): C.validate_reference(self.raw + b' ')

    def test_component_build_excludes_other_models_and_reuses_unchanged_files(self):
        folder = self.component_fixture()
        (self.candidates / 'bass-baseline-kept-on-Mac').write_text('keep')
        destination = self.root / 'build/Probe.app/ModelCandidates'
        first = S.stage_resources(self.candidates, destination)
        self.assertEqual(first['selected_profile'], C.PROFILE)
        self.assertEqual(set(first['files']), {C.BUNDLE + '/' + n for n in C.KEEP | {'reference.json'}})
        self.assertEqual(len(first['files']), 61)
        self.assertFalse((destination / 'bass-baseline-kept-on-Mac').exists())
        self.assertEqual((self.candidates / 'bass-baseline-kept-on-Mac').read_text(), 'keep')
        self.assertEqual((folder / 'reference.json').read_bytes(), self.raw)
        second = S.stage_resources(self.candidates, destination)
        self.assertEqual((second['copied_files'], second['removed_files']), (0, 0))

    def test_bad_report_missing_or_truncated_fixture_preserves_existing_app(self):
        folder = self.component_fixture()
        destination = self.root / 'build/Probe.app/ModelCandidates'; destination.mkdir(parents=True)
        marker = destination / 'keep'; marker.write_text('old build')
        (folder / 'reference.json').write_bytes(self.raw + b' ')
        with self.assertRaises(ValueError): S.stage_resources(self.candidates, destination)
        (folder / 'reference.json').write_bytes(self.raw)
        path = folder / 'capture/initial-noise.f32le'; path.unlink()
        with self.assertRaises(ValueError): S.stage_resources(self.candidates, destination)
        path.write_bytes(b'wrong size')
        with self.assertRaisesRegex(ValueError, 'size mismatch'): S.stage_resources(self.candidates, destination)
        self.assertEqual(marker.read_text(), 'old build')

    def test_build_cli_runs_without_site_packages_or_export_dependencies(self):
        self.component_fixture()
        destination = self.root / 'build/Probe.app/ModelCandidates'
        for profile, count in [(C.PROFILE, 61), (C.DECODER_PROFILE, 7)]:
            (self.candidates / 'active-preview.json').write_text(json.dumps(
                {'schema_version': 1, 'resource_profile': profile, 'audio_seconds': 30.}))
            result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/stage_coreai_app_models.py'),
                '--source', str(self.candidates), '--destination', str(destination), '--dry-run'],
                check=True, capture_output=True, text=True)
            self.assertIn(profile, result.stdout)
            self.assertIn(f'{count} files', result.stdout)
            self.assertFalse(destination.exists())

    def test_decoder_isolation_app_omits_all_DiT_APG_but_preserves_original_assets(self):
        folder = self.component_fixture()
        destination = self.root / 'build/Probe.app/ModelCandidates'
        S.stage_resources(self.candidates, destination)
        (self.candidates / 'active-preview.json').write_text(json.dumps(
            {'schema_version': 1, 'resource_profile': C.DECODER_PROFILE, 'audio_seconds': 30.}))
        plan = S.stage_resources(self.candidates, destination)
        self.assertEqual(len(plan['files']), 7)
        self.assertEqual(plan['removed_files'], 54)
        self.assertEqual(set(plan['files']), {C.BUNDLE + '/' + n for n in C.DECODER_KEEP | {'reference.json'}})
        self.assertFalse((destination / C.BUNDLE / 'front.aimodel').exists())
        self.assertFalse((destination / C.BUNDLE / 'apg-euler.aimodel').exists())
        self.assertTrue((folder / 'front.aimodel/main.mlirb').exists())
        self.assertTrue((folder / 'apg-euler.aimodel/main.mlirb').exists())
        self.assertEqual((destination / C.BUNDLE / 'reference.json').read_bytes(), self.raw)

    def test_selector_rejects_wrong_duration_or_generation_diagnostic_flag(self):
        self.component_fixture()
        path = self.candidates / 'active-preview.json'
        for profile in [C.PROFILE, C.DECODER_PROFILE]:
            for duration, flag in [(10.24, False), (30., True)]:
                path.write_text(json.dumps({'schema_version': 1, 'resource_profile': profile,
                                            'audio_seconds': duration, 'diagnostic_trajectory': flag}))
                with self.assertRaisesRegex(ValueError, 'selector'): S.plan_resources(self.candidates)

    def test_activation_backs_up_selector_preserves_source_and_is_idempotent(self):
        source = self.root / 'original-native'; source.mkdir()
        (source / 'reference.json').write_bytes(self.raw)
        selector = self.candidates / 'active-preview.json'; previous = b'{"old":"exact bytes"}\n'
        selector.write_bytes(previous)
        backups = self.root / 'backups'
        # Full learned binary integrity is separately enforced by verify_bundle;
        # this test exercises selector/copy lifecycle with metadata-only fixtures.
        with patch.object(A, 'ROOT', self.root), patch.object(A, 'verify_bundle', return_value=self.reference) as verify:
            target = A.activate(source, self.candidates, backups)
            self.assertEqual(verify.call_count, 2)
            self.assertEqual((source / 'reference.json').read_bytes(), self.raw)
            self.assertEqual((target / 'reference.json').read_bytes(), self.raw)
            saved = list(backups.glob('selector-*/active-preview.json'))
            self.assertEqual(len(saved), 1); self.assertEqual(saved[0].read_bytes(), previous)
            with patch.object(A.shutil, 'copytree') as copy:
                self.assertEqual(A.activate(source, self.candidates, backups), target)
                copy.assert_not_called()
            self.assertEqual(len(list(backups.iterdir())), 1)

    def test_invalid_source_existing_target_and_copy_failure_preserve_selection(self):
        source = self.root / 'native'; source.mkdir()
        selector = self.candidates / 'active-preview.json'; selector.write_bytes(b'old selection')
        (source / 'reference.json').write_bytes(self.raw + b' ')
        with patch.object(A, 'verify_bundle') as verify:
            with self.assertRaises(ValueError): A.activate(source, self.candidates, self.root / 'backups')
            verify.assert_not_called()
        (source / 'reference.json').write_bytes(self.raw)
        with patch.object(A, 'ROOT', self.root), patch.object(A, 'verify_bundle'), \
                patch.object(A.shutil, 'copytree', side_effect=OSError('interrupted copy')):
            with self.assertRaises(OSError): A.activate(source, self.candidates, self.root / 'backups')
        target = self.candidates / C.BUNDLE; target.mkdir(); (target / 'keep').write_text('original')
        with patch.object(A, 'verify_bundle'):
            with self.assertRaises(FileNotFoundError): A.activate(source, self.candidates, self.root / 'backups')
        self.assertEqual(selector.read_bytes(), b'old selection')
        self.assertEqual((target / 'keep').read_text(), 'original')

    def test_decoder_selection_preserves_component_selector_and_can_switch_back(self):
        source = self.root / 'native'; source.mkdir()
        (source / 'reference.json').write_bytes(self.raw)
        selector = self.candidates / 'active-preview.json'
        component = json.dumps({'schema_version': 1, 'resource_profile': C.PROFILE, 'audio_seconds': 30.}).encode()
        selector.write_bytes(component)
        backups = self.root / 'backups'
        with patch.object(A, 'ROOT', self.root), patch.object(A, 'verify_bundle'):
            A.activate(source, self.candidates, backups, decoder_only=True)
            self.assertEqual(json.loads(selector.read_bytes())['resource_profile'], C.DECODER_PROFILE)
            saved = list(backups.glob('selector-*/active-preview.json'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), component)
            A.activate(source, self.candidates, backups)
            self.assertEqual(json.loads(selector.read_bytes())['resource_profile'], C.PROFILE)
            self.assertEqual((source / 'reference.json').read_bytes(), self.raw)


if __name__ == '__main__':
    unittest.main()

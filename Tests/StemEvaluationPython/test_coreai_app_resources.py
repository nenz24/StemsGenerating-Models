"""Resource selection/incremental-copy tests, using NONEXECUTABLE file fixtures."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import stage_coreai_app_models as M


class AppResourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'repository/ModelCandidates'
        self.source.mkdir(parents=True)
        self.destination = self.root / 'build/CoreAIProbe.app/ModelCandidates'

    def write_bundle(self, name, files, **metadata):
        folder = self.source / name
        folder.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in files:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'NONEXECUTABLE packaging integrity fixture: ' + name.encode())
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        ref = {'artifact_sha256': hashes, **metadata}
        (folder / 'reference.json').write_text(json.dumps(ref, indent=2) + '\n')
        return ref

    def pipeline(self, aot=False):
        self.write_bundle('source-context', ['encoder.aimodel/weights', 'audio.f32le'])
        self.write_bundle('bass-conditioning', ['qwen.aimodel/weights', 'text_ids.f32le'])
        self.write_bundle('bass-preview-512', sorted(M.PREVIEW_FIXTURES) + ['front.aimodel/weights'],
                          source_reference_sha256=M.digest(self.source / 'source-context/reference.json'),
                          condition_reference_sha256=M.digest(self.source / 'bass-conditioning/reference.json'))
        (self.source / 'legacy-probe.aimodel').write_bytes(b'NONEXECUTABLE old test fixture')
        if aot:
            names = ['front', *[f'blocks-{i:02d}-{i+1:02d}' for i in range(0, 24, 2)],
                     'tail', 'apg-euler', 'decoder']
            assets = {n + '.aimodel': n + '.h18p.aimodelc' for n in names}
            # Include both directory-shaped and file-shaped compiled fixtures.
            files = [v + '/payload' if i % 2 else v for i, v in enumerate(assets.values())]
            self.write_bundle('bass-preview-512-aot', files, schema_version=1,
                              profile='bass-preview-512-aot', architecture='h18p', platform='iOS',
                              preferred_compute='none', runtime_policy='default', compiled_asset_count=16,
                              source_reference_sha256=M.digest(self.source / 'bass-preview-512/reference.json'), assets=assets)

    def test_auto_source_excludes_legacy_and_reuses_unchanged_files(self):
        self.pipeline()
        result = M.stage_resources(self.source, self.destination)
        self.assertEqual(result['selected_profile'], 'bass-source')
        self.assertIn('bass-preview-512/front.aimodel/weights', result['files'])
        self.assertFalse((self.destination / 'legacy-probe.aimodel').exists())
        stamp_time = (self.destination / M.STAMP).stat().st_mtime_ns
        second = M.stage_resources(self.source, self.destination)
        self.assertEqual(second['copied_files'], 0)
        self.assertEqual((self.destination / M.STAMP).stat().st_mtime_ns, stamp_time)
        self.assertEqual(second['removed_files'], 0)

    def test_aot_keeps_original_preview_identity_and_every_numerical_fixture(self):
        self.pipeline(aot=True)
        original = (self.source / 'bass-preview-512/reference.json').read_bytes()
        result = M.stage_resources(self.source, self.destination, 'bass-aot')
        self.assertEqual(result['selected_profile'], 'bass-aot')
        self.assertEqual((self.destination / 'bass-preview-512/reference.json').read_bytes(), original)
        self.assertFalse((self.destination / 'bass-preview-512/front.aimodel').exists())
        self.assertEqual({k.split('/', 1)[1] for k in result['files'] if k.startswith('bass-preview-512/')},
                         M.PREVIEW_FIXTURES | {'reference.json'})
        self.assertTrue((self.destination / 'bass-preview-512-aot/front.h18p.aimodelc').is_file())
        self.assertTrue((self.destination / 'bass-conditioning/qwen.aimodel/weights').is_file())
        self.assertTrue((self.source / 'bass-preview-512/front.aimodel/weights').is_file())
        # Explicit source mode never bundles AOT, even when prepared on the Mac.
        source = M.plan_resources(self.source, 'bass-source')
        self.assertFalse(any(k.startswith('bass-preview-512-aot/') for k in source['files']))

    def test_auto_uses_cpu_source_even_when_aot_is_prepared(self):
        self.pipeline(aot=True)
        # Reproduce an existing AOT app before upgrading to the CPU default.
        M.stage_resources(self.source, self.destination, 'bass-aot')
        self.assertTrue((self.destination / 'bass-preview-512-aot').exists())
        result = M.stage_resources(self.source, self.destination)
        self.assertEqual(result['selected_profile'], 'bass-source')
        self.assertTrue((self.destination / 'bass-preview-512/front.aimodel/weights').is_file())
        self.assertFalse((self.destination / 'bass-preview-512-aot').exists())
        self.assertTrue((self.source / 'bass-preview-512-aot/front.h18p.aimodelc').is_file())
        # An unused stale AOT manifest also must not block a CPU-only build.
        (self.source / 'bass-preview-512-aot/reference.json').write_text('unused AOT fixture')
        self.assertEqual(M.plan_resources(self.source)['selected_profile'], 'bass-source')

    def test_missing_selected_payload_and_stale_aot_fail_before_pruning(self):
        self.pipeline(aot=True)
        self.destination.mkdir(parents=True)
        marker = self.destination / 'keep-until-valid'
        marker.write_text('previous build')
        payload = self.source / 'bass-preview-512-aot/front.h18p.aimodelc'
        payload.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or external'):
            M.stage_resources(self.source, self.destination, 'bass-aot')
        self.assertEqual(marker.read_text(), 'previous build')
        payload.write_bytes(b'NONEXECUTABLE fixture')
        ref = self.source / 'bass-preview-512/reference.json'
        ref.write_text(ref.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'identity'):
            M.stage_resources(self.source, self.destination, 'bass-aot')
        self.assertEqual(marker.read_text(), 'previous build')

    def test_profile_switch_prunes_only_build_products_preserving_source_and_cache(self):
        self.pipeline(aot=True)
        cache = self.root / 'repository/.runtime-cache/weights'
        cache.parent.mkdir();cache.write_text('checkpoint fixture')
        M.stage_resources(self.source, self.destination, 'all')
        self.assertTrue((self.destination / 'legacy-probe.aimodel').exists())
        result = M.stage_resources(self.source, self.destination, 'bass-aot')
        self.assertGreater(result['removed_files'], 0)
        self.assertFalse((self.destination / 'legacy-probe.aimodel').exists())
        self.assertFalse((self.destination / 'bass-preview-512/front.aimodel').exists())
        self.assertTrue((self.source / 'legacy-probe.aimodel').is_file())
        self.assertEqual(cache.read_text(), 'checkpoint fixture')

    def test_changed_manifest_forces_copy_even_when_size_and_mtime_match(self):
        self.pipeline()
        M.stage_resources(self.source, self.destination)
        payload = self.source / 'source-context/audio.f32le'
        stamp = payload.stat()
        payload.write_bytes(b'x' * stamp.st_size)
        os.utime(payload, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        ref_path = self.source / 'source-context/reference.json'
        ref = json.loads(ref_path.read_text())
        ref['artifact_sha256']['audio.f32le'] = hashlib.sha256(payload.read_bytes()).hexdigest()
        ref_path.write_text(json.dumps(ref))
        preview_path = self.source / 'bass-preview-512/reference.json'
        preview = json.loads(preview_path.read_text())
        preview['source_reference_sha256'] = M.digest(ref_path)
        preview_path.write_text(json.dumps(preview))
        result = M.stage_resources(self.source, self.destination)
        self.assertEqual(result['copied_files'], 3)
        self.assertEqual((self.destination / 'source-context/audio.f32le').read_bytes(), payload.read_bytes())

    def test_unsafe_output_external_payload_and_manifest_traversal_rejected(self):
        self.pipeline()
        for destination in (self.source, self.root / 'unsafe', self.source / 'nested.app/ModelCandidates'):
            with self.assertRaises(ValueError):
                M.stage_resources(self.source, destination)
        outside = self.root / 'outside'
        outside.write_text('never modify')
        payload = self.source / 'source-context/audio.f32le'
        payload.unlink();payload.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'external'):
            M.stage_resources(self.source, self.destination)
        ref_path = self.source / 'source-context/reference.json'
        ref = json.loads(ref_path.read_text());ref['artifact_sha256']['../outside'] = '0' * 64
        ref_path.write_text(json.dumps(ref))
        with self.assertRaisesRegex(ValueError, 'Invalid artifact'):
            M.stage_resources(self.source, self.destination)
        self.assertEqual(outside.read_text(), 'never modify')

    def test_dry_run_and_fresh_clone_do_not_require_large_models(self):
        plan = M.stage_resources(self.source, self.destination, dry_run=True)
        self.assertEqual(plan['selected_profile'], 'runtime')
        self.assertEqual(plan['included_bytes'], 0)
        self.assertFalse(self.destination.exists())
        model = self.source / 'affine.aimodel'
        model.mkdir();(model / 'model').write_text('NONEXECUTABLE runtime fixture')
        (self.source / 'affine-native-reference.json').write_text('{}')
        result = M.stage_resources(self.source, self.destination)
        self.assertEqual(set(result['files']), {'affine.aimodel/model', 'affine-native-reference.json'})

    def test_xcode_runs_stager_instead_of_copying_entire_candidate_folder(self):
        from openstep_parser import OpenStepDecoder
        project = ROOT / 'prototypes/ios27-coreai/CoreAIProbe.xcodeproj/project.pbxproj'
        objects = OpenStepDecoder.ParseFromString(project.read_text())['objects']
        self.assertEqual(objects['A40000000000000000000003']['files'], [])
        phase = objects['A40000000000000000000004']
        self.assertEqual(phase['isa'], 'PBXShellScriptBuildPhase')
        self.assertIn('--profile "${COREAI_MODEL_PROFILE:-auto}"', phase['shellScript'])
        self.assertIn('$(SRCROOT)/ModelCandidates', phase['inputPaths'])
        self.assertIn('$(TARGET_BUILD_DIR)/$(UNLOCALIZED_RESOURCES_FOLDER_PATH)/ModelCandidates', phase['outputPaths'])
        for config in ('A70000000000000000000003', 'A70000000000000000000004'):
            self.assertEqual(objects[config]['buildSettings']['COREAI_MODEL_PROFILE'], 'auto')
        self.assertNotIn('A10000000000000000000004', objects)


if __name__ == '__main__':
    unittest.main()

"""Shared-weight preview selection and integrity; no Apple/model execution."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import coreai_instrument_preview30_contract as C
import prepare_coreai_instrument_preview30 as P
import stage_coreai_app_models as S
import test_coreai_drum_reference30_generation as generation_tests


def reference(hashes):
    return {'schema_version': 1, 'profile': C.BUNDLE, 'status': 'prepared_instrument_preview30',
            'roles': list(C.ROLES), 'preset_sha256': C.PRESETS, 'condition_tokens': 127,
            'condition_shape': [1, 127, 2048], 'mask_shape': [1, 127],
            'capture_report_sha256': C.CAPTURE_REPORT_SHA256, 'host_drums_control_exact': True,
            'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
            'sampler_weight_storage_dtype': 'bfloat16', 'compute_dtype': 'float32', 'vae_dtype': 'float32',
            'native_exports': 0, 'baseline_modified': False, 'padding_tokens': 0, 'truncated_tokens': 0,
            'full_song_native_readiness': 'NOT EVALUATED', 'bass_guitar_numerical_parity': 'NOT EVALUATED',
            'artifact_sha256': hashes}


class ContractTests(unittest.TestCase):
    def test_geometry_precision_original_gates_and_scope_are_fixed(self):
        ref = reference({n: '0' * 64 for n in C.FILES})
        self.assertEqual(C.validate(ref), ref)
        for key, value in [('condition_tokens', 123), ('roles', ['drums']), ('host_drums_control_exact', False),
                           ('absolute_tolerance', .001), ('sampler_weight_storage_dtype', 'float32'),
                           ('full_song_native_readiness', 'PASS'), ('bass_guitar_numerical_parity', 'PASS')]:
            with self.assertRaises(ValueError): C.validate({**ref, key: value})
        changed = copy.deepcopy(ref)
        changed['preset_sha256']['bass'] = C.PRESETS['drums']
        with self.assertRaises(ValueError): C.validate(changed)
        with self.assertRaises(ValueError): C.validate({**ref, 'artifact_sha256': {'../external': '0'*64}})

    def test_nonfinite_wrong_size_wrong_hash_and_padding_mask_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'condition-mask.f32le'
            def write(values): path.write_bytes(struct.pack('<' + 'f'*len(values), *values))
            write([1]*127)
            P.checked_tensor(path, [1, 127], P.sha256(path))
            with self.assertRaises(ValueError): P.checked_tensor(path, [1, 126], P.sha256(path))
            with self.assertRaises(ValueError): P.checked_tensor(path, [1, 127], '0'*64)
            for invalid in (float('nan'), float('inf'), 0, .5):
                write([invalid] + [1]*126)
                with self.assertRaises(ValueError): P.checked_tensor(path, [1, 127], P.sha256(path))

    def test_default_and_help_do_not_read_models_or_write_output(self):
        with tempfile.TemporaryDirectory() as folder:
            for flags in ([], ['--help']):
                r = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/prepare_coreai_instrument_preview30.py'), *flags],
                                   cwd=folder, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(list(Path(folder).iterdir()), [])


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = generation_tests.Reference30GenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.fixtures()
        self.source = self.fixture.source
        self.bundle = self.source/C.BUNDLE
        self.bundle.mkdir()
        self.hashes = {}
        for name, shape in C.DATA_SHAPES.items():
            path = self.bundle/name
            path.parent.mkdir(exist_ok=True)
            value = 1. if name.endswith('-mask.f32le') else 0.
            path.write_bytes(struct.pack('<f', value)*math.prod(shape))
            self.hashes[name] = P.sha256(path)
        (self.bundle/'conditioning-report.json').write_text('SYNTHETIC TRANSPORT FIXTURE; not neural output')
        self.hashes['conditioning-report.json'] = P.sha256(self.bundle/'conditioning-report.json')
        (self.bundle/'reference.json').write_text(json.dumps(reference(self.hashes)))
        (self.source/'active-preview.json').write_text(json.dumps({'schema_version':1, 'resource_profile':C.PROFILE, 'audio_seconds':30.}))

    def test_shared_sampler_not_duplicated_and_qwen_source_full_decoder_omitted(self):
        self.assertEqual(P.verify_bundle(self.bundle)['roles'], list(C.ROLES))
        out = self.fixture.root/'build/Probe.app/ModelCandidates'
        result = S.stage_resources(self.source, out)
        self.assertEqual(result['selected_profile'], C.PROFILE)
        self.assertEqual(len(result['files']), 73)
        self.assertEqual(sum(n.endswith('main.mlirb') for n in result['files']), 16)
        self.assertFalse(any('qwen' in n.lower() or 'source-context.aimodel' in n for n in result['files']))
        self.assertFalse(any('/decoder.aimodel/' in n and P.SAMPLER_BUNDLE in n for n in result['files']))
        self.assertEqual(S.stage_resources(self.source, out)['copied_files'], 0)

    def test_invalid_new_bundle_preserves_previous_app(self):
        out = self.fixture.root/'build/Probe.app/ModelCandidates'
        out.mkdir(parents=True)
        marker = out/'keep'
        marker.write_text('previous build')
        ref_path = self.bundle/'reference.json'
        good = ref_path.read_bytes()
        invalid = json.loads(good)
        invalid['absolute_tolerance'] = .001
        ref_path.write_text(json.dumps(invalid))
        with self.assertRaises(ValueError): S.stage_resources(self.source, out)
        ref_path.write_bytes(good)
        (self.bundle/'guitar/condition.f32le').write_bytes(b'wrong shape')
        with self.assertRaises(ValueError): S.stage_resources(self.source, out)
        self.assertEqual(marker.read_text(), 'previous build')

    def test_changed_tensor_or_external_reference_rejected(self):
        path = self.bundle/'bass/condition.f32le'
        with path.open('r+b') as stream: stream.write(struct.pack('<f', 1.))
        with self.assertRaises(ValueError): P.verify_bundle(self.bundle)
        path.unlink()
        external = self.fixture.root/'external'
        external.write_bytes(b'external')
        path.symlink_to(external)
        with self.assertRaises(ValueError): P.verify_bundle(self.bundle)

    def test_failed_host_or_model_verification_never_changes_selector(self):
        selector = self.source/'active-preview.json'
        old = selector.read_bytes()
        with patch.object(P, 'verify_host', side_effect=ValueError('host mismatch')):
            with self.assertRaises(ValueError): P.prepare(self.fixture.root/'host', self.source, True)
        with patch.object(P, 'verify_host', return_value={}), \
                patch('prepare_coreai_drum_reference30.verify_bundle', side_effect=ValueError('model mismatch')):
            with self.assertRaises(ValueError): P.prepare(self.fixture.root/'host', self.source, True)
        self.assertEqual(selector.read_bytes(), old)

    def test_copy_and_activate_then_reuse_without_changing_host_or_models(self):
        from activate_coreai_drum_reference30 import select_profile
        host = self.fixture.root/'host'
        self.bundle.rename(host)
        (host/'conditioning-report.json').rename(host/'report.json')
        old = b'{"prior":"source component"}\n'
        selector = self.source/'active-preview.json'
        selector.write_bytes(old)
        config = P.configuration(P.sha256(self.source/P.DECODER_BUNDLE/'reference.json'))
        (self.source/P.CONFIG).write_text(json.dumps(config, indent=2)+'\n')
        before = {str(p.relative_to(host)): P.sha256(p) for p in host.rglob('*') if p.is_file()}
        manifest = {'artifact_sha256': self.hashes}
        backups = self.fixture.root/'backups'
        def choose(candidates, profile): select_profile(candidates, profile, backups)
        with patch.object(P, 'ROOT', self.fixture.root/'repo'), patch.object(P, 'verify_host', return_value=manifest), \
                patch('prepare_coreai_drum_reference30.verify_bundle'), \
                patch('prepare_coreai_drum_reference30_window_decoder.verify_bundle'), \
                patch('activate_coreai_drum_reference30.select_profile', side_effect=choose):
            prepared = P.prepare(host, self.source, True)
            self.assertEqual(prepared, P.verify_bundle(self.bundle))
            self.assertEqual(next(backups.glob('*/active-preview.json')).read_bytes(), old)
            self.assertEqual(json.loads(selector.read_text())['resource_profile'], C.PROFILE)
            self.assertEqual(P.prepare(host, self.source, True), prepared)
            self.assertEqual(before, {str(p.relative_to(host)): P.sha256(p) for p in host.rglob('*') if p.is_file()})
            (self.source/P.CONFIG).write_bytes(b'other configuration')
            with self.assertRaises(FileExistsError): P.prepare(host, self.source, True)
            self.assertEqual((self.source/P.CONFIG).read_bytes(), b'other configuration')


if __name__ == '__main__': unittest.main()

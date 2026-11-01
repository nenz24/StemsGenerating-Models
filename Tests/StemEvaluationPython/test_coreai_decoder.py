"""Integrity/preservation checks; fixture directories are not executable CoreAI models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('decoder_export', ROOT / 'scripts/export_coreai_decoder.py')
MODULE = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class DecoderPreparationTests(unittest.TestCase):
    def fixture(self, root, component='decoder'):
        folder = root / 'fixture';folder.mkdir()
        encoding = component == 'encoder'
        input_name, output_name = ('audio', 'latent') if encoding else ('latent', 'audio')
        input_shape, output_shape = ([1, 2, 48000], [1, 64, 25]) if encoding else ([1, 64, 1], [1, 2, 1920])
        data = {f'{input_name}.f32le': bytes(input_shape[1] * input_shape[2] * 4),
                f'{output_name}-reference.f32le': bytes(output_shape[1] * output_shape[2] * 4),
                f'{component}.aimodel/main.hash': b'fixture only',
                f'{component}.aimodel/main.mlirb': b'not an executable model',
                f'{component}.aimodel/metadata.json': b'{}'}
        for name, value in data.items():
            p = folder / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(value)
        ref = {'schema_version': 1, 'asset_file': f'{component}.aimodel',
               'input_shape': input_shape, 'output_shape': output_shape,
               'input_file': f'{input_name}.f32le', 'expected_file': f'{output_name}-reference.f32le',
               'scalar_type': 'float32', 'byte_order': 'little',
               'checkpoint': {'digest': 'fixture-checkpoint'}, 'checkpoint_full_sha256_verified': True,
               'artifact_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}}
        if encoding:ref.update(component='encoder', posterior_mode='mean')
        (folder / 'reference.json').write_text(json.dumps(ref))
        return folder, ref, {'digest': 'fixture-checkpoint'}

    def test_matching_size_but_wrong_checkpoint_hash_is_rejected_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights';p.write_bytes(b'keep')
            record = {'size': 4, 'digest': hashlib.sha256(b'keep').hexdigest()}
            MODULE.verify_checkpoint(p, record)
            with self.assertRaises(ValueError):
                MODULE.verify_checkpoint(p, dict(record, digest=hashlib.sha256(b'bad!').hexdigest()))
            self.assertEqual(p.read_bytes(), b'keep')

    def test_bundle_digest_coverage_and_reference_size_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, ref, record = self.fixture(Path(tmp))
            MODULE.verify_bundle(folder, record)
            binary = folder / 'audio-reference.f32le';binary.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)
        with tempfile.TemporaryDirectory() as tmp:
            folder, ref, record = self.fixture(Path(tmp))
            ref['artifact_sha256'].pop('decoder.aimodel/main.mlirb')
            (folder / 'reference.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)

    def test_manifest_cannot_request_files_outside_the_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);folder, ref, record = self.fixture(root)
            outside = root / 'outside';outside.write_bytes(b'private')
            ref['artifact_sha256']['../outside'] = hashlib.sha256(b'private').hexdigest()
            (folder / 'reference.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)
            self.assertEqual(outside.read_bytes(), b'private')

    def test_existing_output_is_preserved_before_any_download_or_optional_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'existing';marker.write_text('preserve')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_coreai_decoder.py'),
                                     '--out', tmp], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Output already exists', result.stderr)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_verify_only_rejects_a_different_requested_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, ref, _ = self.fixture(Path(tmp))
            manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())
            ref['checkpoint'] = next(r for r in manifest['files']
                                     if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
            reference = folder / 'reference.json'
            reference.write_text(json.dumps(ref))
            original = reference.read_bytes()
            for frames, expected_code in [(25, 1), (1, 0)]:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_coreai_decoder.py'),
                                         '--out', str(folder), '--frames', str(frames), '--verify-only'],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                if frames == 25:self.assertIn('different window', result.stderr)
                self.assertEqual(reference.read_bytes(), original)

    def test_encoder_requires_mean_latent_contract_and_its_own_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, ref, record = self.fixture(Path(tmp), component='encoder')
            MODULE.verify_bundle(folder, record, component='encoder')
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)
            for key, value in [('posterior_mode', 'sample'), ('output_shape', [1, 128, 25])]:
                changed = dict(ref, **{key: value})
                (folder / 'reference.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record, component='encoder')
            (folder / 'reference.json').write_text(json.dumps(ref))
            (folder / 'latent-reference.f32le').write_bytes(b'bad')
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record, component='encoder')


if __name__ == '__main__':
    unittest.main()

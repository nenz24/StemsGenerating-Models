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
    def fixture(self, root):
        folder = root / 'fixture';folder.mkdir()
        data = {'latent.f32le': bytes(64 * 4), 'audio-reference.f32le': bytes(2 * 1920 * 4),
                'decoder.aimodel/main.hash': b'fixture only',
                'decoder.aimodel/main.mlirb': b'not an executable model',
                'decoder.aimodel/metadata.json': b'{}'}
        for name, value in data.items():
            p = folder / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(value)
        ref = {'schema_version': 1, 'asset_file': 'decoder.aimodel',
               'input_shape': [1, 64, 1], 'output_shape': [1, 2, 1920],
               'input_file': 'latent.f32le', 'expected_file': 'audio-reference.f32le',
               'scalar_type': 'float32', 'byte_order': 'little',
               'checkpoint': {'digest': 'fixture-checkpoint'}, 'checkpoint_full_sha256_verified': True,
               'artifact_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}}
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


if __name__ == '__main__':
    unittest.main()

"""DiT preparation integrity tests; placeholder files are not executable models."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import export_coreai_dit_block as MODULE


class DiTBlockPreparationTests(unittest.TestCase):
    def fixture(self, folder):
        folder.mkdir()
        shapes = {f'{name}.f32le': shape for name, shape in MODULE.INPUT_SHAPES.items()}
        shapes['hidden-reference.f32le'] = MODULE.OUTPUT_SHAPE
        data = {name: bytes(math.prod(shape) * 4) for name, shape in shapes.items()}
        data.update({'dit-block.aimodel/main.hash': b'fixture only',
                     'dit-block.aimodel/main.mlirb': b'not executable',
                     'dit-block.aimodel/metadata.json': b'{}'})
        for name, value in data.items():
            p = folder / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(value)
        record = {'digest': 'fixture checkpoint'}
        ref = {'schema_version': 1, 'asset_file': 'dit-block.aimodel', 'checkpoint': record,
               'checkpoint_full_sha256_verified': True, 'layer_prefix': 'decoder.layers.0.',
               'input_shapes': MODULE.INPUT_SHAPES, 'output_shape': MODULE.OUTPUT_SHAPE,
               'function_name': 'main', 'output_name': 'hidden_out', 'scalar_type': 'float32', 'byte_order': 'little',
               'artifact_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}}
        self.write_reference(folder, ref)
        return ref, record

    def write_reference(self, folder, ref):
        (folder / 'reference.json').write_text(json.dumps(ref))

    def test_manifest_requires_all_seven_inputs_and_correct_binary_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'bundle';ref, record = self.fixture(folder)
            MODULE.verify_bundle(folder, record)
            del ref['artifact_sha256']['cross_mask.f32le']
            self.write_reference(folder, ref)
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'bundle';ref, record = self.fixture(folder)
            (folder / 'hidden-reference.f32le').write_bytes(b'bad')
            ref['artifact_sha256']['hidden-reference.f32le'] = hashlib.sha256(b'bad').hexdigest()
            self.write_reference(folder, ref)
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)

    def test_wrong_layer_checkpoint_or_unverified_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'bundle';ref, record = self.fixture(folder)
            for key, value in [('layer_prefix', 'decoder.layers.1.'), ('checkpoint_full_sha256_verified', False),
                               ('checkpoint', {'digest': 'wrong'})]:
                self.write_reference(folder, dict(ref, **{key: value}))
                with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)

    def test_existing_output_and_modified_source_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);marker = folder / 'marker';marker.write_text('keep')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_coreai_dit_block.py'), '--out', tmp],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Output already exists', result.stderr)
            self.assertEqual(marker.read_text(), 'keep')
            source = folder / 'source.py';source.write_text('changed')
            with self.assertRaises(ValueError):
                MODULE.verified_source({'filename': 'source.py', 'size': 1, 'digest': 'wrong'}, folder)
            self.assertEqual(source.read_text(), 'changed')

    def test_manifest_cannot_read_outside_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'bundle';ref, record = self.fixture(folder)
            outside = Path(tmp) / 'outside';outside.write_bytes(b'keep')
            ref['artifact_sha256']['../outside'] = hashlib.sha256(b'keep').hexdigest()
            self.write_reference(folder, ref)
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder, record)
            self.assertEqual(outside.read_bytes(), b'keep')


if __name__ == '__main__':unittest.main()

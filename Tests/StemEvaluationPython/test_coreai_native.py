"""Native export/package checks; these do not claim iPhone model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'prototypes/ios27-coreai/ModelCandidates'


class CoreAINativeTests(unittest.TestCase):
    def test_native_files_match_exporter_manifest(self):
        ref = json.loads((ASSETS / 'affine-native-reference.json').read_text())
        self.assertEqual(ref['asset_file'], 'affine.aimodel')
        asset = ASSETS / ref['asset_file']
        actual = {str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in asset.rglob('*') if p.is_file()}
        self.assertEqual(ref['artifact_sha256'], actual)
        self.assertGreater((asset / 'main.hash').stat().st_size, 0)
        self.assertGreater((asset / 'main.mlirb').stat().st_size, 0)
        metadata = json.loads((asset / 'metadata.json').read_text())
        self.assertEqual(metadata['producer'], 'coreai-core ' + ref['coreai_core_version'])
        self.assertFalse(ref['coreai_device_inference_verified'])
        self.assertEqual(ref['reference_execution'], 'PyTorch CPU')

    def test_existing_asset_is_preserved_before_importing_export_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'affine.aimodel';asset.mkdir()
            marker = asset / 'existing';marker.write_text('preserve')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_coreai_native.py'),
                                     '--out', tmp], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing to overwrite', result.stderr)
            self.assertEqual(marker.read_text(), 'preserve')

    @unittest.skipUnless(importlib.util.find_spec('coreai_torch'), 'optional native exporter')
    def test_real_export_produces_native_hash_and_pytorch_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_coreai_native.py'),
                                     '--out', tmp], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            ref = json.loads((Path(tmp) / 'affine-native-reference.json').read_text())
            self.assertEqual(ref['expected_values'], [0.25, 0.75, -0.75, 2.25])
            self.assertEqual(ref['input_shape'], [1, 4])
            self.assertEqual(ref['exporter'], 'coreai-torch 0.4.3')
            self.assertTrue((Path(tmp) / 'affine.aimodel/main.hash').is_file())
            self.assertFalse(ref['coreai_device_inference_verified'])


if __name__ == '__main__':
    unittest.main()

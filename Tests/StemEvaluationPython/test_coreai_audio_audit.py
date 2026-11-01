"""Audit invariants and preservation; no fake CoreAI runtime or phone benchmark."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('audio_audit', ROOT / 'scripts/audit_coreai_audio.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AudioAuditTests(unittest.TestCase):
    def test_header_inventory_counts_scalars_and_skips_metadata(self):
        header = {'__metadata__': {'note': 'fixture'},
                  'weight': {'dtype': 'BF16', 'shape': [2, 3]},
                  'scalar': {'dtype': 'F32', 'shape': []}}
        self.assertEqual(AUDIT.header_inventory(header), {'BF16': 6, 'F32': 1})
        with self.assertRaises(ValueError):
            AUDIT.header_inventory({'bad': {'dtype': 'F32', 'shape': [-1, 4]}})

    def test_existing_audit_is_preserved_before_importing_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report.json';report.write_text('preserve evidence')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_coreai_audio.py'),
                                     '--out', tmp], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing to overwrite prior audit', result.stderr)
            self.assertEqual(report.read_text(), 'preserve evidence')

    def test_published_report_keeps_export_header_and_device_evidence_distinct(self):
        r = json.loads((ROOT / 'reviews/coreai-ace-component-audit/report.json').read_text())
        self.assertEqual(sum(r['checkpoint_headers']['vae']['parameters_by_dtype'].values()),
                         r['vae_parameter_counts']['full'])
        self.assertEqual(r['vae_parameter_counts']['encoder'] + r['vae_parameter_counts']['decoder'],
                         r['vae_parameter_counts']['full'])
        self.assertEqual(r['latent_rate_hz'] * r['downsampling_factor'], r['sample_rate'])
        self.assertEqual(r['one_second_decoder_output_shape'], [1, 2, 48000])
        for seconds, size in r['single_tensor_fp32_bytes_by_seconds'].items():
            self.assertEqual(size, int(float(seconds) * r['largest_observed_decoder_tensor_one_second']['elements'] * 4))
        self.assertFalse(r['full_pretrained_checkpoints_downloaded'])
        self.assertFalse(r['pretrained_weights_loaded'])
        self.assertFalse(r['coreai_device_execution_verified'])
        self.assertEqual({p['stride'] for p in r['synthetic_operator_probes']}, {2, 4, 6, 10})
        for p in r['synthetic_operator_probes']:
            self.assertEqual(p['native_export'], 'passed')
            self.assertEqual(p['device_execution'], 'not_run')
            self.assertIn('Synthetic', p['weight_provenance'])
        for header in r['checkpoint_headers'].values():
            self.assertFalse(header['full_checkpoint_checksum_verified'])


if __name__ == '__main__':
    unittest.main()

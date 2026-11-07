"""Inactive750-frame export contract/strict gates, not learned/device parity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_coreai_drum_reference30 as M
from coreai_preview_window import PreviewWindow


class Reference30NativeTests(unittest.TestCase):
    def test_separate_contract_covers_all_layers_without_changing_baseline(self):
        p = M.plan()
        self.assertEqual((p['latent_frames'], p['condition_tokens'], p['asset_count']), (750, 127, 16))
        stages = p['stages']
        self.assertEqual(len(stages), 14)
        self.assertEqual([i for s in stages for i in s['layer_indices']], list(range(24)))
        self.assertEqual(stages[0]['input_shapes']['noise'], [1, 750, 64])
        self.assertEqual(stages[0]['input_shapes']['condition'], [1, 127, 2048])
        self.assertEqual(stages[1]['input_shapes']['hidden'], [2, 375, 2048])
        self.assertEqual(stages[-1]['output_shapes']['velocity'], [2, 750, 64])
        self.assertEqual(p['decoder']['output_shapes']['audio'], [1, 2, 1440000])
        self.assertEqual((p['inference_steps'], p['guidance_scale'], p['shift'], p['apg_momentum'], p['apg_norm_threshold']),
                         (50, 7., 1., -.75, 2.5))
        self.assertEqual(p['apg_projection_dims'], [1])
        self.assertEqual((p['weight_storage_dtype'], p['compute_dtype'], p['VAE_dtype']), ('bfloat16', 'float32', 'float32'))
        self.assertEqual(p['native_parity'], 'NOT EVALUATED')
        self.assertEqual(p['full_trajectory_parity'], 'NOT EVALUATED')
        with self.assertRaises(ValueError): PreviewWindow(750, 'drums')
        self.assertEqual(PreviewWindow(256, 'drums').condition_tokens, 123)

    def test_default_cli_is_read_only_and_needs_no_weights_or_apple_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'never-created'
            r = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_coreai_drum_reference30.py'), '--out', str(out)],
                               check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(r.stdout)['profile'], M.PROFILE)
            self.assertFalse(out.exists())

    def test_gate_requires_shape_finiteness_pointwise_and_rmse_at_original_bounds(self):
        import torch
        report = {'host_checks': {}}
        gold = torch.zeros(100)
        M.require_gate(report, 'zero', gold, gold)
        for name, actual, expected in [('pointwise', torch.ones(100) * .0002, gold),
                                       ('relative_rmse', torch.ones(100) * 1.00105, torch.ones(100)),
                                       ('nonfinite', torch.ones(100) * float('nan'), gold),
                                       ('shape', gold.reshape(10, 10), gold)]:
            with self.assertRaises(ValueError): M.require_gate(report, name, actual, expected)
            self.assertFalse(report['host_checks'][name]['parity_passed'])

    def test_protected_outputs_and_bad_capture_do_not_start_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'capture'; source.mkdir()
            old = Path(tmp) / 'old'; old.mkdir(); (old / 'keep').write_text('original')
            with patch.object(M, 'execute') as execute:
                with self.assertRaises(FileExistsError): M.prepare(source, old)
                for out in [source / 'nested', ROOT / 'baselines/never-create-native30',
                            ROOT / 'prototypes/ios27-coreai/ModelCandidates/never-create-native30']:
                    with self.assertRaises(ValueError): M.prepare(source, out)
                    self.assertFalse(out.exists())
                with self.assertRaises(FileNotFoundError): M.prepare(source, Path(tmp) / 'new')
                execute.assert_not_called()
            self.assertEqual((old / 'keep').read_text(), 'original')

    def test_failed_or_incomplete_export_retains_report_without_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'capture'; source.mkdir()
            for label, error in [('gate', ValueError('strict gate failed')), ('interrupt', KeyboardInterrupt())]:
                out = Path(tmp) / label
                with patch.object(M, 'verify_capture'), patch.object(M, 'execute', side_effect=error):
                    with self.assertRaises(type(error)): M.prepare(source, out)
                r = json.loads((out / 'reference.json').read_text())
                self.assertEqual(r['status'], 'failed')
                self.assertEqual(r['native_exports'], 0)
                self.assertFalse(r['baseline_modified'])
                self.assertEqual(r['native_parity'], 'NOT EVALUATED')
                self.assertFalse((out / 'active-preview.json').exists())
            with patch.object(M, 'verify_capture'), patch.object(M, 'execute'):
                with self.assertRaisesRegex(ValueError, 'Incomplete'): M.prepare(source, Path(tmp) / 'incomplete')
            def incomplete_manifest(capture, out, report, *args):
                report.update(native_exports=16, host_checks={'invalid': {'parity_passed': True}})
            with patch.object(M, 'verify_capture'), patch.object(M, 'execute', side_effect=incomplete_manifest):
                out = Path(tmp) / 'bad-manifest'
                with self.assertRaises(ValueError): M.prepare(source, out)
                self.assertEqual(json.loads((out / 'reference.json').read_text())['status'], 'failed')

    def test_bundle_rejects_claims_of_device_pass_or_false_strict_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            report = {**M.plan(), 'status': 'prepared_inactive_component_diagnostic', 'native_exports': 16,
                      'accepted_report_sha256': M.ACCEPTED_REPORT_SHA256, 'baseline_modified': False,
                      'host_checks': {}}
            for field, value in [('native_parity', 'PASS'), ('weight_storage_dtype', 'float16'),
                                 ('latent_frames', 256), ('condition_tokens', 123)]:
                (folder / 'reference.json').write_text(json.dumps({**report, field: value}))
                with self.assertRaisesRegex(ValueError, 'contract mismatch'): M.verify_bundle(folder)
            metrics = {'finite': True, 'parity_passed': True, 'max_absolute_error': .1, 'rmse': .01,
                       'normalized_rmse': .01, 'elementwise_fraction_over_bound': .1}
            report['host_checks'] = {k: metrics for k in ['FP32_partitions_step1', 'BF16_partitions_step1',
                'APG_step1_latent', 'APG_step1_momentum', 'APG_step1_guided', 'FP32_decoder_saved_final_latent']}
            (folder / 'reference.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): M.verify_bundle(folder)

    def test_asset_manifest_rejects_modified_missing_extra_and_linked_files(self):
        # Placeholder asset bytes exercise transport only; capture validity is
        # independently checked by verify_capture and the actual uploaded audit.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            report = {**M.plan(), 'status': 'prepared_inactive_component_diagnostic', 'native_exports': 16,
                      'accepted_report_sha256': M.ACCEPTED_REPORT_SHA256, 'baseline_modified': False,
                      'checkpoint_sha256': {'base': 'fixture-base', 'vae': 'fixture-vae'}}
            metrics = {'finite': True, 'parity_passed': True, 'max_absolute_error': 0., 'rmse': 0.,
                       'normalized_rmse': 0., 'elementwise_fraction_over_bound': 0.}
            report['host_checks'] = {k: metrics for k in ['FP32_partitions_step1', 'BF16_partitions_step1',
                'APG_step1_latent', 'APG_step1_momentum', 'APG_step1_guided', 'FP32_decoder_saved_final_latent']}
            hashes = {}
            for stage in report['stages'] + [report['apg'], report['decoder']]:
                asset = folder / stage['asset_file']; asset.mkdir()
                for name in ['main.hash', 'main.mlirb', 'metadata.json']:
                    path = asset / name; path.write_bytes(b'placeholder')
                    hashes[str(path.relative_to(folder))] = hashlib.sha256(b'placeholder').hexdigest()
            report['artifact_sha256'] = hashes
            (folder / 'reference.json').write_text(json.dumps(report))
            with patch.object(M, 'verify_capture', return_value={'checkpoint_sha256': report['checkpoint_sha256']}) as verify:
                self.assertEqual(M.verify_bundle(folder)['native_parity'], 'NOT EVALUATED')
                verify.assert_called_once_with(folder / 'capture')
                target = folder / 'front.aimodel/main.hash'
                target.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'integrity'): M.verify_bundle(folder)
                target.unlink()
                with self.assertRaisesRegex(ValueError, 'manifest'): M.verify_bundle(folder)
                target.write_bytes(b'placeholder')
                extra = folder / 'extra'; extra.write_text('unexpected')
                with self.assertRaisesRegex(ValueError, 'manifest'): M.verify_bundle(folder)
                extra.unlink()
                target.unlink(); target.symlink_to(folder / 'tail.aimodel/main.hash')
                with self.assertRaisesRegex(ValueError, 'integrity'): M.verify_bundle(folder)


if __name__ == '__main__':
    unittest.main()

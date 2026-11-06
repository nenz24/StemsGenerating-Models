"""Source-only routing/admission guards; no synthetic music validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_coreai_drum_source_gain as M
from compare_coreai_drum_lyrics import conditioned_predictor


class DrumSourceGainTests(unittest.TestCase):
    def fixture(self, root):
        candidates = root / 'models'
        for name in [M.WINDOW.source_bundle, M.WINDOW.condition_bundle, M.WINDOW.preview_bundle]:
            folder = candidates / name; folder.mkdir(parents=True)
            (folder / 'reference.json').write_text('{"nonexecutable_fixture":true}')
        (candidates / 'active-preview.json').write_text('preserve selector')
        return candidates

    def admission(self, cap=2.5):
        return (patch('export_coreai_source_context.verify_bundle', return_value={
                    'artifact_sha256': {'audio.f32le': 'nonexecutable_PCM', 'epsilon.f32le': 'saved-epsilon'},
                    'recording': {'canonicalization': 'Host ffmpeg -> 48 kHz stereo Float32, planar tensor, no peak normalization; same PCM sent to device'}}),
                patch('export_coreai_bass_conditioning.verify_bundle', return_value={'prompts': {'lyrics': '', 'metadata': '10seconds unchanged'}}),
                patch('export_coreai_bass_preview.verify_bundle', return_value={'apg_norm_threshold': cap, 'host_export_status': 'bounded_diagnostic'}))

    def test_plan_preserves_prompts_provenance_and_inputs_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); a, b, c = self.admission()
            before = {p: p.read_bytes() for p in candidates.rglob('*') if p.is_file()}
            with a, b, c as preview, patch.object(M, 'prepare_context') as encode, patch.object(M, 'execute_host') as execute:
                r = M.compare(candidates, root / 'out', True)
                encode.assert_not_called(); execute.assert_not_called()
                self.assertTrue(preview.call_args.args[-1])
            self.assertEqual(r['status'], 'planned')
            self.assertEqual(r['candidate_source_gain'], .7)
            self.assertEqual(r['control_prompts'], r['candidate_prompts'])
            self.assertEqual(r['control_host_export_status'], 'bounded_diagnostic')
            self.assertEqual((r['absolute_tolerance'], r['relative_tolerance'], r['relative_rmse_tolerance']), (1e-4, 1e-3, 1e-3))
            self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_existing_output_wrong_cap_and_bundle_output_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'out'
            a, b, c = self.admission(1.46059349)
            with a, b, c, patch.object(M, 'prepare_context') as encode:
                with self.assertRaisesRegex(ValueError, 'APG cap2.5'): M.compare(candidates, out, True, True)
                encode.assert_not_called()
            self.assertFalse(out.exists())
            out.mkdir(); (out / 'keep').write_text('keep')
            with patch('export_coreai_source_context.verify_bundle') as verify:
                with self.assertRaises(FileExistsError): M.compare(candidates, out)
                with self.assertRaises(ValueError): M.compare(candidates, candidates / 'new')
                verify.assert_not_called()
            self.assertEqual((out / 'keep').read_text(), 'keep')

    def test_failed_source_replay_retains_report_and_never_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); a, b, c = self.admission()
            with a, b, c, patch.object(M, 'prepare_context', side_effect=ValueError('source control gate failed')), patch.object(M, 'execute_host') as execute:
                with self.assertRaisesRegex(ValueError, 'control gate'): M.compare(candidates, root / 'out', True, True)
                execute.assert_not_called()
            r = json.loads((root / 'out/report.json').read_text())
            self.assertEqual(r['status'], 'failed')
            self.assertEqual((candidates / 'active-preview.json').read_text(), 'preserve selector')

    def test_execution_passes_candidate_context_only_and_rejects_incomplete_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); a, b, c = self.admission(); context = object()
            with a, b, c, patch.object(M, 'prepare_context', return_value=context), patch.object(M, 'execute_host') as execute:
                with self.assertRaisesRegex(ValueError, 'Incomplete'): M.compare(candidates, root / 'out', True, True)
            self.assertIs(execute.call_args.kwargs['candidate_context'], context)
            self.assertTrue(execute.call_args.kwargs['require_unchanged_condition'])
            self.assertNotIn('candidate_apg_factory', execute.call_args.kwargs)
            self.assertEqual(execute.call_args.kwargs['candidate_label'], '02-candidate-source-gain07')

    def test_cfg_predictors_bind_separate_source_contexts(self):
        import torch
        calls = []
        class Recorder:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return (torch.zeros_like(kwargs['hidden_states']),)
        model = Recorder(); cond = torch.ones(1, 3, 4); null = torch.zeros(1, 1, 4)
        control = torch.ones(1, 5, 8); candidate = torch.full_like(control, .7)
        before = control.clone(); z = torch.ones(1, 5, 4); t = torch.ones(1)
        a = conditioned_predictor(model, cond, null, control)
        b = conditioned_predictor(model, cond, null, candidate)
        a(z, t); b(z, t); a(z, t)
        for i, expected in enumerate([control, candidate, control]):
            self.assertTrue(torch.equal(calls[i]['context_latents'], expected.expand(2, -1, -1)))
            self.assertTrue(torch.equal(calls[i]['encoder_hidden_states'][0], cond[0]))
            self.assertTrue(torch.equal(calls[i]['encoder_hidden_states'][1], null[0].expand_as(cond[0])))
            self.assertTrue(torch.equal(calls[i]['hidden_states'][0], z[0]))
            self.assertTrue(torch.equal(calls[i]['hidden_states'][1], z[0]))
            self.assertFalse(calls[i]['use_cache'])
        self.assertTrue(torch.equal(control, before))

    def test_gain_applies_to_PCM_before_encoder_with_saved_epsilon_and_mask(self):
        import hashlib
        import numpy as np
        import torch
        # Analytic fake encoder verifies routing/closed-form posterior only;
        # these arrays are not substituted for any real model's gold data.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); folder = candidates / M.WINDOW.source_bundle
            out = root / 'out'; out.mkdir()
            audio = torch.full((1, 2, M.WINDOW.samples), .5)
            epsilon = torch.full((1, 64, 256), .25); mask = torch.ones(1, 256, 64)
            mean = torch.full((1, 64, 256), 2.)
            std = torch.full_like(mean, float(torch.nn.functional.softplus(torch.tensor(0.)) + 1e-4))
            source = (mean + std * epsilon).transpose(1, 2).contiguous()
            for name, value in [('audio', audio), ('epsilon', epsilon), ('chunk_mask', mask),
                                ('posterior_mean-reference', mean), ('posterior_std-reference', std),
                                ('source_latents-reference', source), ('context_latents-reference', torch.cat([source, mask], dim=-1))]:
                value.numpy().astype('<f4').tofile(folder / (name + '.f32le'))
            before = {p: p.read_bytes() for p in folder.iterdir()}
            seen = []
            def encoder(pcm):
                seen.append(pcm.clone())
                return torch.cat([torch.full_like(mean, float(pcm.mean() * 4)), torch.zeros_like(mean)], dim=1)
            refs = {'source': {'artifact_sha256': {'epsilon.f32le': hashlib.sha256((folder / 'epsilon.f32le').read_bytes()).hexdigest()}}}
            with patch.object(M, 'local_checkpoint', return_value=Path('nonexecutable')), patch.object(M, 'load_encoder', return_value=encoder):
                report = {}; context = M.prepare_context(candidates, out, refs, report, lambda _: None, {})
            self.assertTrue(torch.equal(seen[0], audio))
            self.assertTrue(torch.equal(seen[1], audio * .7))
            self.assertTrue(torch.equal(context[..., 64:], mask))
            expected = float((audio * .7).mean() * 4) + std.transpose(1, 2) * epsilon.transpose(1, 2)
            self.assertTrue(torch.equal(context[..., :64], expected))
            self.assertEqual(report['source_encoder_calls'], {'control': 1, 'candidate': 1})
            self.assertEqual(len(report['control_replay_checks']), 4)
            self.assertTrue(all(x['parity_passed'] for x in report['control_replay_checks'].values()))
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertEqual(np.fromfile(out / 'candidate-source-audio.f32le', dtype='<f4').size, 2 * M.WINDOW.samples)


if __name__ == '__main__':
    unittest.main()

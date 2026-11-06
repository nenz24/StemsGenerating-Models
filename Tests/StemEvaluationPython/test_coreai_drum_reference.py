"""Reference-generation admission/capture invariants, not musical validation."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_coreai_drum_reference as M


class DrumReferenceTests(unittest.TestCase):
    def test_pinned_metadata_and_literal_instrumental_prompt(self):
        r = M.prompts30()
        self.assertEqual(r['audio_seconds'], 30.)
        self.assertEqual(r['lyrics'], '[Instrumental]')
        self.assertIn('- duration: 30 seconds\n', r['metadata'])
        self.assertNotIn('- duration: 10 seconds\n', r['text_prompt'])
        self.assertEqual(r['text_prompt'].count(r['metadata']), 1)
        self.assertEqual(r['instruction'], 'Generate the drums track based on the audio context:')
        self.assertIn('[Instrumental]', r['lyric_prompt'])

    def test_input_resampler_preserves_impulse_time_stereo_and_duration(self):
        import numpy as np
        pcm = np.zeros((960000, 2), dtype='<i2'); pcm[480000, 0] = 16384
        audio = M.resample(pcm.tobytes()).numpy()
        self.assertEqual(audio.shape, (1, 2, 1440000))
        self.assertTrue(np.isfinite(audio).all())
        self.assertEqual(int(np.argmax(np.abs(audio[0, 0]))), 720000)
        self.assertTrue(np.all(audio[0, 1] == 0))
        self.assertLess(float(np.max(np.abs(audio))), 1.)

    def test_bad_source_hash_stops_before_resampling_or_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'different.mp3'; source.write_bytes(b'wrong source')
            out = root / 'out'; out.mkdir()
            with patch.object(M, 'resample') as resample:
                with self.assertRaisesRegex(ValueError, 'source hash'): M.prepare_source(out, source)
                resample.assert_not_called()
            self.assertEqual(list(out.iterdir()), [])
            self.assertEqual(source.read_bytes(), b'wrong source')

    def test_plan_uses_fixed_reference_without_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'plan'
            with patch.object(M, 'execute') as execute:
                r = M.generate(out)
                execute.assert_not_called()
            self.assertEqual(r['status'], 'planned')
            self.assertEqual((r['duration_seconds'], r['latent_frames'], r['steps_requested']), (30, 750, 50))
            self.assertEqual((r['APG_norm_threshold'], r['APG_momentum'], r['guidance_scale']), (2.5, -.75, 7))
            self.assertEqual((r['absolute_tolerance'], r['relative_tolerance'], r['relative_rmse_tolerance']), (1e-4, 1e-3, 1e-3))
            self.assertFalse(r['baseline_modified'])
            self.assertEqual(list(out.iterdir()), [out / 'report.json'])

    def test_output_guards_and_failure_reports_preserve_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root / 'old'; old.mkdir(); (old / 'keep').write_text('keep')
            with patch.object(M, 'execute') as execute:
                with self.assertRaises(FileExistsError): M.generate(old, execute_requested=True)
                for parent in [ROOT / 'baselines', ROOT / 'prototypes/ios27-coreai/ModelCandidates']:
                    with self.assertRaises(ValueError): M.generate(parent / 'never-create')
                execute.assert_not_called()
            with patch.object(M, 'execute', side_effect=FileNotFoundError('local weights missing')):
                with self.assertRaisesRegex(FileNotFoundError, 'weights missing'): M.generate(root / 'failed', execute_requested=True)
            self.assertEqual(json.loads((root / 'failed/report.json').read_text())['status'], 'failed')
            self.assertEqual((old / 'keep').read_text(), 'keep')
            with patch.object(M, 'execute'):
                with self.assertRaisesRegex(ValueError, 'Incomplete'): M.generate(root / 'incomplete', execute_requested=True)

    def test_live_trajectory_captures_real_step_states_without_resets(self):
        import torch
        calls = []; buffers = []; progress = []
        class Buffer:
            running_average = None
        def apg(a, b, scale, momentum_buffer, norm_threshold, dims):
            self.assertEqual((scale, norm_threshold, dims), (7., 2.5, [1]))
            buffers.append(momentum_buffer)
            momentum_buffer.running_average = a-b
            return a
        def predict(latent, time):
            calls.append(latent.clone())
            return torch.full((2, 750, 64), 2.)
        apg_module = types.SimpleNamespace(MomentumBuffer=Buffer, apg_forward=apg)
        noise = torch.ones(1, 750, 64); times = torch.linspace(1., 0., 51)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); report = {}
            result = M.trajectory(predict, noise, times, apg_module, out, progress.append, report)
            torch.testing.assert_close(result, -torch.ones_like(noise))
            self.assertTrue(torch.equal(noise, torch.ones_like(noise)))
            self.assertEqual(len({id(x) for x in buffers}), 1)
            self.assertEqual(len(calls), 50)
            self.assertTrue(all(float(calls[i+1].mean()) < float(calls[i].mean()) for i in range(49)))
            self.assertEqual(report['steps_completed'], 50)
            self.assertEqual(len(list(out.glob('step-*-latent.f32le'))), len(M.CAPTURE_STEPS))
            self.assertEqual((out / 'initial-noise.f32le').stat().st_size, 750 * 64 * 4)
            self.assertEqual(len(progress), 50)

    def test_invalid_velocity_stops_before_first_advance(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            apg = types.SimpleNamespace(MomentumBuffer=lambda: None, apg_forward=lambda a, b, *args, **kwargs: a)
            report = {}; progress = []
            with self.assertRaisesRegex(ValueError, 'Invalid velocity'):
                M.trajectory(lambda *_: torch.full((2, 750, 64), float('nan')), torch.ones(1, 750, 64),
                             torch.linspace(1., 0., 51), apg, Path(tmp), progress.append, report)
            self.assertEqual(progress, [])
            self.assertFalse((Path(tmp) / 'step-01-latent.f32le').exists())

    def test_reference_cache_is_retained_and_fresh_for_each_trajectory(self):
        import torch
        calls = []
        def model(**kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs['hidden_states']), kwargs['past_key_values']
        packed = torch.ones(1, 3, 4); null = torch.zeros(1, 1, 4); context = torch.ones(1, 5, 8)
        noise = torch.ones(1, 5, 4); time = torch.ones(1)
        predictor = M.cached_predictor(model, packed, null, context)
        predictor(noise, time); predictor(noise * .8, time * .8)
        self.assertIs(calls[0]['past_key_values'], calls[1]['past_key_values'])
        self.assertTrue(all(c['use_cache'] for c in calls))
        self.assertTrue(torch.equal(calls[1]['context_latents'], context.expand(2, -1, -1)))
        M.cached_predictor(model, packed, null, context)(noise, time)
        self.assertIsNot(calls[0]['past_key_values'], calls[2]['past_key_values'])


if __name__ == '__main__':
    unittest.main()

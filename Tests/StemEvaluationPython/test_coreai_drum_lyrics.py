"""Control/admission/feedback tests; no real model or native export is mocked as passed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_coreai_drum_lyrics as M


class DrumLyricComparisonTests(unittest.TestCase):
    def prompts(self):
        return {'lyrics': '', 'vocal_language': 'unknown', 'caption': 'unchanged drum caption',
                'lyric_prompt': '# Languages\nunknown\n\n# Lyric\n<|endoftext|>',
                'text_prompt': 'unchanged duration/instruction/caption', 'metadata': '10 seconds'}

    def formatter(self, root):
        path = root / 'formatter.py'
        path.write_text('class PromptMixin:\n'
                        '    def _format_lyrics(self, lyrics: str, language: str) -> str:\n'
                        '        return f"# Languages\\n{language}\\n\\n# Lyric\\n{lyrics}<|endoftext|>"\n')
        return patch.object(M, 'verified_handler_source', return_value=(path, {}))

    def fixture(self, root):
        candidates = root / 'models'
        for name in [M.WINDOW.source_bundle, M.WINDOW.condition_bundle, M.WINDOW.preview_bundle]:
            folder = candidates / name
            folder.mkdir(parents=True)
            (folder / 'reference.json').write_text('{"nonexecutable_admission_fixture":true}')
        (candidates / 'active-preview.json').write_text('preserved selector')
        return candidates

    def boundaries(self):
        return (patch('export_coreai_source_context.verify_bundle', return_value={'artifact_sha256': {'audio.f32le': 'fixture-pcm'}}),
                patch('export_coreai_bass_conditioning.verify_bundle', return_value={'prompts': self.prompts()}),
                patch('export_coreai_bass_preview.verify_bundle', return_value={'host_export_status': 'bounded_diagnostic'}))

    def test_only_lyrics_and_formatted_lyric_text_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = self.prompts()
            with self.formatter(Path(tmp)):
                candidate = M.lyric_variant(old)
                changed = {k for k in old if old[k] != candidate[k]}
                self.assertEqual(changed, {'lyrics', 'lyric_prompt'})
                self.assertIn('[Instrumental]', candidate['lyric_prompt'])
                self.assertEqual(old, self.prompts())
                with self.assertRaises(ValueError): M.lyric_variant({**old, 'lyrics': 'other input'})
                with self.assertRaises(ValueError): M.lyric_variant({**old, 'lyric_prompt': 'wrong formatter'})

    def test_plan_preserves_inputs_and_diagnostic_status_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'comparison'
            before = {p: p.read_bytes() for p in candidates.rglob('*') if p.is_file()}
            a, b, c = self.boundaries()
            with a, b, c as preview, self.formatter(root), patch.object(M, 'execute') as execute:
                report = M.compare(candidates, out, diagnostic=True)
                execute.assert_not_called()
                self.assertTrue(preview.call_args.args[-1])
            self.assertEqual(report['status'], 'planned')
            self.assertEqual(report['control_host_export_status'], 'bounded_diagnostic')
            self.assertEqual(report['native_exports'], 0)
            self.assertEqual(report['relative_rmse_tolerance'], 1e-3)
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertEqual(json.loads((out / 'report.json').read_text()), report)

    def test_failed_admission_or_existing_output_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'comparison'
            a, b, _ = self.boundaries()
            with a, b, patch('export_coreai_bass_preview.verify_bundle', side_effect=ValueError('APG failed')), patch.object(M, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'APG failed'): M.compare(candidates, out, True, True)
                self.assertFalse(out.exists())
                execute.assert_not_called()
            out.mkdir(); saved = out / 'keep'; saved.write_text('prior result')
            with patch('export_coreai_source_context.verify_bundle') as verify:
                with self.assertRaises(FileExistsError): M.compare(candidates, out)
                with self.assertRaises(ValueError): M.compare(candidates, candidates / 'new-child')
                verify.assert_not_called()
            self.assertEqual(saved.read_text(), 'prior result')

    def test_execution_failure_retains_report_without_replacing_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'comparison'
            a, b, c = self.boundaries()
            with a, b, c, self.formatter(root), patch.object(M, 'execute', side_effect=ValueError('control replay failed')):
                with self.assertRaisesRegex(ValueError, 'control replay'): M.compare(candidates, out, True, True)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('control replay failed', report['error'])
            self.assertEqual((candidates / 'active-preview.json').read_text(), 'preserved selector')

    def test_control_replay_gate_retains_failed_metrics(self):
        import torch
        expected = torch.ones(3)
        report = {}
        M.require_parity('exact', expected.clone(), expected, report)
        with self.assertRaisesRegex(ValueError, 'original tolerances unchanged'):
            M.require_parity('different', expected + .1, expected, report)
        self.assertTrue(report['control_replay_checks']['exact']['parity_passed'])
        self.assertFalse(report['control_replay_checks']['different']['parity_passed'])
        with self.assertRaisesRegex(ValueError, 'shape failed'):
            M.require_parity('broadcastable', expected[:1], expected, report)
        self.assertFalse(report['control_replay_checks']['broadcastable']['parity_passed'])

    def test_fifty_steps_feed_back_live_latents_with_fresh_momentum(self):
        import torch
        # Constant velocity gives a closed-form Euler result; this is control-flow
        # validation, not a fake musical model or validation of publisher APG math.
        calls = []; buffers = []
        class Momentum:
            pass
        def apg_forward(a, b, scale, momentum_buffer, dims):
            self.assertEqual(scale, 7.)
            self.assertEqual(dims, [1])
            buffers.append(momentum_buffer)
            return a
        apg = types.SimpleNamespace(MomentumBuffer=Momentum, apg_forward=apg_forward)
        def predict(latent, time):
            calls.append((latent.clone(), float(time[0])))
            return torch.full((2, 3, 2), 2.)
        noise = torch.zeros(1, 3, 2); times = torch.linspace(1., 0., 51)
        progress = []
        result = M.candidate_trajectory(predict, noise, times, apg, progress.append)
        self.assertEqual(progress, list(range(1, 51)))
        self.assertEqual(len({id(b) for b in buffers}), 1)
        self.assertTrue(torch.equal(noise, torch.zeros_like(noise)))
        torch.testing.assert_close(result, torch.full_like(noise, -2.))
        self.assertTrue(all(float(calls[i+1][0].mean()) < float(calls[i][0].mean()) for i in range(49)))
        first_buffer = buffers[0]
        M.candidate_trajectory(predict, noise, times, apg, lambda _: None)
        self.assertIsNot(buffers[-1], first_buffer)
        with self.assertRaisesRegex(ValueError, 'latent shape/values at step 1'):
            M.candidate_trajectory(lambda *_: torch.ones(4, 3, 2), noise, times, apg, lambda _: None)

    def test_local_checkpoint_never_falls_back_to_download(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        record = {'repo_id': 'fixture/no-weights', 'filename': 'no-existing-test-weights.safetensors',
                  'revision': 'fixture', 'destination': 'no-existing-test-weights.safetensors',
                  'size': 4, 'digest': hashlib.sha256(b'test').hexdigest()}
        with patch('huggingface_hub.hf_hub_download', side_effect=LocalEntryNotFoundError('missing')) as download:
            with self.assertRaisesRegex(FileNotFoundError, 'no download attempted'): M.local_checkpoint(record)
            self.assertEqual(download.call_count, 3)
            self.assertTrue(all(c.kwargs['local_files_only'] for c in download.call_args_list))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'; path.write_bytes(b'test')
            self.assertEqual(M.local_checkpoint(record, path), path)
            path.write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'): M.local_checkpoint(record, path)
            self.assertEqual(path.read_bytes(), b'bad!')


if __name__ == '__main__':
    unittest.main()

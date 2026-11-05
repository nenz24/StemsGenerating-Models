"""Derived APG policy and experiment admission; no model/music execution claimed."""
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_coreai_drum_apg as M


class DrumAPGTests(unittest.TestCase):
    def test_cap_preserves_reference_per_frame_bound_and_rejects_bad_geometry(self):
        cap = M.scaled_cap(256, 750)
        self.assertAlmostEqual(cap, 1.4605934866804429)
        self.assertAlmostEqual(cap / math.sqrt(256), 2.5 / math.sqrt(750))
        self.assertEqual(M.scaled_cap(750, 750), 2.5)
        for a, b in [(0, 750), (256, -1), (True, 750), (256., 750), (256, math.inf)]:
            with self.assertRaises(ValueError): M.scaled_cap(a, b)

    def fixture(self, root):
        candidates = root / 'models'
        for name in [M.WINDOW.source_bundle, M.WINDOW.condition_bundle, M.WINDOW.preview_bundle]:
            folder = candidates / name; folder.mkdir(parents=True)
            (folder / 'reference.json').write_text('{"nonexecutable_admission_fixture":true}')
        (candidates / 'active-preview.json').write_text('preserved selector')
        return candidates

    def boundaries(self, cap=2.5):
        return (patch('export_coreai_source_context.verify_bundle', return_value={'artifact_sha256': {'audio.f32le': 'fixture-pcm'}}),
                patch('export_coreai_bass_conditioning.verify_bundle', return_value={'prompts': {'lyrics': '', 'caption': 'unchanged'}}),
                patch('export_coreai_bass_preview.verify_bundle', return_value={'host_export_status': 'bounded_diagnostic', 'apg_norm_threshold': cap}))

    def test_plan_preserves_prompt_status_gates_and_selector_without_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'comparison'
            before = {p: p.read_bytes() for p in candidates.rglob('*') if p.is_file()}
            a, b, c = self.boundaries()
            with a, b, c as preview, patch.object(M, 'execute_host') as execute:
                r = M.compare(candidates, out, True)
                execute.assert_not_called()
                self.assertTrue(preview.call_args.args[-1])
            self.assertEqual(r['control_prompts'], r['candidate_prompts'])
            self.assertEqual(r['control_host_export_status'], 'bounded_diagnostic')
            self.assertEqual((r['absolute_tolerance'], r['relative_tolerance'], r['relative_rmse_tolerance']), (1e-4, 1e-3, 1e-3))
            self.assertEqual(r['status'], 'planned')
            self.assertEqual(r['baseline_apg_norm_threshold'], 2.5)
            self.assertFalse(r['baseline_modified'])
            self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_changed_control_cap_and_existing_output_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'comparison'
            a, b, c = self.boundaries(3.)
            with a, b, c, patch.object(M, 'execute_host') as execute:
                with self.assertRaisesRegex(ValueError, 'cap 2.5'): M.compare(candidates, out, True, True)
                execute.assert_not_called()
            self.assertFalse(out.exists())
            out.mkdir(); (out / 'keep').write_text('preserve')
            with self.assertRaises(FileExistsError): M.compare(candidates, out)
            with self.assertRaises(ValueError): M.compare(candidates, candidates / 'new-child')
            self.assertEqual((out / 'keep').read_text(), 'preserve')

    def test_failed_replay_retains_failure_report_and_requires_unchanged_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidates = self.fixture(root); out = root / 'comparison'
            a, b, c = self.boundaries()
            with a, b, c, patch.object(M, 'execute_host', side_effect=ValueError('control replay failed')) as execute:
                with self.assertRaisesRegex(ValueError, 'control replay'): M.compare(candidates, out, True, True)
            self.assertTrue(execute.call_args.kwargs['require_unchanged_condition'])
            self.assertEqual(execute.call_args.kwargs['candidate_label'], '02-candidate-scaled-apg')
            self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'failed')
            self.assertEqual((candidates / 'active-preview.json').read_text(), 'preserved selector')

    def test_adapter_delegates_original_apg_and_records_candidate_not_control_history(self):
        import torch
        calls = []
        class Buffer:
            running_average = torch.ones(1, 256, 64)
        def apg(a, b, scale, **kwargs):
            calls.append(kwargs)
            return a + .1
        original = types.SimpleNamespace(MomentumBuffer=Buffer, apg_forward=apg)
        cap = M.scaled_cap(256, 750); adapter = M.NormLimitedAPG(original, cap); momentum = Buffer()
        x = torch.zeros(1, 256, 64)
        adapter.apg_forward(x, x, 7., momentum, [1])
        self.assertEqual(calls[0]['norm_threshold'], cap)
        self.assertIs(calls[0]['momentum_buffer'], momentum)
        self.assertEqual(calls[0]['dims'], [1])
        self.assertEqual(adapter.diagnostics[0]['candidate_clip_fraction'], 1.)
        self.assertEqual(adapter.diagnostics[0]['baseline_cap_fraction_for_same_candidate_momentum'], 1.)
        with self.assertRaises(ValueError): adapter.apg_forward(x, x, 7., momentum, [2])
        for invalid in [0, -1, math.nan]:
            with self.assertRaises(ValueError): M.NormLimitedAPG(original, invalid)


if __name__ == '__main__':
    unittest.main()

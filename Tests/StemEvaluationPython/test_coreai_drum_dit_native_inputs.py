"""Temporal alignment tests on synthetic values, not learned-model parity."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_coreai_drum_reference30_dit_inputs as audit


class PlanTests(unittest.TestCase):
    def test_plan_does_not_import_torch_or_read_inputs(self):
        for options in ([],['--input-drift-only']):
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/audit_coreai_drum_reference30_dit_inputs.py'),
                                   '--native-archive','/nonexistent/archive.zip','--out','/nonexistent/never-write-dit-inputs',*options],
                                  text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('PLAN ONLY',result.stdout)
            self.assertFalse(Path('/nonexistent/never-write-dit-inputs').exists())


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional Torch environment')
class ExactInputTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        self.noise=torch.zeros(1,750,64)
        self.times=torch.linspace(1,0,51)
        self.native={f'native-step-{s:02d}-latent.f32le':torch.full_like(self.noise,float(s)) for s in (1,5,10,25)}
        self.native.update({f'native-step-{s:02d}-velocity.f32le':torch.full((2,750,64),float(100+s)) for s in audit.CASE_STEPS})

    def test_native_post_step_is_paired_with_next_velocity_and_timestep(self):
        cases=audit.native_input_cases(self.native,self.noise,self.times)
        self.assertEqual([c['step'] for c in cases],[1,2,6,11,26])
        for c in cases:
            step=c['step']
            self.assertEqual(c['input_post_step'],step-1)
            self.assertEqual(float(c['latent'][0,0,0]),float(step-1))
            self.assertTrue(self.torch.equal(c['timestep'],self.times[step-1:step]))
            self.assertEqual(float(c['native_velocity'][0,0,0]),float(100+step))
        del self.native['native-step-25-latent.f32le']
        with self.assertRaisesRegex(ValueError,'step26'):
            audit.native_input_cases(self.native,self.noise,self.times)

    def test_predictor_may_not_mutate_exact_native_input(self):
        cases=audit.native_input_cases(self.native,self.noise,self.times)
        def bad(x,t):
            x.add_(1)
            return self.torch.zeros(2,750,64)
        with self.assertRaisesRegex(ValueError,'mutated'):
            audit.measure_cases(bad,cases)

    def test_decomposition_adds_vectors_and_handles_zero_control(self):
        tensor=self.torch.tensor
        result=audit.decompose_velocity_error(tensor([5.,5.]),tensor([3.,3.]),tensor([0.,0.]))
        self.assertEqual(result['same_input_operator_rmse'],2)
        self.assertEqual(result['observed_input_drift_effect_rmse'],3)
        self.assertEqual(result['total_rmse'],5)
        self.assertEqual(result['input_drift_to_operator_rmse_ratio'],1.5)
        self.assertEqual(result['reconstruction_max_absolute_error'],0)
        control=audit.decompose_velocity_error(tensor([0.]),tensor([0.]),tensor([0.]))
        self.assertIsNone(control['input_drift_to_operator_rmse_ratio'])
        self.assertIsNone(control['operator_feedback_error_cosine'])

    def test_cancelling_errors_must_not_be_reported_as_additive_rmse(self):
        tensor=self.torch.tensor
        result=audit.decompose_velocity_error(tensor([0.,0.]),tensor([1.,1.]),tensor([0.,0.]))
        self.assertEqual(result['same_input_operator_rmse'],1)
        self.assertEqual(result['observed_input_drift_effect_rmse'],1)
        self.assertEqual(result['total_rmse'],0)
        self.assertEqual(result['operator_feedback_error_cosine'],-1)
        self.assertEqual(result['cross_term_mean'],-2)
        self.assertEqual(result['reconstruction_max_absolute_error'],0)
        with self.assertRaisesRegex(ValueError,'shapes differ'):
            audit.decompose_velocity_error(tensor([0.]),tensor([0.,1.]),tensor([0.]))


if __name__=='__main__':unittest.main()

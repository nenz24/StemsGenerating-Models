"""Control-flow/transport tests; synthetic values are not learned-model evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import replay_coreai_drum_reference30_sampler as replay


class PlanTests(unittest.TestCase):
    def test_latent_pass_does_not_override_failed_momentum_or_accept_audio(self):
        good={'finite':True,'parity_passed':True,'elementwise_fraction_over_bound':0,'normalized_rmse':1e-5}
        latent={str(s):dict(good) for s in replay.STEPS}
        bad=dict(good,parity_passed=False,elementwise_fraction_over_bound=2/48000)
        result=replay.replay_verdicts(latent,bad)
        self.assertEqual(result['latent_checkpoint_verdict'],'PASS')
        self.assertEqual(result['final_momentum_verdict'],'NOT PASSED')
        self.assertEqual(result['sampled_state_verdict'],'NOT PASSED')
        self.assertEqual(result['waveform_verdict'],'NOT EVALUATED')
        del latent['50']
        self.assertEqual(replay.replay_verdicts(latent,good)['latent_checkpoint_verdict'],'NOT PASSED')
    def test_plan_and_help_need_no_site_packages_or_input_files(self):
        for options in ([], ['--help']):
            result = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/replay_coreai_drum_reference30_sampler.py'),
                                     '--out', '/nonexistent/never-created-sampler-output', *options],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(Path('/nonexistent/never-created-sampler-output').exists())
            if not options:
                self.assertIn('PLAN ONLY', result.stdout)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional Torch environment')
class FeedbackTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(2)
        self.noise = torch.ones(1, 750, 64)
        self.times = torch.linspace(1, 0, 51)

    def predictor(self, x, t):
        return self.torch.cat((x*.125+t, x*.1-t))

    def test_live_feedback_and_observer_cannot_replace_loop_state(self):
        checkpoints, seen, progress = [], [], []
        def observe(step, latent, momentum):
            checkpoints.append(step)
            latent.fill_(12345)
            momentum.fill_(12345)
        def predict(x, t):
            seen.append(x.clone())
            return self.predictor(x, t)
        actual = replay.live_replay(predict, self.noise, self.times, observe,
                                    lambda step, seconds: progress.append(step))
        control = replay.live_replay(self.predictor, self.noise, self.times,
                                    lambda *args: None, lambda *args: None)
        self.assertEqual(checkpoints, [1, 5, 10, 25, 50])
        self.assertEqual(progress, list(range(1, 51)))
        self.assertEqual(len(seen), 50)
        self.assertFalse(self.torch.equal(seen[1], self.noise))
        for a, b in zip(actual, control):
            self.assertTrue(self.torch.equal(a, b))
        self.assertTrue(self.torch.equal(self.noise, self.torch.ones_like(self.noise)))

    def test_bad_schedule_rejected_before_prediction(self):
        times = self.times.clone()
        times[20] = times[19]
        with self.assertRaisesRegex(ValueError, 'schedule'):
            replay.live_replay(lambda *args: self.fail('Prediction called'), self.noise, times,
                               lambda *args: None, lambda *args: None)

    def test_invalid_velocity_stops_before_checkpoint(self):
        def invalid(x, t):
            return self.torch.full((2, 750, 64), float('nan'))
        with self.assertRaisesRegex(ValueError, 'velocity at step1'):
            replay.live_replay(invalid, self.noise, self.times,
                               lambda *args: self.fail('Checkpoint reached'), lambda *args: None)

    def test_fixed_velocity_apg_replay_uses_full_ordered_trace_without_mutation(self):
        from replay_coreai_drum_reference30_apg_trace import replay_trace
        velocities=[self.torch.cat((self.noise*(i+1)*.01,self.noise*.005)) for i in range(50)]
        checkpoints,momentum=replay_trace(velocities,self.noise,self.times)
        self.assertEqual(set(checkpoints),{1,5,10,25,50})
        self.assertTrue(self.torch.isfinite(momentum).all())
        altered=[v.clone() for v in velocities]
        altered[-1]*=2
        changed,_=replay_trace(altered,self.noise,self.times)
        self.assertTrue(self.torch.equal(checkpoints[25],changed[25]))
        self.assertFalse(self.torch.equal(checkpoints[50],changed[50]))
        self.assertTrue(self.torch.equal(self.noise,self.torch.ones_like(self.noise)))
        with self.assertRaisesRegex(ValueError,'complete50-step'):
            replay_trace(velocities[:-1],self.noise,self.times)


if __name__ == '__main__':
    unittest.main()

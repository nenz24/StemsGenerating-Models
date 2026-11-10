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


if __name__ == '__main__':
    unittest.main()

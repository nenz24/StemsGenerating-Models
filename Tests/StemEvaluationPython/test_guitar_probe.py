"""Guitar routing/export and the embedded notebook on an older cached bridge."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tools/stem_eval')]
import guitar_probe as guitar
from audio import RATE, write_pcm
from provenance import sha256


class GuitarProbeTests(unittest.TestCase):
    def fixture(self, base):
        source = base / 'piano.wav'
        write_pcm(source, [100, -100] * (10 * RATE + RATE // 8))
        return source

    def fake_worker(self, output, *, wrong_instruction=False, wrong_dtype=False):
        engine = guitar.load_engine(ROOT)
        prepared = output / 'prepared'
        folder = output / 'generation/000_A_guitar_seed0'
        folder.mkdir(parents=True)
        experiment = engine.plan(prepared, 'ace', 'guitar', [0])
        experiment['status'] = 'generated'
        (output / 'generation/experiment.json').write_text(json.dumps(experiment))
        shutil.copyfile(prepared / 'source.wav', folder / 'stem.wav')
        shutil.copyfile(prepared / 'source.wav', folder / 'mix.wav')
        params = {'caption': guitar.PRESET['caption'], 'seed': 0, 'guidance_scale': 7,
                  'inference_steps': 50, 'duration': experiment['jobs'][0]['duration'],
                  'task_type': 'lego', 'instrumental': True, 'enable_normalization': False,
                  'reference_audio': None, 'thinking': False, 'use_cot_caption': False,
                  'use_cot_metas': False, 'use_cot_language': False,
                  'instruction': 'Generate the bass track based on the audio context:' if wrong_instruction else 'Generate the guitar track based on the audio context:'}
        result = dict(experiment['jobs'][0], status='generated', device='cuda',
                      settings={'params': params, 'config': {'seeds': [0], 'batch_size': 1, 'use_random_seed': False}},
                      stem_file='stem.wav', stem_sha256=sha256(folder / 'stem.wav'))
        (folder / 'result.json').write_text(json.dumps(result))
        (output / 'capture').mkdir()
        (output / 'capture/capture.json').write_text(json.dumps({
            'status': 'captured', 'model_dtype': 'torch.float16' if wrong_dtype else 'torch.float32',
            'model_device': 'cuda', 'calls': [{'backend': 'pytorch_vae', 'dtype': 'torch.float32', 'status': 'returned'}]}))

    def test_dry_plan_is_guitar_and_unsupported_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = self.fixture(base); output = base / 'probe'
            with patch('stream_process.run_logged', side_effect=AssertionError('no model')):
                report = guitar.generate(ROOT, source, output, duration=10.125)
            experiment = json.loads((output / 'generation/experiment.json').read_text())
            self.assertEqual(report['status'], 'planned')
            self.assertEqual([(j['instrument'], j['duration']) for j in experiment['jobs']], [('guitar', 11)])
            self.assertFalse((output / 'guitar_feedback.zip').exists())
            with self.assertRaisesRegex(ValueError, 'requires the ACE-Step'):
                guitar.load_engine(ROOT).plan(output / 'prepared', 'stage', 'guitar', [0])
            with self.assertRaisesRegex(ValueError, '10–30'):
                guitar.generate(ROOT, source, base / 'invalid', duration=90)

    def test_verified_export_trims_padding_and_rejects_wrong_role_or_precision(self):
        for mode in ('correct', 'wrong_instruction', 'wrong_dtype'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); source = self.fixture(base); output = base / 'probe'
                source_hash = sha256(source)
                def worker(command, **kwargs):
                    self.assertIn('--worker', command)
                    self.assertEqual(Path(command[1]).name, 'guitar_probe.py')
                    self.fake_worker(output, **({mode: True} if mode != 'correct' else {}))
                with patch.object(guitar.single, 'model_python', return_value=Path(sys.executable)), patch('stream_process.run_logged', side_effect=worker) as run:
                    if mode != 'correct':
                        with self.assertRaises(ValueError):
                            guitar.generate(ROOT, source, output, duration=10.125, execute=True)
                        self.assertFalse((output / 'guitar.wav').exists())
                        self.assertFalse((output / 'guitar_result.zip').exists())
                    else:
                        report = guitar.generate(ROOT, source, output, duration=10.125, execute=True)
                        self.assertEqual(report['status'], 'generated')
                        self.assertEqual(guitar.single.frames(output / 'guitar.wav'), 10 * RATE + RATE // 8)
                        with zipfile.ZipFile(output / 'guitar_feedback.zip') as z:
                            self.assertIsNone(z.testzip())
                            self.assertIn('guitar_preview.mp3', z.namelist())
                            self.assertNotIn('guitar.wav', z.namelist())
                        with zipfile.ZipFile(output / 'guitar_result.zip') as z:
                            self.assertEqual(z.read('guitar.wav'), (output / 'guitar.wav').read_bytes())
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual(sha256(source), source_hash)

    def test_notebook_dry_run_uses_new_engine_without_mutating_old_bridge(self):
        script = r'''
import contextlib, io, json, shutil, sys, types
from pathlib import Path
repo, base = map(Path, sys.argv[1:])
root = base / 'ace_context_source_old'
for directory in ('tools', 'scripts', 'configs'):
 shutil.copytree(repo / directory, root / directory, ignore=shutil.ignore_patterns('__pycache__'))
engine = root / 'tools/stem_eval/run.py'
old = engine.read_text().replace('if instrument not in ("bass", "drums", "guitar"):', 'if instrument not in ("bass", "drums"):')
engine.write_text(old)
sys.path[:0] = [str(root / 'tools/stem_eval'), str(root / 'scripts')]
from audio import RATE, write_pcm
source = base / 'piano.wav'; write_pcm(source, [100, -100] * (31 * RATE))
def forbidden(*a, **kw): raise AssertionError('no upload/download in dry-run')
colab = types.ModuleType('google.colab'); colab.files = types.SimpleNamespace(upload=forbidden, download=forbidden)
sys.modules['google.colab'] = colab
book = json.loads((repo / 'notebooks/18_colab_clean_electric_guitar.ipynb').read_text())
scope = {'ROOT': root, 'SOURCE_PATH': source}
with contextlib.redirect_stdout(io.StringIO()):
 for c in book['cells']:
  if c['cell_type'] == 'code':
   exec(compile(''.join(c['source']).replace('DRY_RUN = False', 'DRY_RUN = True'), 'guitar-notebook', 'exec'), scope)
assert scope['report']['status'] == 'planned'
out = scope['GUITAR_OUT']
plan = json.loads((out / 'generation/experiment.json').read_text())
assert [(j['instrument'], j['duration']) for j in plan['jobs']] == [('guitar', 30)]
assert engine.read_text() == old
assert not (root / 'vendor').exists()
assert not list(out.rglob('guitar.wav'))
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', script, str(ROOT), tmp],
                                    cwd=tmp, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

"""Guitar routing/export and the embedded notebook on an older cached bridge."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
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

    def fake_worker(self, output, *, wrong_instruction=False, wrong_dtype=False, variant='original', reference_audio=None):
        engine = guitar.load_engine(ROOT)
        prepared = output / 'prepared'
        folder = output / 'generation/000_A_guitar_seed0'
        folder.mkdir(parents=True)
        experiment = engine.plan(prepared, 'ace', 'guitar', [0])
        experiment['status'] = 'generated'
        (output / 'generation/experiment.json').write_text(json.dumps(experiment))
        shutil.copyfile(prepared / 'source.wav', folder / 'stem.wav')
        shutil.copyfile(prepared / 'source.wav', folder / 'mix.wav')
        params = {'caption': guitar.PRESETS[variant]['caption'], 'src_audio': str(prepared / 'source.wav'), 'seed': 0, 'guidance_scale': 7,
                  'inference_steps': 50, 'duration': experiment['jobs'][0]['duration'],
                  'task_type': 'lego', 'instrumental': True, 'enable_normalization': False,
                  'reference_audio': str(reference_audio) if reference_audio is not None else None,
                  'thinking': False, 'use_cot_caption': False,
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
            with self.assertRaisesRegex(ValueError, '10–90'):
                guitar.generate(ROOT, source, base / 'invalid', duration=120)

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

    def test_90_second_context_plans_one_guitar_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base / 'piano.wav'
            with wave.open(str(source), 'wb') as audio:
                audio.setparams((2, 2, RATE, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0' * (90 * RATE * 4))
            report = guitar.generate(ROOT, source, base / 'probe', duration=90)
            experiment = json.loads((base / 'probe/generation/experiment.json').read_text())
            self.assertEqual(report['input']['valid_frames'], 90 * RATE)
            self.assertEqual([(j['instrument'], j['duration']) for j in experiment['jobs']], [('guitar', 90)])
            self.assertEqual(experiment['max_job_seconds'], 90)

    def test_matched_articulation_exports_and_rejects_unrelated_setting_changes(self):
        import guitar_articulation
        for unrelated_change in (False, True):
            with self.subTest(unrelated_change=unrelated_change), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); source = base / 'piano.wav'; output = base / 'comparison'
                with wave.open(str(source), 'wb') as audio:
                    audio.setparams((2, 2, RATE, 0, 'NONE', 'not compressed'))
                    audio.writeframes(b'\0' * (31 * RATE * 4))
                source_hash = sha256(source)
                def worker(command, **kwargs):
                    variant = command[command.index('--variant') + 1]
                    folder = Path(command[command.index('--out') + 1])
                    self.fake_worker(folder, variant=variant)
                    if unrelated_change and variant == 'picked':
                        path = folder / 'generation/000_A_guitar_seed0/result.json'
                        result = json.loads(path.read_text())
                        result['settings']['params']['shift'] = 2
                        path.write_text(json.dumps(result))
                with patch.object(guitar.single, 'model_python', return_value=Path(sys.executable)), patch('stream_process.run_logged', side_effect=worker) as run:
                    if unrelated_change:
                        with self.assertRaisesRegex(ValueError, 'Unexpected differences'):
                            guitar_articulation.compare(ROOT, source, output, start=0, execute=True)
                        self.assertEqual(json.loads((output / 'comparison.json').read_text())['status'], 'failed')
                        self.assertFalse((output / 'guitar_articulation_feedback.zip').exists())
                    else:
                        report = guitar_articulation.compare(ROOT, source, output, start=0, execute=True)
                        self.assertEqual(report['status'], 'generated')
                        self.assertEqual(report['actual_parameter_differences'], ['caption', 'src_audio'])
                        a, b = [json.loads((output / variant / 'guitar_probe.json').read_text()) for variant in ('original', 'picked')]
                        self.assertEqual(a['input'], b['input'])
                        self.assertEqual(b['preset'], guitar.PICKED_PRESET)
                        with zipfile.ZipFile(output / 'guitar_articulation_feedback.zip') as z:
                            self.assertIsNone(z.testzip())
                            self.assertIn('picked/guitar_preview.mp3', z.namelist())
                            self.assertNotIn('picked/guitar.wav', z.namelist())
                        with zipfile.ZipFile(output / 'guitar_articulation_result.zip') as z:
                            self.assertIn('original/guitar.wav', z.namelist())
                            self.assertIn('picked/guitar.wav', z.namelist())
                    self.assertEqual(run.call_count, 2)
                    self.assertEqual(sha256(source), source_hash)
                    self.assertEqual(guitar.PRESET['id'], 'clean-electric-guitar-v1')

    def test_notebook_dry_run_uses_new_engine_without_mutating_old_bridge(self):
        script = r'''
import contextlib, io, json, shutil, sys, types, wave
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
source = base / 'piano.wav'
with wave.open(str(source), 'wb') as audio:
 audio.setparams((2, 2, RATE, 0, 'NONE', 'not compressed')); audio.writeframes(b'\0' * (121 * RATE * 4))
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
book = json.loads((repo / 'notebooks/19_colab_guitar_context_90.ipynb').read_text())
for repeat in range(2):
 with contextlib.redirect_stdout(io.StringIO()):
  for c in book['cells']:
   if c['cell_type'] == 'code':
    exec(compile(''.join(c['source']).replace('DRY_RUN = False', 'DRY_RUN = True'), 'guitar-chunk-notebook', 'exec'), scope)
 if repeat == 0: chunk_out = scope['GUITAR_CHUNK_OUT']
 assert scope['GUITAR_CHUNK_OUT'] == chunk_out
plan = scope['guitar_plan']
assert plan['instrument'] == 'guitar' and plan['context_cap_seconds'] == 90
assert plan['total_frames'] == 121 * RATE
assert engine.read_text() == old
assert not list(chunk_out.rglob('guitar.wav'))
book = json.loads((repo / 'notebooks/20_colab_guitar_articulation.ipynb').read_text())
with contextlib.redirect_stdout(io.StringIO()):
 for c in book['cells']:
  if c['cell_type'] == 'code':
   exec(compile(''.join(c['source']).replace('DRY_RUN = False', 'DRY_RUN = True'), 'guitar-articulation-notebook', 'exec'), scope)
assert scope['report']['status'] == 'planned'
assert scope['report']['start_seconds'] == 77.130625
assert scope['report']['preset_differences'] == ['caption', 'id']
assert engine.read_text() == old
assert not list(scope['ARTICULATION_OUT'].rglob('guitar.wav'))
book = json.loads((repo / 'notebooks/21_colab_guitar_layers.ipynb').read_text())
with contextlib.redirect_stdout(io.StringIO()):
 for c in book['cells']:
  if c['cell_type'] == 'code':
   exec(compile(''.join(c['source']).replace('DRY_RUN = False', 'DRY_RUN = True'), 'guitar-layers-notebook', 'exec'), scope)
assert scope['report']['status'] == 'planned' and scope['report']['start_seconds'] == 0
assert scope['report']['reference_audio_sha256'] == scope['hashlib'].sha256(scope['STYLE_REFERENCE'].read_bytes()).hexdigest()
assert scope['report']['model_calls_requested'] == 2
assert engine.read_text() == old
assert not list(scope['LAYERS_OUT'].rglob('guitar.wav'))
with contextlib.redirect_stdout(io.StringIO()):
 for c in book['cells']:
  if c['cell_type'] == 'code':
   text = ''.join(c['source']).replace('DRY_RUN = False', 'DRY_RUN = True').replace('MODE = "preview30"', 'MODE = "full90"')
   exec(compile(text, 'guitar-full-layers-notebook', 'exec'), scope)
assert scope['report']['status'] == 'planned'
assert scope['report']['export_frames'] == 121 * RATE
assert scope['report']['model_calls_total_if_uncached'] == 4
assert engine.read_text() == old
assert not list(scope['LAYERS_OUT'].rglob('guitar.wav'))
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', script, str(ROOT), tmp],
                                    cwd=tmp, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

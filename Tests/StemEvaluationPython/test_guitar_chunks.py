"""Guitar-only chunk routing, exact-length export, resume and notebook bundles."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tools/stem_eval')]
import ace_context_chunks as chunks
import guitar_probe
import drum_probe


class GuitarChunkTests(unittest.TestCase):
    def test_guitar_resume_export_and_notebook_packaging_preserve_other_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base / 'piano.wav'; output = base / 'guitar'
            frames = 7786580
            sf.write(source, np.full((frames, 2), .02, dtype=np.float32), chunks.RATE, subtype='PCM_16')
            presets = json.dumps([chunks.single.PRESET, drum_probe.PRESET], sort_keys=True)
            plan = chunks.prepare(source, output, instrument='guitar')
            self.assertEqual(plan['preset'], guitar_probe.PRESET)
            self.assertEqual(len(plan['chunks']), 3)
            self.assertTrue(all(c['model_duration'] <= 90 for c in plan['chunks']))
            calls = []
            def generate(root, original, folder, *, start, duration, execute):
                calls.append(start); folder.mkdir()
                if len(calls) == 2:
                    raise RuntimeError('interrupted guitar worker')
                sf.write(folder / 'guitar.wav', np.full((round(duration * chunks.RATE), 2), .03, dtype=np.float32), chunks.RATE, subtype='PCM_16')
                params = {'instruction': 'Generate the guitar track based on the audio context:',
                          'caption': guitar_probe.PRESET['caption'], 'seed': 0, 'guidance_scale': 7,
                          'inference_steps': 50, 'duration': int(np.ceil(duration)), 'task_type': 'lego',
                          'reference_audio': None}
                chunks.write_json(folder / 'guitar_probe.json', {
                    'status': 'generated', 'instrument': 'guitar', 'preset': plan['preset'],
                    'stem_sha256': chunks.sha256(folder / 'guitar.wav'), 'settings': {'params': params},
                    'input': {'original_sha256': plan['generation_source_sha256'], 'start': start,
                              'requested_duration': duration, 'source_gain': .7, 'model_duration': int(np.ceil(duration))}})
            with patch.object(chunks.single, 'model_python', return_value=Path(sys.executable)), \
                 patch.object(chunks.single, 'generate', side_effect=AssertionError('do not regenerate bass')), \
                 patch.object(drum_probe, 'generate', side_effect=AssertionError('do not regenerate drums')), \
                 patch.object(guitar_probe, 'generate', side_effect=generate):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    chunks.run_chunks(ROOT, output / 'plan.json')
                progress = chunks.run_chunks(ROOT, output / 'plan.json')
                self.assertEqual(len(calls), 4)
                chunks.run_chunks(ROOT, output / 'plan.json')
                self.assertEqual(len(calls), 4)
            with patch.object(chunks, 'pitch', side_effect=AssertionError('monophonic scoring invalid for guitar')):
                report = chunks.stitch_run(output / 'plan.json')
            self.assertEqual(report['export_frames'], frames)
            self.assertEqual(report['stem_file'], 'guitar_full.wav')
            self.assertEqual(sf.info(output / 'stitched/guitar_full.wav').frames, frames)
            self.assertTrue(all(s['pitch_change_cents'] is None for s in report['seams']))
            book = json.loads((ROOT / 'notebooks/19_colab_guitar_context_90.ipynb').read_text())
            downloads = []
            scope = {'GUITAR_CHUNK_OUT': output, 'GUITAR_CHUNK_PLAN': output / 'plan.json',
                     'guitar_plan': plan, 'context': chunks, 'json': json, 'zipfile': zipfile,
                     'files': SimpleNamespace(download=downloads.append)}
            exec(compile(''.join(book['cells'][-1]['source']), 'guitar-package', 'exec'), scope)
            self.assertEqual(len(downloads), 2)
            with zipfile.ZipFile(output / 'guitar_chunked_feedback.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertIn('stitched/guitar_full.mp3', z.namelist())
                self.assertEqual(len([n for n in z.namelist() if n.startswith('stitched/chunk_reports/')]), 3)
                self.assertFalse(any(n.startswith('chunks/') for n in z.namelist()))
            with zipfile.ZipFile(output / 'guitar_chunked_result.zip') as z:
                self.assertEqual(len([n for n in z.namelist() if n.startswith('chunks/')]), 3)
                self.assertIn('stitched/guitar_full.wav', z.namelist())
            self.assertEqual(json.dumps([chunks.single.PRESET, drum_probe.PRESET], sort_keys=True), presets)
            attempt = output / progress['0']
            path = attempt / 'guitar_probe.json'; wrong = json.loads(path.read_text())
            wrong['settings']['params']['instruction'] = 'Generate the bass track based on the audio context:'
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'guitar chunk settings'):
                chunks.verify_chunk(attempt, plan, plan['chunks'][0])

    def test_rejects_oversized_guitar_context_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'cannot exceed 90'):
                chunks.prepare(base / 'missing.wav', base / 'bad', context=120, instrument='guitar')
            self.assertFalse((base / 'bad').exists())


if __name__ == '__main__':
    unittest.main()

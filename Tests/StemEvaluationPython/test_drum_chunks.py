"""Instrument routing and cached drum stitching; GPU inference is mocked."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import ace_context_chunks as chunks
import drum_probe


class DrumChunkTests(unittest.TestCase):
    def test_drum_worker_resume_reports_and_export_without_bass_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'piano.wav'
            sf.write(source,np.full((61*chunks.RATE,2),.02),chunks.RATE,subtype='PCM_16')
            plan=chunks.prepare(source,root/'drums',context=30,margin=3,instrument='drums')
            self.assertEqual(plan['preset'],drum_probe.PRESET)
            self.assertEqual(chunks.instrument_of(plan),'drums')
            calls=[]
            def generate(_,source,output,*,start,duration,execute):
                calls.append(start);output.mkdir()
                if len(calls)==2:raise RuntimeError('interrupted drum worker')
                sf.write(output/'drums.wav',np.full((round(duration*chunks.RATE),2),.03),chunks.RATE,subtype='PCM_16')
                chunks.write_json(output/'drum_probe.json',{'status':'generated','preset':plan['preset'],
                    'instrument':'drums','stem_sha256':chunks.sha256(output/'drums.wav'),
                    'input':{'original_sha256':plan['generation_source_sha256'],'start':start,
                        'requested_duration':duration,'source_gain':.7,'model_duration':int(np.ceil(duration))}})
            with patch.object(chunks.single,'model_python',return_value=Path(sys.executable)),patch.object(chunks.single,'generate',side_effect=AssertionError('must not run bass')),patch.object(drum_probe,'generate',side_effect=generate):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    chunks.run_chunks(ROOT,root/'drums/plan.json')
                progress=chunks.run_chunks(ROOT,root/'drums/plan.json')
                self.assertEqual(len(calls),4)
                chunks.run_chunks(ROOT,root/'drums/plan.json')
                self.assertEqual(len(calls),4)
            with patch.object(chunks,'pitch',side_effect=AssertionError('bass pitch score is invalid for drums')):
                report=chunks.stitch_run(root/'drums/plan.json')
            self.assertEqual(report['instrument'],'drums')
            self.assertEqual(report['export_frames'],61*chunks.RATE)
            self.assertEqual(report['stem_file'],'drums_full.wav')
            self.assertFalse((root/'drums/stitched/electric_bass.wav').exists())
            self.assertEqual(len(list((root/'drums/stitched/chunk_reports').glob('*.json'))),3)
            self.assertEqual(len(list((root/'drums/stitched').glob('*_before_stitch.mp3'))),3)
            self.assertTrue(all(s['left_pitch']['hz'] is None for s in report['seams']))
            book=json.loads((ROOT/'notebooks/15_colab_drum_context_90.ipynb').read_text())
            downloads=[]
            scope={'DRUM_CHUNK_OUT':root/'drums','DRUM_CHUNK_PLAN':root/'drums/plan.json',
                   'drum_plan':plan,'context':chunks,'json':json,'zipfile':zipfile,
                   'files':SimpleNamespace(download=lambda path:downloads.append(path))}
            exec(compile(''.join(book['cells'][-1]['source']),'drum-chunk-package-cell','exec'),scope)
            self.assertEqual(len(downloads),2)
            with zipfile.ZipFile(root/'drums/drum_chunked_feedback.zip') as z:
                self.assertEqual(len([n for n in z.namelist() if n.endswith('_before_stitch.mp3')]),3)
                self.assertEqual(len([n for n in z.namelist() if n.startswith('stitched/chunk_reports/')]),3)
                self.assertFalse(any(n.startswith('chunks/') for n in z.namelist()))
            with zipfile.ZipFile(root/'drums/drum_chunked_result.zip') as z:
                self.assertEqual(len([n for n in z.namelist() if n.startswith('chunks/')]),3)
            attempt=root/'drums'/progress['0'];wrong=json.loads((attempt/'drum_probe.json').read_text())
            wrong['preset']=chunks.single.PRESET
            (attempt/'drum_probe.json').write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                chunks.verify_chunk(attempt,plan,plan['chunks'][0])

    def test_rejects_wrong_instrument_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):chunks.prepare(root/'none.wav',root/'bad',instrument='piano')
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()

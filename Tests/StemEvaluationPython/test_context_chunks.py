"""Real PCM fixtures and seam mathematics; these tests do not validate GPU quality."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import ace_context_chunks as chunks


class ContextChunkTests(unittest.TestCase):
    def test_selected_90_second_cap_yields_three_contexts_for_this_song(self):
        plan=chunks.layout(round(243.330625*chunks.RATE))
        self.assertEqual(plan['context_cap_seconds'],90)
        self.assertEqual(len(plan['chunks']),3)
        self.assertTrue(all(c['model_duration']<=90 for c in plan['chunks']))

    def test_context_cap_includes_margins_and_covers_fractional_song(self):
        for seconds in [10,23,59.9,60,60.125,130,243.330625,600]:
            for context,margin in [(22,6),(60,6),(90,6),(120,6)]:
                plan=chunks.layout(round(seconds*chunks.RATE),context,margin)
                parts=plan['chunks']
                self.assertEqual(parts[0]['core_start'],0)
                self.assertEqual(parts[-1]['core_end'],plan['total_frames'])
                self.assertTrue(all(a['core_end']==b['core_start'] for a,b in zip(parts,parts[1:])))
                for c in parts:
                    self.assertLessEqual(c['model_duration'],context)
                    self.assertGreaterEqual(c['model_duration'],10)
                    self.assertLessEqual(c['input_start'],c['core_start'])
                    self.assertGreaterEqual(c['input_end'],c['core_end'])

    def test_actual_provided_boundaries_are_preserved_at_seams(self):
        plan=chunks.layout(130*chunks.RATE,60,6,cues=[44.,86.])
        self.assertEqual(plan['boundary_policies'],['provided_boundary']*2)
        signal=np.full((130*chunks.RATE,2),.02,dtype=np.float32)
        parts=[signal[c['input_start']:c['input_end']] for c in plan['chunks']]
        _,report=chunks.stitch_arrays(plan,parts)
        self.assertEqual([s['center_seconds'] for s in report['seams']],[44.,86.])

    def test_equal_power_is_not_a_phase_cancellation_fix(self):
        width=round(.58*chunks.RATE)
        a,b=chunks.fade_weights(width)
        np.testing.assert_allclose(a*a+b*b,1.,atol=1e-12)
        left=np.full((3*chunks.RATE,2),.1,dtype=np.float32)
        fake={'input_start':0}
        _,_,blend,stats=chunks.seam_window(left,-left,fake,fake,chunks.RATE,width)
        self.assertLess(abs(float(blend[width//2,0])),1e-4)
        self.assertLess(stats['blend_relative_rms_db'],-3)
        self.assertLess(stats['correlation'],-.99)

    def test_stitch_has_exact_length_no_time_shift_and_unblended_interiors(self):
        plan=chunks.layout(130*chunks.RATE)
        signal=(.06*np.sin(2*np.pi*82.4*np.arange(plan['total_frames'])/chunks.RATE)).astype(np.float32)
        stereo=np.column_stack([signal,signal])
        parts=[stereo[c['input_start']:c['input_end']].copy() for c in plan['chunks']]
        output,report=chunks.stitch_arrays(plan,parts)
        self.assertEqual(output.shape,stereo.shape)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[5*chunks.RATE:10*chunks.RATE],stereo[5*chunks.RATE:10*chunks.RATE])
        self.assertEqual(len(report['seams']),len(parts)-1)
        self.assertEqual(report['global_headroom_gain'],1.)
        with self.assertRaises(ValueError):chunks.stitch_arrays(plan,parts[:-1])

    def test_plan_checks_boundary_source_identity_and_does_not_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.wav'
            sf.write(source,np.zeros((12*chunks.RATE,2)),chunks.RATE,subtype='PCM_16')
            cues=root/'cues.json';cues.write_text(json.dumps({'source_sha256':'wrong','downbeats':[5]}))
            with self.assertRaisesRegex(ValueError,'source hash'):
                chunks.prepare(source,root/'invalid',cues_file=cues)
            with patch.object(chunks.single,'generate',side_effect=AssertionError('must not generate')):
                report=chunks.prepare(source,root/'valid')
                self.assertEqual(report['status'],'planned')
                self.assertEqual(chunks.sha256(chunks.generation_source(root/'valid',report)),chunks.sha256(source))
                chunks.load_plan(root/'valid/plan.json')
                saved=chunks.generation_source(root/'valid',report)
                saved.write_bytes(b'changed original')
                with self.assertRaisesRegex(ValueError,'Generation source changed'):
                    chunks.load_plan(root/'valid/plan.json')
                saved.write_bytes(source.read_bytes())
                legacy=dict(report)
                legacy.pop('generation_source_file');legacy.pop('generation_source_sha256')
                self.assertEqual(chunks.generation_source(root/'valid',legacy),root/'valid/source.wav')
            (root/'valid/source.wav').write_bytes(b'changed')
            with self.assertRaises(ValueError):chunks.load_plan(root/'valid/plan.json')

    def test_resume_after_failure_reuses_only_verified_completed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.wav'
            sf.write(source,np.full((61*chunks.RATE,2),.02),chunks.RATE,subtype='PCM_16')
            plan=chunks.prepare(source,root/'run',context=30,margin=3)
            calls=[]
            fail=[True]
            def generate(_,source,output,*,start,duration,execute):
                self.assertEqual(chunks.sha256(source),plan['generation_source_sha256'])
                calls.append(start);output.mkdir()
                if len(calls)==2 and fail[0]:raise RuntimeError('simulated interrupted generation')
                sf.write(output/'electric_bass.wav',np.full((round(duration*chunks.RATE),2),.03),chunks.RATE,subtype='PCM_16')
                chunks.write_json(output/'single_pass.json',{'status':'generated','preset':plan['preset'],
                    'stem_sha256':chunks.sha256(output/'electric_bass.wav'),
                    'input':{'original_sha256':plan['generation_source_sha256'],'start':start,'requested_duration':duration,
                             'source_gain':.7,'model_duration':int(np.ceil(duration))}})
            with patch.object(chunks.single,'model_python',return_value=Path(sys.executable)),patch.object(chunks.single,'generate',side_effect=generate):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    chunks.run_chunks(ROOT,root/'run/plan.json')
                fail[0]=False
                progress=chunks.run_chunks(ROOT,root/'run/plan.json')
                self.assertEqual(len(progress),3)
                self.assertEqual(len(calls),4)
                chunks.run_chunks(ROOT,root/'run/plan.json')
                self.assertEqual(len(calls),4)
                stitched=chunks.stitch_run(root/'run/plan.json')
                self.assertEqual(stitched['export_frames'],61*chunks.RATE)
                self.assertEqual(len(stitched['seams']),2)
                self.assertTrue((root/'run/stitched/mix_full.mp3').is_file())
                stem=root/'run'/progress['0']/'electric_bass.wav'
                with stem.open('ab') as stream:stream.write(b'tamper')
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    chunks.run_chunks(ROOT,root/'run/plan.json')


if __name__=='__main__':unittest.main()

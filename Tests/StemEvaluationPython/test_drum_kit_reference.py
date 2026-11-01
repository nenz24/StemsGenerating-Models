"""A/B input/settings identity and safe WAV recovery; no real GPU inference."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tools/stem_eval')]
import drum_kit_reference as kit


class DrumKitReferenceTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'piano.wav';sf.write(source,np.full((61*32000,2),.03),32000,subtype='PCM_16')
        plan=kit.chunks.prepare(source,root/'original',context=30,margin=3,instrument='drums')
        kit.drum.single.attach(ROOT);reports=[];stems=[]
        for c in plan['chunks'][:2]:
            folder=root/f'fixture_{c["index"]}';folder.mkdir()
            m=kit.drum.single.prepare(source,folder/'prepared',start=c['input_start']/32000,duration=(c['input_end']-c['input_start'])/32000)
            shutil.copyfile(folder/'prepared/original.wav',folder/'drums.wav')
            report={'status':'generated','preset':plan['preset'],'input':m,'stem_sha256':kit.chunks.sha256(folder/'drums.wav'),
                    'settings':{'params':{'caption':plan['preset']['caption'],'seed':0,'guidance_scale':7,
                        'duration':m['model_duration'],'src_audio':str(folder/'prepared/source.wav'),'reference_audio':None},
                        'config':{'seeds':[0],'batch_size':1,'use_random_seed':False}}}
            reports.append(report);stems.append(folder/'drums.wav')
        archive=root/'result.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('plan.json',json.dumps(plan))
            for i,(report,stem) in enumerate(zip(reports,stems)):
                z.write(stem,f'chunks/chunk_{i:03d}_drums.wav')
                z.writestr(f'stitched/chunk_reports/{i:03d}.json',json.dumps({'generation_report':report}))
        return source,archive,plan,reports

    def test_lossless_result_restore_and_plan_do_not_launch_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,archive,plan,reports=self.fixture(root)
            original_hash=kit.chunks.sha256(archive)
            with patch('stream_process.run_logged',side_effect=AssertionError('must not launch GPU')):
                report=kit.compare(ROOT,root/'comparison',result_archive=archive,source=source)
            self.assertEqual(report['status'],'planned')
            self.assertEqual(report['control_stem_sha256'],reports[0]['stem_sha256'])
            self.assertEqual(kit.chunks.sha256(archive),original_hash)
            self.assertEqual(kit.chunks.sha256(root/'comparison/candidate/prepared/source.wav'),reports[0]['input']['source_sha256'])
            progress={}
            for i,r in enumerate(reports):
                folder=root/f'original/chunk_{i:03d}/attempt_000';folder.mkdir(parents=True)
                shutil.copyfile(root/f'fixture_{i}/drums.wav',folder/'drums.wav')
                kit.chunks.write_json(folder/'drum_probe.json',r)
                progress[str(i)]=str(folder.relative_to(root/'original'))
            kit.chunks.write_json(root/'original/progress.json',progress)
            with patch('stream_process.run_logged',side_effect=AssertionError('must not launch GPU')):
                cached=kit.compare(ROOT,root/'cached_comparison',plan_path=root/'original/plan.json')
            self.assertEqual(cached['control_stem_sha256'],report['control_stem_sha256'])
            self.assertEqual(cached['reference_excerpt_sha256'],report['reference_excerpt_sha256'])

    def test_only_reference_and_input_path_may_change(self):
        for wrong in [False,True]:
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,archive,plan,reports=self.fixture(root)
                def generate(_,source,output,*,start,duration,reference_audio,execute):
                    output.mkdir();m=kit.drum.single.prepare(source,output/'prepared',start=start,duration=duration)
                    shutil.copyfile(output/'prepared/original.wav',output/'drums.wav')
                    shutil.copyfile(output/'prepared/original.wav',output/'mix.wav')
                    r=copy.deepcopy(reports[0]);r.update(input=m,stem_sha256=kit.chunks.sha256(output/'drums.wav'))
                    r['settings']['params'].update(reference_audio=str(reference_audio),src_audio=str(output/'prepared/source.wav'))
                    if wrong:r['settings']['params']['guidance_scale']=8
                    kit.chunks.write_json(output/'drum_probe.json',r)
                    return r
                with patch.object(kit.drum,'generate',side_effect=generate) as worker:
                    if wrong:
                        with self.assertRaisesRegex(ValueError,'settings differences'):
                            kit.compare(ROOT,root/'comparison',result_archive=archive,source=source,execute=True)
                    else:
                        r=kit.compare(ROOT,root/'comparison',result_archive=archive,source=source,execute=True)
                        self.assertEqual(r['status'],'generated')
                        self.assertTrue(r['same_piano_model_input'])
                        self.assertEqual(r['actual_parameter_differences'],['reference_audio','src_audio'])
                        self.assertTrue((root/'comparison/drum_kit_feedback.zip').is_file())
                        self.assertTrue((root/'comparison/drum_kit_result.zip').is_file())
                    self.assertEqual(worker.call_count,1)


if __name__=='__main__':unittest.main()

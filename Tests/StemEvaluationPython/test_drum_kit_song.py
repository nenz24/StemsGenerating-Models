"""Shared-reference reuse and full-song routing; synthetic WAVs, mocked inference."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import wave
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tools/stem_eval')]
import drum_kit_song as song


class DrumKitSongTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'piano.wav'
        with wave.open(str(source),'wb') as w:
            w.setparams((2,2,32000,0,'NONE','not compressed'))
            for _ in range(243):w.writeframesraw(b'\xe8\x03\xe8\x03'*32000)
            w.writeframesraw(b'\xe8\x03\xe8\x03'*10580)
        plan=song.chunks.prepare(source,root/'old_plan',instrument='drums')
        old=root/'approved';(old/'candidate').mkdir(parents=True)
        song.drum.single.attach(ROOT)
        c=plan['chunks'][0]
        m=song.drum.single.prepare(source,old/'candidate/prepared',start=c['input_start']/32000,duration=(c['input_end']-c['input_start'])/32000)
        shutil.copyfile(old/'candidate/prepared/original.wav',old/'candidate/drums.wav')
        song.drum.single.export_audio(old/'candidate/drums.wav',old/'reference.wav',30*32000)
        shutil.copyfile(old/'reference.wav',old/'candidate/reference.wav')
        digest=song.chunks.sha256(old/'reference.wav');candidate_sha=song.chunks.sha256(old/'candidate/drums.wav')
        report={'status':'generated','preset':plan['preset'],'input':m,'stem_sha256':candidate_sha,
            'reference_audio':{'path':str(old/'candidate/reference.wav'),'sha256':digest},
            'settings':{'params':{'reference_audio':str(old/'candidate/reference.wav')}}}
        comparison={'status':'generated','target_chunk':0,'preset':plan['preset'],
            'original_source_sha256':plan['original_sha256'],'reference_excerpt_sha256':digest,'candidate_stem_sha256':candidate_sha}
        song.chunks.write_json(old/'comparison.json',comparison)
        song.chunks.write_json(old/'candidate/drum_probe.json',report)
        archive=root/'drum_kit_result.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name in ['comparison.json','reference.wav','candidate/drum_probe.json','candidate/drums.wav']:z.write(old/name,name)
        return source,old,archive,digest,candidate_sha

    def test_cached_approved_zero_is_reused_and_only_two_chunks_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,old,archive,digest,approved=self.fixture(root);out=root/'song'
            plan=song.prepare(ROOT,source,out,comparison_folder=old,expected_candidate_sha=approved,expected_reference_sha=digest)
            self.assertEqual(len(plan['chunks']),3)
            self.assertEqual(song.chunks.sha256(out/'chunk_000/attempt_000/drums.wav'),approved)
            calls=[]
            def generate(_,source,output,*,start,duration,execute,reference_audio):
                calls.append(start);self.assertEqual(song.chunks.sha256(reference_audio),digest)
                output.mkdir()
                m=song.drum.single.prepare(source,output/'prepared',start=start,duration=duration)
                shutil.copyfile(output/'prepared/original.wav',output/'drums.wav')
                shutil.copyfile(reference_audio,output/'reference.wav')
                r={'status':'generated','preset':plan['preset'],'input':m,
                   'stem_sha256':song.chunks.sha256(output/'drums.wav'),
                   'reference_audio':{'path':str(output/'reference.wav'),'sha256':digest},
                   'settings':{'params':{'reference_audio':str(output/'reference.wav')}}}
                song.chunks.write_json(output/'drum_probe.json',r)
            with patch.object(song.chunks.single,'model_python',return_value=Path(sys.executable)),patch.object(song.drum,'generate',side_effect=generate):
                song.chunks.run_chunks(ROOT,out/'plan.json')
                song.chunks.run_chunks(ROOT,out/'plan.json')
            self.assertEqual(len(calls),2);self.assertNotIn(0,calls)
            result=song.chunks.stitch_run(out/'plan.json')
            self.assertEqual(result['export_frames'],plan['total_frames'])
            self.assertEqual(len(result['seams']),2)
            self.assertEqual(song.chunks.sha256(out/'chunk_000/attempt_000/drums.wav'),approved)
            book=json.loads((ROOT/'notebooks/17_colab_drum_shared_kit_song.ipynb').read_text())
            downloads=[]
            exec(compile(''.join(book['cells'][-1]['source']),'shared-kit-package','exec'),
                 {'SHARED_KIT_OUT':out,'SHARED_KIT_PLAN':out/'plan.json','shared_plan':plan,
                  'context':song.chunks,'json':json,'zipfile':zipfile,
                  'files':SimpleNamespace(download=lambda path:downloads.append(path))})
            self.assertEqual(len(downloads),2)
            with zipfile.ZipFile(out/'drum_shared_kit_result.zip') as z:
                self.assertIn('shared_reference.wav',z.namelist())
                self.assertEqual(len([n for n in z.namelist() if n.startswith('chunks/')]),3)
            rpath=out/'chunk_001/attempt_000/drum_probe.json';r=json.loads(rpath.read_text());r['reference_audio']['sha256']='wrong'
            rpath.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'shared reference mismatch'):
                song.chunks.verify_chunk(rpath.parent,plan,plan['chunks'][1])
            with (out/plan['shared_reference']['file']).open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Shared reference'):
                song.chunks.load_plan(out/'plan.json')

    def test_lossless_archive_restore_and_wrong_approval_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,old,archive,digest,approved=self.fixture(root)
            plan=song.prepare(ROOT,source,root/'restored',result_archive=archive,
                              expected_candidate_sha=approved,expected_reference_sha=digest)
            self.assertEqual(song.chunks.sha256(root/'restored/chunk_000/attempt_000/drums.wav'),approved)
            self.assertEqual(plan['shared_reference']['sha256'],digest)
            with self.assertRaisesRegex(ValueError,'Not the approved'):
                song.prepare(ROOT,source,root/'wrong',result_archive=archive,expected_candidate_sha='wrong')
            self.assertFalse((root/'wrong').exists())
            path=old/'candidate/drum_probe.json';bad=json.loads(path.read_text())
            bad['input']['source_sha256']='wrong';path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'piano input differs'):
                song.prepare(ROOT,source,root/'incomplete',comparison_folder=old)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                song.chunks.load_plan(root/'incomplete/plan.json')


if __name__=='__main__':unittest.main()

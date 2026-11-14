"""Existing-file review transport/OLA checks; synthetic data, no musical evidence."""
import array
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import package_coreai_full_song_quality_review as P
from coreai_listening_audio import write_float_wav


def wav_header(frames):
    return (b'RIFF'+struct.pack('<I',50+frames*8)+b'WAVEfmt '+struct.pack('<IHHIIHHH',18,3,2,48000,384000,8,32,0)
        +b'fact'+struct.pack('<II',4,frames)+b'data'+struct.pack('<I',frames*8))


class AudioOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.audio=self.root/'piano-with-band.wav';self.out=self.root/'review.zip'
        self.values=array.array('f',[0.,.25,-.125,.1,-.5,.125])
        write_float_wav(self.audio,self.values,3)

    def test_float32_samples_positions_and_source_are_preserved(self):
        before=self.audio.read_bytes();manifest=P.package(self.out,audio=self.audio)
        self.assertEqual(manifest['mode'],'audio-only');self.assertEqual(manifest['new_inference_calls'],0)
        self.assertEqual(len(manifest['clips']),3);self.assertEqual(self.audio.read_bytes(),before)
        self.assertIn('NOT PROVIDED',manifest['missing_evidence'])
        with zipfile.ZipFile(self.out) as archive:
            for clip in manifest['clips']:
                data=archive.read(clip['path'])
                self.assertEqual(data,before)
                self.assertEqual(clip['start_seconds'],0);self.assertEqual(clip['duration_seconds'],3/48000)
                self.assertEqual(hashlib.sha256(data).hexdigest(),manifest['artifacts'][clip['path']]['sha256'])
        self.assertEqual(P.regions(48000*40,8),[('beginning',0,384000),('middle',768000,384000),('end',1536000,384000)])

    def test_pcm16_decode_and_nonfinite_or_truncated_audio_rejection(self):
        import wave
        self.audio.unlink()
        with wave.open(str(self.audio),'wb') as f:
            f.setparams((2,2,48000,0,'NONE','not compressed'));f.writeframes(struct.pack('<hhhh',-32768,32767,16384,-8192))
        self.assertEqual(list(P.Wave(self.audio).read(0,2)),[-1.,.5,32767/32768,-.25])
        self.audio.write_bytes(self.audio.read_bytes()[:-1])
        with self.assertRaises(ValueError):P.package(self.out,audio=self.audio)
        self.assertFalse(self.out.exists())
        self.audio.unlink();write_float_wav(self.audio,self.values,3)
        raw=bytearray(self.audio.read_bytes());struct.pack_into('<f',raw,58,float('nan'));self.audio.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):P.package(self.out,audio=self.audio)
        self.assertFalse(self.out.exists())

    def test_existing_output_limit_and_input_are_preserved(self):
        self.out.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):P.package(self.out,audio=self.audio)
        self.assertEqual(self.out.read_bytes(),b'keep');self.out.unlink()
        from unittest.mock import patch
        with patch.object(P,'LIMIT',1):
            with self.assertRaisesRegex(ValueError,'32MiB'):P.package(self.out,audio=self.audio)
        self.assertFalse(self.out.exists());self.assertFalse(list(self.root.glob('.full-song-quality-*')))

    def test_cli_needs_no_site_packages_or_models(self):
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/package_coreai_full_song_quality_review.py'),
            '--audio',str(self.audio),'--out',str(self.out)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('audio-only',result.stdout)
        self.assertTrue(self.out.is_file())


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional synthetic container fixture')
class ContainerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import coreai_full_song_assembly as O
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.container=self.root/'Downloads/saved.xcappdata'
        self.execution='a'*64;self.job='b'*64
        self.folder=self.container/'AppData/Documents/FullSongGeneration'/self.execution;self.folder.mkdir(parents=True)
        self.out=self.root/'review.zip';self.plan=P.schedule(961280);n=self.plan['target_samples48']
        self.gain=struct.unpack('<f',struct.pack('<f',.30444536))[0]
        stems={};self.children={}
        for role_index,role in enumerate(P.ROLES):
            assembler=O.OverlapAssembler(self.plan['windows'],n)
            for w in self.plan['windows']:
                i=w['index'];child='attempt-00000000-0000-4000-8000-%012d'%(role_index*10+i)
                parent=self.folder/role/('window-%03d'%i);folder=parent/child;folder.mkdir(parents=True)
                self.children[role,i]=folder
                values=np.full((2,P.FRAMES),(.07+role_index*.03)*(i+1),dtype='<f4');values[1]*=-.5
                values.tofile(folder/('raw-'+role+'.f32le'));assembler.add(i,values)
                latent=np.full((1,750,64),role_index+i,dtype='<f4');latent.tofile(folder/'final-latent.f32le')
                (folder/'report.txt').write_text('LIVE final latent SHA256: '+P.sha256(folder/'final-latent.f32le')+'\n')
                (parent/'completed.json').write_text(json.dumps({'schema_version':1,'execution_sha256':self.execution,
                    'job_reference_sha256':self.job,'instrument':role,'window_index':i,'child_directory':child,
                    'raw_sha256':P.sha256(folder/('raw-'+role+'.f32le')),'report_sha256':P.sha256(folder/'report.txt')}))
            stems[role]=assembler.finish();stems[role].tofile(self.folder/('raw-'+role+'.f32le'))
            (self.folder/(role+'-assembled.json')).write_text(json.dumps({'schema_version':1,'execution_sha256':self.execution,
                'instrument':role,'target_samples48':n,'raw_sha256':P.sha256(self.folder/('raw-'+role+'.f32le'))}))
        stems['piano']=np.full((2,n),.12345,dtype='<f4')
        # Match the device Float32 order and original round-trip Float gain.
        stems['piano-with-band']=(stems['piano']+stems['bass']+stems['drums']+stems['guitar'])*np.float32(.25)
        for name,values in stems.items():
            scaled=(values*np.float32(self.gain)).T.copy().astype('<f4')
            (self.folder/(name+'.wav')).write_bytes(wav_header(n)+scaled.tobytes())
        (self.folder/'completed.json').write_text(json.dumps({'schema_version':1,'execution_sha256':self.execution,
            'job_reference_sha256':self.job,'target_samples48':n,'completed_windows':6,'listening_gain':.30444536,
            'artifact_sha256':{name:P.sha256(self.folder/name) for name in P.WAVS}}))
        (self.folder/'report.txt').write_text('Runtime: CoreAI full-song prepared-input diagnostic\nFull-song collection: COMPLETED\n')

    def test_container_binds_slices_replays_overlap_and_preserves_all_files(self):
        before={p:P.sha256(p) for p in self.container.rglob('*') if p.is_file()}
        manifest=P.package(self.out,source=self.root/'Downloads',execution=self.execution)
        self.assertEqual(manifest['mode'],'container');self.assertEqual(len(manifest['clips']),15)
        self.assertEqual(len(manifest['checks']),12);self.assertTrue(all(c['passed'] for c in manifest['checks'].values()))
        self.assertEqual(manifest['listening_gain_float32'],self.gain)
        self.assertLess(self.out.stat().st_size,32*1024*1024)
        self.assertEqual({p:P.sha256(p) for p in before},before)
        with zipfile.ZipFile(self.out) as archive:
            self.assertEqual(len(archive.namelist()),len(manifest['artifacts'])+1)
            for name,entry in manifest['artifacts'].items():
                data=archive.read(name);self.assertEqual(len(data),entry['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),entry['sha256'])
        with self.assertRaisesRegex(ValueError,'outside'):
            P.package(self.container/'review.zip',source=self.container)

    def test_numerically_wrong_assembly_is_exported_as_failed_evidence(self):
        import numpy as np
        path=self.folder/'raw-bass.f32le';values=np.fromfile(path,dtype='<f4');values+=np.float32(.01);values.tofile(path)
        receipt=self.folder/'bass-assembled.json';r=json.loads(receipt.read_text());r['raw_sha256']=P.sha256(path);receipt.write_text(json.dumps(r))
        manifest=P.package(self.out,source=self.folder)
        self.assertFalse(manifest['checks']['bass_selected_overlap_replay']['passed'])
        self.assertFalse(manifest['checks']['bass_middle_wav_binding']['passed'])
        self.assertTrue(manifest['sampled_transport_or_assembly'].startswith('NOT PASSED'))
        self.assertEqual(manifest['full_song_numerical_parity'],'NOT EVALUATED')
        self.assertTrue(self.out.exists())

    def test_changed_selected_raw_or_receipt_identity_is_rejected_without_export(self):
        path=self.children['bass',0]/'raw-bass.f32le'
        with path.open('r+b') as stream:stream.write(struct.pack('<f',.9))
        with self.assertRaisesRegex(ValueError,'raw SHA'):P.package(self.out,source=self.container)
        self.assertFalse(self.out.exists())
        receipt=self.folder/'completed.json';r=json.loads(receipt.read_text());r['execution_sha256']='c'*64;receipt.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'receipt'):P.package(self.out,source=self.folder)
        self.assertFalse(self.out.exists())

    def test_wrong_requested_run_does_not_fallback_or_repeat_generation(self):
        with self.assertRaisesRegex(ValueError,'do not repeat generation'):
            P.discover(self.root/'Downloads','c'*64)
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()

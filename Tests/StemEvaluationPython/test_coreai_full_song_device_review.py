"""Synthetic device-focus ZIP protocol tests; Apple ZIP/UI execution is separate."""
import array
import hashlib
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
import audit_coreai_full_song_focus_review as A
from coreai_listening_audio import write_float_wav


class DeviceFocusReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.path=self.root/'focus.zip';self.files={};self.execution='a'*64;self.job='b'*64;self.count=96000
        self.m={'schema_version':1,'profile':'private-full-song-quality-review-v1','mode':'device-focus',
            'execution_sha256':self.execution,'job_reference_sha256':self.job,'completed_windows':3,
            'completed_receipt':{'target_samples48':1440000},'focus':{'start_sample':9*48000,'frames':self.count,'window_indices':[0]},
            'schedule':[{'index':0,'sample_start48':0,'sample_end48':1440000}],
            'new_inference_calls':0,'native_exports':0,'baseline_modified':False,'absolute_tolerance':1e-4,
            'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,'full_song_numerical_parity':'NOT EVALUATED',
            'music_acceptance':'NOT EVALUATED BY EXPORT','artifacts':{}}
        hashes={r+'.wav':hashlib.sha256(r.encode()).hexdigest() for r in ('piano','bass','drums','guitar','piano-with-band')}
        self.add('completed.json',json.dumps({'schema_version':1,'execution_sha256':self.execution,'job_reference_sha256':self.job,
            'target_samples48':1440000,'completed_windows':3,'listening_gain':.3,'artifact_sha256':hashes}).encode())
        self.add('device-report.txt',b'SYNTHETIC TRANSPORT FIXTURE; not device or musical evidence')
        gain=struct.unpack('<f',struct.pack('<f',.3))[0]
        for role,value in [('piano',.4),('bass',.1),('drums',.2),('guitar',.3),('piano-with-band',.25)]:
            wav=self.root/(role+'.wav');write_float_wav(wav,array.array('f',[value]*(144000*2)),144000,gain)
            self.add('listening/focus/'+role+'.wav',wav.read_bytes(),frames=144000,start_sample=408000,source_file_sha256=hashes[role+'.wav'])
            if role in ('piano','piano-with-band'):continue
            raw=array.array('f',[value]*(self.count*2)).tobytes();source=hashlib.sha256((role+'raw').encode()).hexdigest()
            self.add(role+'/assembled-focus.f32le',raw,shape=[1,2,self.count],dtype='float32_le',global_start_sample=432000,source_file_sha256=source)
            self.add(role+'/assembled.json',json.dumps({'schema_version':1,'execution_sha256':self.execution,'instrument':role,
                'target_samples48':1440000,'raw_sha256':source}).encode())
            prefix=role+'/window-000/';latent=bytes(48000*4)
            report=('LIVE final latent SHA256: '+hashlib.sha256(latent).hexdigest()+'\n').encode()
            self.add(prefix+'report.txt',report)
            self.add(prefix+'final-latent.f32le',latent,shape=[1,750,64],dtype='float32_le')
            self.add(prefix+'completed.json',json.dumps({'schema_version':1,'execution_sha256':self.execution,'job_reference_sha256':self.job,
                'instrument':role,'window_index':0,'raw_sha256':source,'report_sha256':hashlib.sha256(report).hexdigest()}).encode())
            self.add(prefix+'focus.f32le',raw,shape=[1,2,self.count],dtype='float32_le',global_start_sample=432000,local_start_sample=432000,source_file_sha256=source)

    def add(self,name,data,**properties):
        self.files[name]=data;self.m['artifacts'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),**properties}

    def write(self,prefix='payload/'):
        with zipfile.ZipFile(self.path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,data in self.files.items():z.writestr(prefix+name,data)
            z.writestr(prefix+'review-manifest.json',json.dumps(self.m))

    def test_wrapped_archive_binds_raw_windows_assembly_and_saved_float32_gain(self):
        self.write();before=self.path.read_bytes();r=A.audit(self.path)
        self.assertTrue(r['selected_checks_passed']);self.assertEqual(len(r['checks']),6)
        self.assertEqual(self.path.read_bytes(),before);self.assertEqual(r['new_inference_calls'],0)
        self.assertEqual(r['native_decoder_parity'],'NOT EVALUATED')

    def test_bad_assembly_remains_failed_despite_matching_archive_hashes(self):
        name='bass/assembled-focus.f32le';raw=bytearray(self.files[name]);struct.pack_into('<f',raw,0,.25)
        properties={k:v for k,v in self.m['artifacts'][name].items() if k not in ('sha256','bytes')}
        self.add(name,bytes(raw),**properties);self.write('')
        r=A.audit(self.path);self.assertFalse(r['selected_checks_passed']);self.assertFalse(r['checks']['bass_raw_assembly']['passed'])
        self.assertFalse(r['checks']['bass_wav_binding']['passed'])

    def test_wrong_latent_and_changed_tolerances_are_rejected(self):
        name='bass/window-000/final-latent.f32le';self.add(name,b'bad!'+self.files[name][4:],shape=[1,750,64],dtype='float32_le')
        self.write()
        with self.assertRaisesRegex(ValueError,'latent/report'):A.audit(self.path)
        self.m['absolute_tolerance']=.1;self.write()
        with self.assertRaisesRegex(ValueError,'bounds'):A.audit(self.path)

    def test_corrupt_artifact_and_unsafe_path_are_rejected(self):
        name='drums/assembled-focus.f32le';self.files[name]=b'bad!'+self.files[name][4:];self.write()
        with self.assertRaisesRegex(ValueError,'SHA'):A.audit(self.path)
        self.write('../')
        with self.assertRaisesRegex(ValueError,'Unsafe'):A.audit(self.path)

    def test_audit_cli_is_stdlib_only(self):
        self.write()
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/audit_coreai_full_song_focus_review.py'),
            '--archive',str(self.path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['selected_checks_passed'])


if __name__=='__main__':unittest.main()

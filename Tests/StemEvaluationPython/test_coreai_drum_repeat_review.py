"""Synthetic repeatability ZIP contracts; no device execution or musical evidence."""
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import audit_coreai_drum_repeat_review as A


class RepeatReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'repeat.zip';self.files={}
        self.m={'schema_version':1,'profile':'full-song-one-window-repeatability-v1','instrument':'drums','new_generation_windows':1,
            'new_inference_calls':756,'native_exports':0,'baseline_modified':False,'absolute_tolerance':1e-4,'relative_tolerance':1e-3,
            'relative_rmse_tolerance':1e-3,'sampler_weight_storage_dtype':'bfloat16','compute_dtype':'float32','vae_dtype':'float32',
            'inference_steps':50,'guidance_scale':7,'apg_momentum':-.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
            'music_acceptance':'NOT EVALUATED BY TEST','execution_sha256':'a'*64,'job_reference_sha256':'b'*64,
            'latent_start':0,'latent_end':750,'focus':{'local_start_sample':48000,'global_start_sample':48000,'frames':96000,'listening_gain':.3},
            'used_input_sha256':{},'repeatability_passed':True,'artifacts':{}}
        for name,shape in [('context',[1,750,128]),('noise',[1,750,64]),('condition',[1,127,2048]),('timesteps',[51])]:
            raw=bytes(math.prod(shape)*4) if name!='timesteps' else array.array('f',(1-i/50 for i in range(51))).tobytes()
            self.add('input/'+name+'.f32le',raw,shape);self.m['used_input_sha256'][name]=hashlib.sha256(raw).hexdigest()
        latent=bytes(48000*4);raw=bytes(2880000*4);n=96000
        wav=(b'RIFF'+struct.pack('<I',50+n*8)+b'WAVEfmt '+struct.pack('<IHHIIHHH',18,3,2,48000,384000,8,32,0)
             +b'fact'+struct.pack('<II',4,n)+b'data'+struct.pack('<I',n*8)+bytes(n*8))
        for label in ('captured','repeat'):
            self.add(label+'/final-latent.f32le',latent,[1,750,64]);self.add(label+'/raw-drums.f32le',raw,[1,2,1440000])
            self.add(label+'/report.txt',('LIVE final latent SHA256: '+hashlib.sha256(latent).hexdigest()+'\n').encode())
            self.add(label+'/focus.wav',wav)
        self.add('repeat/final-momentum.f32le',latent,[1,750,64])
        for name,key in [('captured/raw-drums.f32le','original_captured_raw_sha256'),('captured/final-latent.f32le','original_captured_latent_sha256'),
                         ('captured/report.txt','original_captured_report_sha256')]:self.m[key]=self.m['artifacts'][name]['sha256']

    def add(self,name,raw,shape=None):
        self.files[name]=raw;e={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        if shape:e.update(shape=shape,dtype='float32_le')
        self.m['artifacts'][name]=e

    def write(self):
        with zipfile.ZipFile(self.path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,raw in self.files.items():z.writestr('payload/'+name,raw)
            z.writestr('payload/review-manifest.json',json.dumps(self.m))

    def test_same_output_passes_repeatability_without_accepting_music(self):
        self.write();r=A.audit(self.path)
        self.assertTrue(r['repeatability_passed']);self.assertTrue(r['focus_wav_bindings_passed'])
        self.assertEqual(r['music_acceptance'],'NOT EVALUATED BY AUDIT');self.assertFalse(r['baseline_modified'])

    def test_difference_outside_focus_still_fails_full_audio_comparison(self):
        name='repeat/raw-drums.f32le';self.add(name,struct.pack('<f',1.)+self.files[name][4:],[1,2,1440000])
        self.m['repeatability_passed']=False;self.write();r=A.audit(self.path)
        self.assertFalse(r['repeatability_passed']);self.assertEqual(r['checks']['raw-drums']['pointwise_violations'],1)
        self.assertTrue(r['focus_wav_bindings_passed'])

    def test_changed_used_input_is_rejected_despite_new_archive_checksum(self):
        name='input/noise.f32le';self.add(name,b'bad!'+self.files[name][4:],[1,750,64]);self.write()
        with self.assertRaisesRegex(ValueError,'input hash'):A.audit(self.path)

    def test_precision_or_bounds_changes_are_rejected(self):
        for field,value in [('relative_tolerance',.01),('sampler_weight_storage_dtype','float32'),('inference_steps',20)]:
            previous=self.m[field];self.m[field]=value;self.write()
            with self.assertRaisesRegex(ValueError,'bounds'):A.audit(self.path)
            self.m[field]=previous


if __name__=='__main__':unittest.main()

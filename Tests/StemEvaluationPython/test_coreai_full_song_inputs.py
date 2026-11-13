"""Synthetic timeline assembly and inactive input preparation; no learned inference."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import wave

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_coreai_full_song_inputs as J
import coreai_full_song_assembly as O
import prepare_coreai_full_song_context as C
import test_coreai_source_window as source_tests


class PlanAndProofTests(unittest.TestCase):
    def test_plan_needs_no_site_packages_and_makes_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'absent'
            r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/prepare_coreai_full_song_inputs.py'),'--out',str(out)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('PLAN ONLY',r.stdout);self.assertFalse(out.exists())

    def test_missing_standalone_audit_reuses_verified_bundle_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=source_tests.metadata();ref['reference_convolution_backend']='aten'
            with patch('prepare_coreai_source_window.verify_bundle',return_value=ref) as verify:
                proof,path=C.source_proof(root/'missing.json',root/'source-bundle')
                verify.assert_called_once_with(root/'source-bundle')
            self.assertEqual(path,root/'source-bundle/reference.json')
            self.assertEqual(proof['checks'],ref['host_checks'])
            self.assertEqual(proof['reference_convolution_backend'],'aten')
            self.assertEqual(proof['comparisons_passed'],32)

    def test_existing_failed_audit_is_not_replaced_with_successful_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json';path.write_text(json.dumps({'strict_verdict':'NOT PASSED'}))
            with patch('prepare_coreai_source_window.verify_bundle') as verify:
                proof,used=C.source_proof(path,Path(tmp)/'bundle');verify.assert_not_called()
            self.assertEqual(proof['strict_verdict'],'NOT PASSED');self.assertEqual(used,path)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy timeline tests')
class AssemblyTests(unittest.TestCase):
    def test_exact_stereo_reconstruction_with_padding_and_three_way_overlap(self):
        import numpy as np
        for frames32 in (960000,960001,1312047,1920000):
            plan=J.schedule(frames32);n=plan['target_samples48']
            timeline=np.stack([np.arange(n,dtype=np.float32)/n,-np.arange(n,dtype=np.float32)/n]).astype(np.float32)
            output=O.OverlapAssembler(plan['windows'],n)
            for w in plan['windows']:
                part=np.zeros((2,1440000),dtype=np.float32)
                a,b=w['sample_start48'],w['sample_end48'];part[:,:b-a]=timeline[:,a:b]
                output.add(w['index'],part)
            np.testing.assert_array_equal(output.finish(),timeline)
            self.assertTrue((output.weight>0).all())

    def test_missing_duplicate_invalid_or_nonfinite_window_is_not_accepted(self):
        import numpy as np
        p=J.schedule(1920000);output=O.OverlapAssembler(p['windows'],p['target_samples48'])
        part=np.zeros((2,1440000),dtype=np.float32)
        with self.assertRaises(ValueError):output.finish()
        output.add(0,part)
        before=output.weight.copy()
        with self.assertRaises(ValueError):output.add(0,part)
        np.testing.assert_array_equal(output.weight,before)
        part[0,0]=np.nan
        with self.assertRaises(ValueError):output.add(1,part)
        bad=copy.deepcopy(p['windows']);bad[1]['sample_start48']=bad[0]['sample_end48']+1
        with self.assertRaises(ValueError):O.OverlapAssembler(bad,p['target_samples48'])


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('torch'),'Optional NumPy/Torch input transport')
class InputTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from coreai_source_window_contract import tile_plan,VAE_SHA
        from coreai_instrument_preview30_contract import PRESETS
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.prepared=self.root/'prepared';self.prepared.mkdir()
        self.context=self.root/'context';self.context.mkdir();self.conditioning=self.root/'condition';self.conditioning.mkdir()
        self.capture=self.root/'capture';self.capture.mkdir();self.out=self.root/'job'
        wav=self.prepared/'source32.wav'
        with wave.open(str(wav),'wb') as f:f.setparams((2,2,32000,0,'NONE','not compressed'));f.writeframes(bytes(960000*4))
        import hashlib
        self.source={'status':'source_and_schedule_prepared','source_gain':.7,'baseline_modified':False,
                     'source32_wav_sha256':J.sha256(wav),'source32_pcm_sha256':hashlib.sha256(bytes(960000*4)).hexdigest(),
                     'source32_frames':960000,'source_sha256':'0'*64}
        (self.prepared/'report.json').write_text(json.dumps(self.source));(self.prepared/'plan.json').write_text(json.dumps(J.schedule(960000)))
        arrays={'audio48-model.f32le':np.zeros((1,2,1440000),dtype='<f4'),'epsilon-global.f32le':np.ones((1,64,750),dtype='<f4'),
                'source-mask.f32le':np.ones((1,750,64),dtype='<f4'),'posterior_mean.f32le':np.zeros((1,64,750),dtype='<f4'),
                'posterior_std.f32le':np.full((1,64,750),1e-4,dtype='<f4'),'source_latents.f32le':np.full((1,750,64),1e-4,dtype='<f4'),
                'context_latents.f32le':np.concatenate([np.full((1,750,64),1e-4,dtype='<f4'),np.ones((1,750,64),dtype='<f4')],axis=-1)}
        for name,value in arrays.items():value.tofile(self.context/name)
        self.context_report={'status':'host_source_context_captured','source32_wav_sha256':self.source['source32_wav_sha256'],
            'source_gain_already_applied':.7,'additional_gain':1.,'sample_rate':48000,'samples':1440000,'latent_frames':750,
            'vae_checkpoint_sha256':VAE_SHA,'window_frames':256,'core_frames':224,'halo_frames':16,'reference_convolution_backend':'aten',
            'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,'mask_preserved_exact':True,
            'baseline_modified':False,'ready_for_native_execution':False,'source_encoder_calls':4,'expected_source_encoder_calls':4,
            'DiT_calls':0,'Qwen_calls':0,'native_exports':0,'original_source_proof_comparisons_passed':32,'window_plan':tile_plan(750),
            'global_right_padding_samples':0,'artifact_sha256':{name:J.sha256(self.context/name) for name in arrays}}
        (self.context/'report.json').write_text(json.dumps(self.context_report))
        self.cond={'status':'host_conditioning_captured','geometry':{'text':111,'lyric':15,'packed':127},
                   'prompts':{role:{'preset_sha256':PRESETS[role]} for role in J.ROLES},'artifact_sha256':{}}
        for i,role in enumerate(J.ROLES):
            (self.conditioning/role).mkdir()
            for name,value in [('condition.f32le',np.full((1,127,2048),i,dtype='<f4')),('condition-mask.f32le',np.ones((1,127),dtype='<f4'))]:
                value.tofile(self.conditioning/role/name);self.cond['artifact_sha256'][role+'/'+name]=J.sha256(self.conditioning/role/name)
        (self.conditioning/'report.json').write_text(json.dumps(self.cond))
        original=ROOT/'reviews/coreai-drums-host30s-listening-accepted/report.json'
        (self.capture/'report.json').write_bytes(original.read_bytes())
        torch.linspace(1,0,51).numpy().astype('<f4').tofile(self.capture/'timesteps.f32le')

    def build(self):
        with patch('prepare_coreai_instrument_preview30.verify_host',return_value=self.cond):
            return J.build_job(self.prepared,self.context,self.conditioning,self.capture,self.out)

    def test_prepares_three_distinct_noises_and_preserves_legacy_baseline_and_scope(self):
        import torch
        result=self.build();self.assertEqual(J.verify_job(self.out),result)
        self.assertEqual(len(result['runs']),3);self.assertFalse(result['ready_for_native_execution'])
        self.assertEqual(result['native_batch_runtime'],'NOT IMPLEMENTED')
        self.assertEqual(result['sampler_weight_storage_dtype'],'bfloat16');self.assertEqual(result['guidance_scale'],7.)
        before=J.sha256(self.out/'reference.json')
        with patch.object(torch,'randn',side_effect=AssertionError('No new RNG')):
            reused=J.execute(SimpleNamespace(out=self.out,reuse_verified=True,source=None))
        self.assertEqual(reused,result);self.assertEqual(J.sha256(self.out/'reference.json'),before)

    def test_exact_global_slices_are_used_across_overlap(self):
        import numpy as np
        plan=J.schedule(1312047);frames=plan['latent_frames']
        context=np.arange(frames*128,dtype='<f4').reshape(1,frames,128)
        piano=np.zeros((1,2,frames*1920),dtype='<f4');epsilon=np.zeros((1,64,frames),dtype='<f4')
        noises={role:np.arange(frames*64,dtype='<f4').reshape(1,frames,64)+i for i,role in enumerate(J.ROLES)}
        records=J.window_records(plan,context,piano,epsilon,noises,{r:'0'*64 for r in J.ROLES})
        self.assertEqual(len(records),3*len(plan['windows']))
        for i,w in enumerate(plan['windows']):
            self.assertEqual(records[i]['initial_noise_sha256'],J.slice_sha(noises['bass'][:,w['latent_start']:w['latent_end'],:]))
            if i:
                a=w['latent_start'];b=plan['windows'][i-1]['latent_end']
                previous=context[:,plan['windows'][i-1]['latent_start']:b,:]
                current=context[:,a:w['latent_end'],:]
                np.testing.assert_array_equal(previous[:,-(b-a):,:],current[:,:b-a,:])

    def test_legacy_missing_padding_is_derived_without_rewriting_and_bad_position_rejected(self):
        p=self.prepared/'plan.json';plan=json.loads(p.read_text())
        for w in plan['windows']:del w['right_padding_samples48']
        p.write_text(json.dumps(plan));before=p.read_bytes();J.verify_source(self.prepared);self.assertEqual(p.read_bytes(),before)
        plan['windows'][0]['sample_end48']-=1;p.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):J.verify_source(self.prepared)

    def test_changed_copied_condition_or_schedule_cannot_be_accepted(self):
        import numpy as np
        self.build();p=self.out/'bass/condition.f32le';value=np.fromfile(p,dtype='<f4');value[0]+=.5;value.tofile(p)
        ref=json.loads((self.out/'reference.json').read_text());ref['artifact_sha256']['bass/condition.f32le']=J.sha256(p)
        (self.out/'reference.json').write_text(json.dumps(ref))
        with self.assertRaisesRegex(ValueError,'Copied conditioning'):J.verify_job(self.out)

    def test_native_or_music_status_cannot_be_promoted_by_input_preparation(self):
        self.build();path=self.out/'reference.json';original=path.read_text();ref=json.loads(original)
        for key,value in [('ready_for_native_execution',True),('native_batch_runtime','PASS'),('full_song_music_acceptance','PASS'),('guidance_scale',5.)]:
            path.write_text(json.dumps({**ref,key:value}))
            with self.assertRaises(ValueError):J.verify_job(self.out)
        path.write_text(original)

    def test_inconsistent_context_rejected_before_any_output(self):
        import numpy as np
        p=self.context/'context_latents.f32le';value=np.fromfile(p,dtype='<f4');value[0]+=.5;value.tofile(p)
        self.context_report['artifact_sha256'][p.name]=J.sha256(p)
        (self.context/'report.json').write_text(json.dumps(self.context_report))
        with self.assertRaisesRegex(ValueError,'consistency'):self.build()
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()

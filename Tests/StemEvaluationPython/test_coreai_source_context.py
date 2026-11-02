"""Posterior math, PCM and bundle integrity; no emulated CoreAI inference."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import export_coreai_source_context as MODULE


class SourceContextTests(unittest.TestCase):
    def test_sample_matches_diffusers_distribution_and_preserves_source(self):
        import torch
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution
        parameters=torch.linspace(-3,3,128*8).reshape(1,128,8)
        epsilon=torch.randn((1,64,8),generator=torch.Generator().manual_seed(31))
        mask=(torch.arange(8)%2).reshape(1,8,1).expand(1,8,64).float()
        mean,std,source,context=MODULE.posterior_context(parameters,epsilon,mask)
        distribution=OobleckDiagonalGaussianDistribution(parameters)
        expected=distribution.sample(generator=torch.Generator().manual_seed(31)).transpose(1,2)
        torch.testing.assert_close(source,expected,rtol=0,atol=0)
        torch.testing.assert_close(std,distribution.std,rtol=0,atol=0)
        self.assertTrue(torch.all(std>0))
        self.assertFalse(torch.equal(source,mean.transpose(1,2)))
        torch.testing.assert_close(context[:,:,:64],source,rtol=0,atol=0)
        torch.testing.assert_close(context[:,:,64:],mask,rtol=0,atol=0)

    def fixture(self,folder):
        folder.mkdir()
        data={f'{n}.f32le':bytes(math.prod(s)*4) for n,s in MODULE.INPUTS.items()}
        data['chunk_mask.f32le']=struct.pack('<f',1.)*math.prod(MODULE.INPUTS['chunk_mask'])
        data.update({f'{n}-reference.f32le':bytes(math.prod(s)*4) for n,s in MODULE.OUTPUTS.items()})
        for n in ['main.hash','main.mlirb','metadata.json']:data['source-context.aimodel/'+n]=b'integrity fixture; NOT an executable model'
        for name,raw in data.items():
            p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        record=next(r for r in json.loads((ROOT/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
        ref={'schema_version':1,'profile':'source-context','asset_file':'source-context.aimodel','function_name':'main',
             'input_shapes':MODULE.INPUTS,'output_shapes':MODULE.OUTPUTS,'scalar_type':'float32','byte_order':'little',
             'checkpoint_digest':record['digest'],'checkpoint_full_sha256_verified':True,'task_type':'lego','is_covers':False,
             'posterior_mode':'sample_with_explicit_epsilon','sample_rate':48000,'audio_seconds':5.12,'parameter_count':84264832,
             'handler_source_revision':json.loads((ROOT/'configs/coreai-lego-preprocessing.json').read_text())['revision'],
             'recording':{'name':'fixture.wav','sha256':'a'*64,'start_seconds':0.},'lego_mask':{'source_preservation_exact':True},
             'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
             'host_distribution_sample_parity_exact':True,'host_weight_norm_folding_parity_exact':True,
             'artifact_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in data.items()}}
        (folder/'reference.json').write_text(json.dumps(ref));return ref

    def test_contract_and_corrupt_binary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref=self.fixture(folder);MODULE.verify_bundle(folder)
            for key,value in [('posterior_mode','mean'),('is_covers',True),('task_type','cover'),('absolute_tolerance',.1),('input_shapes',{'audio':[1,2,48000]})]:
                changed=copy.deepcopy(ref);changed[key]=value
                (folder/'reference.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError,msg=key):MODULE.verify_bundle(folder)
            (folder/'reference.json').write_text(json.dumps(ref))
            p=folder/'source_latents-reference.f32le';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'integrity mismatch'):MODULE.verify_bundle(folder)

    def test_nonfinite_and_non_full_mask_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref=self.fixture(folder)
            p=folder/'epsilon.f32le';p.write_bytes(struct.pack('<f',float('nan'))+p.read_bytes()[4:])
            ref['artifact_sha256']['epsilon.f32le']=MODULE.sha256(p)
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):MODULE.verify_bundle(folder)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref=self.fixture(folder);p=folder/'chunk_mask.f32le'
            p.write_bytes(bytes(p.stat().st_size));ref['artifact_sha256']['chunk_mask.f32le']=MODULE.sha256(p)
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'full-range'):MODULE.verify_bundle(folder)

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required for actual PCM canonicalization')
    def test_actual_pcm_channel_order_and_short_audio_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'stereo.wav'
            with wave.open(str(p),'wb') as w:
                w.setnchannels(2);w.setsampwidth(2);w.setframerate(48000)
                w.writeframes(struct.pack('<hh',3276,-1638)*MODULE.SAMPLES)
            result=MODULE.canonical_audio(p,0)
            self.assertEqual(list(result.shape),MODULE.INPUTS['audio'])
            self.assertAlmostEqual(float(result[0,0,100]),3276/32768,places=6)
            self.assertAlmostEqual(float(result[0,1,100]),-1638/32768,places=6)
            with self.assertRaisesRegex(ValueError,'less than'):MODULE.canonical_audio(p,5.)

    def test_existing_destination_preserved_and_shell_syntax_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'keep';marker.write_text('original')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/export_coreai_source_context.py'),'--out',tmp],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'original')
        result=subprocess.run(['bash','-n',str(ROOT/'scripts/prepare_coreai_source_context.sh')],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

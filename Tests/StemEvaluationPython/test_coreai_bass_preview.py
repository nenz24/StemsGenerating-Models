"""Integration contract/APG checks; no simulated native inference evidence."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import export_coreai_bass_preview as M


class BassPreviewTests(unittest.TestCase):
    def test_apg_feedback_matches_double_projection_and_is_not_plain_cfg(self):
        import torch
        g=torch.Generator().manual_seed(16)
        velocity=torch.randn((2,128,64),generator=g)
        latent=torch.randn((1,128,64),generator=g)
        momentum=torch.randn((1,128,64),generator=g)
        dt=torch.tensor([.02])
        actual,next_momentum,guided=M.apg_euler(velocity,latent,momentum,dt)
        a,b=velocity.chunk(2);expected_momentum=(a-b)+(-.75*momentum)
        torch.testing.assert_close(next_momentum,expected_momentum,rtol=0,atol=0)
        diff=expected_momentum*torch.minimum(torch.ones_like(momentum),2.5/expected_momentum.norm(p=2,dim=[1],keepdim=True))
        unit=torch.nn.functional.normalize(a.double(),dim=[1])
        orthogonal=(diff.double()-(diff.double()*unit).sum(dim=[1],keepdim=True)*unit).float()
        expected_guided=a+6*orthogonal
        torch.testing.assert_close(guided,expected_guided,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(actual,latent-guided*dt,rtol=0,atol=0)
        self.assertFalse(torch.allclose(guided,b+7*(a-b)))
        result=M.apg_euler(torch.zeros_like(velocity),latent,torch.zeros_like(momentum),dt)
        torch.testing.assert_close(result[0],latent,rtol=0,atol=0)
        self.assertTrue(all(torch.isfinite(v).all() for v in result))

    def test_stage_plan_covers_24_layers_and_both_guidance_branches(self):
        plan=M.stage_plan();self.assertEqual(len(plan),14)
        self.assertEqual([i for s in plan for i in s['layer_indices']],list(range(24)))
        state=dict(M.INPUTS)
        for stage in plan:
            for n,s in stage['input_shapes'].items():self.assertEqual(state[n],s)
            for n,s in stage['output_shapes'].items():state[stage['output_bindings'][n]]=s
        self.assertEqual(state['velocity'],[2,128,64])
        self.assertEqual(M.APG_OUTPUTS['latent_next'],[1,128,64])

    def fixture(self,folder,source,condition):
        folder.mkdir();source.mkdir();condition.mkdir()
        (source/'reference.json').write_text('source reference identity only; no inference')
        (condition/'reference.json').write_text('condition reference identity only; no inference')
        required={'initial-noise.f32le':[1,128,64],'timesteps.f32le':[51]}
        for step in M.CHECKPOINTS:
            for kind in ['reference','fp32-reference']:required[f'step-{step}-latent-{kind}.f32le']=[1,128,64]
        for kind in ['reference','fp32-reference']:required[f'audio-{kind}.f32le']=[1,2,245760]
        for n,s in required.items():(folder/n).write_bytes(bytes(4*math.prod(s)))
        (folder/'timesteps.f32le').write_bytes(struct.pack('<51f',*(1-i/50 for i in range(51))))
        for asset in [*[s['asset_file'] for s in M.stage_plan()],'apg-euler.aimodel','decoder.aimodel']:
            (folder/asset).mkdir()
            for n in ['main.hash','main.mlirb','metadata.json']:(folder/asset/n).write_bytes(b'NOT executable; integrity-only fixture')
        ref={'schema_version':1,'profile':'bass-preview-512','stages':M.stage_plan(),'inference_steps':50,'guidance_scale':7,'shift':1,
             'method':'euler_ode_apg','apg_momentum':-.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
             'apg_projection_precision':'float32_validated_against_original_float64','checkpoints':M.CHECKPOINTS,'function_name':'main',
             'latent_frames':128,'condition_tokens':117,'sample_rate':48000,'audio_seconds':5.12,'scalar_type':'float32','byte_order':'little',
             'checkpoint_digest':M.records()['model.safetensors']['digest'],
             'vae_checkpoint_digest':'da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0',
             'all_checkpoints_verified':True,'host_dual_trajectory_parity_passed':True,'host_decoder_folding_exact':True,
             'host_comparisons':[{'step':i+1,'latent':{'parity_passed':True}} for i in range(50)],
             'apg_same_input_vs_publisher':[{'parity_passed':True} for _ in range(50)],'host_audio_vs_fp32':{'parity_passed':True},
             'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
             'source_reference_sha256':M.sha256(source/'reference.json'),'condition_reference_sha256':M.sha256(condition/'reference.json'),
             'artifact_sha256':{str(p.relative_to(folder)):M.sha256(p) for p in folder.rglob('*') if p.is_file()}}
        (folder/'reference.json').write_text(json.dumps(ref));return ref

    def test_rejects_changed_component_identity_guidance_and_missing_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'bundle';source=root/'source';condition=root/'condition'
            ref=self.fixture(folder,source,condition);M.verify_bundle(folder,source,condition)
            for k,v in {'guidance_scale':1,'inference_steps':4,'stages':list(reversed(ref['stages'])),
                        'apg_momentum':0,'source_reference_sha256':'0'*64,'host_audio_vs_fp32':{'parity_passed':False}}.items():
                changed=copy.deepcopy(ref);changed[k]=v;(folder/'reference.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError,msg=k):M.verify_bundle(folder,source,condition)
            (folder/'reference.json').write_text(json.dumps(ref));(source/'reference.json').write_text('changed recording bundle')
            with self.assertRaisesRegex(ValueError,'changed component'):M.verify_bundle(folder,source,condition)

    def test_rejects_corruption_nonfinite_and_invalid_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'bundle';source=root/'source';condition=root/'condition';ref=self.fixture(folder,source,condition)
            path=folder/'initial-noise.f32le';raw=path.read_bytes();path.write_bytes(struct.pack('<f',float('nan'))+raw[4:])
            ref['artifact_sha256'][path.name]=M.sha256(path);(folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):M.verify_bundle(folder,source,condition)
            path.write_bytes(raw);ref['artifact_sha256'][path.name]=M.sha256(path)
            path=folder/'timesteps.f32le';path.write_bytes(struct.pack('<51f',*([1.]*50+[0.])))
            ref['artifact_sha256'][path.name]=M.sha256(path);(folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'schedule'):M.verify_bundle(folder,source,condition)

    def test_existing_output_preserved_and_shell_syntax(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'keep';marker.write_text('original')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/export_coreai_bass_preview.py'),'--out',tmp],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'original')
        result=subprocess.run(['bash','-n',str(ROOT/'scripts/prepare_coreai_bass_preview.sh')],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

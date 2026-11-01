"""Full-denoiser preparation checks; integrity fixtures are not executable models."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import export_coreai_denoiser as MODULE
from export_coreai_dit_block import convert_storage16


class FullDenoiserTests(unittest.TestCase):
    def fixture(self,folder):
        folder.mkdir()
        shapes={f'{n}.f32le':s for n,s in MODULE.INPUT_SHAPES.items()}
        shapes.update({'velocity-reference.f32le':MODULE.OUTPUT_SHAPE,'velocity-fp32-reference.f32le':MODULE.OUTPUT_SHAPE})
        data={n:bytes(math.prod(s)*4) for n,s in shapes.items()}
        data.update({'denoiser.aimodel/main.hash':b'fixture only','denoiser.aimodel/main.mlirb':b'not executable',
                     'denoiser.aimodel/metadata.json':b'{}'})
        for name,value in data.items():
            p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
        record={'digest':'fixture checkpoint'}
        ref={'schema_version':1,'profile':'denoiser','asset_file':'denoiser.aimodel','function_name':'main','output_name':'velocity',
             'input_shapes':MODULE.INPUT_SHAPES,'output_shape':MODULE.OUTPUT_SHAPE,'layer_count':24,'checkpoint_prefix':'decoder.',
             'latent_frames':25,'weight_storage_dtype':'float16','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
             'checkpoint':record,'checkpoint_full_sha256_verified':True,'original_fp32_reference_file':'velocity-fp32-reference.f32le',
             'storage16_vs_original_fp32':{'parity_passed':True},
             'artifact_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in data.items()}}
        (folder/'reference.json').write_text(json.dumps(ref))
        return ref,record

    def test_full_model_contract_requires_all_layers_and_original_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref,record=self.fixture(folder)
            MODULE.verify_bundle(folder,record)
            for key,value in [('layer_count',2),('checkpoint_prefix','encoder.'),('output_shape',[1,26,64]),
                              ('storage16_vs_original_fp32',{'parity_passed':False})]:
                (folder/'reference.json').write_text(json.dumps(dict(ref,**{key:value})))
                with self.assertRaises(ValueError):MODULE.verify_bundle(folder,record)
            (folder/'reference.json').write_text(json.dumps(ref))
            (folder/'velocity-fp32-reference.f32le').write_bytes(b'bad')
            ref['artifact_sha256']['velocity-fp32-reference.f32le']=hashlib.sha256(b'bad').hexdigest()
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder,record)

    def test_existing_output_is_preserved_before_imports_or_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'marker';marker.write_text('keep')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/export_coreai_denoiser.py'),'--out',tmp],
                                  capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'keep')

    def test_manifest_symlink_cannot_read_outside_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref,record=self.fixture(folder)
            outside=Path(tmp)/'outside';outside.write_bytes(b'keep')
            (folder/'noise.f32le').unlink();(folder/'noise.f32le').symlink_to(outside)
            ref['artifact_sha256']['noise.f32le']=hashlib.sha256(b'keep').hexdigest()
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder,record)
            self.assertEqual(outside.read_bytes(),b'keep')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Requires the actual CPU export environment')
    def test_storage16_preserves_analytical_buffers_and_fp32_projection_calculations(self):
        import torch
        model=torch.nn.Sequential(torch.nn.Conv1d(2,4,2,stride=2),torch.nn.ConvTranspose1d(4,2,2,stride=2),
                                  torch.nn.Flatten(),torch.nn.Linear(12,3)).eval().requires_grad_(False)
        model.register_buffer('inv_freq',torch.tensor([0.123456789,1e-10]))
        original_buffer=model.inv_freq.clone()
        for p in model.parameters():p.data.fill_(0.25)
        x=torch.arange(12,dtype=torch.float32).reshape(1,2,6)/13
        with torch.no_grad():reference=model(x)
        convert_storage16(model)
        self.assertTrue(all(p.dtype==torch.float16 for p in model.parameters()))
        self.assertTrue(torch.equal(model.inv_freq,original_buffer))
        self.assertEqual(model.inv_freq.dtype,torch.float32)
        with torch.no_grad():actual=model(x)
        self.assertEqual(actual.dtype,torch.float32)
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()

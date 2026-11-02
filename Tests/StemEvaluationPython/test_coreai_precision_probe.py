"""Integrity/selection gates using NONEXECUTABLE payloads, not native inference."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_coreai_precision_probe as PROBE
import stage_coreai_app_models as STAGE


class PrecisionProbeTests(unittest.TestCase):
    def test_small_normalized_rmse_does_not_override_pointwise_failure(self):
        import torch
        # The largest absolute error (0.5 at a large reference) is allowed.
        # The smaller error at zero violates the unchanged absolute bound.
        reference=torch.tensor([1000.,0.],dtype=torch.float32)
        actual=torch.tensor([1000.5,0.0002],dtype=torch.float32)
        metrics=PROBE.precision_metrics(actual,reference)
        self.assertLess(metrics['normalized_rmse'],1e-3)
        self.assertEqual(metrics['max_absolute_error'],0.5)
        self.assertEqual(metrics['elementwise_fraction_over_bound'],0.5)
        self.assertFalse(metrics['parity_passed'])

    def fixture(self,candidates):
        folder=candidates/PROBE.PROFILE;folder.mkdir(parents=True)
        for asset in PROBE.ASSETS.values():
            for name in PROBE.NATIVE_FILES:
                p=folder/asset/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'NONEXECUTABLE precision-probe integrity fixture '+asset.encode()+name.encode())
        for name,shape in PROBE.FILES.items():(folder/name).write_bytes(bytes(4*math.prod(shape)))
        zero={'finite':True,'parity_passed':True,'max_absolute_error':0.,'rmse':0.,'normalized_rmse':0.,'elementwise_fraction_over_bound':0.}
        ref={'schema_version':1,'profile':PROBE.PROFILE,'function_name':'main','output_name':'hidden_out',
             'layer_indices':[0,1],'input_shapes':PROBE.INPUTS,'output_shape':PROBE.OUTPUT,'assets':PROBE.ASSETS,
             'pass_order':PROBE.ORDER,'runs_per_pass':4,'scalar_type':'float32','byte_order':'little',
             'compute_dtype':'float32','checkpoint_digest':PROBE.records()['model.safetensors']['digest'],
             'checkpoint_full_sha256_verified':True,'learned_parameter_values_identical':True,
             'reference_file':'hidden-reference.f32le','absolute_tolerance':1e-4,'relative_tolerance':1e-3,
             'relative_rmse_tolerance':1e-3,'source_pair_manifest_profile':'bass-preview-1024',
             'source_weight_storage_dtype':'bfloat16','source_pair_manifest_sha256':'a'*64,
             'host_bf16_vs_original_fp32':zero,'host_expanded_fp32_vs_bf16':zero,
             'artifact_sha256':{p.relative_to(folder).as_posix():PROBE.sha256(p) for p in folder.rglob('*') if p.is_file()}}
        ref['source_pair_artifact_sha256']={n:ref['artifact_sha256'][f'{PROBE.ASSETS["bfloat16"]}/{n}'] for n in PROBE.NATIVE_FILES}
        self.write(folder,ref)
        return folder,ref

    def write(self,folder,ref):(folder/'reference.json').write_text(json.dumps(ref))

    def test_input_geometry_matches_actual_long_cfg_pair_and_verified_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,ref=self.fixture(Path(tmp))
            self.assertEqual(PROBE.INPUTS['hidden'],[2,128,2048])
            self.assertEqual(PROBE.INPUTS['condition'],[2,118,2048])
            self.assertEqual(PROBE.INPUTS['cross_mask'],[1,1,128,118])
            self.assertEqual(PROBE.verify_bundle(folder),ref)

    def test_changed_values_precision_geometry_bounds_or_order_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,ref=self.fixture(Path(tmp))
            cases=[('learned_parameter_values_identical',False),('source_weight_storage_dtype','float16'),
                   ('checkpoint_full_sha256_verified',1),('layer_indices',[2,3]),('pass_order',['float32','bfloat16']),
                   ('runs_per_pass',3),('relative_rmse_tolerance',.01),('input_shapes',{'hidden':[1,128,2048]})]
            for key,value in cases:
                changed=copy.deepcopy(ref);changed[key]=value;self.write(folder,changed)
                with self.assertRaises(ValueError,msg=key):PROBE.verify_bundle(folder)
            changed=copy.deepcopy(ref);changed['host_expanded_fp32_vs_bf16']['max_absolute_error']=1e-7;self.write(folder,changed)
            with self.assertRaisesRegex(ValueError,'exact host values'):PROBE.verify_bundle(folder)

    def test_source_asset_identity_missing_payload_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,ref=self.fixture(Path(tmp))
            changed=copy.deepcopy(ref);changed['source_pair_artifact_sha256']['main.hash']='f'*64;self.write(folder,changed)
            with self.assertRaisesRegex(ValueError,'identity changed'):PROBE.verify_bundle(folder)
            self.write(folder,ref)
            p=folder/'blocks-fp32.aimodel/main.mlirb';p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity mismatch'):PROBE.verify_bundle(folder)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):PROBE.verify_bundle(folder)

    def test_probe_selection_excludes_full_models_and_preserves_local_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ModelCandidates';self.fixture(source)
            old=source/'bass-preview-1024/large-placeholder';old.parent.mkdir();old.write_text('preserve old weights')
            PROBE.activate(source)
            dest=root/'build/Probe.app/ModelCandidates'
            staged=STAGE.stage_resources(source,dest)
            self.assertEqual(staged['selected_profile'],PROBE.PROFILE)
            self.assertTrue(all(n.startswith(PROBE.PROFILE+'/') for n in staged['files']))
            self.assertFalse((dest/'bass-preview-1024').exists())
            self.assertEqual(old.read_text(),'preserve old weights')
            self.assertEqual(STAGE.stage_resources(source,dest)['copied_files'],0)

    def test_failed_activation_preserves_previous_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);folder,_=self.fixture(source)
            selected=source/'active-preview.json';selected.write_text('previous selector')
            (folder/'hidden-reference.f32le').write_bytes(b'broken')
            with self.assertRaises(ValueError):PROBE.activate(source)
            self.assertEqual(selected.read_text(),'previous selector')

    def test_extra_probe_payload_is_rejected_before_pruning_built_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ModelCandidates';folder,ref=self.fixture(source)
            dest=root/'build/Probe.app/ModelCandidates'
            STAGE.stage_resources(source,dest,PROBE.PROFILE)
            marker=dest/PROBE.PROFILE/'hidden-reference.f32le';saved=marker.read_bytes()
            extra=folder/'unneeded-generator';extra.write_text('exclude this payload')
            ref['artifact_sha256'][extra.name]=PROBE.sha256(extra);self.write(folder,ref)
            with self.assertRaisesRegex(ValueError,'Unexpected precision-probe payload'):
                STAGE.stage_resources(source,dest,PROBE.PROFILE)
            self.assertEqual(marker.read_bytes(),saved)

    def test_existing_invalid_bundle_is_preserved_without_reexport(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/PROBE.PROFILE;folder.mkdir();marker=folder/'keep';marker.write_text('preserve')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_coreai_precision_probe.py'),'--out',str(folder)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(marker.read_text(),'preserve')


if __name__=='__main__':unittest.main()

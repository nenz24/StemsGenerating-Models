"""Geometry, strict gates and NONEXECUTABLE transport; no device inference."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import coreai_source_window_contract as C
import prepare_coreai_source_window as W
import stage_coreai_app_models as S


def metadata():
    metric={'finite':True,'parity_passed':True,'max_absolute_error':0.,'rmse':0.,'normalized_rmse':0.,'elementwise_fraction_over_bound':0.}
    names={f'core-{i}-{n}' for i in range(4) for n in C.GOLD_FILES}|{f'seam-{i}-{n}' for i in range(1,4) for n in C.GOLD_FILES}|{'overall-'+n for n in C.GOLD_FILES}
    hashes={n:'0'*64 for n in C.FILES};hashes['capture-report.json']=C.ACCEPTED_REPORT_SHA256
    return {'schema_version':1,'profile':C.PROFILE,'status':'prepared_source_window_encoder',
        'asset_file':'source-context.aimodel','function_name':'main','latent_frames':750,'window_frames':256,'core_frames':224,'halo_frames':16,
        'input_shapes':C.INPUT_SHAPES,'output_shapes':C.OUTPUT_SHAPES,'vae_checkpoint_sha256':C.VAE_SHA,
        'capture_report_sha256':C.ACCEPTED_REPORT_SHA256,'weight_storage_dtype':'float32','compute_dtype':'float32',
        'scalar_type':'float32','byte_order':'little','absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
        'window_plan':C.tile_plan(750),'dependency_sample_bounds':[C.dependency_bounds(t,750) for t in C.tile_plan(750)],
        'host_numerical_gate_passed':True,'coverage_exactly_once':True,'architecture_guard_passed':True,
        'epsilon_regenerated':False,'native_exports':0,'baseline_modified':False,'native_parity':'NOT EVALUATED',
        'native_memory_gate':'NOT EVALUATED','full_song_native_readiness':'NOT EVALUATED','artifact_sha256':hashes,
        'original_source_reference_sha256':'0'*64,'host_checks':{n:dict(metric) for n in names},
        'tile_input_sha256':[{n:'0'*64 for n in C.INPUT_SHAPES} for _ in range(4)]}


class SourceWindowTests(unittest.TestCase):
    def test_every_retained_frame_is_covered_once_with_true_edge_dependencies(self):
        for width in (128,256):
            for total in (width,width+1,750,1024,6000):
                covered=[]
                for tile in C.tile_plan(total,width):
                    covered.extend(range(tile['core_start'],tile['core_end']))
                    lo,hi=C.dependency_bounds(tile,total)
                    self.assertTrue(0<=lo<=hi<total*1920)
                self.assertEqual(covered,list(range(total)))
        self.assertEqual([t['window_start'] for t in C.tile_plan(750)],[0,208,432,494])

    def test_inadequate_context_or_bad_crop_cannot_pass_receptive_field_proof(self):
        good=C.tile_plan(750)[1]
        for bad in [dict(good,window_start=224,crop_start=0),dict(good,window_start=192,crop_start=32)]:
            with self.assertRaisesRegex(ValueError,'Artificial'):C.dependency_bounds(bad,750)
        with self.assertRaises(ValueError):C.dependency_bounds(dict(good,crop_start=15),750)
        with self.assertRaises(ValueError):C.tile_plan(127,128)

    def test_source_contract_never_promotes_native_or_relaxes_original_limits(self):
        ref=metadata();self.assertEqual(C.validate_reference(ref),ref)
        for key,value in [('weight_storage_dtype','bfloat16'),('absolute_tolerance',1e-3),('epsilon_regenerated',True),
                          ('native_parity','PASS'),('native_exports',1),('full_song_native_readiness','PASS')]:
            with self.assertRaises(ValueError):C.validate_reference({**ref,key:value})
        bad=copy.deepcopy(ref);bad['host_checks']['overall-source_latents']['elementwise_fraction_over_bound']=1/48000
        with self.assertRaises(ValueError):C.validate_reference(bad)
        bad=copy.deepcopy(ref);del bad['host_checks']['seam-1-posterior_mean']
        with self.assertRaises(ValueError):C.validate_reference(bad)

    def test_source_encoder_reuse_requires_intact_original_fp32_256_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'source-context.aimodel').mkdir();hashes={}
            for name in C.NATIVE_FILES:
                path=folder/'source-context.aimodel'/name;path.write_bytes(b'NONEXECUTABLE source transport fixture')
                hashes['source-context.aimodel/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
            ref={'profile':'source-context-1024','asset_file':'source-context.aimodel','function_name':'main',
                 'input_shapes':C.INPUT_SHAPES,'output_shapes':C.OUTPUT_SHAPES,'scalar_type':'float32','byte_order':'little',
                 'checkpoint_digest':C.VAE_SHA,'checkpoint_full_sha256_verified':True,
                 'host_distribution_sample_parity_exact':True,'host_weight_norm_folding_parity_exact':True,'artifact_sha256':hashes}
            path=folder/'reference.json';path.write_text(json.dumps(ref));W.verify_reusable(folder)
            path.write_text(json.dumps({**ref,'scalar_type':'float16'}))
            with self.assertRaises(ValueError):W.verify_reusable(folder)
            path.write_text(json.dumps(ref));(folder/'source-context.aimodel/main.hash').write_bytes(b'changed')
            with self.assertRaises(ValueError):W.verify_reusable(folder)

    def test_missing_or_bad_reuse_fails_before_any_source_model_or_selector_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with (patch.object(W,'verify_reusable',side_effect=ValueError('wrong source asset')),
                 patch('audit_coreai_source_window.execute') as run,
                 patch('activate_coreai_drum_reference30.select_profile') as select):
                with self.assertRaises(ValueError):W.prepare(root/'capture',root/'out',root/'old')
                run.assert_not_called();select.assert_not_called()
            self.assertFalse((root/'out').exists())

    def test_thirteen_file_stdlib_build_rejects_wrong_contract_before_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'repo/ModelCandidates';folder=source/C.BUNDLE;folder.mkdir(parents=True)
            ref=metadata()
            for name in C.FILES:
                path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
                if name in C.DATA_SHAPES:
                    with path.open('wb') as stream:stream.truncate(math.prod(C.DATA_SHAPES[name])*4)
                else:path.write_bytes(b'NONEXECUTABLE packaging fixture')
            path=folder/'reference.json';path.write_text(json.dumps(ref))
            (source/'active-preview.json').write_text(json.dumps({'schema_version':1,'resource_profile':C.PROFILE,'audio_seconds':30.}))
            dest=root/'build/Probe.app/ModelCandidates'
            result=S.stage_resources(source,dest);self.assertEqual(len(result['files']),13)
            self.assertEqual(S.stage_resources(source,dest)['copied_files'],0)
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/stage_coreai_app_models.py'),
                '--source',str(source),'--destination',str(dest),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('13 files',result.stdout)
            before=(dest/C.BUNDLE/'reference.json').read_bytes();ref['native_parity']='PASS';path.write_text(json.dumps(ref))
            sentinel=dest/'preserve';sentinel.write_text('prior app output')
            with self.assertRaises(ValueError):S.stage_resources(source,dest)
            self.assertTrue(sentinel.exists());self.assertEqual((dest/C.BUNDLE/'reference.json').read_bytes(),before)

    def test_host_and_preparation_plan_need_no_site_packages_and_do_not_execute(self):
        for name in ('audit_coreai_source_window.py','prepare_coreai_source_window.py','prepare_coreai_full_song_context.py'):
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts'/name)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('PLAN ONLY',result.stdout)


if __name__=='__main__':unittest.main()

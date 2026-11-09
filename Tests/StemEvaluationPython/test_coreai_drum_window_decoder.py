"""Bounded decoder guards/transport; NONEXECUTABLE files, no musical evidence."""
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
import coreai_drum_reference30_window_contract as C
import prepare_coreai_drum_reference30_window_decoder as W
import stage_coreai_app_models as S


def metadata():
    m={'finite':True,'parity_passed':True,'max_absolute_error':0.,'rmse':0.,'normalized_rmse':0.,'elementwise_fraction_over_bound':0.}
    checks={k:dict(m) for k in {'overall'}|{f'tile-{i}' for i in range(6)}|{f'seam-{i}' for i in range(1,6)}}
    return {'schema_version':1,'profile':C.PROFILE,'status':'prepared_window_decoder','asset_file':'decoder.aimodel','function_name':'main',
        'input_shapes':{'latent':[1,64,256]},'output_shapes':{'audio':[1,2,491520]},'sample_rate':48000,'audio_seconds':30.,
        'latent_frames':750,'window_frames':256,'core_frames':128,'halo_frames':64,'weight_storage_dtype':'float32',
        'compute_dtype':'float32','scalar_type':'float32','byte_order':'little','vae_checkpoint_sha256':C.VAE_SHA,
        'capture_report_sha256':C.CAPTURE_SHA,'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
        'window_plan':C.tile_plan(),'host_numerical_gate_passed':True,'architecture_guard_passed':True,'coverage_exactly_once':True,
        'crossfade':False,'extra_latent_padding':False,'native_parity':'NOT EVALUATED','native_memory_gate':'NOT EVALUATED',
        'full_trajectory_parity':'NOT EVALUATED','baseline_modified':False,'host_checks':checks,
        'tile_input_sha256':['0'*64]*6,'dependency_bounds':[C.dependency_bounds(t) for t in C.tile_plan()],
        'artifact_sha256':{**C.DATA_HASHES,**{'decoder.aimodel/'+n:'0'*64 for n in C.NATIVE_FILES}}}


class WindowDecoderTests(unittest.TestCase):
    def test_receptive_field_plan_has_complete_nonoverlapping_cores_and_true_clip_edges(self):
        covered=[]
        for t in C.tile_plan():
            covered.extend(range(t['core_start'],t['core_end']))
            lo,hi=C.dependency_bounds(t)
            self.assertTrue(t['window_start']<=lo<=hi<t['window_start']+256)
        self.assertEqual(covered,list(range(750)))
        self.assertEqual(C.tile_plan()[-1],{'window_start':494,'window_frames':256,'core_start':640,'core_end':750,'crop_start':146})
        bad=dict(C.tile_plan()[2],window_start=256,crop_start=0)
        with self.assertRaisesRegex(ValueError,'Artificial'):C.dependency_bounds(bad)

    def test_contract_rejects_dtype_sampler_claims_shape_changes_and_relaxed_bounds(self):
        ref=metadata();self.assertEqual(C.validate(ref),ref)
        for k,v in [('weight_storage_dtype','bfloat16'),('input_shapes',{'latent':[1,64,750]}),
                    ('absolute_tolerance',1e-3),('relative_tolerance',1e-2),('native_parity','PASS'),
                    ('crossfade',True),('extra_latent_padding',True),('native_memory_gate','PASS')]:
            with self.assertRaises(ValueError):C.validate({**ref,k:v})
        wrong=copy.deepcopy(ref);wrong['window_plan'][-1]['window_start']=495
        with self.assertRaises(ValueError):C.validate(wrong)

    def test_false_pass_metrics_missing_regions_and_malformed_hashes_rejected(self):
        ref=metadata()
        for k,v in [('normalized_rmse',.00101),('elementwise_fraction_over_bound',.0001),('rmse',float('nan')),('finite',False)]:
            wrong=copy.deepcopy(ref);wrong['host_checks']['overall'][k]=v
            with self.assertRaises(ValueError):C.validate(wrong)
        wrong=copy.deepcopy(ref);del wrong['host_checks']['seam-5']
        with self.assertRaises(ValueError):C.validate(wrong)
        wrong=copy.deepcopy(ref);wrong['tile_input_sha256'][0]='bad'
        with self.assertRaises(ValueError):C.validate(wrong)

    def test_default_plan_is_read_only_without_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'never-created'
            r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/prepare_coreai_drum_reference30_window_decoder.py'),'--out',str(out)],
                check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(r.stdout)['native_calls'],6);self.assertFalse(out.exists())

    def test_missing_capture_and_protected_or_existing_output_never_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);capture=root/'capture';capture.mkdir();old=root/'old';old.mkdir();(old/'keep').write_text('old')
            with patch.object(W,'execute') as run:
                with self.assertRaises(FileNotFoundError):W.prepare(capture,old)
                for out in [capture/'nested',ROOT/'baselines/never-window',ROOT/'prototypes/ios27-coreai/ModelCandidates/bass-preview-1024/never']:
                    with self.assertRaises(ValueError):W.prepare(capture,out)
                    self.assertFalse(out.exists())
                with self.assertRaises(ValueError):W.prepare(capture,root/'new')
                run.assert_not_called()
            self.assertEqual((old/'keep').read_text(),'old')

    def test_existing_complete_bundle_is_verified_without_repeat_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'done';out.mkdir()
            with patch.object(W,'verify_bundle') as verify,patch.object(W,'execute') as run:
                W.prepare(Path(tmp)/'capture',out)
                verify.assert_called_once_with(out);run.assert_not_called()

    def test_failed_or_incomplete_prepare_retains_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,side in enumerate([ValueError('host gate failed'),KeyboardInterrupt(),None]):
                out=root/str(i)
                with patch.object(W,'verify_capture'),patch.object(W,'execute',side_effect=side):
                    with self.assertRaises(BaseException):W.prepare(root/'capture',out)
                ref=json.loads((out/'reference.json').read_text());self.assertEqual(ref['status'],'failed')
                self.assertEqual(ref['native_parity'],'NOT EVALUATED');self.assertFalse(ref['baseline_modified'])

    def test_reuse_requires_256_frame_original_fp32_VAE_provenance_and_intact_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);asset=folder/'decoder.aimodel';asset.mkdir();hashes={}
            for n in C.NATIVE_FILES:
                p=asset/n;p.write_bytes(b'NONEXECUTABLE decoder reuse transport fixture')
                hashes['decoder.aimodel/'+n]=hashlib.sha256(p.read_bytes()).hexdigest()
            ref={'profile':'drums-preview-1024','latent_frames':256,'audio_seconds':10.24,'function_name':'main',
                'vae_checkpoint_digest':C.VAE_SHA,'host_decoder_folding_exact':True,'all_checkpoints_verified':True,
                'scalar_type':'float32','byte_order':'little','weight_storage_dtype':'bfloat16','artifact_sha256':hashes}
            path=folder/'reference.json';path.write_text(json.dumps(ref))
            self.assertEqual(W.reusable_decoder(folder)['profile'],'drums-preview-1024')
            path.write_text(json.dumps({**ref,'latent_frames':750}))
            with self.assertRaises(ValueError):W.reusable_decoder(folder)
            path.write_text(json.dumps(ref));(asset/'main.hash').write_bytes(b'changed')
            with self.assertRaises(ValueError):W.reusable_decoder(folder)

    def test_seven_file_build_profile_stdlib_cli_and_bad_contract_preserve_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'repo/ModelCandidates';folder=source/C.BUNDLE;folder.mkdir(parents=True)
            ref=metadata()
            for name in C.FILES:
                p=folder/name;p.parent.mkdir(parents=True,exist_ok=True)
                if name in C.DATA_SHAPES:
                    with p.open('wb') as f:f.truncate(math.prod(C.DATA_SHAPES[name])*4)
                else:p.write_bytes(b'NONEXECUTABLE packaging fixture')
            path=folder/'reference.json';path.write_text(json.dumps(ref))
            (source/'active-preview.json').write_text(json.dumps({'schema_version':1,'resource_profile':C.PROFILE,'audio_seconds':30.}))
            dest=root/'build/Probe.app/ModelCandidates'
            r=S.stage_resources(source,dest);self.assertEqual(len(r['files']),7)
            self.assertEqual(S.stage_resources(source,dest)['copied_files'],0)
            cli=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/stage_coreai_app_models.py'),
                '--source',str(source),'--destination',str(dest),'--dry-run'],check=True,capture_output=True,text=True)
            self.assertIn('7 files',cli.stdout)
            before=(dest/C.BUNDLE/'reference.json').read_bytes();ref['crossfade']=True;path.write_text(json.dumps(ref))
            with self.assertRaises(ValueError):S.stage_resources(source,dest)
            self.assertEqual((dest/C.BUNDLE/'reference.json').read_bytes(),before)

    def test_auto_reuse_skips_ineligible_sources_without_mutating_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bad=root/'drums-preview-1024';bad.mkdir();(bad/'reference.json').write_text('invalid JSON')
            good=root/'bass-preview-1024';good.mkdir();(good/'reference.json').write_text('{}')
            def inspect(folder):
                if folder==bad:raise ValueError('ineligible')
                return {'profile':'bass-preview-1024'}
            with patch.object(W,'reusable_decoder',side_effect=inspect):
                folder,info,skipped=W.find_reusable(root)
            self.assertEqual(folder,good);self.assertEqual(info['profile'],'bass-preview-1024')
            self.assertEqual(skipped,[{'profile':'drums-preview-1024','reason':'ValueError'}])
            self.assertEqual((bad/'reference.json').read_text(),'invalid JSON')

    def test_package_integrity_verifier_accepts_complete_transport_and_rejects_tampering(self):
        # Synthetic data identities are patched only for this transport test.
        # These files are NONEXECUTABLE and never represent model acceptance.
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);ref=metadata();hashes={}
            for name in C.FILES:
                p=folder/name;p.parent.mkdir(parents=True,exist_ok=True)
                if name in C.DATA_SHAPES:
                    with p.open('wb') as f:f.truncate(math.prod(C.DATA_SHAPES[name])*4)
                else:p.write_bytes(b'NONEXECUTABLE integrity fixture')
                hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            ref['artifact_sha256']=hashes
            data_hashes={n:hashes[n] for n in C.DATA_HASHES}
            (folder/'reference.json').write_text(json.dumps(ref))
            with patch.object(C,'DATA_HASHES',data_hashes):
                self.assertEqual(W.verify_bundle(folder)['native_parity'],'NOT EVALUATED')
                target=folder/'decoder.aimodel/main.hash';old=target.read_bytes();target.write_bytes(b'tamper')
                with self.assertRaisesRegex(ValueError,'integrity'):W.verify_bundle(folder)
                target.write_bytes(old);extra=folder/'extra';extra.write_text('unexpected')
                with self.assertRaisesRegex(ValueError,'payload'):W.verify_bundle(folder)
                extra.unlink();target.unlink();target.symlink_to(folder/'decoder.aimodel/main.mlirb')
                with self.assertRaisesRegex(ValueError,'integrity'):W.verify_bundle(folder)

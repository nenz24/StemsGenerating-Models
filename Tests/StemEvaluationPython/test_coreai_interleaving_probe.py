"""Schedule, integrity and activation tests; payloads are NOT native models."""
import copy
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import coreai_interleaving_contract as C
import prepare_coreai_interleaving_probe as P
import stage_coreai_app_models as S
from export_coreai_dit_block import records


class InterleavingTests(unittest.TestCase):
    def fixture(self, root, count=2):
        folder = root/C.PROFILE
        folder.mkdir(parents=True)
        for name in C.payload(count):
            path = folder/name; path.parent.mkdir(parents=True,exist_ok=True)
            shape = C.INPUTS.get(name.removesuffix('.f32le'))
            if name.startswith('reference-'): shape = C.OUTPUT
            if shape:
                with path.open('wb') as f: f.truncate(math.prod(shape)*4)
            else: path.write_text('NONEXECUTABLE integrity fixture '+name)
        hashes = {n:C.sha256(folder/n) for n in C.payload(count)}
        ref = {'schema_version':1,'profile':C.PROFILE,'pair_count':count,'assets':C.assets(count),
               'input_shapes':C.INPUTS,'output_shape':C.OUTPUT,'calls_per_asset':20,
               'weight_storage_dtype':'bfloat16','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
               'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
               'checkpoint_digest':records()['model.safetensors']['digest'],'checkpoint_full_sha256_verified':True,
               'host_exact':[True]*count,'artifact_sha256':hashes,'source_manifest_sha256':'a'*64,
               'source_asset_sha256':{n:h for n,h in hashes.items() if '/' in n}}
        (folder/'reference.json').write_text(json.dumps(ref))
        return folder, ref

    def test_matched_work_and_switch_counts(self):
        for k in C.SIZES:
            grouped, interleaved = C.schedule('grouped',k), C.schedule('interleaved',k)
            self.assertEqual(sorted(grouped),sorted(interleaved))
            self.assertEqual(len(grouped),20*k)
            for i in range(k): self.assertEqual(interleaved.count(i),20)
            self.assertEqual(sum(a!=b for a,b in zip(grouped,grouped[1:])),k-1)
            self.assertEqual(sum(a!=b for a,b in zip(interleaved,interleaved[1:])),20*k-1)
            for mode in ('single-before','pool-only','single-after'):
                self.assertEqual(C.schedule(mode,k),[0]*20)
            self.assertEqual(sum(len(C.schedule(m,k)) for m in
                ['single-before','pool-only','grouped','interleaved','single-after']),60+40*k)

    def test_contract_matches_existing_long_pair(self):
        from prepare_coreai_precision_probe import INPUTS
        self.assertEqual(C.INPUTS,INPUTS)

    def test_corruption_rejected_and_selector_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder,ref=self.fixture(root)
            self.assertEqual(C.verify_bundle(folder),ref)
            selector=root/'active-preview.json';selector.write_text('previous selector')
            (folder/'hidden.f32le').write_bytes(b'broken')
            with self.assertRaises(ValueError): P.activate(root)
            self.assertEqual(selector.read_text(),'previous selector')

    def test_dtype_tolerance_host_identity_and_duplicate_assets_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,ref=self.fixture(Path(tmp))
            cases=[('weight_storage_dtype','float32'),('relative_tolerance',.01),
                   ('host_exact',[True,False]),('host_exact',[1,1]),('pair_count',True),
                   ('source_manifest_sha256','bad'),('checkpoint_digest','b'*64)]
            for key,value in cases:
                changed=copy.deepcopy(ref);changed[key]=value
                with self.assertRaises(ValueError,msg=key): C.validate(changed)
            changed=copy.deepcopy(ref)
            name=C.assets(2)[1]+'/main.mlirb'
            changed['source_asset_sha256'][name]='f'*64
            with self.assertRaisesRegex(ValueError,'source changed'): C.validate(changed)
            changed=copy.deepcopy(ref)
            duplicate=changed['artifact_sha256'][C.assets(2)[0]+'/main.mlirb']
            changed['artifact_sha256'][name]=duplicate;changed['source_asset_sha256'][name]=duplicate
            with self.assertRaisesRegex(ValueError,'distinct'): C.validate(changed)

    def test_only_probe_staged_and_generation_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source';folder,_=self.fixture(source)
            marker=source/'bass-preview-1024/keep';marker.parent.mkdir();marker.write_text('old baseline')
            P.activate(source)
            dest=root/'build/Probe.app/ModelCandidates'
            result=S.stage_resources(source,dest)
            self.assertEqual(result['selected_profile'],C.PROFILE)
            self.assertTrue(all(n.startswith(C.PROFILE+'/') for n in result['files']))
            self.assertEqual(marker.read_text(),'old baseline')
            self.assertFalse((dest/'bass-preview-1024').exists())
            self.assertEqual(S.stage_resources(source,dest)['copied_files'],0)
            prior=(dest/C.PROFILE/'reference.json').read_bytes()
            ref=json.loads((folder/'reference.json').read_text());ref['artifact_sha256']['../escape']='a'*64
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError): S.stage_resources(source,dest)
            self.assertEqual((dest/C.PROFILE/'reference.json').read_bytes(),prior)

    def test_extra_payload_and_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_=self.fixture(Path(tmp))
            extra=folder/'unexpected';extra.write_text('preserved')
            with self.assertRaises(ValueError): C.verify_bundle(folder)
            with self.assertRaises(FileExistsError): P.prepare(folder,folder,folder,2)
            self.assertEqual(extra.read_text(),'preserved')

    def test_expansion_archives_old_bundle_and_preserves_selector_and_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,_=self.fixture(root/'candidates');prepared,_=self.fixture(root/'fresh',4)
            before=(out/'reference.json').read_bytes()
            selector=out.parent/'active-preview.json';selector.write_text('previous selector')
            baseline=out.parent/'bass-preview-1024/keep';baseline.parent.mkdir();baseline.write_text('generation unchanged')
            archive=P.install_expansion(out,prepared,root/'archives',C.sha256(out/'reference.json'))
            self.assertEqual(C.verify_bundle(out)['pair_count'],4)
            self.assertEqual(C.verify_bundle(archive)['pair_count'],2)
            self.assertEqual((archive/'reference.json').read_bytes(),before)
            self.assertEqual(selector.read_text(),'previous selector')
            self.assertEqual(baseline.read_text(),'generation unchanged')
            self.assertFalse(prepared.exists())
            P.activate(out.parent)
            self.assertEqual(S.plan_resources(out.parent)['selected_profile'],C.PROFILE)

    def test_failed_new_bundle_install_restores_old_and_preserves_prepared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,_=self.fixture(root/'candidates');prepared,_=self.fixture(root/'fresh',4)
            before=(out/'reference.json').read_bytes();rename=P.os.rename
            def fail_install(source,target):
                if Path(source)==prepared: raise OSError('simulated install failure')
                return rename(source,target)
            with mock.patch.object(P.os,'rename',side_effect=fail_install):
                with self.assertRaisesRegex(OSError,'simulated'):
                    P.install_expansion(out,prepared,root/'archives',C.sha256(out/'reference.json'))
            self.assertEqual((out/'reference.json').read_bytes(),before)
            self.assertEqual(C.verify_bundle(out)['pair_count'],2)
            self.assertEqual(C.verify_bundle(prepared)['pair_count'],4)

    def test_expansion_rejects_changed_shared_payload_source_or_concurrent_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,_=self.fixture(root/'candidates');prepared,ref=self.fixture(root/'fresh',4)
            old_digest=C.sha256(out/'reference.json')
            with self.assertRaisesRegex(ValueError,'changed during preparation'):
                P.install_expansion(out,prepared,root/'archives','f'*64)
            changed=copy.deepcopy(ref);changed['source_manifest_sha256']='b'*64
            (prepared/'reference.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Source manifest changed'):
                P.install_expansion(out,prepared,root/'archives',old_digest)
            (prepared/'reference-00.f32le').write_bytes(bytes(4*math.prod(C.OUTPUT)-1)+b'1')
            ref['artifact_sha256']['reference-00.f32le']=C.sha256(prepared/'reference-00.f32le')
            (prepared/'reference.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'assets/inputs/references changed'):
                P.install_expansion(out,prepared,root/'archives',old_digest)
            self.assertEqual(C.verify_bundle(out)['pair_count'],2)
            self.assertFalse((root/'archives').exists())

    def test_expansion_preparation_failure_never_moves_current_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,_=self.fixture(root/'candidates')
            saved=(out/'reference.json').read_bytes()
            selector=out.parent/'active-preview.json';selector.write_text('old selector')
            with mock.patch.object(P,'ROOT',root/'runtime'), mock.patch.object(P,'prepare',side_effect=ValueError('host failed')):
                with self.assertRaisesRegex(ValueError,'host failed'):
                    P.expand_bundle(root/'source',root/'fixture',out,4)
            self.assertEqual((out/'reference.json').read_bytes(),saved)
            self.assertEqual(selector.read_text(),'old selector')
            self.assertFalse((root/'runtime/.runtime-cache/coreai-interleaving-archives').exists())

    def test_cli_requires_expansion_flag_and_preserves_existing_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,_=self.fixture(Path(tmp));saved=(out/'reference.json').read_bytes()
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_coreai_interleaving_probe.py'),
                '--out',str(out),'--pairs','4'],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Use --expand-existing',result.stderr)
            self.assertEqual((out/'reference.json').read_bytes(),saved)


if __name__ == '__main__': unittest.main()

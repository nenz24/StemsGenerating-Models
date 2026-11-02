"""Schedule, integrity and activation tests; payloads are NOT native models."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import coreai_interleaving_contract as C
import prepare_coreai_interleaving_probe as P
import stage_coreai_app_models as S
from export_coreai_dit_block import records


class InterleavingTests(unittest.TestCase):
    def fixture(self, root):
        folder = root/C.PROFILE
        folder.mkdir(parents=True)
        for name in C.payload(2):
            path = folder/name; path.parent.mkdir(parents=True,exist_ok=True)
            shape = C.INPUTS.get(name.removesuffix('.f32le'))
            if name.startswith('reference-'): shape = C.OUTPUT
            if shape:
                with path.open('wb') as f: f.truncate(math.prod(shape)*4)
            else: path.write_text('NONEXECUTABLE integrity fixture '+name)
        hashes = {n:C.sha256(folder/n) for n in C.payload(2)}
        ref = {'schema_version':1,'profile':C.PROFILE,'pair_count':2,'assets':C.assets(2),
               'input_shapes':C.INPUTS,'output_shape':C.OUTPUT,'calls_per_asset':20,
               'weight_storage_dtype':'bfloat16','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
               'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
               'checkpoint_digest':records()['model.safetensors']['digest'],'checkpoint_full_sha256_verified':True,
               'host_exact':[True,True],'artifact_sha256':hashes,'source_manifest_sha256':'a'*64,
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


if __name__ == '__main__': unittest.main()

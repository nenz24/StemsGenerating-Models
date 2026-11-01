"""Partition-order/integrity/retirement checks; fixture native files cannot execute."""
import contextlib
import copy
import hashlib
import io
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
import export_coreai_denoiser_sharded as MODULE


class ShardedDenoiserTests(unittest.TestCase):
    def fixture(self,folder):
        folder.mkdir()
        data={f'{n}.f32le':bytes(math.prod(s)*4) for n,s in MODULE.INPUT_SHAPES.items()}
        for n in ['velocity-reference.f32le','velocity-fp32-reference.f32le']:data[n]=bytes(1600*4)
        for stage in MODULE.stage_plan():
            for n in ['main.hash','main.mlirb','metadata.json']:data[stage['asset_file']+'/'+n]=b'fixture, not executable'
        for n,v in data.items():
            p=folder/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
        record={'digest':'fixture checkpoint'}
        ref={'schema_version':1,'profile':'denoiser-sharded','asset_file':'sharded-pipeline','function_name':'main','output_name':'velocity',
             'input_shapes':MODULE.INPUT_SHAPES,'output_shape':MODULE.OUTPUT_SHAPE,'layer_count':24,'stages':MODULE.stage_plan(),
             'parameter_count':1575458880,'parameter_coverage_exactly_once':True,
             'checkpoint':record,'checkpoint_full_sha256_verified':True,'partition_vs_monolithic':{'parity_passed':True},
             'storage16_vs_original_fp32':{'parity_passed':True},'weight_storage_dtype':'float16','compute_dtype':'float32',
             'scalar_type':'float32','byte_order':'little','original_fp32_reference_file':'velocity-fp32-reference.f32le',
             'artifact_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in data.items()}}
        (folder/'reference.json').write_text(json.dumps(ref))
        return ref,record

    def test_plan_covers_24_layers_once_and_rejects_wrong_order_or_wiring(self):
        plan=MODULE.stage_plan()
        self.assertEqual(len(plan),14)
        self.assertEqual([i for s in plan for i in s['layer_indices']],list(range(24)))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref,record=self.fixture(folder)
            MODULE.verify_bundle(folder,record)
            for kind in ['order','wiring','missing']:
                changed=copy.deepcopy(ref)
                if kind=='order':changed['stages'][1],changed['stages'][2]=changed['stages'][2],changed['stages'][1]
                elif kind=='wiring':changed['stages'][0]['output_bindings']['condition_projected']='noise'
                else:changed['stages'].pop()
                (folder/'reference.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):MODULE.verify_bundle(folder,record)

    def test_reference_hash_and_complete_stage_manifest_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref,record=self.fixture(folder)
            (folder/'velocity-reference.f32le').write_bytes(b'bad')
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder,record)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bundle';ref,record=self.fixture(folder)
            del ref['artifact_sha256']['blocks-22-23.aimodel/main.mlirb']
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError):MODULE.verify_bundle(folder,record)

    def test_monolith_retirement_preserves_files_and_never_overwrites_archive(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(MODULE,'ROOT',Path(tmp)),contextlib.redirect_stdout(io.StringIO()):
            source=Path(tmp)/'prototypes/ios27-coreai/ModelCandidates/denoiser-pretrained'
            source.mkdir(parents=True);(source/'keep').write_bytes(b'original')
            destination=MODULE.retire_monolith()
            self.assertFalse(source.exists());self.assertEqual((destination/'keep').read_bytes(),b'original')
            self.assertIsNone(MODULE.retire_monolith())
            source.mkdir();(source/'keep').write_bytes(b'new')
            second=MODULE.retire_monolith()
            self.assertNotEqual(destination,second)
            self.assertEqual((destination/'keep').read_bytes(),b'original');self.assertEqual((second/'keep').read_bytes(),b'new')

    def test_existing_output_is_preserved_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'keep';marker.write_text('original')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/export_coreai_denoiser_sharded.py'),'--out',tmp],
                                  text=True,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'original')


if __name__=='__main__':unittest.main()

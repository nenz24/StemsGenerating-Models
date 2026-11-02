"""Contract and corruption tests; fixtures never simulate CoreAI inference."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import export_coreai_bass_conditioning as M


class BassConditioningTests(unittest.TestCase):
    def test_fixed_plan_covers_all_40_layers_and_connects_every_input(self):
        plan=M.stage_plan(105,11)
        self.assertEqual(len(plan),27)
        self.assertEqual([s['asset_file'] for s in plan if s['asset_file'].startswith('qwen-') and s['asset_file']!='qwen-norm.aimodel'],
                         [f'qwen-{i:02d}-{i+1:02d}.aimodel' for i in range(0,28,2)])
        state={'text_ids':[1,105],'lyric_ids':[1,11],'reference_audio':[1,750,64]}
        for stage in plan:
            for name,shape in stage['input_shapes'].items():self.assertEqual(state[name],shape)
            for name,shape in stage['output_shapes'].items():state[stage['output_bindings'][name]]=shape
        self.assertEqual(state['condition'],[1,117,2048])
        self.assertEqual(state['condition_mask'],[1,117])
        for lengths in [(0,11),(257,11),(105,0),(105,2049)]:
            with self.assertRaises(ValueError):M.stage_plan(*lengths)

    def fixture(self,folder):
        """Small integrity-only asset files are explicitly NOT executable models."""
        folder.mkdir();plan=M.stage_plan(105,11)
        shapes={**plan[0]['input_shapes'],'reference_audio':[1,750,64]}
        required={f'{n}.f32le':s for n,s in shapes.items()}
        for n,s in {'text_embeddings':[1,105,1024],**plan[-1]['output_shapes']}.items():
            for kind in ['reference','fp32-reference']:required[f'{n}-{kind}.f32le']=s
        for n,s in required.items():
            raw=(struct.pack('<f',1.) if n.startswith('condition_mask') else bytes(4))*math.prod(s)
            (folder/n).write_bytes(raw)
        for stage in plan:
            asset=folder/stage['asset_file'];asset.mkdir()
            for n in ['main.hash','main.mlirb','metadata.json']:(asset/n).write_bytes(b'NOT an executable model; integrity-only fixture')
        ref={'schema_version':1,'profile':M.PROFILE,'function_name':'main','batch_size':1,'text_tokens':105,'lyric_tokens':11,
             'qwen_layer_count':28,'lyric_layer_count':8,'timbre_layer_count':4,'stages':plan,
             'qwen_checkpoint_digest':M.QWEN_DIGEST,'base_checkpoint_digest':M.BASE_DIGEST,'silence_digest':M.SILENCE_DIGEST,
             'all_checkpoint_hashes_verified':True,'scalar_type':'float32','byte_order':'little','weight_storage_dtype':'float16','compute_dtype':'float32',
             'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
             'all_valid_single_reference_specialization':True,'reference_audio_mode':'publisher_silence_750','tokenization_on_device':False,
             'host_partition_vs_original_fp32':{'parity_passed':True},'host_storage16_vs_original_fp32':{'parity_passed':True},
             'prompts':{'test':'isolated validation fixture'},
             'artifact_sha256':{str(p.relative_to(folder)):M.sha256(p) for p in folder.rglob('*') if p.is_file()}}
        self.write_reference(folder,ref)
        return ref

    @staticmethod
    def write_reference(folder,ref):(folder/'reference.json').write_text(json.dumps(ref))

    def test_rejects_missing_reordered_or_wrong_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(M,'format_prompts',return_value={'test':'isolated validation fixture'}):
            folder=Path(tmp)/'bundle';ref=self.fixture(folder);M.verify_bundle(folder)
            changes={'stages':list(reversed(ref['stages'])),'qwen_layer_count':27,'lyric_layer_count':6,
                     'timbre_layer_count':2,'qwen_checkpoint_digest':'0'*64,'text_tokens':104,
                     'tokenization_on_device':True,'reference_audio_mode':'random',
                     'host_storage16_vs_original_fp32':{'parity_passed':False},'relative_tolerance':.1}
            for key,value in changes.items():
                changed=copy.deepcopy(ref);changed[key]=value;self.write_reference(folder,changed)
                with self.assertRaises(ValueError,msg=key):M.verify_bundle(folder)
            self.write_reference(folder,ref)
            (folder/'qwen-26-27.aimodel/main.hash').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):M.verify_bundle(folder)

    def test_corruption_and_rehashed_invalid_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(M,'format_prompts',return_value={'test':'isolated validation fixture'}):
            folder=Path(tmp)/'bundle';ref=self.fixture(folder)
            for name,value,error in [('text_ids.f32le',.5,'Invalid token'),('lyric_ids.f32le',151669.,'Invalid token'),
                                     ('reference_audio.f32le',float('nan'),'Nonfinite'),('condition_mask-reference.f32le',0.,'all-valid')]:
                path=folder/name;original=path.read_bytes()
                path.write_bytes(struct.pack('<f',value)+original[4:])
                changed=copy.deepcopy(ref);changed['artifact_sha256'][name]=M.sha256(path);self.write_reference(folder,changed)
                with self.assertRaisesRegex(ValueError,error):M.verify_bundle(folder)
                path.write_bytes(original)
            self.write_reference(folder,ref)
            path=folder/'condition-reference.f32le';raw=bytearray(path.read_bytes());raw[0]^=1;path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):M.verify_bundle(folder)

    def test_existing_destination_preserved_before_heavy_imports_and_shell_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'keep';marker.write_text('original')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/export_coreai_bass_conditioning.py'),'--out',tmp],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'original')
        result=subprocess.run(['bash','-n',str(ROOT/'scripts/prepare_coreai_bass_conditioning.sh')],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

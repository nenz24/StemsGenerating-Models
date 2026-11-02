"""Window/selection integrity tests. File fixtures are NONEXECUTABLE, not inference."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from coreai_preview_window import SHORT,LONG,PreviewWindow
import export_coreai_source_context as SOURCE
import export_coreai_bass_preview as PREVIEW
import export_coreai_bass_conditioning as CONDITION
import stage_coreai_app_models as STAGE
import prepare_coreai_preview_window as PREPARE

class PreviewWindowTests(unittest.TestCase):
    def test_only_two_supported_windows_and_duration_matching_prompt(self):
        for seconds,window in [(5.12,SHORT),(10.24,LONG)]:
            self.assertEqual(PreviewWindow.for_seconds(seconds),window)
            self.assertEqual(window.samples,round(seconds*48000))
            self.assertEqual(window.frames*1920,window.samples)
            self.assertEqual(window.condition_tokens,window.text_tokens+12)
            prompt=CONDITION.format_prompts(seconds)
            self.assertEqual(prompt['audio_seconds'],seconds)
            self.assertIn(f'duration: {int(seconds)} seconds',prompt['text_prompt'])
        for seconds in [0,5,10,30,float('nan'),float('inf'),True,'10.24']:
            with self.assertRaises(ValueError):PreviewWindow.for_seconds(seconds)
        for frames in [0,128.0,True,255,512]:
            with self.assertRaises(ValueError):PreviewWindow(frames)

    def test_full_24_layers_and_cross_attention_when_queries_exceed_keys(self):
        for window in [SHORT,LONG]:
            plan=PREVIEW.stage_plan(window)
            self.assertEqual([i for s in plan for i in s['layer_indices']],list(range(24)))
            state=dict(plan[0]['input_shapes'])
            for stage in plan:
                for name,shape in stage['input_shapes'].items():self.assertEqual(state[name],shape)
                for name,shape in stage['output_shapes'].items():state[stage['output_bindings'][name]]=shape
            self.assertEqual(state['velocity'],[2,window.frames,64])
            self.assertEqual(state['cross_mask'],[1,1,window.hidden_tokens,window.condition_tokens])
            inputs,outputs=SOURCE.shapes(window)
            self.assertEqual(inputs['audio'],[1,2,window.samples])
            self.assertEqual(outputs['context_latents'],[1,window.frames,128])
        self.assertGreater(LONG.hidden_tokens,LONG.condition_tokens)
        self.assertEqual(PREVIEW.stage_plan()[0]['input_shapes'],PREVIEW.INPUTS)
        self.assertEqual(SOURCE.shapes(),(SOURCE.INPUTS,SOURCE.OUTPUTS))

    def test_long_reference_scope_shapes_and_unchanged_tolerances_are_enforced(self):
        # Reuse the explicitly NONEXECUTABLE integrity fixture; no backend is mocked.
        from test_coreai_bass_preview import BassPreviewTests
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'preview';source=root/'source';condition=root/'condition'
            ref=BassPreviewTests().fixture(folder,source,condition)
            ref.update(profile=LONG.preview_bundle,stages=PREVIEW.stage_plan(LONG),latent_frames=LONG.frames,
                       condition_tokens=LONG.condition_tokens,audio_seconds=LONG.seconds,
                       reference_conditioning_scope='original_fp32_vs_storage16_conditioning',weight_storage_dtype=LONG.weight_storage_dtype)
            for name in ref['artifact_sha256']:
                if name=='initial-noise.f32le' or name.startswith('step-'):
                    (folder/name).write_bytes(bytes(4*LONG.frames*64))
                elif name.startswith('audio-'):
                    (folder/name).write_bytes(bytes(4*LONG.samples*2))
            ref['artifact_sha256']={name:PREVIEW.sha256(folder/name) for name in ref['artifact_sha256']}
            manifest=folder/'reference.json';manifest.write_text(json.dumps(ref))
            PREVIEW.verify_bundle(folder,source,condition,LONG)
            for key,value in [('reference_conditioning_scope','identical_verified_storage16_conditioning'),
                              ('weight_storage_dtype','float16'),
                              ('condition_tokens',SHORT.condition_tokens),('audio_seconds',SHORT.seconds),
                              ('relative_rmse_tolerance',.01)]:
                changed={**ref,key:value};manifest.write_text(json.dumps(changed))
                with self.assertRaises(ValueError,msg=key):PREVIEW.verify_bundle(folder,source,condition,LONG)

    def test_bfloat16_storage_preserves_tiny_bf16_values_and_float32_calculations(self):
        import copy
        import torch
        from export_coreai_dit_block import convert_storage16
        model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False)).eval()
        values=torch.tensor([[1e-8,2e-8],[-1e-10,1.]],dtype=torch.bfloat16).float()
        with torch.no_grad():model[0].weight.copy_(values)
        x=torch.tensor([[1.,-1.],[2.,3.]])
        expected=model(x)
        fp16=convert_storage16(copy.deepcopy(model))
        bf16=convert_storage16(copy.deepcopy(model),LONG.weight_storage_dtype)
        self.assertEqual(bf16[0].weight.dtype,torch.bfloat16)
        self.assertEqual(bf16[0].weight.element_size(),2)
        self.assertEqual(bf16(x).dtype,torch.float32)
        torch.testing.assert_close(bf16[0].weight.float(),values,rtol=0,atol=0)
        torch.testing.assert_close(bf16(x),expected,rtol=0,atol=0)
        self.assertFalse(torch.equal(fp16[0].weight.float(),values))
        self.assertFalse(torch.equal(fp16(x),expected))
        with self.assertRaises(ValueError):convert_storage16(copy.deepcopy(model),'float64')

    def bundle(self,source,name,metadata):
        folder=source/name;folder.mkdir(parents=True)
        data=b'NONEXECUTABLE window-selection integrity fixture'
        (folder/'payload').write_bytes(data)
        ref={'artifact_sha256':{'payload':hashlib.sha256(data).hexdigest()},**metadata}
        (folder/'reference.json').write_text(json.dumps(ref))
        return ref

    def pipeline(self,source,window):
        self.bundle(source,window.source_bundle,{'profile':window.source_bundle,'audio_seconds':window.seconds,'input_shapes':{'audio':[1,2,window.samples]}})
        self.bundle(source,window.condition_bundle,{'profile':window.condition_bundle,'weight_storage_dtype':window.weight_storage_dtype,'text_tokens':window.text_tokens,'prompts':{'audio_seconds':window.seconds}})
        self.bundle(source,window.preview_bundle,{'profile':window.preview_bundle,'reference_conditioning_scope':'original_fp32_vs_storage16_conditioning','weight_storage_dtype':window.weight_storage_dtype,'audio_seconds':window.seconds,'latent_frames':window.frames,'condition_tokens':window.condition_tokens,'source_reference_sha256':STAGE.digest(source/window.source_bundle/'reference.json'),'condition_reference_sha256':STAGE.digest(source/window.condition_bundle/'reference.json')})

    def select(self,source,window):
        (source/'active-preview.json').write_text(json.dumps({'schema_version':1,'resource_profile':window.resource_profile,'audio_seconds':window.seconds}))

    def test_active_long_profile_excludes_old_assets_and_can_return_to_short(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'ModelCandidates';dest=root/'build/Probe.app/ModelCandidates'
            for w in [SHORT,LONG]:self.pipeline(source,w)
            STAGE.stage_resources(source,dest)
            self.assertTrue((dest/SHORT.preview_bundle).is_dir())
            self.select(source,LONG)
            long=STAGE.stage_resources(source,dest)
            self.assertEqual(long['selected_profile'],LONG.resource_profile)
            self.assertTrue((dest/LONG.preview_bundle).is_dir())
            self.assertFalse((dest/SHORT.preview_bundle).exists())
            self.assertFalse((dest/SHORT.condition_bundle).exists())
            self.assertTrue((source/SHORT.preview_bundle).is_dir())
            again=STAGE.stage_resources(source,dest)
            self.assertEqual(again['copied_files'],0)
            # Explicit short profile overrides the local auto selector.
            STAGE.stage_resources(source,dest,SHORT.resource_profile)
            self.assertTrue((dest/SHORT.preview_bundle).is_dir())
            self.assertFalse((dest/LONG.preview_bundle).exists())
            self.assertTrue((source/LONG.preview_bundle).is_dir())

    def test_invalid_selector_missing_or_mixed_long_bundle_never_prunes_app(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'ModelCandidates';dest=root/'build/Probe.app/ModelCandidates'
            self.pipeline(source,SHORT);STAGE.stage_resources(source,dest)
            saved=(dest/SHORT.preview_bundle/'payload').read_bytes()
            (source/'active-preview.json').write_text('{broken')
            with self.assertRaises(ValueError):STAGE.stage_resources(source,dest)
            self.select(source,LONG)
            with self.assertRaises(ValueError):STAGE.stage_resources(source,dest)
            self.pipeline(source,LONG)
            ref=source/LONG.preview_bundle/'reference.json';d=json.loads(ref.read_text());d['audio_seconds']=SHORT.seconds;ref.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Mixed'):STAGE.stage_resources(source,dest)
            self.assertEqual((dest/SHORT.preview_bundle/'payload').read_bytes(),saved)

    def test_failed_activation_preserves_current_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.select(root,SHORT);saved=(root/'active-preview.json').read_bytes()
            with self.assertRaises(FileNotFoundError):PREPARE.activate(root,LONG)
            self.assertEqual((root/'active-preview.json').read_bytes(),saved)

if __name__=='__main__':unittest.main()

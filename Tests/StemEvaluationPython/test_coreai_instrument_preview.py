"""Instrument isolation, model-contract reuse and failure preservation, not native inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from coreai_preview_window import PreviewWindow,SHORT,LONG
import export_coreai_bass_conditioning as CONDITION
import export_coreai_bass_preview as PREVIEW
import prepare_coreai_instrument_preview as DRIVER
import prepare_coreai_preview_window as ACTIVATE
import stage_coreai_app_models as STAGE


class InstrumentPreviewTests(unittest.TestCase):
    def test_pinned_presets_and_geometry_are_independent_and_baseline_unchanged(self):
        for instrument,tokens in [('bass',106),('drums',111),('guitar-strum',106),('guitar-pick',111)]:
            w=PreviewWindow(256,instrument)
            self.assertEqual(w.text_tokens,tokens)
            self.assertEqual(w.condition_tokens,tokens+12)
            self.assertEqual(w.source_bundle,LONG.source_bundle)
            self.assertEqual(CONDITION.sha256(ROOT/'configs'/w.preset_file),w.preset_sha256)
            prompts=CONDITION.format_prompts(10.24,instrument)
            self.assertEqual(prompts['instruction'],w.instruction)
            self.assertEqual(prompts['caption'],json.loads((ROOT/'configs'/w.preset_file).read_text())['caption'])
            state=dict(PREVIEW.stage_plan(w)[0]['input_shapes'])
            for stage in PREVIEW.stage_plan(w):
                for n,s in stage['input_shapes'].items(): self.assertEqual(state[n],s)
                for n,s in stage['output_shapes'].items(): state[stage['output_bindings'][n]]=s
            self.assertEqual(state['velocity'],[2,256,64])
            self.assertEqual(state['cross_mask'],[1,1,128,tokens+12])
        self.assertEqual(PreviewWindow.for_seconds(5.12),SHORT)
        self.assertEqual(LONG.preview_bundle,'bass-preview-1024')
        for name in DRIVER.INSTRUMENTS:
            with self.assertRaises(ValueError):PreviewWindow(128,name)
        with self.assertRaises(ValueError):PreviewWindow(256,'vocals')

    def test_reuse_rejects_different_geometry_dtype_or_bindings(self):
        drums=PreviewWindow(256,'drums');strum=PreviewWindow(256,'guitar-strum')
        previous={'stages':PREVIEW.stage_plan(LONG),'weight_storage_dtype':'bfloat16'}
        for stage in previous['stages']:
            name=stage['asset_file']
            self.assertTrue(PREVIEW.native_reuse_compatible(previous,LONG,strum,name))
            self.assertEqual(PREVIEW.native_reuse_compatible(previous,LONG,drums,name),name=='tail.aimodel')
        for name in ['apg-euler.aimodel','decoder.aimodel']:
            self.assertTrue(PREVIEW.native_reuse_compatible(previous,LONG,drums,name))
            self.assertFalse(PREVIEW.native_reuse_compatible(previous,SHORT,drums,name))
        changed=copy.deepcopy(previous);changed['stages'][1]['output_bindings']={'hidden_out':'wrong'}
        self.assertFalse(PREVIEW.native_reuse_compatible(changed,LONG,strum,'blocks-00-01.aimodel'))
        changed['weight_storage_dtype']='float16'
        self.assertFalse(PREVIEW.native_reuse_compatible(changed,LONG,strum,'tail.aimodel'))
        same={'stages':PREVIEW.stage_plan(drums),'weight_storage_dtype':'bfloat16'}
        for s in same['stages']:
            self.assertTrue(PREVIEW.native_reuse_compatible(same,drums,PreviewWindow(256,'guitar-pick'),s['asset_file']))

    def fixture(self,root,w):
        from test_coreai_preview_window import PreviewWindowTests
        test=PreviewWindowTests();test.pipeline(root,w)
        p=root/w.condition_bundle/'reference.json';ref=json.loads(p.read_text())
        ref['prompts'].update(instruction=w.instruction,preset_sha256=w.preset_sha256)
        p.write_text(json.dumps(ref))
        p=root/w.preview_bundle/'reference.json';ref=json.loads(p.read_text());ref['condition_reference_sha256']=STAGE.digest(root/w.condition_bundle/'reference.json');p.write_text(json.dumps(ref))

    def test_staging_isolates_each_instrument_and_rejects_wrong_preset_before_pruning(self):
        for name in DRIVER.INSTRUMENTS:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'ModelCandidates';dest=root/'build/App.app/ModelCandidates';w=PreviewWindow(256,name)
                self.fixture(source,w)
                (source/'active-preview.json').write_text(json.dumps({'schema_version':1,'resource_profile':w.resource_profile,'audio_seconds':10.24}))
                result=STAGE.stage_resources(source,dest)
                self.assertEqual(result['selected_profile'],w.resource_profile)
                self.assertEqual({n.split('/')[0] for n in result['files']},{w.source_bundle,w.condition_bundle,w.preview_bundle})
                saved=(dest/w.condition_bundle/'reference.json').read_bytes()
                p=source/w.condition_bundle/'reference.json';ref=json.loads(p.read_text());ref['prompts']['preset_sha256']=LONG.preset_sha256;p.write_text(json.dumps(ref))
                with self.assertRaisesRegex(ValueError,'preset identity'):STAGE.stage_resources(source,dest)
                self.assertEqual((dest/w.condition_bundle/'reference.json').read_bytes(),saved)

    def test_failed_preparation_keeps_selector_and_never_activates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selector=root/'active-preview.json';selector.write_text('old bass selector');commands=[]
            def failed(command,**kw):commands.append(command);raise RuntimeError('host parity fails')
            with patch('export_coreai_source_context.verify_bundle'),patch.object(DRIVER,'activate') as activate:
                with self.assertRaisesRegex(RuntimeError,'parity'):
                    DRIVER.prepare('drums',True,candidates=root,runner=failed)
                activate.assert_not_called()
            self.assertEqual(selector.read_text(),'old bass selector')
            self.assertEqual(commands[0][commands[0].index('--instrument')+1],'drums')

    def test_admission_failure_does_not_replace_active_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selector=root/'active-preview.json';selector.write_text('old selector')
            with patch('export_coreai_source_context.verify_bundle'),patch('export_coreai_bass_conditioning.verify_bundle'),patch('export_coreai_bass_preview.verify_bundle',side_effect=ValueError('strict component failed')):
                with self.assertRaisesRegex(ValueError,'component'):ACTIVATE.activate(root,PreviewWindow(256,'drums'),True)
            self.assertEqual(selector.read_text(),'old selector')

    def test_reuse_prefers_matching_token_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for w in [LONG,PreviewWindow(256,'drums')]:
                p=root/w.preview_bundle;p.mkdir();(p/'reference.json').write_text('not yet validated')
            self.assertEqual(DRIVER.reuse_candidate(root,PreviewWindow(256,'guitar-pick'),'preview_bundle'),root/'drums-preview-1024')
            self.assertEqual(DRIVER.reuse_candidate(root,PreviewWindow(256,'guitar-strum'),'preview_bundle'),root/LONG.preview_bundle)

    def test_preview_command_carries_explicit_diagnostic_and_new_component_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);commands=[]
            def stop_at_preview(command,**kw):
                commands.append(command)
                if command[2].endswith('export_coreai_bass_preview.py'): raise RuntimeError('trajectory rejected')
            with patch('export_coreai_source_context.verify_bundle'),patch.object(DRIVER,'activate') as activate:
                with self.assertRaisesRegex(RuntimeError,'trajectory rejected'):
                    DRIVER.prepare('drums',True,candidates=root,runner=stop_at_preview)
                activate.assert_not_called()
            self.assertEqual(len(commands),2)
            self.assertNotIn('--diagnostic-trajectory',commands[0])
            self.assertIn('--diagnostic-trajectory',commands[1])
            i=commands[1].index('--condition-bundle')
            self.assertEqual(commands[1][i+1],str(root/'drums-conditioning-1024'))

    def test_activate_only_validates_then_writes_instrument_selector_without_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runner=[]
            with patch('export_coreai_source_context.verify_bundle'),patch('export_coreai_bass_conditioning.verify_bundle'),patch('export_coreai_bass_preview.verify_bundle',return_value={'host_trajectory_diagnostic_requested':True}):
                payload=DRIVER.prepare('guitar-pick',True,True,root,runner=lambda *a,**k:runner.append(a))
            self.assertEqual(runner,[])
            self.assertEqual(payload,{'schema_version':1,'resource_profile':'guitar-pick-source-1024','audio_seconds':10.24,'diagnostic_trajectory':True})
            self.assertEqual(json.loads((root/'active-preview.json').read_text()),payload)


if __name__=='__main__':unittest.main()

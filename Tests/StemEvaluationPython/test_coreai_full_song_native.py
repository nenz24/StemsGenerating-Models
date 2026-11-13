"""Native batch transport/selection contracts using NONEXECUTABLE model fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import coreai_full_song_contract as C
import activate_coreai_full_song as A
import stage_coreai_app_models as S
import test_coreai_full_song_inputs as input_tests
import test_coreai_drum_reference30_generation as generation_tests


class FullSongBuildTests(unittest.TestCase):
    def test_activation_plan_needs_no_site_packages_or_input_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent=Path(tmp)/'absent'
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/activate_coreai_full_song.py'),
                '--inputs',str(absent)],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('PLAN ONLY',result.stdout);self.assertFalse(absent.exists())

    def test_xcode_compiles_full_song_support_with_existing_probe(self):
        from openstep_parser import OpenStepDecoder
        project=ROOT/'prototypes/ios27-coreai/CoreAIProbe.xcodeproj/project.pbxproj'
        objects=OpenStepDecoder.ParseFromString(project.read_text())['objects']
        refs={key for key,value in objects.items() if value.get('isa')=='PBXFileReference'
              and value.get('path')=='CoreAIFullSongSupport.swift'}
        self.assertEqual(len(refs),1)
        builds={key for key,value in objects.items() if value.get('isa')=='PBXBuildFile' and value.get('fileRef') in refs}
        phases=[value for value in objects.values() if value.get('isa')=='PBXSourcesBuildPhase']
        self.assertTrue(any(builds.intersection(phase['files']) for phase in phases))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),'Optional input fixture dependencies')
class FullSongNativeTests(unittest.TestCase):
    def setUp(self):
        self.inputs=input_tests.InputTests();self.inputs.setUp();self.addCleanup(self.inputs.doCleanups)
        self.job=self.inputs.build()
        self.models=generation_tests.Reference30GenerationTests();self.models.setUp();self.addCleanup(self.models.doCleanups)
        self.models.fixtures();self.candidates=self.models.source
        self.backups=self.models.root/'activation-backups'

    def activate(self):
        with patch('prepare_coreai_drum_reference30.verify_bundle'),patch('prepare_coreai_drum_reference30_window_decoder.verify_bundle'), \
                patch.object(A,'ROOT',self.models.root/'repo'):
            return A.activate(self.inputs.out,self.candidates,self.backups)

    def test_shared_weights_and_exact_global_inputs_are_staged_without_qwen_or_source_models(self):
        # Match canonical configuration bytes used by real30s activation.
        old=self.candidates/'drums-reference30-generation.json';old.write_text(json.dumps(json.loads(old.read_text()),indent=2)+'\n')
        config=self.activate();destination=self.models.root/'build/Probe.app/ModelCandidates'
        r=S.stage_resources(self.candidates,destination)
        self.assertEqual(r['selected_profile'],C.PROFILE)
        self.assertEqual(len(r['files']),83)
        self.assertEqual(sum(n.endswith('main.mlirb') for n in r['files']),16)
        self.assertIn(config['job_bundle']+'/piano48.f32le',r['files'])
        self.assertFalse(any('qwen' in n.lower() or 'source-context.aimodel' in n for n in r['files']))
        self.assertEqual(S.stage_resources(self.candidates,destination)['copied_files'],0)
        self.assertEqual(json.loads((self.candidates/'active-preview.json').read_text())['resource_profile'],C.PROFILE)
        self.assertTrue(list(self.backups.glob('*/active-preview.json')))

    def test_configuration_and_input_cannot_promote_native_parity_or_change_bounds(self):
        config=C.configuration('1'*64,'2'*64);self.assertEqual(C.validate_config(config),config)
        for key,value in [('full_song_strict_parity','PASS'),('scope','production'),('sampler_bundle','other'),('baseline_modified',True)]:
            with self.assertRaises(ValueError):C.validate_config({**config,key:value})
        for key,value in [('guidance_scale',5.),('ready_for_native_execution',True),('relative_tolerance',.01),('frames32',960001)]:
            with self.assertRaises(ValueError):C.validate_inputs({**self.job,key:value})
        changed=copy.deepcopy(self.job);changed['runs'][0]['initial_noise_sha256']='not-a-hash'
        with self.assertRaises(ValueError):C.validate_inputs(changed)

    def test_invalid_activation_preserves_selection_and_existing_config(self):
        selector=self.candidates/'active-preview.json';before=selector.read_bytes()
        (self.candidates/C.CONFIG).write_bytes(b'existing execution config')
        path=self.inputs.out/'bass/noise-global.f32le'
        with path.open('r+b') as stream:stream.write(b'bad!')
        with self.assertRaises(ValueError):self.activate()
        self.assertEqual(selector.read_bytes(),before)
        self.assertEqual((self.candidates/C.CONFIG).read_bytes(),b'existing execution config')

    def test_failed_selector_update_rolls_back_config_and_keeps_backup(self):
        old=self.candidates/'drums-reference30-generation.json';old.write_text(json.dumps(json.loads(old.read_text()),indent=2)+'\n')
        path=self.candidates/C.CONFIG;path.write_bytes(b'prior generated config')
        selector=self.candidates/'active-preview.json';before=selector.read_bytes()
        def failed_selection(*args):
            selector.write_bytes(b'partially selected')
            raise RuntimeError('selector failure')
        with patch('activate_coreai_drum_reference30.select_profile',side_effect=failed_selection):
            with self.assertRaises(RuntimeError):self.activate()
        self.assertEqual(path.read_bytes(),b'prior generated config')
        self.assertEqual(selector.read_bytes(),before)
        self.assertEqual(next(self.backups.glob('*/'+C.CONFIG)).read_bytes(),b'prior generated config')

    def test_mismatched_job_file_does_not_prune_existing_app(self):
        old=self.candidates/'drums-reference30-generation.json';old.write_text(json.dumps(json.loads(old.read_text()),indent=2)+'\n')
        config=self.activate();destination=self.models.root/'build/Probe.app/ModelCandidates';destination.mkdir(parents=True)
        marker=destination/'keep';marker.write_bytes(b'old build')
        (self.candidates/config['job_bundle']/'bass/noise-global.f32le').write_bytes(b'wrong size')
        with self.assertRaises(ValueError):S.stage_resources(self.candidates,destination)
        self.assertEqual(marker.read_bytes(),b'old build')

    def test_stage_build_remains_stdlib_only(self):
        old=self.candidates/'drums-reference30-generation.json';old.write_text(json.dumps(json.loads(old.read_text()),indent=2)+'\n')
        self.activate()
        destination=self.models.root/'build/Probe.app/ModelCandidates'
        r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/stage_coreai_app_models.py'),'--source',str(self.candidates),
                          '--destination',str(destination),'--dry-run'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('83 files',r.stdout);self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()

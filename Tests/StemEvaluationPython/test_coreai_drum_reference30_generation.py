"""Frozen generation selection/transport, not Apple execution or music evidence."""
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
import coreai_drum_reference30_generation_contract as C
import activate_coreai_drum_reference30_generation as A
import stage_coreai_app_models as S
from test_coreai_drum_window_decoder import metadata as window_metadata


class Reference30GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'repo/ModelCandidates';self.source.mkdir(parents=True)

    def fixtures(self):
        # NONEXECUTABLE native bytes and synthetic arrays exercise packaging only.
        sampler=self.source/C.SAMPLER_BUNDLE;sampler.mkdir()
        (sampler/'reference.json').write_bytes((ROOT/'reviews/coreai-drums-reference30-export-completed/reference.json').read_bytes())
        for name in C.KEEP:
            path=sampler/name;path.parent.mkdir(parents=True,exist_ok=True)
            if name in C.SHAPES:
                with path.open('wb') as f:f.truncate(math.prod(C.SHAPES[name])*4)
            else:path.write_bytes(b'NONEXECUTABLE sampler transport fixture')
        decoder=self.source/C.DECODER_BUNDLE;decoder.mkdir()
        w=window_metadata();(decoder/'reference.json').write_text(json.dumps(w))
        from coreai_drum_reference30_window_contract import FILES,DATA_SHAPES
        for name in FILES:
            path=decoder/name;path.parent.mkdir(parents=True,exist_ok=True)
            if name in DATA_SHAPES:
                with path.open('wb') as f:f.truncate(math.prod(DATA_SHAPES[name])*4)
            else:path.write_bytes(b'NONEXECUTABLE decoder transport fixture')
        sha=hashlib.sha256((decoder/'reference.json').read_bytes()).hexdigest()
        config=C.configuration(sha)
        (self.source/C.CONFIG).write_text(json.dumps(config))
        (self.source/'active-preview.json').write_text(json.dumps({'schema_version':1,'resource_profile':C.PROFILE,'audio_seconds':30.}))
        return config

    def test_original_sampler_feedback_duration_gain_and_tolerances_fixed(self):
        r=C.configuration('0'*64);self.assertEqual(C.validate(r,'0'*64),r)
        self.assertEqual((r['inference_steps'],r['latent_frames'],r['guidance_scale'],r['apg_norm_threshold']), (50,750,7.,2.5))
        self.assertEqual(r['checkpoints'],[1,5,10,25,50])
        self.assertEqual(r['feedback'],'live_native_latent_and_momentum_no_gold_resets')
        self.assertEqual(r['sampler_weight_storage_dtype'],'bfloat16')
        self.assertFalse(any(n.startswith('decoder.aimodel/') for n in C.KEEP))
        self.assertEqual(sum(n.endswith('main.mlirb') for n in C.KEEP),15)
        for name,value in [('inference_steps',25),('absolute_tolerance',1e-3),('piano_mix_gain',.78),
                           ('feedback','gold_resets'),('native_full_trajectory','PASS')]:
            with self.assertRaises(ValueError):C.validate({**r,name:value},'0'*64)

    def test_build_contains_only_sampling_assets_bounded_decoder_and_reference_data(self):
        self.fixtures();out=self.root/'build/Probe.app/ModelCandidates'
        r=S.stage_resources(self.source,out)
        self.assertEqual(r['selected_profile'],C.PROFILE)
        self.assertEqual(len(r['files']),65)
        self.assertFalse(any(n.startswith(C.SAMPLER_BUNDLE+'/decoder.aimodel/') for n in r['files']))
        self.assertIn(C.SAMPLER_BUNDLE+'/capture/source48.f32le',r['files'])
        self.assertIn(C.DECODER_BUNDLE+'/decoder.aimodel/main.mlirb',r['files'])
        self.assertEqual(S.stage_resources(self.source,out)['copied_files'],0)

    def test_changed_decoder_identity_missing_checkpoint_or_wrong_shape_preserves_app(self):
        config=self.fixtures();out=self.root/'build/Probe.app/ModelCandidates';out.mkdir(parents=True)
        marker=out/'keep';marker.write_text('previous app')
        d=self.source/C.DECODER_BUNDLE/'reference.json';old=d.read_bytes();d.write_bytes(old+b' ')
        with self.assertRaises(ValueError):S.stage_resources(self.source,out)
        d.write_bytes(old)
        p=self.source/C.SAMPLER_BUNDLE/'capture/step-25-latent.f32le';p.unlink()
        with self.assertRaises(ValueError):S.stage_resources(self.source,out)
        p.write_bytes(b'wrong shape')
        with self.assertRaises(ValueError):S.stage_resources(self.source,out)
        self.assertEqual(marker.read_text(),'previous app')

    def test_build_cli_remains_stdlib_only(self):
        self.fixtures();out=self.root/'build/Probe.app/ModelCandidates'
        r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/stage_coreai_app_models.py'),
            '--source',str(self.source),'--destination',str(out),'--dry-run'],check=True,capture_output=True,text=True)
        self.assertIn('65 files',r.stdout);self.assertFalse(out.exists())

    def test_activation_help_never_attempts_models_or_changes_selection(self):
        r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/activate_coreai_drum_reference30_generation.py'),'--help'],
            cwd=self.root,check=True,capture_output=True,text=True)
        self.assertIn('usage:',r.stdout)
        self.assertEqual(list(self.source.iterdir()),[])

    def test_activation_verifies_both_bundles_then_preserves_selector_and_configs(self):
        self.fixtures();selector=self.source/'active-preview.json';old=b'{"old":"selection"}\n';selector.write_bytes(old)
        (self.source/C.CONFIG).unlink()
        from activate_coreai_drum_reference30 import select_profile
        backups=self.root/'backups'
        def select(candidates,profile):select_profile(candidates,profile,backups)
        with patch.object(A,'verify_sampler') as sampler,patch.object(A,'verify_decoder') as decoder, \
                patch.object(A,'sha256',side_effect=lambda p:C.SOURCE_SHA if Path(p).name=='source48.f32le' else hashlib.sha256(Path(p).read_bytes()).hexdigest()), \
                patch.object(A,'select_profile',side_effect=select):
            A.activate(self.source)
            self.assertEqual(sampler.call_count,1);self.assertEqual(decoder.call_count,1)
            self.assertEqual(json.loads(selector.read_bytes())['resource_profile'],C.PROFILE)
            self.assertEqual(next(backups.glob('*/active-preview.json')).read_bytes(),old)
            with patch.object(A,'select_profile') as choose:
                config=self.source/C.CONFIG;config.write_bytes(b'different existing config')
                with self.assertRaises(FileExistsError):A.activate(self.source)
                choose.assert_not_called();self.assertEqual(config.read_bytes(),b'different existing config')

    def test_missing_or_changed_bundle_never_selects_generation(self):
        self.fixtures();selector=self.source/'active-preview.json';before=selector.read_bytes()
        with patch.object(A,'verify_sampler'),patch.object(A,'verify_decoder',side_effect=ValueError('changed decoder')),patch.object(A,'select_profile') as select:
            with self.assertRaises(ValueError):A.activate(self.source)
            select.assert_not_called()
        self.assertEqual(selector.read_bytes(),before)


if __name__=='__main__':unittest.main()

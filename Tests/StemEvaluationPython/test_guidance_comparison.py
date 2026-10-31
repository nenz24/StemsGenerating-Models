"""One-setting generation diagnostic and guidance-aware cache protection, without ML inference."""
import json
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
from backends import AceBackend,validate_guidance
from compare_guidance import prepare
import compare_guidance
from provenance import sha256
from run import cached_job
from test_full_song import fixture


class GuidanceComparisonTests(unittest.TestCase):
    def make_comparison(self,root):
        run=root/'decoder_comparison_fixture'
        (run/'prepared').mkdir(parents=True)
        fixture(run/'prepared/source.wav',30)
        profile=json.loads((ROOT/'configs/mac-full-song.json').read_text())
        source=run/'prepared/source.wav'
        job={'condition':'A','instrument':'bass','seed':0,'duration':30,'source':str(source),'source_sha256':sha256(source)}
        folder=run/'default_result/000_A_bass_seed0';folder.mkdir(parents=True)
        (folder/'stem.wav').write_bytes(source.read_bytes())
        params={'task_type':'lego','inference_steps':50,'guidance_scale':7.0,'duration':30.0,'seed':0,
            'instruction':'Generate the bass track based on the audio context:',
            'caption':'Isolated bass accompaniment following the source harmony, timing and dynamics.',
            'thinking':False,'enable_normalization':False,'instrumental':True,'lyrics':'[Instrumental]',
            'repainting_start':0.0,'repainting_end':-1,'use_cot_caption':False,'use_cot_metas':False,'use_cot_language':False}
        result=dict(job,status='generated',stem_file='stem.wav',stem_sha256=sha256(folder/'stem.wav'),execution_signature=None,
                    settings={'params':params,'config':{'fixture_only':True}})
        (folder/'result.json').write_text(json.dumps(result))
        (run/'default_result/experiment.json').write_text(json.dumps({'status':'generated','jobs':[job]}))
        (run/'comparison.json').write_text(json.dumps({'status':'decoded','profile':profile,'source_sha256':sha256(source),
            'assets_sha256':sha256(ROOT/'configs/assets-base.json')}))
        return run,job,folder

    def test_invalid_guidance_fails_before_model_import(self):
        for value in (True,False,float('nan'),float('inf'),0,16,'1'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                AceBackend(Path('/no-model'),guidance_scale=value)
        self.assertEqual(validate_guidance(1),1.0)
        self.assertEqual(validate_guidance(7),7.0)

    def test_backend_sends_cfg1_with_same_source_seed_steps_and_caption(self):
        handler=MagicMock(device='mps')
        handler.initialize_service.return_value=('ok',True)
        params,config=MagicMock(),MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audio=root/'audio.wav';audio.touch()
            generate=MagicMock(return_value=types.SimpleNamespace(success=True,audios=[{'path':str(audio)}]))
            modules={'acestep':types.ModuleType('acestep'),
                'acestep.handler':types.SimpleNamespace(AceStepHandler=lambda:handler),
                'acestep.inference':types.SimpleNamespace(GenerationParams=params,GenerationConfig=config,generate_music=generate)}
            calls=[]
            with patch.dict(sys.modules,modules):
                for guidance in (7.0,1.0):
                    backend=AceBackend(root,device='mps',guidance_scale=guidance)
                    backend.generate(audio,30,'bass',0,root)
                    calls.append(dict(params.call_args.kwargs))
            self.assertEqual(calls[0].pop('guidance_scale'),7)
            self.assertEqual(calls[1].pop('guidance_scale'),1)
            self.assertEqual(calls[0],calls[1])
            self.assertEqual(calls[0]['inference_steps'],50)

    def test_baseline_snapshot_matches_source_and_plan_runs_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,job,folder=self.make_comparison(root)
            report=prepare(run,root/'snapshot')
            self.assertEqual(sha256(root/'snapshot/prepared/source.wav'),job['source_sha256'])
            self.assertEqual(sha256(root/'snapshot/baseline_cfg7.wav'),sha256(folder/'stem.wav'))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/compare_guidance.py'),
                '--run',str(run),'--out',str(root/'planned')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads((root/'planned/cfg1_result/experiment.json').read_text())
            self.assertEqual(plan['status'],'planned')
            self.assertEqual(plan['guidance_scale'],1)
            self.assertEqual(len(plan['jobs']),1)
            self.assertEqual(plan['jobs'][0]['guidance_scale'],1)
            self.assertFalse(list((root/'planned/cfg1_result').glob('**/stem.wav')))
            self.assertEqual(report['baseline_guidance'],7)

    def test_legacy_cfg7_cache_cannot_be_reused_as_cfg1(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,job,folder=self.make_comparison(root)
            self.assertIsNotNone(cached_job(folder,dict(job,guidance_scale=7),None))
            with self.assertRaisesRegex(ValueError,'different guidance'):
                cached_job(folder,dict(job,guidance_scale=1),None)

    def test_changed_baseline_or_input_is_rejected_before_output(self):
        for name in ('source','guidance'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);run,_,folder=self.make_comparison(root)
                if name=='source':
                    with (run/'prepared/source.wav').open('ab') as stream:stream.write(b'changed')
                else:
                    result=json.loads((folder/'result.json').read_text())
                    result['settings']['params']['guidance_scale']=3
                    (folder/'result.json').write_text(json.dumps(result))
                with self.assertRaises(ValueError):prepare(run,root/'bad')
                self.assertFalse((root/'bad').exists())

    def test_cli_rejects_nan_and_preserves_default_guidance7(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,_,_=self.make_comparison(root)
            prepare(run,root/'snapshot')
            command=[sys.executable,str(ROOT/'tools/stem_eval/run.py'),'--backend','ace','--prepared',str(root/'snapshot/prepared'),
                     '--out',str(root/'default'),'--seeds','0','--max-job-seconds','30']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/'default/experiment.json').read_text())['guidance_scale'],7)
            denied=subprocess.run(command+['--guidance-scale','nan'],capture_output=True,text=True)
            self.assertNotEqual(denied.returncode,0)
            self.assertIn('finite ACE guidance',denied.stderr)

    def test_execute_generates_only_cfg1_and_flags_other_parameter_changes(self):
        for extra_change in (False,True):
            with self.subTest(extra_change=extra_change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);run,_,_=self.make_comparison(root)
                output=root/'comparison'
                calls=[]
                def fake_generation(command,**kwargs):
                    calls.append(command)
                    self.assertEqual(command[-2:],['--guidance-scale','1'])
                    prepared=output/'prepared'
                    folder=output/'cfg1_result/000_A_bass_seed0';folder.mkdir(parents=True)
                    shutil.copyfile(prepared/'source.wav',folder/'stem.wav')
                    settings=json.loads((output/'comparison.json').read_text())['baseline_settings']
                    settings['params'].update(guidance_scale=1,src_audio=str(prepared/'source.wav'))
                    if extra_change:settings['params']['inference_steps']=100
                    job={'condition':'A','instrument':'bass','seed':0,'duration':30,'source':str(prepared/'source.wav'),
                         'source_sha256':sha256(prepared/'source.wav'),'guidance_scale':1}
                    result=dict(job,status='generated',stem_file='stem.wav',stem_sha256=sha256(folder/'stem.wav'),
                                execution_signature=None,settings=settings)
                    (folder/'result.json').write_text(json.dumps(result))
                    (output/'cfg1_result/experiment.json').write_text(json.dumps({'status':'generated','jobs':[job]}))
                argv=['compare_guidance.py','--run',str(run),'--out',str(output),'--execute']
                with patch.object(sys,'argv',argv),patch('compare_guidance.check_model_runtime',return_value={'fixture_only':True}),patch('compare_guidance.run_logged',side_effect=fake_generation):
                    if extra_change:
                        with self.assertRaisesRegex(ValueError,'Additional generation settings'):
                            compare_guidance.main()
                    else:compare_guidance.main()
                self.assertEqual(len(calls),1)
                report=json.loads((output/'comparison.json').read_text())
                self.assertEqual(report['status'],'failed' if extra_change else 'generated')
                self.assertTrue((output/'baseline_cfg7.wav').is_file())
                if not extra_change:
                    self.assertTrue((output/'cfg7_listen.wav').is_file())
                    self.assertTrue((output/'cfg1_listen.wav').is_file())


if __name__=='__main__':unittest.main()

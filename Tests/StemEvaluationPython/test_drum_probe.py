"""Drum targeting, actual CLI plan and checked export; inference is mocked."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tools/stem_eval')]
import drum_probe as drum
from audio import RATE,write_pcm
from provenance import sha256
from run import plan


class DrumProbeTests(unittest.TestCase):
    def source(self,root):
        path=root/'piano.wav'
        write_pcm(path,[100,-100]*(30*RATE))
        return path

    def test_real_dry_run_targets_one_drum_job_without_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=self.source(p)
            report=drum.generate(ROOT,source,p/'probe')
            experiment=json.loads((p/'probe/generation/experiment.json').read_text())
            self.assertEqual(report['status'],'planned')
            self.assertEqual(len(experiment['jobs']),1)
            self.assertEqual(experiment['jobs'][0]['instrument'],'drums')
            self.assertEqual(experiment['jobs'][0]['duration'],30)
            self.assertFalse((p/'probe/drums.wav').exists())
            self.assertFalse((p/'probe/drum_feedback.zip').exists())

    def fake(self,out,wrong=False):
        prepared=out/'prepared';generation=out/'generation';folder=generation/'000_A_drums_seed0'
        folder.mkdir(parents=True)
        exp=plan(prepared,'ace','drums',[0]);exp['status']='generated'
        (generation/'experiment.json').write_text(json.dumps(exp))
        shutil.copyfile(prepared/'source.wav',folder/'stem.wav')
        shutil.copyfile(prepared/'source.wav',folder/'mix.wav')
        params={'caption':drum.PRESET['caption'],'seed':0,'guidance_scale':7,'inference_steps':50,
                'duration':30,'task_type':'lego','instrumental':True,'enable_normalization':False,
                'thinking':False,'use_cot_caption':False,'use_cot_metas':False,'use_cot_language':False,
                'instruction':'Generate the bass track based on the audio context:' if wrong else 'Generate the drums track based on the audio context:'}
        settings={'params':params,'config':{'seeds':[0],'batch_size':1,'use_random_seed':False}}
        result=dict(exp['jobs'][0],status='generated',device='cuda',execution_signature=None,
                    settings=settings,stem_file='stem.wav',stem_sha256=sha256(folder/'stem.wav'))
        (folder/'result.json').write_text(json.dumps(result))
        (out/'capture').mkdir()
        (out/'capture/capture.json').write_text(json.dumps({'status':'captured','model_dtype':'torch.float32','model_device':'cuda','calls':[{'backend':'pytorch_vae','dtype':'torch.float32','status':'returned'}]}))

    def test_export_requires_drums_instruction_and_fp32_worker(self):
        for wrong in [False,True]:
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);source=self.source(p);out=p/'probe'
                def run(command,**kwargs):
                    self.assertEqual(command[1],'-c')
                    self.assertIn(drum.PRESET['caption'],command[2])
                    self.assertEqual(command[command.index('--instrument')+1],'drums')
                    self.fake(out,wrong)
                with patch.object(drum.single,'model_python',return_value=Path(sys.executable)),patch('stream_process.run_logged',side_effect=run):
                    if wrong:
                        with self.assertRaisesRegex(ValueError,'Actual drum settings differ'):
                            drum.generate(ROOT,source,out,execute=True)
                        self.assertFalse((out/'drums.wav').exists())
                    else:
                        report=drum.generate(ROOT,source,out,execute=True)
                        self.assertEqual(report['status'],'generated')
                        self.assertEqual(drum.single.frames(out/'drums.wav'),30*RATE)
                        self.assertTrue((out/'drum_feedback.zip').is_file())
                        self.assertTrue((out/'drum_result.zip').is_file())

    def test_discovers_the_existing_fresh_vm_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'ace_context_source_existing'
            for name in ['configs/assets-base.json','tools/stem_eval/run.py','scripts/setup_runtime.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('marker')
            self.assertEqual(drum.single.discover_root(base),root)


if __name__=='__main__':unittest.main()

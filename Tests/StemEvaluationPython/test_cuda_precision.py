"""Verify the worker's loader/offload dtype boundary without Torch or a GPU."""
from pathlib import Path
import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from cuda_precision import FLOAT32_WORKER,float32_command


class CudaPrecisionTests(unittest.TestCase):
    def test_real_run_entrypoint_imports_siblings_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source'
            scripts=source/'tools/stem_eval'
            shutil.copytree(ROOT/'tools/stem_eval',scripts,ignore=shutil.ignore_patterns('__pycache__'))
            upstream=source/'vendor/ACE-Step-1.5'
            package=upstream/'acestep';package.mkdir(parents=True)
            (package/'__init__.py').touch()
            (package/'handler.py').write_text('class AceStepHandler: pass\n')
            (upstream/'torch.py').write_text('float32 = "float32"\n')
            prepared=root/'prepared';prepared.mkdir()
            # Plan mode verifies hashes, without decoding or inferring audio.
            audio=b'hash-only input fixture; no model output'
            conditioning=b'{}'
            (prepared/'source.wav').write_bytes(audio)
            (prepared/'conditioning.json').write_bytes(conditioning)
            (prepared/'prepared.json').write_text(json.dumps({
                'schema_version':1,'duration':30,'conditions':{'A':'source.wav'},
                'condition_sha256':{'A':hashlib.sha256(audio).hexdigest()},
                'conditioning_sha256':hashlib.sha256(conditioning).hexdigest()}))
            unrelated=root/'content';unrelated.mkdir()
            output=root/'plan'
            command=float32_command([
                sys.executable,str(scripts/'run.py'),'--prepared',str(prepared),
                '--backend','ace','--seeds','0','--device','cuda',
                '--max-job-seconds','30','--out',str(output)])
            environment=dict(os.environ)
            environment.pop('PYTHONPATH',None)
            result=subprocess.run(command,cwd=unrelated,env=environment,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            experiment=json.loads((output/'experiment.json').read_text())
            self.assertEqual(experiment['status'],'planned')
            self.assertEqual(experiment['device_requested'],'cuda')
            self.assertEqual(len(experiment['jobs']),1)
            self.assertEqual(experiment['jobs'][0]['duration'],30)

    def test_override_precedes_weight_loading_and_survives_vae_offload(self):
        events=[]
        class Handler:
            def initialize_service(self,device):
                # Match pinned upstream: initializer overwrites dtype, then loads
                # DiT, VAE and text, in that order.
                self.dtype='float16'
                self._load_main_model_from_checkpoint(device=device)
                events.append(('text',self.dtype))
                for target in ('cpu','cuda',None,'cpu'):
                    events.append(('vae',self._get_vae_dtype(target)))
            def _load_main_model_from_checkpoint(self,**kwargs):
                events.append(('weights',self.dtype))
            def _get_vae_dtype(self,device=None):
                return 'float16'
        handler_module=types.SimpleNamespace(AceStepHandler=Handler)
        modules={'torch':types.SimpleNamespace(float32='float32'),
                 'acestep':types.ModuleType('acestep'),'acestep.handler':handler_module}
        script=Path('/fixture/source/tools/stem_eval/run.py')
        command=[sys.executable,str(script),'--device','cuda','--guidance-scale','7']
        def entry(path,run_name):
            self.assertEqual(path,str(script))
            self.assertEqual(run_name,'__main__')
            self.assertEqual(sys.argv,command[1:])
            self.assertEqual(sys.path[:2],['/fixture/source/tools/stem_eval','/fixture/source/vendor/ACE-Step-1.5'])
            handler_module.AceStepHandler().initialize_service('cuda')
            with self.assertRaisesRegex(RuntimeError,'requires device=cuda'):
                handler_module.AceStepHandler().initialize_service('mps')
        before=list(sys.path)
        try:
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',['-c',*command[1:]]),patch('runpy.run_path',side_effect=entry):
                exec(compile(FLOAT32_WORKER,'fp32-worker','exec'),{})
        finally:
            sys.path[:]=before
        self.assertEqual(events,[('weights','float32'),('text','float32')]+[('vae','float32')]*4)
        self.assertEqual(float32_command(command),[sys.executable,'-c',FLOAT32_WORKER,*command[1:]])

    def test_existing_colab_recovery_cell_reuses_worker_and_preserves_failed_run(self):
        doc=(ROOT/'docs/colab-fp32-recovery.md').read_text()
        cell=doc.split('```python\n',1)[1].split('```',1)[0]
        for fails in (False,True):
            with self.subTest(fails=fails),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                previous=root/'runs/previous_failure';previous.mkdir(parents=True)
                (previous/'cuda_reference.json').write_text('preserve original failure')
                original=Mock()
                module=types.SimpleNamespace(run_logged=original)
                def run(reference,output,execute):
                    self.assertEqual(reference,root/'reference')
                    self.assertTrue(execute)
                    self.assertNotEqual(output,previous)
                    output.mkdir()
                    status={'status':'failed' if fails else 'generated'}
                    (output/'cuda_reference.json').write_text(json.dumps(status))
                    module.run_logged(['/runtime/python','/source/tools/stem_eval/run.py','--device','cuda'])
                    if fails:raise RuntimeError('fixture CUDA failure')
                    return status
                module.run_reference=run
                namespace={'ROOT':root,'REFERENCE':root/'reference'}
                with patch.dict(sys.modules,{'cuda_reference':module}):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError,'fixture CUDA failure'):
                            exec(compile(cell,'colab-recovery-cell','exec'),namespace)
                    else:exec(compile(cell,'colab-recovery-cell','exec'),namespace)
                command=original.call_args.args[0]
                self.assertEqual(command[:2],['/runtime/python','-c'])
                self.assertEqual(command[2].strip(),FLOAT32_WORKER.strip())
                self.assertEqual(command[3:],['/source/tools/stem_eval/run.py','--device','cuda'])
                self.assertIs(module.run_logged,original)
                self.assertEqual((previous/'cuda_reference.json').read_text(),'preserve original failure')
                report=json.loads((namespace['WORK']/'cuda_reference.json').read_text())
                self.assertEqual(report['precision_override'],'float32')
                self.assertEqual(report['status'],'failed' if fails else 'generated')


if __name__=='__main__':unittest.main()

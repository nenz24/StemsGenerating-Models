"""Portable exact-B CUDA reference protocol; no GPU or model weights run in cloud tests."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
import cuda_reference
from cuda_reference import export_bundle,run_reference,verify_bundle
from provenance import sha256
import test_guidance_comparison


class CudaReferenceTests(unittest.TestCase):
    def make_bundle(self,root,bad_manifest=False):
        run,_,_=test_guidance_comparison.GuidanceComparisonTests().make_comparison(root)
        source=io.BytesIO()
        with zipfile.ZipFile(source,'w') as archive:
            for name in ('scripts/cuda_reference.py','notebooks/08_cuda_reference.ipynb','configs/assets-base.json'):
                data=(ROOT/name).read_bytes()
                if bad_manifest and name=='configs/assets-base.json':data=b'changed'
                archive.writestr(name,data)
        def git(command,**kwargs):
            if 'archive' in command:return source.getvalue()
            if 'rev-parse' in command:return 'fixture-committed-source\n'
            raise AssertionError(command)
        archive=root/'cuda_reference.zip'
        with patch('cuda_reference.subprocess.check_output',side_effect=git):
            export_bundle(run,archive)
        destination=root/'unpacked';destination.mkdir()
        with zipfile.ZipFile(archive) as bundle:bundle.extractall(destination)
        return run,archive,destination

    def test_bundle_copies_full_B_and_baseline_without_models_or_git_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,archive,bundle=self.make_bundle(root)
            manifest=verify_bundle(bundle)
            self.assertEqual(manifest['duration'],30)
            self.assertEqual(manifest['guidance_scale'],7)
            self.assertEqual(manifest['seed'],0)
            self.assertEqual(sha256(bundle/'reference/prepared/source.wav'),sha256(run/'prepared/source.wav'))
            protocol=json.loads((bundle/'reference/comparison.json').read_text())
            self.assertEqual(protocol['assets_sha256'],sha256(ROOT/'configs/assets-base.json'))
            self.assertEqual(protocol['profile']['device'],'mps')
            with zipfile.ZipFile(archive) as contents:
                self.assertFalse(any('checkpoints/' in name or '/.git/' in name or name.endswith('.safetensors') for name in contents.namelist()))
                self.assertIn('source/notebooks/08_cuda_reference.ipynb',contents.namelist())

    def test_tampered_bundle_or_reference_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,bundle=self.make_bundle(root)
            with (bundle/'reference/prepared/source.wav').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):verify_bundle(bundle)
            with self.assertRaises(ValueError):run_reference(bundle/'reference',root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_jupyter_saved_outputs_are_allowed_but_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,bundle=self.make_bundle(root)
            path=bundle/'source/notebooks/08_cuda_reference.ipynb'
            book=json.loads(path.read_text())
            code=next(cell for cell in book['cells'] if cell['cell_type']=='code')
            code['execution_count']=1
            code['outputs']=[{'output_type':'stream','name':'stdout','text':['fixture output\n']}]
            code['source']=''.join(code['source']).replace('DRY_RUN = False','DRY_RUN = True')
            path.write_text(json.dumps(book))
            verify_bundle(bundle)
            code['source']+='\nprint("changed code")\n'
            path.write_text(json.dumps(book))
            with self.assertRaisesRegex(ValueError,'content changed'):verify_bundle(bundle)

    def test_changed_source_snapshot_manifest_is_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                self.make_bundle(root,bad_manifest=True)
            self.assertFalse((root/'cuda_reference.zip').exists())

    def test_plan_is_exactly_one_cuda_job_and_never_runs_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,bundle=self.make_bundle(root)
            report=run_reference(bundle/'reference',root/'plan')
            self.assertEqual(report['status'],'planned')
            plan=json.loads((root/'plan/generation/experiment.json').read_text())
            self.assertEqual(plan['device_requested'],'cuda')
            self.assertEqual(len(plan['jobs']),1)
            self.assertEqual(plan['jobs'][0]['duration'],30)
            self.assertEqual(plan['jobs'][0]['seed'],0)
            self.assertEqual(plan['guidance_scale'],7)
            self.assertFalse((root/'plan/capture').exists())

    def test_no_cuda_failure_happens_before_model_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,bundle=self.make_bundle(root)
            with patch('cuda_reference.check_cuda',side_effect=RuntimeError('CUDA unavailable')),patch('cuda_reference.run_logged') as model:
                with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):
                    run_reference(bundle/'reference',root/'failed',execute=True)
                model.assert_not_called()
            self.assertEqual(json.loads((root/'failed/cuda_reference.json').read_text())['status'],'failed')

    def test_cuda_runtime_error_keeps_the_actual_driver_diagnostic(self):
        failure=subprocess.CompletedProcess([],1,stdout='',stderr='CUDA driver incompatible with runtime')
        with patch('cuda_reference.subprocess.run',return_value=failure):
            with self.assertRaisesRegex(RuntimeError,'CUDA driver incompatible'):
                cuda_reference.check_cuda(Path('/model/python'))

    def test_single_candidate_must_report_cuda_and_matching_generation_parameters(self):
        for device in ('cuda','mps'):
            with self.subTest(device=device),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);_,_,bundle=self.make_bundle(root)
                reference=bundle/'reference';output=root/'result';calls=[]
                def fake_generation(command,**kwargs):
                    calls.append(command)
                    self.assertEqual(command[command.index('--device')+1],'cuda')
                    folder=output/'generation/000_A_bass_seed0';folder.mkdir(parents=True)
                    shutil.copyfile(reference/'prepared/source.wav',folder/'stem.wav')
                    original=json.loads((reference/'comparison.json').read_text())
                    settings=original['baseline_settings']
                    settings['params']['src_audio']=str(reference/'prepared/source.wav')
                    job={'condition':'A','instrument':'bass','seed':0,'duration':30,'source_sha256':original['source_sha256'],'guidance_scale':7}
                    result=dict(job,status='generated',device=device,execution_signature=None,stem_file='stem.wav',
                                stem_sha256=sha256(folder/'stem.wav'),settings=settings)
                    (folder/'result.json').write_text(json.dumps(result))
                    (output/'generation/experiment.json').write_text(json.dumps({'status':'generated','jobs':[job]}))
                    (output/'generation/session.json').write_text(json.dumps({'device':device,'fixture_only':True}))
                    (output/'capture').mkdir()
                    (output/'capture/capture.json').write_text(json.dumps({'status':'captured','model_device':device,'model_dtype':'fixture'}))
                with patch('cuda_reference.check_cuda',return_value={'fixture_only':True}),patch('cuda_reference.run_logged',side_effect=fake_generation):
                    if device=='cuda':run_reference(reference,output,execute=True)
                    else:
                        with self.assertRaises(ValueError):run_reference(reference,output,execute=True)
                self.assertEqual(len(calls),1)
                report=json.loads((output/'cuda_reference.json').read_text())
                self.assertEqual(report['status'],'generated' if device=='cuda' else 'failed')
                self.assertEqual((output/'cuda_listen.wav').exists(),device=='cuda')


if __name__=='__main__':unittest.main()

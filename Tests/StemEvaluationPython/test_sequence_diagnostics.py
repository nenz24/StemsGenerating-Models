"""Reverse-order diagnostic inputs and real engine orchestration, without a real model."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools/stem_eval'))
from audio import read_pcm
from provenance import sha256
from run_song import song_plan
from sequence_diagnostics import prepare_sequence, run_sequence
from test_full_song import fixture
import run as engine


class SequenceDiagnosticsTests(unittest.TestCase):
    def make_run(self, root):
        fixture(root/'source.wav',65)
        profile = json.loads((ROOT/'configs/mac-full-song.json').read_text())
        run = root/'original'
        experiment, _ = song_plan(root/'source.wav',run,profile,ROOT/'configs/assets-base.json',30,3)
        for index, job in enumerate(experiment['jobs']):
            folder = run/f'{index:03d}_A_bass_seed0'
            attempt = folder/'attempt_fixture'
            attempt.mkdir(parents=True)
            shutil.copyfile(job['source'],attempt/'stem.wav')
            shutil.copyfile(job['source'],attempt/'model_raw.wav')
            (folder/'result.json').write_text(json.dumps(dict(job,status='generated',
                stem_file='attempt_fixture/stem.wav',stem_sha256=sha256(attempt/'stem.wav'),
                execution_signature=experiment['execution_signature'],settings={'fixture_only':True})))
        experiment['status']='generated'
        (run/'experiment.json').write_text(json.dumps(experiment))
        return run

    def test_reverse_order_reuses_exact_bytes_and_leaves_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_run(root)
            before={str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}
            comparison=prepare_sequence(run,root/'diagnostic')
            self.assertEqual([j['original_chunk_index'] for j in comparison['jobs']],[1,0])
            self.assertEqual([j['start_frame']/32000 for j in comparison['jobs']],[27,0])
            original=json.loads((run/'experiment.json').read_text())
            for position,index in enumerate((1,0)):
                job=comparison['jobs'][position]
                self.assertEqual(sha256(job['source']),original['jobs'][index]['source_sha256'])
                self.assertEqual(read_pcm(job['source']),read_pcm(original['jobs'][index]['source']))
                self.assertEqual(job['duration'],30)
                self.assertEqual(job['seed'],0)
                self.assertEqual(job['source'],job['clean_source'])
                self.assertTrue(Path(job['source']).is_relative_to(root/'diagnostic'))
                self.assertTrue((root/'diagnostic'/f'cached_{index:03d}'/'raw_0.wav').is_file())
            after={str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            with self.assertRaises(FileExistsError):
                prepare_sequence(run,root/'diagnostic')

    def test_changed_source_or_stem_is_rejected_before_creating_output(self):
        for changed in ('prepared/chunk_001.wav','001_A_bass_seed0/attempt_fixture/stem.wav'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                run=self.make_run(root)
                with (run/changed).open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaises(ValueError):
                    prepare_sequence(run,root/'diagnostic')
                self.assertFalse((root/'diagnostic').exists())

    def test_cli_defaults_to_cache_only_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_run(root)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/sequence_diagnostics.py'),
                '--run',str(run),'--out',str(root/'diagnostic')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('PLAN ONLY',result.stdout)
            self.assertEqual(json.loads((root/'diagnostic/comparison.json').read_text())['status'],'planned')
            self.assertFalse((root/'diagnostic/runtime').exists())
            self.assertFalse((root/'diagnostic/000_A_bass_seed0').exists())

    def run_with_fake_model(self, root, fail_second=False):
        run=self.make_run(root)
        output=root/'diagnostic'
        report=prepare_sequence(run,output)
        calls=[]
        class Model:
            def generate(self, source, duration, instrument, seed, target):
                calls.append(str(source))
                if fail_second and len(calls)==2:
                    raise RuntimeError('fixture second-call failure')
                raw=target/'raw.wav'
                shutil.copyfile(source,raw)
                return raw,{'fixture_only':True}
        factory=MagicMock(return_value=Model())
        torch=types.ModuleType('torch')
        torch.inference_mode=nullcontext
        with patch.dict(os.environ), patch.dict(sys.modules,{'torch':torch}), patch('backends.AceBackend',factory), patch('backends.weight_inventory',return_value=[]), patch('model_assets.process'), patch('model_assets.materialize',return_value=root/'runtime/checkpoints'), patch.object(engine,'check_upstream'), patch.object(engine,'select_device',return_value='mps'), patch.object(engine,'synchronize'), patch.object(engine,'reset_memory_stats'), patch.object(engine,'memory_stats',return_value={}):
            if fail_second:
                with self.assertRaisesRegex(RuntimeError,'fixture second-call failure'):
                    run_sequence(output,report)
            else:
                run_sequence(output,report)
        self.assertEqual(factory.call_count,1)
        self.assertEqual(calls,[j['source'] for j in report['jobs']])
        return output,report

    def test_real_engine_calls_one_handler_in_reverse_order_with_aligned_mixes(self):
        with tempfile.TemporaryDirectory() as directory:
            output,report=self.run_with_fake_model(Path(directory))
            self.assertEqual(report['status'],'generated')
            for position,job in enumerate(report['jobs']):
                folder=output/f'{position:03d}_A_bass_seed0'
                result=json.loads((folder/'result.json').read_text())
                self.assertEqual(result['original_chunk_index'],job['original_chunk_index'])
                self.assertEqual(read_pcm(folder/'stem.wav'),read_pcm(job['source']))
                self.assertEqual(read_pcm(folder/'mix.wav'),read_pcm(job['source']))

    def test_second_job_failure_keeps_first_result_and_comparison_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output,report=self.run_with_fake_model(Path(directory),fail_second=True)
            self.assertEqual(json.loads((output/'comparison.json').read_text())['status'],'failed')
            self.assertEqual(json.loads((output/'000_A_bass_seed0/result.json').read_text())['status'],'generated')
            self.assertEqual(json.loads((output/'001_A_bass_seed0/result.json').read_text())['status'],'failed')

    def test_notebook_is_unexecuted_and_only_replays_two_jobs(self):
        book=json.loads((ROOT/'notebooks/07_sequence_diagnostics.ipynb').read_text())
        source=''.join(''.join(c['source']) for c in book['cells'])
        self.assertIn('full_song30_',source)
        self.assertIn('RUN_REPLAY = True',source)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),'sequence-cell','exec')


if __name__=='__main__': unittest.main()

"""Source-only diagnostic contract; actual VAE weights are not run in cloud tests."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
from check_source_roundtrip import prepare,reconstruct
from audio import RATE,read_pcm
from provenance import sha256
from test_full_song import fixture
from test_decoder_comparison import Tensor


class SourceRoundtripTests(unittest.TestCase):
    def make_run(self,root):
        run=root/'decoder_comparison_fixture'
        (run/'prepared').mkdir(parents=True)
        source=run/'prepared/source.wav'
        fixture(source,30)
        (run/'comparison.json').write_text(json.dumps({'status':'decoded','source_sha256':sha256(source)}))
        return run

    def test_same_feedback_interval_snapshot_has_no_second_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_run(root)
            source=run/'prepared/source.wav'
            original_hash=sha256(source)
            output=root/'roundtrip'
            metadata=prepare(run,output)
            self.assertEqual(read_pcm(output/'source.wav'),read_pcm(source)[2*3*RATE:2*13*RATE])
            self.assertEqual(metadata['duration'],10)
            self.assertEqual(sha256(source),original_hash)
            with self.assertRaises(FileExistsError):prepare(run,output)

    def test_tampered_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_run(root)
            with (run/'prepared/source.wav').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):prepare(run,root/'roundtrip')
            self.assertFalse((root/'roundtrip').exists())

    def test_plan_cli_works_without_loading_torch_or_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_run(root)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/check_source_roundtrip.py'),
                '--run',str(run),'--out',str(root/'roundtrip')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('PLAN ONLY',result.stdout)
            self.assertFalse(list((root/'roundtrip').glob('*encoder*')))

    def test_common_cpu_decoder_receives_the_original_latent_values(self):
        values=np.arange(400,dtype=np.float32).reshape(1,2,200)/1000
        calls=[]
        class Vae:
            def decode(self,tensor):
                calls.append(tensor.values.copy())
                return types.SimpleNamespace(sample=Tensor(np.repeat(tensor.values,4,axis=-1)))
        torch=types.SimpleNamespace(from_numpy=Tensor,inference_mode=nullcontext)
        with patch.dict(sys.modules,{'torch':torch}):
            actual=reconstruct(Vae(),values)
        np.testing.assert_array_equal(actual,np.repeat(values,4,axis=-1))
        self.assertGreater(len(calls),1)
        self.assertTrue(all(window.shape[1]==2 and window.shape[2]<=128 for window in calls))
        np.testing.assert_array_equal(values,np.arange(400,dtype=np.float32).reshape(1,2,200)/1000)


if __name__=='__main__': unittest.main()

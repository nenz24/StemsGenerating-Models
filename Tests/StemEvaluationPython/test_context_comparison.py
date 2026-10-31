"""Controlled input identity and explicit longer-context opt-in, without real model inference."""
import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, canonicalize, read_pcm, write_pcm
from context_comparison import prepare_comparison
from mac_workflow import run_command
from provenance import sha256
from song_audio import prepare_song


class ContextComparisonTests(unittest.TestCase):
    def make_run(self, root):
        source = root / 'input.wav'
        write_pcm(source, array.array('h',[1000,-1000])*(32*RATE))
        run = root / 'original'
        song = prepare_song(source, run/'prepared')
        profile = json.loads((ROOT/'configs/mac-full-song.json').read_text())
        (run/'identity.json').write_text(json.dumps({'profile':profile,'assets_sha256':sha256(ROOT/'configs/assets-base.json')}))
        chunk=song['chunks'][2]
        job=run/'002_A_bass_seed0';job.mkdir()
        stem=job/'stem.wav'
        write_pcm(stem, array.array('h',[800,-800])*(10*RATE))
        (job/'result.json').write_text(json.dumps({'status':'generated','stem_file':'stem.wav','stem_sha256':sha256(stem)}))
        return run,song,profile

    def test_replay_keeps_exact_input_bytes_and_context_preserves_existing_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run,song,profile=self.make_run(root)
            comparison=prepare_comparison(run,root/'comparison')
            fresh=Path(comparison['fresh_chunk'])/'source.wav'
            self.assertEqual(sha256(fresh),song['chunks'][2]['sha256'])
            context=Path(comparison['context'])/'source.wav'
            self.assertEqual(len(read_pcm(context)),60*RATE)
            self.assertTrue(read_pcm(context)==read_pcm(run/'prepared/source.wav')[:60*RATE])
            self.assertEqual(comparison['profile'],profile)
            self.assertTrue(comparison['same_input_sha256'])

    def test_30_second_job_requires_explicit_limit_and_plan_does_not_run_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run,_,profile=self.make_run(root)
            comparison=prepare_comparison(run,root/'comparison')
            command=run_command(ROOT,profile,Path(comparison['context']),root/'plan')
            denied=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(denied.returncode,0)
            self.assertFalse((root/'plan').exists())
            accepted=subprocess.run(command+['--max-job-seconds','30'],capture_output=True,text=True)
            self.assertEqual(accepted.returncode,0,accepted.stderr)
            plan=json.loads((root/'plan/experiment.json').read_text())
            self.assertEqual(plan['jobs'][0]['duration'],30)
            self.assertEqual(plan['status'],'planned')
            self.assertFalse(list((root/'plan').glob('**/stem.wav')))

    def test_canonical_export_default_stays_10_but_explicit_30_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.wav'
            write_pcm(source,array.array('h',[1000,-1000])*(30*RATE))
            with self.assertRaises(ValueError):
                canonicalize(source,root/'default.wav',duration=30,gain=1)
            canonicalize(source,root/'explicit.wav',duration=30,gain=1,max_duration=30)
            self.assertEqual(len(read_pcm(root/'explicit.wav')),60*RATE)

    def test_comparison_notebook_has_no_saved_outputs(self):
        book=json.loads((ROOT/'notebooks/05_context_comparison.ipynb').read_text())
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),'context-comparison-cell','exec')


if __name__=='__main__':unittest.main()

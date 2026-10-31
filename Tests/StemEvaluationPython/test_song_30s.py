"""Longer-context timeline and cache isolation; no inference quality claims."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools/stem_eval'))
from audio import RATE, read_pcm
from provenance import COMMITS, sha256
from run_song import song_identity, song_plan
from song_audio import prepare_song, stitch_song
from test_full_song import fixture


class Song30SecondTests(unittest.TestCase):
    def test_180_seconds_round_trip_with_30_second_context_and_final_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root/'source.wav', 180)
            profile = json.loads((ROOT/'configs/mac-full-song.json').read_text())
            plan, song = song_plan(root/'source.wav', root/'run', profile,
                                   ROOT/'configs/assets-base.json', 30, 3)
            self.assertEqual([job['duration'] for job in plan['jobs']], [30]*7)
            self.assertEqual([chunk['start_frame']//RATE for chunk in song['chunks']], [0,27,54,81,108,135,150])
            stitched = root/'stitched.wav'
            stitch_song(song, [root/'run/prepared'/chunk['file'] for chunk in song['chunks']], stitched)
            self.assertEqual(read_pcm(stitched), read_pcm(root/'run/prepared/source.wav'))
            self.assertEqual(len(read_pcm(stitched)), 2*180*RATE)

    def test_legacy_identity_unchanged_and_30s_cannot_reuse_10s_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root/'source.wav', 2)
            source, assets = root/'source.wav', ROOT/'configs/assets-base.json'
            profile = json.loads((ROOT/'configs/mac-full-song.json').read_text())
            old_identity = {'schema_version':1, 'source_sha256':sha256(source), 'profile':profile,
                            'assets_sha256':sha256(assets), 'upstream_commit':COMMITS['ace'],
                            'chunk_seconds':10, 'overlap_seconds':1}
            legacy, _ = song_plan(source, root/'legacy', profile, assets)
            self.assertEqual(legacy['execution_signature'], hashlib.sha256(json.dumps(old_identity,sort_keys=True).encode()).hexdigest())
            self.assertEqual(song_identity(source,profile,assets,10,1.0), old_identity)
            with self.assertRaises(ValueError):
                song_plan(source, root/'legacy', profile, assets, 30, 3)
            newer, song = song_plan(source, root/'longer', profile, assets, 30, 3)
            self.assertNotEqual(newer['execution_signature'], legacy['execution_signature'])
            self.assertEqual(song['chunks'][0]['frames'], 30*RATE)
            stitch_song(song, [root/'longer/prepared/chunk_000.wav'], root/'short.wav')
            self.assertEqual(len(read_pcm(root/'short.wav')), 2*2*RATE)

    def test_cli_accepts_30s_plan_and_rejects_invalid_overlap_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root/'source.wav', 40)
            command = [sys.executable, str(ROOT/'scripts/run_song.py'), '--source',str(root/'source.wav'),
                       '--out',str(root/'run'), '--chunk-seconds','30','--overlap-seconds','3']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan = json.loads((root/'run/plan.json').read_text())
            self.assertEqual(plan['status'],'planned')
            self.assertEqual([job['duration'] for job in plan['jobs']], [30,30])
            self.assertFalse(list((root/'run').glob('**/stem.wav')))
            for overlap in (0, 30, float('nan'), float('inf'), 1e-9):
                with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                    prepare_song(root/'source.wav',root/'invalid',30,overlap)
                self.assertFalse((root/'invalid').exists())

    def test_notebook_uses_new_namespace_and_explicit_30_second_cli(self):
        book = json.loads((ROOT/'notebooks/06_mac_full_song_30s.ipynb').read_text())
        source = ''.join(''.join(cell['source']) for cell in book['cells'])
        self.assertIn('full_song30_',source)
        self.assertIn('CHUNK_SECONDS = 30',source)
        self.assertIn('--chunk-seconds',source)
        self.assertIn('DRY_RUN = False',source)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),'song-30s-cell','exec')


if __name__=='__main__': unittest.main()

"""Cached audio diagnostics must preserve existing results and avoid model inference."""
import array
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, write_pcm
from provenance import sha256
from seam_diagnostics import inspect_run, signal_stats
from song_audio import chunk_starts, stitch_song


class SeamDiagnosticTests(unittest.TestCase):
    def test_clipping_measurement_is_signal_data_not_an_automatic_quality_claim(self):
        stats = signal_stats(array.array('h', [32767, -32768, 0, 0]))
        self.assertEqual(stats['full_scale_fraction'], 0.5)
        self.assertEqual(signal_stats(array.array('h', [0, 0]))['rms_dbfs'], None)

    def test_cached_previews_and_unblended_sample_comparison_do_not_change_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            prepared = run / 'prepared'
            prepared.mkdir()
            frames = 30*RATE
            starts = chunk_starts(frames)
            chunks=[]
            stems=[]
            for index, start in enumerate(starts):
                job = run/f'{index:03d}_A_bass_seed0'
                job.mkdir()
                stem = job/'stem.wav'
                write_pcm(stem, array.array('h',[1000+index*100,-1000-index*100])*(10*RATE))
                chunks.append({'index': index, 'start_frame': start, 'valid_frames':10*RATE, 'frames':10*RATE})
                stems.append(stem)
                (job/'result.json').write_text(json.dumps({'status':'generated','stem_file':'stem.wav','stem_sha256':sha256(stem)}))
            song={'sample_rate':RATE,'total_frames':frames,'chunks':chunks}
            (prepared/'song.json').write_text(json.dumps(song))
            write_pcm(prepared/'source.wav', array.array('h',[500,-500])*frames)
            bass=run/'bass_full.wav'
            stitch_song(song,stems,bass)
            (run/'experiment.json').write_text(json.dumps({'stem':str(bass),'stem_sha256':sha256(bass)}))
            before={p:sha256(p) for p in stems+[bass,prepared/'source.wav']}
            output, report=inspect_run(run)
            self.assertTrue((output/'report.json').is_file())
            self.assertEqual(len(report['chunks']),3)
            chunk=report['chunks'][1]
            self.assertEqual(chunk['unblended_region']['song_start_seconds'],10)
            self.assertEqual(chunk['unblended_region']['song_end_seconds'],18)
            self.assertTrue(chunk['unblended_region']['matches_cached_chunk_exactly'])
            for file,digest in before.items():
                self.assertEqual(sha256(file),digest)
            (run/'002_A_bass_seed0/result.json').write_text(json.dumps({'status':'failed'}))
            _, partial = inspect_run(run)
            self.assertEqual(len(partial['chunks']), 2)


if __name__=='__main__':unittest.main()

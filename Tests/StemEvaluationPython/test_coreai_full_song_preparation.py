"""Full-song geometry and frontend invariants; no neural model tests."""
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_coreai_full_song as full_song


class ScheduleTests(unittest.TestCase):
    def test_windows_cover_exact_duration_and_keep_fixed_model_geometry(self):
        for frames in [1,160000,960000,992000,1440007,1920000,5760000]:
            plan=full_song.schedule(frames);windows=plan['windows']
            self.assertEqual(windows[0]['sample_start48'],0)
            self.assertEqual(windows[-1]['sample_end48'],plan['target_samples48'])
            for i,w in enumerate(windows):
                self.assertEqual(w['latent_end']-w['latent_start'],750)
                self.assertEqual(w['sample_end48']-w['sample_start48']+w['right_padding_samples48'],1440000)
                if i:
                    self.assertGreater(w['latent_start'],windows[i-1]['latent_start'])
                    self.assertGreaterEqual(w['overlap_with_previous_frames'],200)
                    self.assertLessEqual(w['sample_start48'],windows[i-1]['sample_end48'])
            self.assertEqual(plan['proposed_calls']['generation_runs'],len(windows)*3)
            self.assertFalse(plan['ready_for_native_execution'])
            self.assertIsNone(plan['latency_estimate'])

    def test_invalid_frame_counts_rejected(self):
        for count in [0,-1,True,1.25,None]:
            with self.assertRaises(ValueError):full_song.schedule(count)

    def test_plan_mode_needs_no_site_packages_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'no-output'
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/prepare_coreai_full_song.py'),
                '--frames32','5760000','--out',str(out)],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'],'PLAN ONLY; no model execution')
            self.assertFalse(out.exists())


@unittest.skipUnless(shutil.which('ffmpeg') and importlib.util.find_spec('numpy'),'Optional FFmpeg/NumPy frontend')
class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();(self.root/'inputs').mkdir()
        self.source=self.root/'inputs/source.wav';self.reference=self.root/'reference.wav';self.out=self.root/'prepared'
        self.write_wave(self.source,1000,32000)
        self.write_wave(self.reference,700,3200)

    def write_wave(self,path,amplitude,frames):
        with wave.open(str(path),'wb') as audio:
            audio.setparams((2,2,32000,0,'NONE','not compressed'))
            audio.writeframes(struct.pack('<hh',amplitude,amplitude)*frames)

    def test_source_gain_applied_once_and_reference_prefix_is_exact(self):
        before=full_song.sha256(self.source)
        report=full_song.prepare(self.source,self.out,self.reference)
        self.assertEqual(report['baseline_prefix_comparison'],'PASS')
        self.assertEqual(report['source32_frames'],32000)
        self.assertEqual(report['model_calls'],0)
        self.assertFalse(report['ready_for_native_execution'])
        with wave.open(str(self.out/'source32.wav'),'rb') as result:
            self.assertEqual(struct.unpack('<hh',result.readframes(1)),(700,700))
        self.assertEqual(full_song.sha256(self.source),before)
        with self.assertRaises(FileExistsError):full_song.prepare(self.source,self.out,self.reference)

    def test_prefix_mismatch_preserves_failure_and_blocks_plan(self):
        self.write_wave(self.reference,800,3200)
        before=full_song.sha256(self.reference)
        with self.assertRaisesRegex(ValueError,'prefix differs'):
            full_song.prepare(self.source,self.out,self.reference)
        report=json.loads((self.out/'report.json').read_text())
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['baseline_prefix_comparison'],'NOT PASSED')
        self.assertFalse((self.out/'plan.json').exists())
        self.assertEqual(full_song.sha256(self.reference),before)


if __name__=='__main__':unittest.main()

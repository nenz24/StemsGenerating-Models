"""Measure real float WAV overshoot and produce safe previews without model inference."""
import array
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
import audit_audio
from audit_audio import audit_run, sample_stats
from audio import RATE, canonicalize, read_pcm, write_pcm
from provenance import sha256


def float_wav(path, amplitude):
    rate=48000
    samples=array.array('f')
    for frame in range(rate):
        value=amplitude*math.sin(2*math.pi*110*frame/rate)
        samples.extend((value,value))
    if sys.byteorder!='little': samples.byteswap()
    data=samples.tobytes()
    fmt=struct.pack('<HHIIHH',3,2,rate,rate*8,8,32)
    payload=b'WAVEfmt '+struct.pack('<I',len(fmt))+fmt+b'data'+struct.pack('<I',len(data))+data
    path.write_bytes(b'RIFF'+struct.pack('<I',len(payload))+payload)


class AudioAuditTests(unittest.TestCase):
    def make_replay(self, run):
        run.mkdir(parents=True)
        jobs=[]
        for position,index,amplitude in ((0,1,2.0),(1,0,0.4)):
            source_dir=run/f'cached_{index:03d}'
            source_dir.mkdir()
            source=source_dir/'source.wav'
            write_pcm(source,array.array('h',[1000,-1000])*RATE)
            folder=run/f'{position:03d}_A_bass_seed0'
            folder.mkdir()
            raw=folder/'model.wav'
            float_wav(raw,amplitude)
            canonicalize(raw,folder/'stem.wav',duration=1,gain=1)
            job={'condition':'A','instrument':'bass','seed':0,'duration':1,'source':str(source),
                 'source_sha256':sha256(source),'original_chunk_index':index,'start_frame':index*27*RATE}
            result=dict(job,status='generated',execution_signature=None,stem_file='stem.wav',
                        stem_sha256=sha256(folder/'stem.wav'),settings={'fixture_only':True})
            (folder/'result.json').write_text(json.dumps(result))
            jobs.append(job)
        (run/'comparison.json').write_text(json.dumps({'status':'generated','jobs':jobs}))
        return run

    def test_float_overshoot_measured_before_pcm16_and_preview_only_attenuates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_replay(root/'reversed')
            before={str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}
            report=audit_run(run,root/'audit')
            raw_b=report['clips'][0]['raw_files'][0]
            self.assertAlmostEqual(raw_b['native']['finite_peak'],2,places=4)
            self.assertGreater(raw_b['native']['outside_unit_range_fraction'],0.5)
            self.assertGreater(raw_b['float_before_pcm16']['outside_unit_range_fraction'],0.5)
            self.assertGreater(report['clips'][0]['stored_stem']['near_full_scale_fraction'],0.5)
            self.assertLess(raw_b['attenuation_gain'],0.36)
            preview=raw_b['preview_stats']
            self.assertEqual(preview['outside_unit_range_samples'],0)
            self.assertLess(preview['finite_peak'],0.71)
            self.assertGreater(preview['finite_peak'],0.70)
            self.assertEqual(len(read_pcm(raw_b['preview'])),2*RATE)
            raw_a=report['clips'][1]['raw_files'][0]
            self.assertEqual(raw_a['attenuation_gain'],1)
            self.assertEqual(raw_a['native']['outside_unit_range_samples'],0)
            self.assertEqual(before,{str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError): audit_run(run,root/'audit')

    def test_nonfinite_samples_remain_explicit_and_json_serializable(self):
        stats=sample_stats(array.array('f',[float('nan'),float('inf'),-float('inf'),0.25,-0.25]))
        self.assertEqual(stats['nonfinite_samples'],3)
        self.assertEqual(stats['finite_peak'],0.25)
        self.assertEqual(stats['finite_rms'],0.25)
        json.dumps(stats,allow_nan=False)
        self.assertEqual(sample_stats(array.array('f',[float('nan')]))['finite_rms'],None)

    def test_real_nan_raw_is_reported_without_creating_repaired_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_replay(root/'reversed')
            float_wav(run/'000_A_bass_seed0/model.wav',float('nan'))
            report=audit_run(run,root/'audit')
            raw=report['clips'][0]['raw_files'][0]
            self.assertGreater(raw['native']['nonfinite_samples'],0)
            self.assertIsNone(raw['preview'])
            self.assertIn('Nonfinite',raw['preview_skipped'])
            self.assertFalse((root/'audit/chunk_001_raw_0_headroom.wav').exists())
            json.dumps(report,allow_nan=False)

    def test_changed_cached_source_or_stem_rejected_before_output(self):
        for changed in ('cached_001/source.wav','000_A_bass_seed0/stem.wav'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                run=self.make_replay(root/'reversed')
                with (run/changed).open('ab') as stream: stream.write(b'changed')
                with self.assertRaises(ValueError): audit_run(run,root/'audit')
                self.assertFalse((root/'audit').exists())

    def test_cli_uses_cached_audio_without_runtime_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_replay(root/'reversed')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/audit_audio.py'),'--run',str(run),
                                   '--out',str(root/'audit')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Tanpa inferensi/unduhan',result.stdout)
            self.assertIn('peak=2',result.stdout)
            self.assertTrue((root/'audit/report.json').is_file())
            self.assertFalse((run/'runtime').exists())

    def test_automatic_selection_uses_latest_completed_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_replay(root/'runs/sequence_diagnostics_fixture/reversed')
            failed=root/'runs/sequence_diagnostics_failed/reversed'
            failed.mkdir(parents=True)
            (failed/'comparison.json').write_text(json.dumps({'status':'failed'}))
            with patch.object(audit_audio,'ROOT',root):
                self.assertEqual(audit_audio.latest_replay(),run)


if __name__=='__main__': unittest.main()

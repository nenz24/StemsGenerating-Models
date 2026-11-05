"""Listening copies preserve source PCM; synthetic audio is not native/model validation."""
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from coreai_listening_audio import read_planar,write_float_wav,headroom_gain
from coreai_preview_window import PreviewWindow
from export_coreai_listening_references import export_references


def samples(path):
    data=path.read_bytes()
    assert data[:4]==b'RIFF' and data[8:16]==b'WAVEfmt '
    assert struct.unpack_from('<I',data,4)[0]==len(data)-8
    assert struct.unpack_from('<HHIIHH',data,20)==(3,2,48000,384000,8,32)
    assert data[38:42]==b'fact' and data[50:54]==b'data'
    frames=struct.unpack_from('<I',data,46)[0]
    assert struct.unpack_from('<I',data,54)[0]==8*frames
    return struct.unpack_from('<'+'f'*(2*frames),data,58)


class ListeningAudioTests(unittest.TestCase):
    def test_raw_float_wav_preserves_over_unity_and_stereo_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'raw.wav'
            write_float_wav(out,[1.375,-.5,.25,-1.125],2)
            self.assertEqual(samples(out),(1.375,.25,-.5,-1.125))
            before=out.read_bytes()
            with self.assertRaises(FileExistsError):write_float_wav(out,[0]*4,2)
            self.assertEqual(out.read_bytes(),before)

    def test_attenuation_only_keeps_silence_and_existing_headroom(self):
        for peak in [0,.1,.5]:self.assertEqual(headroom_gain(peak),1)
        gain=headroom_gain(1.375)
        self.assertAlmostEqual(gain*1.375,10**(-3/20))
        for peak in [-1,math.inf,math.nan]:
            with self.assertRaises(ValueError):headroom_gain(peak)

    def test_reject_invalid_pcm_and_gain_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'invalid.wav'
            for values,frames,gain in [([0],1,1),([0,math.nan],1,1),([0,math.inf],1,1),
                                       ([0,0],1,1.01),([0,0],1,0),([],0,1)]:
                with self.assertRaises(ValueError):write_float_wav(out,values,frames,gain)
                self.assertFalse(out.exists())
            src=Path(tmp)/'audio.f32le';src.write_bytes(struct.pack('<ff',0,math.inf))
            with self.assertRaises(ValueError):read_planar(src,1)
            src.write_bytes(bytes(4))
            with self.assertRaises(ValueError):read_planar(src,1)

    def fixture(self,root):
        # Verifiers below are mocked admission boundaries, not fake executable assets.
        w=PreviewWindow(256,'drums')
        for name in [w.source_bundle,w.preview_bundle]:(root/name).mkdir(parents=True)
        for path,left,right in [(root/w.source_bundle/'audio.f32le',.125,-.25),
                                (root/w.preview_bundle/'audio-fp32-reference.f32le',1.375,-.5),
                                (root/w.preview_bundle/'audio-reference.f32le',.5,-.25)]:
            data=array.array('f',[left]*w.samples+[right]*w.samples)
            if sys.byteorder!='little':data.byteswap()
            path.write_bytes(data.tobytes())
        (root/w.preview_bundle/'reference.json').write_text('{"nonexecutable_test_fixture":true}')
        return w

    def boundaries(self):
        return (patch('export_coreai_source_context.verify_bundle',return_value={'artifact_sha256':{'audio.f32le':'fixture-source-hash'}}),
                patch('export_coreai_bass_conditioning.verify_bundle'),
                patch('export_coreai_bass_preview.verify_bundle',return_value={'host_export_status':'bounded_diagnostic'}))

    def test_export_common_gain_leaves_inputs_and_selector_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidates=root/'models';w=self.fixture(candidates);out=root/'listening'
            selector=candidates/'active-preview.json';selector.write_text('original selector')
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates.rglob('*') if p.is_file()}
            a,b,c=self.boundaries()
            with a as source,b as condition,c as preview:
                report=export_references(candidates,'drums',out,True)
                source.assert_called_once_with(candidates/w.source_bundle,w)
                condition.assert_called_once_with(candidates/w.condition_bundle,w)
                preview.assert_called_once_with(candidates/w.preview_bundle,candidates/w.source_bundle,candidates/w.condition_bundle,w,True)
            self.assertEqual(report['host_export_status'],'bounded_diagnostic')
            self.assertEqual(report['inference_calls'],0)
            self.assertEqual({p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before},before)
            self.assertEqual(len(list(out.glob('*.wav'))),5)
            self.assertEqual(samples(out/'drums-original-fp32-raw.wav')[:2],(1.375,-.5))
            gold=samples(out/'drums-original-fp32-headroom.wav')[0]
            host=samples(out/'drums-partition-host-headroom.wav')[0]
            self.assertAlmostEqual(gold,10**(-3/20),places=6)
            self.assertAlmostEqual(host/gold,.5/1.375,places=6)
            self.assertEqual(json.loads((out/'listening-report.json').read_text()),report)

    def test_admission_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidates=root/'models';self.fixture(candidates);out=root/'listening'
            a,b,_=self.boundaries()
            with a,b,patch('export_coreai_bass_preview.verify_bundle',side_effect=ValueError('strict APG gate failed')):
                with self.assertRaisesRegex(ValueError,'APG gate'):export_references(candidates,'drums',out)
            self.assertFalse(out.exists())

    def test_existing_destination_or_bundle_child_rejected_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'listening';out.mkdir();saved=out/'keep';saved.write_text('keep')
            with patch('export_coreai_source_context.verify_bundle') as verify:
                with self.assertRaises(FileExistsError):export_references(root/'models','drums',out)
                self.assertEqual(saved.read_text(),'keep')
                with self.assertRaises(ValueError):export_references(root/'models','drums',root/'models/new')
                verify.assert_not_called()


if __name__=='__main__':unittest.main()

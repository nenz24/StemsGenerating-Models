"""Two separately exported guitar performances share piano timing and style bytes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'tools/stem_eval')]
import guitar_layers
import guitar_probe
import ace_context_chunks as chunks
import test_guitar_probe as fixtures
from provenance import sha256


class GuitarLayersTests(unittest.TestCase):
    def test_two_time_aligned_layers_reference_and_mix_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'piano.wav';reference=base/'style.wav';out=base/'layers'
            rate=32000;t=np.arange(31*rate)/rate
            piano=np.repeat((.02*np.sin(2*np.pi*220*t))[:,None],2,axis=1)
            sf.write(source,piano,rate,subtype='PCM_16')
            sf.write(reference,piano[:30*rate],rate,subtype='PCM_16')
            source_hash=sha256(source);ref_hash=sha256(reference);calls=[]
            fixture=fixtures.GuitarProbeTests()
            def worker(command,**kwargs):
                variant=command[command.index('--variant')+1];folder=Path(command[command.index('--out')+1])
                frozen=Path(command[command.index('--reference-audio')+1]);calls.append((variant,sha256(frozen)))
                fixture.fake_worker(folder,variant=variant,reference_audio=frozen)
                job=folder/'generation/000_A_guitar_seed0';frequency=330 if variant=='strum-layer' else 660
                tone=np.repeat((.1*np.sin(2*np.pi*frequency*t[:30*rate]))[:,None],2,axis=1)
                sf.write(job/'stem.wav',tone,rate,subtype='PCM_16')
                p=job/'result.json';r=json.loads(p.read_text());r['stem_sha256']=sha256(job/'stem.wav');p.write_text(json.dumps(r))
            with patch.object(guitar_probe.single,'model_python',return_value=Path(sys.executable)),patch('stream_process.run_logged',side_effect=worker):
                report=guitar_layers.generate(ROOT,source,out,reference,execute=True)
            self.assertEqual(calls,[('strum-layer',ref_hash),('pick-layer',ref_hash)])
            self.assertEqual(report['status'],'generated')
            self.assertEqual(report['export_frames'],30*rate)
            self.assertEqual(report['musical_quality'],'unreviewed')
            a=sf.read(out/'strum/guitar.wav',dtype='float32',always_2d=True)[0]
            b=sf.read(out/'pick/guitar.wav',dtype='float32',always_2d=True)[0]
            combined,sr=sf.read(out/'guitars_layered.wav',dtype='float32',always_2d=True)
            self.assertEqual(sr,rate);self.assertEqual(combined.shape,(30*rate,2))
            np.testing.assert_allclose(combined,(.45*a+.45*b)*report['global_headroom_gain'],atol=2e-7)
            spectrum=np.abs(np.fft.rfft(combined[:,0]));freq=np.fft.rfftfreq(len(combined),1/rate)
            self.assertGreater(spectrum[np.argmin(abs(freq-330))],100)
            self.assertGreater(spectrum[np.argmin(abs(freq-660))],100)
            mix=sf.read(out/'piano_guitars_mix.wav',dtype='float32',always_2d=True)[0]
            prepared=sf.read(out/'strum/prepared/original.wav',dtype='float32',always_2d=True)[0]
            np.testing.assert_allclose(mix,.5*(prepared+.45*a+.45*b)*report['global_headroom_gain'],atol=2e-7)
            with zipfile.ZipFile(out/'guitar_layers_feedback.zip') as z:
                self.assertIsNone(z.testzip());self.assertIn('guitars_layered.mp3',z.namelist())
                self.assertNotIn('strum/guitar.wav',z.namelist())
            with zipfile.ZipFile(out/'guitar_layers_result.zip') as z:
                self.assertIn('strum/guitar.wav',z.namelist());self.assertIn('pick/guitar.wav',z.namelist())
                self.assertEqual(z.read('style_reference.wav'),reference.read_bytes())
            self.assertEqual(sha256(source),source_hash);self.assertEqual(sha256(reference),ref_hash)

    def test_layer_reference_parameter_must_match_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'piano.wav';reference=base/'style.wav'
            sf.write(source,np.full((31*32000,2),.02),32000,subtype='PCM_16')
            sf.write(reference,np.full((30*32000,2),.02),32000,subtype='PCM_16')
            fixture=fixtures.GuitarProbeTests()
            def worker(command,**kwargs):
                variant=command[command.index('--variant')+1];folder=Path(command[command.index('--out')+1])
                fixture.fake_worker(folder,variant=variant)
            with patch.object(guitar_probe.single,'model_python',return_value=Path(sys.executable)),patch('stream_process.run_logged',side_effect=worker):
                with self.assertRaisesRegex(ValueError,'Actual guitar settings differ'):
                    guitar_layers.generate(ROOT,source,base/'layers',reference,execute=True)
            report=json.loads((base/'layers/layers.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertFalse((base/'layers/guitar_layers_result.zip').exists())

    def test_full_song_two_referenced_plans_resume_and_stay_synchronized(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'piano.wav';reference=base/'style.wav';out=base/'full'
            frames=7786580
            sf.write(source,np.full((frames,2),.02,dtype=np.float32),32000,subtype='PCM_16')
            sf.write(reference,np.full((30*32000,2),.03,dtype=np.float32),32000,subtype='PCM_16')
            calls=[]
            def generate(root,original,folder,*,start,duration,execute,variant,reference_audio):
                calls.append((variant,start));folder.mkdir()
                if len(calls)==3:raise RuntimeError('interrupted guitar layer')
                local=folder/'reference.wav';local.write_bytes(Path(reference_audio).read_bytes())
                sf.write(folder/'guitar.wav',np.full((round(duration*32000),2),.03 if variant=='strum-layer' else .05,dtype=np.float32),32000,subtype='PCM_16')
                preset=guitar_probe.PRESETS[variant]
                chunks.write_json(folder/'guitar_probe.json',{'status':'generated','instrument':'guitar','preset':preset,
                    'stem_sha256':sha256(folder/'guitar.wav'),'reference_audio':{'path':str(local),'sha256':sha256(local)},
                    'input':{'original_sha256':sha256(original),'start':start,'requested_duration':duration,'source_gain':.7,'model_duration':int(np.ceil(duration))},
                    'settings':{'params':{'instruction':'Generate the guitar track based on the audio context:',
                        'caption':preset['caption'],'seed':0,'guidance_scale':7,'inference_steps':50,
                        'duration':int(np.ceil(duration)),'task_type':'lego','reference_audio':str(local)}}})
            with patch.object(guitar_probe.single,'model_python',return_value=Path(sys.executable)),patch.object(guitar_probe,'generate',side_effect=generate):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    guitar_layers.generate_full(ROOT,source,out,reference,execute=True)
                report=guitar_layers.generate_full(ROOT,source,out,reference,execute=True)
                self.assertEqual(len(calls),7)
                again=guitar_layers.generate_full(ROOT,source,out,reference,execute=True)
                self.assertEqual(len(calls),7);self.assertEqual(again,report)
            self.assertEqual(report['export_frames'],frames)
            self.assertEqual(report['model_calls_total_if_uncached'],6)
            plans=[json.loads((out/role/'plan.json').read_text()) for role in ('strum','pick')]
            self.assertEqual(plans[0]['chunks'],plans[1]['chunks'])
            self.assertEqual({p['shared_reference']['sha256'] for p in plans},{sha256(reference)})
            self.assertEqual({p['guitar_variant'] for p in plans},{'strum-layer','pick-layer'})
            mixed=out/report['mix_export_folder']
            self.assertEqual(sf.info(mixed/'guitars_layered.wav').frames,frames)
            self.assertEqual(sf.info(mixed/'piano_guitars_mix.wav').frames,frames)
            with zipfile.ZipFile(out/'guitar_layers_full_feedback.zip') as z:
                self.assertIsNone(z.testzip());self.assertIn('guitars_layered.mp3',z.namelist())
                self.assertIn('strum/stitched/chunk_reports/000.json',z.namelist())
            with zipfile.ZipFile(out/'guitar_layers_full_result.zip') as z:
                self.assertIn('strum/stitched/guitar_full.wav',z.namelist())
                self.assertIn('pick/stitched/guitar_full.wav',z.namelist())
            source_hash=sha256(source);reference.write_bytes(reference.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'input/reference differs'):
                guitar_layers.generate_full(ROOT,source,out,reference)
            self.assertEqual(sha256(source),source_hash)


if __name__=='__main__':unittest.main()

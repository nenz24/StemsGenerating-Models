"""Same-latent decoder capture and bounded CPU reference; no real ML inference."""
from contextlib import nullcontext
import array
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
from decoder_capture import install_capture
from vae_reference import decode_tiled
from provenance import sha256
import test_sequence_diagnostics
from compare_decoders import prepare, decode_cpu
from mac_workflow import run_command


def write_float(path, data, rate, **kwargs):
    values=np.ascontiguousarray(data,dtype='<f4')
    channels=values.shape[1]
    fmt=struct.pack('<HHIIHH',3,channels,rate,rate*channels*4,channels*4,32)
    payload=b'WAVEfmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',values.nbytes)+values.tobytes()
    Path(path).write_bytes(b'RIFF'+struct.pack('<I',len(payload))+payload)


class Tensor:
    def __init__(self, values): self.values=np.asarray(values,dtype=np.float32)
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self
    def numpy(self): return self.values


class DecoderComparisonTests(unittest.TestCase):
    def test_tiled_decode_preserves_all_samples_across_context_and_tail(self):
        latents=np.arange(2*513,dtype=np.float32).reshape(1,2,513)/10000
        calls=[]
        def decoder(window):
            calls.append(window.shape[-1])
            return np.repeat(window,4,axis=-1)
        actual=decode_tiled(latents,decoder)
        np.testing.assert_array_equal(actual,np.repeat(latents,4,axis=-1))
        self.assertTrue(all(size<=128 for size in calls))
        self.assertGreater(len(calls),2)

    def test_tiling_rejects_nonfinite_and_changed_sample_ratio(self):
        with self.assertRaises(ValueError): decode_tiled(np.full((1,2,5),np.nan),lambda x:x)
        with self.assertRaises(ValueError): decode_tiled(np.zeros((1,2,5)),lambda x:np.full((1,2,5),np.inf))
        calls=[]
        def changed(window):
            calls.append(1)
            return np.repeat(window,2 if len(calls)==1 else 3,axis=-1)
        with self.assertRaises(ValueError): decode_tiled(np.zeros((1,2,300)),changed)

    def capture_fixture(self, directory, fallback=False, frames=60, hop=24000):
        values=np.arange(frames*2,dtype=np.float32).reshape(1,frames,2)/max(1000,frames*4)
        waveform=Tensor(np.repeat(values.transpose(0,2,1),hop,axis=-1))
        class Vae:
            def parameters(self): return iter([types.SimpleNamespace(device='mps',dtype='torch.float16')])
            def decode(self,*args): return waveform
        class Handler:
            device='mps'; dtype='torch.float32'; use_mlx_vae=True; mlx_vae=object()
            _mlx_vae_dtype='mlx.float32'; sample_rate=48000
            def __init__(self): self.vae=Vae()
            def _mlx_vae_decode(self,latents):
                if fallback: raise RuntimeError('fixture mlx fallback')
                return waveform
            def _decode_generate_music_pred_latents(self,pred_latents,**kwargs):
                try: decoded=self._mlx_vae_decode(pred_latents)
                except RuntimeError: decoded=self.vae.decode(pred_latents)
                return (decoded,None,{'fixture':True})
        handler=Handler()
        with patch.dict(sys.modules,{'soundfile':types.SimpleNamespace(write=write_float)}):
            install_capture(handler,directory)
            result=handler._decode_generate_music_pred_latents(pred_latents=Tensor(values))
        return values,waveform,result

    def test_capture_preserves_latents_result_and_records_actual_mlx_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'capture'
            values,waveform,result=self.capture_fixture(output)
            np.testing.assert_array_equal(np.load(output/'latents.npy',allow_pickle=False),values)
            self.assertIs(result[0],waveform)
            report=json.loads((output/'capture.json').read_text())
            self.assertEqual(report['status'],'captured')
            self.assertEqual(report['latent_sha256'],sha256(output/'latents.npy'))
            self.assertEqual(report['calls'],[{'backend':'mlx_vae','status':'returned'}])

    def test_capture_records_mlx_failure_and_actual_torch_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'capture'
            self.capture_fixture(output,fallback=True)
            calls=json.loads((output/'capture.json').read_text())['calls']
            self.assertEqual(calls[0]['status'],'failed')
            self.assertEqual(calls[1],{'backend':'pytorch_vae','status':'returned','device':'mps','dtype':'torch.float16'})

    def test_snapshot_and_capture_plan_keep_B_input_and_only_one_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=test_sequence_diagnostics.SequenceDiagnosticsTests().make_run(root)
            # prepare() consumes the completed reversed replay format.
            old=json.loads((run/'experiment.json').read_text())
            from sequence_diagnostics import prepare_sequence
            reverse=root/'reversed'
            comparison=prepare_sequence(run,reverse)
            for position,index in enumerate((1,0)):
                folder=reverse/f'{position:03d}_A_bass_seed0'
                folder.mkdir()
                source_result=json.loads((run/f'{index:03d}_A_bass_seed0/result.json').read_text())
                (folder/'stem.wav').write_bytes((run/f'{index:03d}_A_bass_seed0'/source_result['stem_file']).read_bytes())
                (folder/'result.json').write_text(json.dumps(dict(comparison['jobs'][position],status='generated',
                    stem_file='stem.wav',stem_sha256=sha256(folder/'stem.wav'),execution_signature=None)))
            comparison['status']='generated'
            (reverse/'comparison.json').write_text(json.dumps(comparison))
            output=root/'comparison'
            report=prepare(reverse,output)
            self.assertEqual(sha256(output/'prepared/source.wav'),old['jobs'][1]['source_sha256'])
            command=run_command(ROOT,report['profile'],output/'prepared',output/'plan')
            result=subprocess.run(command+['--max-job-seconds','30','--capture-decoder',str(output/'capture')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads((output/'plan/experiment.json').read_text())
            self.assertEqual(len(plan['jobs']),1)
            self.assertEqual(plan['jobs'][0]['seed'],0)
            self.assertFalse((output/'capture').exists())
            denied=subprocess.run(command+['--seeds','0','1','--max-job-seconds','30','--capture-decoder',str(output/'bad_capture')],capture_output=True,text=True)
            self.assertNotEqual(denied.returncode,0)
            self.assertIn('exactly one ACE job',denied.stderr)
            dry=subprocess.run([sys.executable,str(ROOT/'scripts/compare_decoders.py'),
                '--run',str(reverse),'--out',str(root/'dry')],capture_output=True,text=True)
            self.assertEqual(dry.returncode,0,dry.stderr)
            self.assertIn('PLAN ONLY',dry.stdout)
            self.assertFalse((root/'dry/default_result').exists())

    def cpu_dependencies(self,hop=24000):
        import compare_decoders
        class Vae:
            config=types.SimpleNamespace(decoder_input_channels=2)
            def to(self,**kwargs):
                self.placement=kwargs
                return self
            def eval(self): return self
            def decode(self,tensor):
                return types.SimpleNamespace(sample=Tensor(np.repeat(tensor.values,hop,axis=-1)))
        vae=Vae()
        loader=MagicMock(return_value=vae)
        modules={'torch':types.SimpleNamespace(from_numpy=Tensor,inference_mode=nullcontext,float32='torch.float32',__version__='fixture'),
                 'soundfile':types.SimpleNamespace(write=write_float),
                 'diffusers':types.ModuleType('diffusers'),
                 'diffusers.models':types.SimpleNamespace(AutoencoderOobleck=types.SimpleNamespace(from_pretrained=loader))}
        return vae,loader,modules

    def test_cpu_reference_uses_identical_latents_only_local_vae_and_real_wav_exports(self):
        for frames,hop in ((60,24000),(703,2048)):
            with self.subTest(frames=frames),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                output=root/'comparison';output.mkdir()
                self.capture_fixture(output/'capture',frames=frames,hop=hop)
                (output/'comparison.json').write_text(json.dumps({'assets_sha256':sha256(ROOT/'configs/assets-base.json')}))
                vae,loader,modules=self.cpu_dependencies(hop)
                with patch.dict(os.environ), patch.dict(sys.modules,modules), patch('compare_decoders.check_upstream'), patch('model_assets.verify'):
                    decode_cpu(output)
                self.assertEqual(vae.placement,{'device':'cpu','dtype':'torch.float32'})
                self.assertTrue(loader.call_args.kwargs['local_files_only'])
                self.assertTrue(loader.call_args.args[0].endswith('/checkpoints/vae'))
                report=json.loads((output/'comparison.json').read_text())
                self.assertEqual(report['status'],'decoded')
                self.assertEqual(report['same_latent_sha256'],sha256(output/'capture/latents.npy'))
                self.assertLessEqual(report['listening_duration'],min(30,frames*hop/48000))
                from audio import read_pcm
                self.assertEqual(read_pcm(output/'default_decoder_listen.wav'),read_pcm(output/'cpu_decoder_listen.wav'))

    def test_cpu_reference_rejects_modified_latents_before_loading_vae(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'comparison';output.mkdir()
            self.capture_fixture(output/'capture')
            (output/'comparison.json').write_text(json.dumps({'assets_sha256':sha256(ROOT/'configs/assets-base.json')}))
            with (output/'capture/latents.npy').open('ab') as stream: stream.write(b'changed')
            _,loader,modules=self.cpu_dependencies()
            with patch.dict(os.environ), patch.dict(sys.modules,modules), patch('compare_decoders.check_upstream'), self.assertRaises(ValueError):
                decode_cpu(output)
            loader.assert_not_called()

    def test_real_engine_installs_capture_only_when_explicitly_requested(self):
        import run as engine
        from audio import RATE, write_pcm
        from context_comparison import create_input
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.wav'
            write_pcm(source,array.array('h',[1000,-1000])*RATE)
            prepared=create_input(root/'prepared',source,1,'fixture',copy_exact=True)
            class Model:
                handler=object()
                def generate(self,source,duration,instrument,seed,output):
                    path=output/'raw.wav'
                    path.write_bytes(source.read_bytes())
                    return path,{'fixture_only':True}
            torch=types.ModuleType('torch');torch.inference_mode=nullcontext
            for enabled in (False,True):
                with self.subTest(enabled=enabled):
                    output=root/str(enabled);output.mkdir()
                    capture=output/'capture' if enabled else None
                    settings=types.SimpleNamespace(upstream=root/'upstream',backend='ace',checkpoint=None,
                        ace_model='acestep-v15-base',assets_manifest=ROOT/'configs/assets-base.json',device='mps',
                        offload=True,out=output,prepared=prepared,capture_decoder=capture)
                    model=Model()
                    with patch.dict(os.environ), patch.dict(sys.modules,{'torch':torch}), patch('backends.AceBackend',return_value=model), patch('backends.weight_inventory',return_value=[]), patch('model_assets.process'), patch('model_assets.materialize',return_value=root/'runtime/checkpoints'), patch.object(engine,'check_upstream'), patch.object(engine,'select_device',return_value='mps'), patch.object(engine,'synchronize'), patch.object(engine,'reset_memory_stats'), patch.object(engine,'memory_stats',return_value={}), patch('decoder_capture.install_capture') as install:
                        experiment=engine.plan(prepared,'ace','bass',[0])
                        engine.execute(settings,experiment)
                        self.assertEqual(experiment['status'],'generated')
                        if enabled: install.assert_called_once_with(model.handler,capture)
                        else: install.assert_not_called()


if __name__=='__main__': unittest.main()

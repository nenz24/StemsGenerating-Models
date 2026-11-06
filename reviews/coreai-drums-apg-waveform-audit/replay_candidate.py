from pathlib import Path
import gc,hashlib,json,sys,time
import numpy as np
import torch
from safetensors import safe_open
from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
root=Path.cwd();folder=root/'work/drum-waveforms'
sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root/'tools/stem_eval'))
from export_coreai_dit_block import precision_metrics
from coreai_listening_audio import write_float_wav,headroom_gain
from vae_reference import decode_tiled
cfg=json.loads((root/'configs/coreai-vae-config.json').read_text());torch.set_num_threads(2)
record=next(r for r in json.loads((root/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
p=folder/'vae/diffusion_pytorch_model.safetensors';assert p.stat().st_size==record['size']
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==record['digest']
print('Pinned VAE verified; load decoder only',flush=True)
with torch.device('meta'):
 decoder=OobleckDecoder(channels=cfg['decoder_channels'],input_channels=cfg['decoder_input_channels'],audio_channels=cfg['audio_channels'],upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),channel_multiples=cfg['channel_multiples']).eval()
with safe_open(str(p),framework='pt',device='cpu') as w:
 state={k[len('decoder.'):]:w.get_tensor(k) for k in w.keys() if k.startswith('decoder.')}
decoder.load_state_dict(state,strict=True,assign=True);del state
decoder.to(device='cpu',dtype=torch.float32).requires_grad_(False)
result={'checkpoint_sha256':h.hexdigest(),'torch':torch.__version__,'mode':'CPU FP32; decoder only; no encoder or DiT inference; no native code','absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,'sampling_steps':0}
folder=root/'work/drum-apg-waveforms'
z=np.fromfile(folder/'candidate-latent.f32le',dtype='<f4').reshape(1,256,64).transpose(0,2,1).copy()
start=time.perf_counter()
with torch.no_grad(): a=decoder(torch.from_numpy(z)).contiguous()
expected=torch.from_numpy(np.fromfile(folder/'candidate-audio.f32le',dtype='<f4').copy().reshape(1,2,491520))
result.update(decoder_calls=1,encoder_calls=0,seconds=time.perf_counter()-start,uploaded_latent_sha256=hashlib.sha256((folder/'candidate-latent.f32le').read_bytes()).hexdigest(),uploaded_audio_sha256=hashlib.sha256((folder/'candidate-audio.f32le').read_bytes()).hexdigest(),replay_vs_uploaded=precision_metrics(a,expected))
(folder/'candidate-replay-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
assert result['replay_vs_uploaded']['parity_passed']

from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from compare_coreai_drum_source_gain import load_encoder,encode,SOURCE_GAIN
from compare_coreai_drum_lyrics import local_checkpoint
from export_coreai_dit_block import precision_metrics
torch.set_num_threads(2)
p=root/'work/drum-waveforms';out=root/'work/drum-source-gain-v1'
ref=json.loads((p/'reference.json').read_text())
record=next(r for r in json.loads((root/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
checkpoint=local_checkpoint(record,p/'vae/diffusion_pytorch_model.safetensors')
audio=torch.from_numpy(np.fromfile(p/'audio.f32le',dtype='<f4').copy().reshape(1,2,491520))
epsilon=torch.randn((1,64,256),generator=torch.Generator(device='cpu').manual_seed(31))
mask=torch.ones((1,256,64))
for label,t in [('audio.f32le',audio),('chunk_mask.f32le',mask)]:
 assert hashlib.sha256(t.numpy().astype('<f4').tobytes()).hexdigest()==ref['artifact_sha256'][label],label
expected=torch.from_numpy(np.fromfile(p/'source_latents-reference.f32le',dtype='<f4').copy().reshape(1,256,64))
gold_context=torch.cat([expected,mask],dim=-1)
assert hashlib.sha256(gold_context.numpy().astype('<f4').tobytes()).hexdigest()==ref['artifact_sha256']['context_latents-reference.f32le']
report={'checkpoint_sha256':record['digest'],'source_PCM_mask_and_uploaded_gold_context_hashes_verified':True,'epsilon_fixture_sha256':hashlib.sha256(epsilon.numpy().astype('<f4').tobytes()).hexdigest(),'Mac_saved_epsilon_sha256':ref['artifact_sha256']['epsilon.f32le'],'epsilon_bytes_match_Mac':False,'torch':torch.__version__,'mode':'CPU FP32 VAE encoder smoke only; source-gain0.7 candidate; no DiT/native','encoder_calls':0,'decoder_calls':0,'sampling_steps':0,'source_gain':SOURCE_GAIN,'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,'scope':'Actual uploaded PCM with fresh CPU seed31 epsilon reused for both branches. CPU epsilon hash differs from Mac; original epsilon/mean/std arrays absent. This is NOT matched source replay/parity. Mac script requires exact saved epsilon and four existing gold arrays before generation. No full candidate50-step generation or musical acceptance here.'}
encoder=load_encoder(checkpoint)
print('Encode actual uploaded piano using explicit CPU epsilon fixture (not Mac gold replay)',flush=True)
start=time.perf_counter();mean,std,source,context=encode(encoder,audio,epsilon,mask);report['encoder_calls']+=1
report['descriptive_vs_uploaded_Mac_latent_not_a_matched_parity_gate']=precision_metrics(source,expected)
report['control_seconds']=time.perf_counter()-start
print('CPU epsilon fixture encoded; original Mac epsilon bytes remain unavailable here',flush=True)
print('Encode fixed gain0.7 with same epsilon/mask',flush=True)
start=time.perf_counter();mean,std,source,context=encode(encoder,(audio*SOURCE_GAIN).contiguous(),epsilon,mask);report['encoder_calls']+=1
assert torch.equal(context[...,64:],mask)
for label,t in [('candidate-context',context),('candidate-source-latent',source)]:t.numpy().astype('<f4').tofile(out/(label+'.f32le'))
report['candidate']={'seconds':time.perf_counter()-start,'audio_peak':float((audio*SOURCE_GAIN).abs().max()),'latent_peak':float(source.abs().max()),'latent_RMS':float(source.square().mean().sqrt()),'source_context_changed':not torch.equal(context,gold_context),'full_mask_identical':True,'context_sha256':hashlib.sha256(context.numpy().astype('<f4').tobytes()).hexdigest()}
(out/'encoder-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

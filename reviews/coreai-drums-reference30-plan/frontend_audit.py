from pathlib import Path
import json,sys,time,hashlib
import torch
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
import generate_coreai_drum_reference as M
from compare_coreai_drum_lyrics import local_checkpoint
torch.set_num_threads(2)
out=root/'work/drum-reference30-v1/frontend';out.mkdir(exist_ok=False)
source=Path('/workspace/scratch/guitar_layers_preview30_review/piano_input/source.wav')
record=next(x for x in json.loads((root/'configs/assets-base.json').read_text())['files'] if x['filename']=='vae/diffusion_pytorch_model.safetensors')
checkpoint=local_checkpoint(record,root/'work/drum-waveforms/vae/diffusion_pytorch_model.safetensors')
audio,metadata=M.prepare_source(out,prepared_source=source)
start=time.perf_counter();context=M.encode_source(audio,checkpoint,out,print)
report={'schema_version':1,'source_preparation':metadata,'source_original_accepted_WAV_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_encoder_calls':1,'decoder_calls':0,'DiT_steps':0,'Qwen_calls':0,'context_shape':list(context.shape),'finite':bool(torch.isfinite(context).all()),'full_mask_identical':bool(torch.equal(context[...,64:],torch.ones(1,750,64))),'seconds':time.perf_counter()-start,'checkpoint_sha256':record['digest'],'scope':'Actual CPU FP32 frontend encoding only. Fresh CPU epsilon31 saved; no claim of original CUDA posterior/RNG identity, complete reference generation, numerical parity or musical quality.'}
(out.parent/'frontend-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

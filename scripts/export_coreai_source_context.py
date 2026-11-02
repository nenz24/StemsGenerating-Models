#!/usr/bin/env python3
"""Real recording -> sampled VAE source latents + full-range lego context.

Only the audio side of conditioning. No text encoder, ConditionEncoder or bass
generation. Large asset and user-derived tensors stay in an ignored local bundle.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import urllib.request

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from coreai_preview_window import SHORT, PreviewWindow

FRAMES=128
SAMPLES=FRAMES*1920
INPUTS={'audio':[1,2,SAMPLES],'epsilon':[1,64,FRAMES],'chunk_mask':[1,FRAMES,64]}
OUTPUTS={'posterior_mean':[1,64,FRAMES],'posterior_std':[1,64,FRAMES],
         'source_latents':[1,FRAMES,64],'context_latents':[1,FRAMES,128]}


def shapes(window=SHORT):
    return ({'audio':[1,2,window.samples],'epsilon':[1,64,window.frames],'chunk_mask':[1,window.frames,64]},
            {'posterior_mean':[1,64,window.frames],'posterior_std':[1,64,window.frames],
             'source_latents':[1,window.frames,64],'context_latents':[1,window.frames,128]})


def posterior_context(parameters,epsilon,chunk_mask):
    import torch
    mean,scale=parameters.chunk(2,dim=1)
    std=torch.nn.functional.softplus(scale)+1e-4
    source=(mean+std*epsilon).transpose(1,2).contiguous()
    context=torch.cat([source,chunk_mask],dim=-1)
    return mean.contiguous(),std.contiguous(),source,context


def verified_handler_source(name):
    manifest=json.loads((ROOT/'configs/coreai-lego-preprocessing.json').read_text())
    record=next(r for r in manifest['files'] if r['path'].endswith('/'+name))
    checkout=ROOT/'.runtime-cache/ace-conditioning-upstream'/record['path']
    cache=ROOT/'.runtime-cache/coreai-conditioning-source'/name
    path=checkout if checkout.is_file() else cache
    if not path.exists():
        url=f'https://raw.githubusercontent.com/{manifest["repo"]}/{manifest["revision"]}/{record["path"]}'
        with urllib.request.urlopen(url,timeout=60) as response:raw=response.read(record['size']+1)
        if len(raw)!=record['size'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Pinned handler source mismatch')
        cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(raw)
    raw=path.read_bytes()
    if (len(raw)!=record['size'] or hashlib.sha256(raw).hexdigest()!=record['sha256']
            or hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=record['git_blob_sha1']):
        raise ValueError(f'Handler source integrity mismatch: {path}; preserved')
    return path,manifest


def lego_mask(audio,source_latents):
    frames=source_latents.shape[1]; samples=audio.shape[-1]
    """Execute the actual pinned mask helper, no simulated inference/handler."""
    path,_=verified_handler_source('conditioning_masks.py')
    spec=importlib.util.spec_from_file_location('coreai_verified_lego_masks',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    host=module.ConditioningMaskMixin();host.device='cpu';host.sample_rate=48000
    masks,spans,covers,source,repaint=host._build_chunk_masks_and_src_latents(
        1,frames,['Generate the bass track based on the audio context:'],[None],audio,source_latents,
        [0.],[samples/48000],None,chunk_mask_modes=['auto'],task_type='lego')
    import torch
    if covers.any() or not torch.equal(source,source_latents) or not masks.all():
        raise ValueError('Pinned lego source preservation/full mask contract changed')
    expanded=masks.to(torch.float32).unsqueeze(-1).repeat(1,1,64)
    return expanded,{'is_covers':covers.tolist(),'spans':spans,'mask_dtype_before_expand':str(masks.dtype),
                     'source_preservation_exact':True,'full_range_mask':True}


def canonical_audio(path,start_seconds,window=SHORT):
    inputs,_=shapes(window); samples=window.samples
    import numpy as np
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg required; on Mac: brew install ffmpeg')
    # Decode/resample on the host, then share those exact PCM bytes with iPhone.
    # No claim of bitwise MP3/resampler equivalence to torchaudio or AVFoundation.
    result=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-ss',str(start_seconds),'-t',str(samples/48000),
                           '-ar','48000','-ac','2','-f','f32le','pipe:1'],check=True,capture_output=True)
    values=np.frombuffer(result.stdout,dtype='<f4').copy()
    if values.size!=2*samples:raise ValueError(f'Recording has less than {window.seconds} seconds at requested start; no silent padding')
    audio=values.reshape(samples,2).T.copy().reshape(inputs['audio'])
    if not np.isfinite(audio).all() or np.max(np.abs(audio))<=1e-6:raise ValueError('Nonfinite or silent recording segment')
    return audio


def verify_bundle(folder,window=SHORT):
    inputs,outputs=shapes(window)
    ref=json.loads((folder/'reference.json').read_text())
    vae_record=next(r for r in json.loads((ROOT/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    if (ref.get('schema_version')!=1 or ref.get('profile')!=window.source_bundle
            or ref.get('asset_file')!='source-context.aimodel' or ref.get('function_name')!='main'
            or ref.get('input_shapes')!=inputs or ref.get('output_shapes')!=outputs
            or ref.get('scalar_type')!='float32' or ref.get('byte_order')!='little'
            or ref.get('checkpoint_digest')!=vae_record['digest'] or ref.get('checkpoint_full_sha256_verified') is not True
            or ref.get('task_type')!='lego' or ref.get('is_covers') is not False
            or ref.get('posterior_mode')!='sample_with_explicit_epsilon' or ref.get('sample_rate')!=48000
            or ref.get('audio_seconds')!=window.seconds or ref.get('parameter_count')!=84264832
            or ref.get('handler_source_revision')!=json.loads((ROOT/'configs/coreai-lego-preprocessing.json').read_text())['revision']
            or ref.get('absolute_tolerance')!=1e-4 or ref.get('relative_tolerance')!=1e-3 or ref.get('relative_rmse_tolerance')!=1e-3
            or ref.get('host_distribution_sample_parity_exact') is not True
            or ref.get('host_weight_norm_folding_parity_exact') is not True):
        raise ValueError('Unexpected source-context contract')
    recording=ref.get('recording',{})
    if (not isinstance(recording.get('name'),str) or not recording['name']
            or not isinstance(recording.get('sha256'),str) or len(recording['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in recording['sha256'])
            or not isinstance(recording.get('start_seconds'),(int,float))
            or not math.isfinite(recording['start_seconds']) or recording['start_seconds']<0
            or ref.get('lego_mask',{}).get('source_preservation_exact') is not True):
        raise ValueError('Invalid recording/source preservation metadata')
    required={f'{n}.f32le':s for n,s in inputs.items()}
    required.update({f'{n}-reference.f32le':s for n,s in outputs.items()})
    hashes=ref['artifact_sha256']
    native={f'source-context.aimodel/{n}' for n in ['main.hash','main.mlirb','metadata.json']}
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if not (set(required)|native).issubset(hashes) or actual!=set(hashes):raise ValueError('Incomplete source-context bundle')
    for name,digest in hashes.items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or sha256(p)!=digest:raise ValueError(f'Source-context integrity mismatch: {name}')
    for name,shape in required.items():
        if (folder/name).stat().st_size!=math.prod(shape)*4:raise ValueError(f'Incorrect Float32 size: {name}')
        values=[v[0] for v in struct.iter_unpack('<f',(folder/name).read_bytes())]
        if not all(math.isfinite(v) for v in values):raise ValueError(f'Nonfinite source-context data: {name}')
        if name=='chunk_mask.f32le' and not all(v==1 for v in values):raise ValueError('Expected full-range lego mask')
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,choices=[5.12,10.24],default=5.12)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--start-seconds',type=float,default=0.)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    window=PreviewWindow.for_seconds(args.seconds); inputs,outputs=shapes(window)
    args.out=args.out or ROOT/'prototypes/ios27-coreai/ModelCandidates'/window.source_bundle
    if args.verify_only:
        ref=verify_bundle(args.out,window)
        if args.source and (sha256(args.source)!=ref['recording']['sha256'] or args.start_seconds!=ref['recording']['start_seconds']):
            raise ValueError('Existing bundle belongs to another recording/offset; preserve it and use a new --out')
        print('Source-context bundle VERIFIED; existing source/model retained');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    if args.source is None or not args.source.is_file():parser.error('--source must name an existing recording')
    if not math.isfinite(args.start_seconds) or args.start_seconds<0:parser.error('--start-seconds must be finite and nonnegative')
    audio_values=canonical_audio(args.source,args.start_seconds,window)
    manifest=json.loads((ROOT/'configs/assets-base.json').read_text())
    record=next(r for r in manifest['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    cfg_record=next(r for r in manifest['files'] if r['filename']=='vae/config.json')
    raw_cfg=(ROOT/'configs/coreai-vae-config.json').read_bytes()
    if hashlib.sha1(f'blob {len(raw_cfg)}\0'.encode()+raw_cfg).hexdigest()!=cfg_record['digest']:raise ValueError('VAE config mismatch')
    checkpoint=checkpoint_path(record,args.checkpoint)
    import numpy as np
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckEncoder,OobleckDiagonalGaussianDistribution
    from coreai_torch import TorchConverter,get_decomp_table
    torch.set_num_threads(2);cfg=json.loads(raw_cfg)
    with torch.device('meta'):
        encoder=OobleckEncoder(encoder_hidden_size=cfg['encoder_hidden_size'],audio_channels=cfg['audio_channels'],
                               downsampling_ratios=cfg['downsampling_ratios'],channel_multiples=cfg['channel_multiples']).eval()
    with safe_open(str(checkpoint),framework='pt',device='cpu') as weights:
        state={k[len('encoder.'):]:weights.get_tensor(k) for k in weights.keys() if k.startswith('encoder.')}
    encoder.load_state_dict(state,strict=True,assign=True);del state
    encoder.to(device='cpu',dtype=torch.float32).requires_grad_(False)
    audio=torch.from_numpy(audio_values)
    epsilon=torch.randn(inputs['epsilon'],generator=torch.Generator(device='cpu').manual_seed(31),dtype=torch.float32)
    print(f'Verified full VAE checkpoint; encoding actual recording, {window.seconds} s stereo',flush=True)
    with torch.no_grad():
        parameters=encoder(audio)
        distribution=OobleckDiagonalGaussianDistribution(parameters)
        sampled=distribution.sample(generator=torch.Generator(device='cpu').manual_seed(31)).transpose(1,2).contiguous()
        mask,mask_info=lego_mask(audio,sampled)
        expected=posterior_context(parameters,epsilon,mask)
    torch.testing.assert_close(sampled,expected[2],rtol=0,atol=0)
    class SourceContext(torch.nn.Module):
        def __init__(self):super().__init__();self.encoder=encoder
        def forward(self,audio,epsilon,chunk_mask):return posterior_context(self.encoder(audio),epsilon,chunk_mask)
    model=SourceContext().eval()
    for module in encoder.modules():
        if hasattr(module,'weight_g') and hasattr(module,'weight_v'):torch.nn.utils.remove_weight_norm(module)
    with torch.no_grad():folded=model(audio,epsilon,mask)
    for a,b in zip(folded,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    parent=ROOT/'.runtime-cache/coreai-source-context-staging';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='recording-',dir=parent))
    for name,value in zip(inputs,[audio,epsilon,mask]):value.numpy().astype('<f4',copy=False).tofile(stage/f'{name}.f32le')
    for name,value in zip(outputs,expected):value.numpy().astype('<f4',copy=False).tofile(stage/f'{name}-reference.f32le')
    print('Distribution sampling and weight-norm folding parity EXACT; exporting audio-context asset',flush=True)
    exported=torch.export.export(model,(audio,epsilon,mask)).run_decompositions(get_decomp_table())
    program=TorchConverter().add_exported_program(exported,input_names=list(inputs),output_names=list(outputs)).to_coreai()
    program.save_asset(stage/'source-context.aimodel')
    ref={'schema_version':1,'profile':window.source_bundle,'asset_file':'source-context.aimodel','function_name':'main',
         'input_shapes':inputs,'output_shapes':outputs,'scalar_type':'float32','byte_order':'little',
         'sample_rate':48000,'audio_seconds':window.seconds,'task_type':'lego','is_covers':False,
         'posterior_mode':'sample_with_explicit_epsilon','epsilon_seed_host_fixture':31,
         'recording':{'name':args.source.name,'sha256':sha256(args.source),'start_seconds':args.start_seconds,
                      'canonicalization':'Host ffmpeg -> 48 kHz stereo Float32, planar tensor, no peak normalization; same PCM sent to device'},
         'checkpoint_digest':record['digest'],'checkpoint_full_sha256_verified':True,
         'host_distribution_sample_parity_exact':True,'host_weight_norm_folding_parity_exact':True,'lego_mask':mask_info,
         'handler_source_revision':json.loads((ROOT/'configs/coreai-lego-preprocessing.json').read_text())['revision'],
         'parameter_count':sum(p.numel() for p in encoder.parameters()),
         'versions':{n:importlib.metadata.version(n) for n in ['torch','diffusers','coreai-torch','coreai-core']},
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'artifact_sha256':{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()},
         'scope':'Real-recording audio side of lego conditioning only; text/ConditionEncoder/full generation NOT implemented here',
         'device_execution_verified':False}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_bundle(stage,window)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(f'Destination appeared; staging retained: {stage}')
    shutil.move(str(stage),str(args.out));print('Source-context prepared locally:',args.out,flush=True)


if __name__=='__main__':main()

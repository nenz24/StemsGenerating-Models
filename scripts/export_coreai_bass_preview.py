#!/usr/bin/env python3
"""Prepare a fixed 5.12 or 10.24 s piano-conditioned bass pipeline.

Uses duration-matched source and full-conditioning bundles. Exports 14 shape-matched
DiT shards, one APG/Euler update and a full-window decoder; no cloud inference
is used by the iPhone caller. Large/user-derived assets remain local and ignored.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from export_coreai_decoder import ROOT,checkpoint_path,sha256
from coreai_preview_window import SHORT, PreviewWindow
from export_coreai_dit_block import records,verified_source,convert_storage16,precision_metrics
from export_coreai_denoiser import load_full_upstream
from export_coreai_source_context import verify_bundle as verify_source
from export_coreai_bass_conditioning import verify_bundle as verify_condition
from prepare_coreai_denoiser_loop import verify_sampling_source

FRAMES=128
TOKENS=117
STEPS=50
CHECKPOINTS=[1,10,25,50]
INPUTS={'noise':[1,128,64],'condition':[1,117,2048],'context':[1,128,128],'timestep':[1]}
SHARED={'hidden':[2,64,2048],'condition':[2,117,2048],'temb':[2,6,2048],
        'rope_cos':[1,64,128],'rope_sin':[1,64,128],'self_mask':[1,1,64,64],
        'local_mask':[1,1,64,64],'cross_mask':[1,1,64,117]}
APG_INPUTS={'velocity':[2,128,64],'latent':[1,128,64],'momentum':[1,128,64],'dt':[1]}
APG_OUTPUTS={'latent_next':[1,128,64],'momentum_next':[1,128,64],'guided_velocity':[1,128,64]}
SOURCE=ROOT/'prototypes/ios27-coreai/ModelCandidates/source-context'
CONDITION=ROOT/'prototypes/ios27-coreai/ModelCandidates/bass-conditioning'


def stage_plan(window=SHORT):
    h,f,n=window.hidden_tokens,window.frames,window.condition_tokens
    inputs={'noise':[1,f,64],'condition':[1,n,2048],'context':[1,f,128],'timestep':[1]}
    shared={'hidden':[2,h,2048],'condition':[2,n,2048],'temb':[2,6,2048],
            'rope_cos':[1,h,128],'rope_sin':[1,h,128],'self_mask':[1,1,h,h],
            'local_mask':[1,1,h,h],'cross_mask':[1,1,h,n]}
    front_outputs={('condition_projected' if name=='condition' else name):shape for name,shape in shared.items()}
    front_outputs['final_temb']=[2,2048]
    plan=[{'asset_file':'front.aimodel','layer_indices':[],'input_shapes':inputs,'output_shapes':front_outputs,
           'output_bindings':{name:('condition' if name=='condition_projected' else name) for name in front_outputs}}]
    for i in range(0,24,2):
        plan.append({'asset_file':f'blocks-{i:02d}-{i+1:02d}.aimodel','layer_indices':[i,i+1],
                     'input_shapes':shared,'output_shapes':{'hidden_out':[2,h,2048]},'output_bindings':{'hidden_out':'hidden'}})
    plan.append({'asset_file':'tail.aimodel','layer_indices':[],
                 'input_shapes':{'hidden':[2,h,2048],'final_temb':[2,2048]},
                 'output_shapes':{'velocity':[2,f,64]},'output_bindings':{'velocity':'velocity'}})
    return plan


def apg_euler(velocity,latent,momentum,dt):
    """Publisher APG semantics, FP32 projection validated against FP64 gold."""
    import torch
    cond,uncond=velocity.chunk(2,dim=0)
    next_momentum=(cond-uncond)+(-0.75*momentum)
    norm=torch.linalg.vector_norm(next_momentum,ord=2,dim=1,keepdim=True)
    diff=next_momentum*torch.minimum(torch.ones_like(next_momentum),2.5/norm)
    unit=torch.nn.functional.normalize(cond,p=2,dim=1)
    parallel=(diff*unit).sum(dim=1,keepdim=True)*unit
    guided=cond+6.0*(diff-parallel)
    return (latent-guided*dt.reshape(1,1,1)).contiguous(),next_momentum.contiguous(),guided.contiguous()


def block_pair(layers):
    """Shared implementation for generation shards and the precision experiment."""
    import torch
    class Pair(torch.nn.Module):
        def __init__(self):super().__init__();self.layers=torch.nn.ModuleList(layers)
        def forward(self,hidden,condition,temb,rope_cos,rope_sin,self_mask,local_mask,cross_mask):
            for layer in self.layers:
                hidden=layer(hidden_states=hidden,position_embeddings=(rope_cos,rope_sin),temb=temb,
                    attention_mask=local_mask if layer.attention_type=='sliding_attention' else self_mask,
                    encoder_hidden_states=condition,encoder_attention_mask=cross_mask,use_cache=False,past_key_value=None)[0]
            return hidden.contiguous()
    return Pair()


def partitions(root,null,make_mask,window=SHORT):
    import torch
    class Front(torch.nn.Module):
        def __init__(self):
            super().__init__();self.proj=root.proj_in;self.condition=root.condition_embedder
            self.time=root.time_embed;self.time_r=root.time_embed_r;self.rotary=root.rotary_emb
            self.register_buffer('null',null)
        def forward(self,noise,condition,context,timestep):
            noise=torch.cat([noise,noise],dim=0);context=torch.cat([context,context],dim=0)
            condition=torch.cat([condition,self.null.float().expand_as(condition)],dim=0)
            t=timestep.expand(2);a,p=self.time(t);b,q=self.time_r(torch.zeros_like(t))
            hidden=self.proj(torch.cat([context,noise],dim=-1))
            cond=self.condition(condition);cos,sin=self.rotary(hidden,torch.arange(window.hidden_tokens).unsqueeze(0))
            full=make_mask(window.hidden_tokens,torch.float32,hidden.device,is_causal=False)
            local=make_mask(window.hidden_tokens,torch.float32,hidden.device,sliding_window=128,is_sliding_window=True,is_causal=False)
            cross=torch.zeros((1,1,window.hidden_tokens,window.condition_tokens),dtype=torch.float32,device=hidden.device)
            return tuple(v.contiguous() for v in [hidden,cond,p+q,cos,sin,full,local,cross,a+b])
    class Tail(torch.nn.Module):
        def __init__(self):super().__init__();self.table=root.scale_shift_table;self.norm=root.norm_out;self.proj=root.proj_out
        def forward(self,hidden,final_temb):
            shift,scale=(self.table+final_temb.unsqueeze(1)).chunk(2,dim=1)
            return self.proj(self.norm(hidden)*(1+scale)+shift).contiguous()
    return [Front()]+[block_pair(root.layers[i:i+2]) for i in range(0,24,2)]+[Tail()]


def bounded_comparison(stats):
    """Keep strict parity distinct from permission to export a diagnostic.

    The pointwise bounds do not change. A diagnostic may fail them, but must
    remain finite with normalized RMSE <= the existing 1e-3 limit.
    """
    keys=['max_absolute_error','rmse','normalized_rmse','elementwise_fraction_over_bound']
    if (stats.get('finite') is not True or type(stats.get('parity_passed')) is not bool
            or any(type(stats.get(k)) not in (int,float) or not math.isfinite(stats[k]) or stats[k]<0 for k in keys)
            or stats['elementwise_fraction_over_bound']>1 or stats['normalized_rmse']>1e-3):
        raise ValueError('Host comparison is nonfinite, incomplete or exceeds diagnostic normalized-RMSE limit; no export')
    strict=stats['elementwise_fraction_over_bound']==0 and stats['normalized_rmse']<=1e-3
    if stats['parity_passed'] is not strict:raise ValueError('Inconsistent strict host parity flag')
    return strict


def validate_host_trajectory(comparisons,apg_comparisons):
    if (len(comparisons)!=50 or [r.get('step') for r in comparisons]!=list(range(1,51))
            or len(apg_comparisons)!=50):raise ValueError('Incomplete host trajectory/APG comparisons')
    # APG at identical inputs must still pass the full strict bound.
    if not all([bounded_comparison(row) for row in apg_comparisons]):
        raise ValueError('Identical-input APG gate failed; no export')
    return all([bounded_comparison(row.get('latent',{})) for row in comparisons])


def classify_host_references(comparisons,apg_comparisons,audio_metrics):
    trajectory=validate_host_trajectory(comparisons,apg_comparisons)
    audio=bounded_comparison(audio_metrics)
    passed=trajectory and audio
    return {'host_dual_trajectory_parity_passed':trajectory,'host_numerical_gate_passed':passed,
            'host_export_status':'strict_pass' if passed else 'bounded_diagnostic'}


def diagnostic_comparison(stats):
    """Validate measurements without turning a failed trajectory into PASS."""
    keys=['max_absolute_error','rmse','normalized_rmse','elementwise_fraction_over_bound']
    if (stats.get('finite') is not True or type(stats.get('parity_passed')) is not bool
            or any(type(stats.get(k)) not in (int,float) or not math.isfinite(stats[k]) or stats[k]<0 for k in keys)
            or stats['elementwise_fraction_over_bound']>1):
        raise ValueError('Invalid/nonfinite diagnostic comparison; no export')
    strict=stats['elementwise_fraction_over_bound']==0 and stats['normalized_rmse']<=1e-3
    if stats['parity_passed'] is not strict:raise ValueError('Inconsistent strict diagnostic parity flag')
    return strict


def classify_trajectory_diagnostic(comparisons,apg_comparisons,denoiser_checks,audio_metrics):
    if (len(comparisons)!=50 or [r.get('step') for r in comparisons]!=list(range(1,51))
            or len(apg_comparisons)!=50 or [r.get('step') for r in denoiser_checks]!=CHECKPOINTS):
        raise ValueError('Incomplete diagnostic trajectory/component comparisons')
    if not all([bounded_comparison(row) for row in apg_comparisons]):
        raise ValueError('Identical-input APG gate failed; no diagnostic export')
    if not all([bounded_comparison(row.get('comparison',{})) for row in denoiser_checks]):
        raise ValueError('Identical-input denoiser gate failed; no diagnostic export')
    strict_rows=[diagnostic_comparison(row.get('latent',{})) for row in comparisons]
    audio_strict=diagnostic_comparison(audio_metrics)
    exceeds=any(row['latent']['normalized_rmse']>1e-3 for row in comparisons) or audio_metrics['normalized_rmse']>1e-3
    trajectory=all(strict_rows);passed=trajectory and audio_strict
    status='trajectory_drift_diagnostic' if exceeds else ('strict_pass' if passed else 'bounded_diagnostic')
    return {'host_dual_trajectory_parity_passed':trajectory,'host_numerical_gate_passed':passed,'host_export_status':status}


def same_input_checks(root,parts,plan,fixtures,condition,context,null,times,window,original):
    """Run actual original/partition models serially; caller supplies one kind.

    Fixtures are states AFTER each named gold step, evaluated at times[step].
    No reference is fed back into the independent candidate sampling loop.
    """
    import torch
    result={}
    with torch.no_grad():
        for step,x in fixtures.items():
            t=times[step:step+1]
            if original:
                condition2=torch.cat([condition,null.expand_as(condition)])
                velocity=root(hidden_states=torch.cat([x,x]),timestep=t.expand(2),timestep_r=t.expand(2),attention_mask=None,
                    encoder_hidden_states=condition2,encoder_attention_mask=None,context_latents=torch.cat([context,context]),use_cache=False)[0]
            else:
                state={'noise':x,'condition':condition,'context':context,'timestep':t}
                for part,stage in zip(parts,plan):
                    out=part(*(state[n] for n in stage['input_shapes']));out=out if isinstance(out,tuple) else (out,)
                    for n,v in zip(stage['output_shapes'],out):state[stage['output_bindings'][n]]=v
                velocity=state['velocity']
            result[step]=velocity.clone()
    return result


def verify_bundle(folder,source_folder=None,condition_folder=None,window=SHORT,allow_trajectory_diagnostic=False):
    source_folder=source_folder or ROOT/'prototypes/ios27-coreai/ModelCandidates'/window.source_bundle
    condition_folder=condition_folder or ROOT/'prototypes/ios27-coreai/ModelCandidates'/window.condition_bundle
    ref=json.loads((folder/'reference.json').read_text())
    requested=ref.get('host_trajectory_diagnostic_requested',False)
    if type(requested) is not bool:raise ValueError('Invalid diagnostic request marker')
    if requested:
        if window.frames!=256 or not allow_trajectory_diagnostic:
            raise ValueError('Trajectory diagnostic requires explicit --diagnostic-trajectory for 10.24 seconds')
        status=classify_trajectory_diagnostic(ref.get('host_comparisons',[]),ref.get('apg_same_input_vs_publisher',[]),
            ref.get('host_denoiser_same_input',[]),ref.get('host_audio_vs_fp32',{}))
        for row in ref['host_denoiser_same_input']:
            name=f'step-{row["step"]}-latent-fp32-reference.f32le'
            if (row.get('phase')!='after_gold_step' or row.get('input_latent_file')!=name
                    or row.get('input_latent_sha256')!=ref.get('artifact_sha256',{}).get(name)
                    or row.get('conditioning_input')!='original_fp32_condition_reference'):
                raise ValueError('Diagnostic denoiser input identity mismatch')
    else:
        status=classify_host_references(ref.get('host_comparisons',[]),ref.get('apg_same_input_vs_publisher',[]),ref.get('host_audio_vs_fp32',{}))
    # Existing strictly passing bundles remain reusable without any re-export.
    for key,value in status.items():
        if key not in ref and status['host_numerical_gate_passed'] and key!='host_dual_trajectory_parity_passed':continue
        if ref.get(key)!=value:raise ValueError('Inconsistent host export status; preserved')
    if (ref.get('schema_version')!=1 or ref.get('profile')!=window.preview_bundle or ref.get('stages')!=stage_plan(window)
            or ref.get('inference_steps')!=50 or ref.get('guidance_scale')!=7 or ref.get('shift')!=1
            or ref.get('method')!='euler_ode_apg' or ref.get('apg_momentum')!=-0.75
            or ref.get('apg_norm_threshold')!=2.5 or ref.get('apg_projection_dims')!=[1]
            or ref.get('apg_projection_precision')!=window.apg_projection_precision
            or (window.frames==256 and (ref.get('reference_conditioning_scope')!='original_fp32_vs_storage16_conditioning' or ref.get('weight_storage_dtype')!=window.weight_storage_dtype))
            or ref.get('checkpoints')!=CHECKPOINTS or ref.get('function_name')!='main'
            or ref.get('latent_frames')!=window.frames or ref.get('condition_tokens')!=window.condition_tokens
            or ref.get('sample_rate')!=48000 or ref.get('audio_seconds')!=window.seconds
            or ref.get('scalar_type')!='float32' or ref.get('byte_order')!='little'
            or ref.get('checkpoint_digest')!=records()['model.safetensors']['digest']
            or ref.get('all_checkpoints_verified') is not True
            or ref.get('vae_checkpoint_digest')!='da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0'
            or len(ref.get('host_comparisons',[]))!=50
            or len(ref.get('apg_same_input_vs_publisher',[]))!=50
            or not all(r.get('parity_passed') for r in ref.get('apg_same_input_vs_publisher',[]))
            or ref.get('host_decoder_folding_exact') is not True
            or ref.get('absolute_tolerance')!=1e-4 or ref.get('relative_tolerance')!=1e-3 or ref.get('relative_rmse_tolerance')!=1e-3
            or ref.get('source_reference_sha256')!=sha256(source_folder/'reference.json')
            or ref.get('condition_reference_sha256')!=sha256(condition_folder/'reference.json')):
        raise ValueError('Unexpected bass-preview contract or changed component bundle; preserved')
    required={'initial-noise.f32le':[1,window.frames,64],'timesteps.f32le':[51]}
    for step in CHECKPOINTS:
        for name in ['latent']:
            for kind in ['reference','fp32-reference']:required[f'step-{step}-{name}-{kind}.f32le']=[1,window.frames,64]
    for kind in ['reference','fp32-reference']:required[f'audio-{kind}.f32le']=[1,2,window.samples]
    native={f'{name}/{n}' for name in [*[s['asset_file'] for s in stage_plan(window)],'apg-euler.aimodel','decoder.aimodel'] for n in ['main.hash','main.mlirb','metadata.json']}
    hashes=ref['artifact_sha256'];actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if not (set(required)|native).issubset(hashes) or set(hashes)!=actual:raise ValueError('Incomplete preview bundle')
    for name,digest in hashes.items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=digest:raise ValueError(f'Preview integrity mismatch: {name}')
    for name,shape in required.items():
        data=(folder/name).read_bytes()
        if len(data)!=math.prod(shape)*4:raise ValueError('Preview reference size mismatch')
        vals=[v[0] for v in struct.iter_unpack('<f',data)]
        if not all(math.isfinite(v) for v in vals):raise ValueError('Nonfinite preview reference')
    times=[v[0] for v in struct.iter_unpack('<f',(folder/'timesteps.f32le').read_bytes())]
    if times[0]!=1 or times[-1]!=0 or not all(a>b for a,b in zip(times,times[1:])):raise ValueError('Invalid sampling schedule')
    if requested:
        for row in ref['host_denoiser_same_input']:
            if row.get('timestep')!=times[row['step']]:raise ValueError('Diagnostic denoiser timestep mismatch')
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,choices=[5.12,10.24],default=5.12)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--source-bundle',type=Path,default=None)
    parser.add_argument('--condition-bundle',type=Path,default=None)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--diagnostic-trajectory',action='store_true',help='Explicit 10.24 s component-gated experiment; failed free-trajectory comparisons remain failed')
    args=parser.parse_args()
    window=PreviewWindow.for_seconds(args.seconds)
    if args.diagnostic_trajectory and window.frames!=256:parser.error('--diagnostic-trajectory is restricted to 10.24 seconds')
    candidates=ROOT/'prototypes/ios27-coreai/ModelCandidates'
    args.out=args.out or candidates/window.preview_bundle
    args.source_bundle=args.source_bundle or candidates/window.source_bundle
    args.condition_bundle=args.condition_bundle or candidates/window.condition_bundle
    if args.verify_only:
        ref=verify_bundle(args.out,args.source_bundle,args.condition_bundle,window,args.diagnostic_trajectory)
        print('Integrated bass preview integrity VERIFIED; host status:',ref.get('host_export_status','strict_pass'),'; no re-export');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    verify_source(args.source_bundle,window);verify_condition(args.condition_bundle,window)
    import numpy as np
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    from coreai_torch import TorchConverter,get_decomp_table
    torch.set_num_threads(2)
    publisher=records();path=checkpoint_path(publisher['model.safetensors'])
    source=ROOT/'.runtime-cache/coreai-generator-source-publisher';source.mkdir(parents=True,exist_ok=True)
    paths={n:verified_source(publisher[n],source) for n in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json','apg_guidance.py']}
    _,sampling_source=verify_sampling_source(paths['modeling_acestep_v15_base.py'])
    spec=importlib.util.spec_from_file_location('coreai_original_apg',paths['apg_guidance.py']);apg=importlib.util.module_from_spec(spec);spec.loader.exec_module(apg)
    cfg,kind,rotary=load_full_upstream(paths);make_mask=sys.modules[kind.__module__].create_4d_mask
    with torch.device('meta'):root=kind(cfg).eval()
    with safe_open(str(path),framework='pt',device='cpu') as w:
        state={k[len('decoder.'):]:w.get_tensor(k) for k in w.keys() if k.startswith('decoder.')};null=w.get_tensor('null_condition_emb').float()
    root.load_state_dict(state,strict=True,assign=True);del state
    root.rotary_emb=rotary(cfg,device='cpu');root.to(device='cpu',dtype=torch.float32).requires_grad_(False)
    def tensor(folder,name,shape):return torch.from_numpy(np.fromfile(folder/name,dtype='<f4').copy()).reshape(shape)
    context=tensor(args.source_bundle,'context_latents-reference.f32le',[1,window.frames,128])
    full_condition=tensor(args.condition_bundle,'condition-fp32-reference.f32le',[1,window.condition_tokens,2048])
    cond=tensor(args.condition_bundle,'condition-reference.f32le',[1,window.condition_tokens,2048])
    gold_cond=full_condition
    noise=torch.randn((1,window.frames,64),generator=torch.Generator(device='cpu').manual_seed(0))
    times=torch.linspace(1.,0.,51,dtype=torch.float32)
    def original_predict(x,t):
        conditions=torch.cat([gold_cond,null.expand_as(gold_cond)],dim=0)
        return root(hidden_states=torch.cat([x,x]),timestep=t.expand(2),timestep_r=t.expand(2),attention_mask=None,
            encoder_hidden_states=conditions,encoder_attention_mask=None,context_latents=torch.cat([context,context]),use_cache=False)[0]
    print('Full FP32 gold: real piano, actual bass condition, 50 APG-guided Euler steps.',flush=True)
    gold={};x=noise.clone();momentum=apg.MomentumBuffer()
    with torch.no_grad():
        for i in range(50):
            velocity=original_predict(x,times[i:i+1]);a,b=velocity.chunk(2)
            guided=apg.apg_forward(a,b,7.,momentum_buffer=momentum,dims=[1])
            x=x-guided*(times[i]-times[i+1]);gold[i+1]=(x.clone(),guided.clone())
            if i+1 in CHECKPOINTS or (window.frames==256 and (i+1)%5==0):print(f'Gold step {i+1}/50',flush=True)
    gold_latent=x.clone()
    fixtures={step:gold[step][0].clone() for step in CHECKPOINTS}
    original_checks=same_input_checks(root,None,None,fixtures,gold_cond,context,null,times,window,True) if args.diagnostic_trajectory else {}
    convert_storage16(root,window.weight_storage_dtype);parts=partitions(root,null,make_mask,window);plan=stage_plan(window)
    original_ids={id(p) for p in root.parameters()};partition_ids=[id(p) for part in parts for p in part.parameters()]
    if set(partition_ids)!=original_ids or len(partition_ids)!=len(original_ids):raise ValueError('Denoiser parameter coverage mismatch')
    class APG(torch.nn.Module):
        def forward(self,velocity,latent,momentum,dt):return apg_euler(velocity,latent,momentum,dt)
    update=APG().eval()
    def partition_predict(x,t):
        state={'noise':x,'condition':cond,'context':context,'timestep':t}
        for part,s in zip(parts,plan):
            out=part(*(state[n] for n in s['input_shapes']));out=out if isinstance(out,tuple) else (out,)
            for n,v in zip(s['output_shapes'],out):state[s['output_bindings'][n]]=v
        return state['velocity']
    denoiser_checks=[]
    if args.diagnostic_trajectory:
        candidate_checks=same_input_checks(None,parts,plan,fixtures,gold_cond,context,null,times,window,False)
        for step in CHECKPOINTS:
            stats=precision_metrics(candidate_checks[step],original_checks[step])
            print('Same-input denoiser after gold step',step,stats,flush=True)
            if not bounded_comparison(stats):raise ValueError('Identical-input denoiser gate failed; no diagnostic export')
            raw=fixtures[step].detach().numpy().astype('<f4').tobytes()
            denoiser_checks.append({'step':step,'phase':'after_gold_step','timestep':float(times[step]),
                'input_latent_file':f'step-{step}-latent-fp32-reference.f32le','input_latent_sha256':hashlib.sha256(raw).hexdigest(),
                'conditioning_input':'original_fp32_condition_reference','comparison':stats})
    print('Independent storage16 partition + FP32 APG trajectory.',flush=True)
    expected={};comparisons=[];apg_comparisons=[];x=noise.clone();momentum=torch.zeros_like(x)
    with torch.no_grad():
        for i in range(50):
            velocity=partition_predict(x,times[i:i+1])
            # Compare APG itself at identical velocity, latent and momentum,
            # independently of divergence between the two full trajectories.
            original_momentum=apg.MomentumBuffer();original_momentum.running_average=momentum.clone()
            a,b=velocity.chunk(2);original_guided=apg.apg_forward(a,b,7.,momentum_buffer=original_momentum,dims=[1])
            x,momentum,guided=update(velocity,x,momentum,(times[i]-times[i+1]).reshape(1))
            apg_comparisons.append(precision_metrics(guided,original_guided))
            expected[i+1]=(x.clone(),guided.clone())
            row={'step':i+1,'latent':precision_metrics(x,gold[i+1][0]),'guided':precision_metrics(guided,gold[i+1][1])};comparisons.append(row)
            if i+1 in CHECKPOINTS or (window.frames==256 and (i+1)%5==0):print('Partition step',i+1,'latent',row['latent'],'guided',row['guided'],flush=True)
    try:
        if args.diagnostic_trajectory:
            trajectory_passed=all([diagnostic_comparison(row['latent']) for row in comparisons])
            if not all([bounded_comparison(row) for row in apg_comparisons]):raise ValueError('Identical-input APG gate failed')
        else:
            trajectory_passed=validate_host_trajectory(comparisons,apg_comparisons)
    except ValueError:
        diagnostics=ROOT/'.runtime-cache/coreai-bass-preview-diagnostics';diagnostics.mkdir(parents=True,exist_ok=True)
        failed=Path(tempfile.mkdtemp(prefix=window.suffix+'-',dir=diagnostics))
        payload={'profile':window.preview_bundle,'audio_seconds':window.seconds,
                 'host_export_status':'rejected','absolute_tolerance':1e-4,'relative_tolerance':1e-3,
                 'relative_rmse_tolerance':1e-3,'host_comparisons':comparisons,
                 'apg_same_input_vs_publisher':apg_comparisons,
                 'reference_conditioning_scope':'original_fp32_vs_storage16_conditioning',
                 'weight_storage_dtype':window.weight_storage_dtype,
                 'scope':'Host gate failed; no sampling assets exported; active app profile not changed'}
        (failed/'comparisons.json').write_text(json.dumps(payload,indent=2)+'\n')
        print('Host gate rejected; comparisons saved:',failed/'comparisons.json',flush=True)
        raise
    print('Host latent strict parity:', 'PASS' if trajectory_passed else 'NOT PASSED; diagnostic status remains separate',flush=True)
    expected_latent=x.clone()
    del root,parts,original_predict,partition_predict;gc.collect()
    manifest=json.loads((ROOT/'configs/assets-base.json').read_text());vae=next(r for r in manifest['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    vpath=checkpoint_path(vae);raw_cfg=(ROOT/'configs/coreai-vae-config.json').read_bytes()
    cfg_record=next(r for r in manifest['files'] if r['filename']=='vae/config.json')
    if hashlib.sha1(f'blob {len(raw_cfg)}\0'.encode()+raw_cfg).hexdigest()!=cfg_record['digest']:raise ValueError('VAE config checksum mismatch')
    vcfg=json.loads(raw_cfg)
    with torch.device('meta'):
        decoder=OobleckDecoder(channels=vcfg['decoder_channels'],input_channels=vcfg['decoder_input_channels'],audio_channels=vcfg['audio_channels'],
            upsampling_ratios=list(reversed(vcfg['downsampling_ratios'])),channel_multiples=vcfg['channel_multiples']).eval()
    with safe_open(str(vpath),framework='pt',device='cpu') as w:state={k[len('decoder.'):]:w.get_tensor(k) for k in w.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(state,strict=True,assign=True);del state
    decoder.to(device='cpu',dtype=torch.float32).requires_grad_(False)
    with torch.no_grad():gold_audio=decoder(gold_latent.transpose(1,2)).clone();unfolded=decoder(expected_latent.transpose(1,2)).clone()
    for m in decoder.modules():
        if hasattr(m,'weight_g') and hasattr(m,'weight_v'):torch.nn.utils.remove_weight_norm(m)
    with torch.no_grad():audio=decoder(expected_latent.transpose(1,2))
    torch.testing.assert_close(audio,unfolded,rtol=0,atol=0)
    audio_metrics=precision_metrics(audio,gold_audio)
    host_status=(classify_trajectory_diagnostic(comparisons,apg_comparisons,denoiser_checks,audio_metrics) if args.diagnostic_trajectory
                 else classify_host_references(comparisons,apg_comparisons,audio_metrics))
    print('Decoded waveform strict parity:',audio_metrics,flush=True)
    print('Host preparation status:',host_status['host_export_status'],'; tolerances unchanged',flush=True)
    parent=ROOT/'.runtime-cache/coreai-bass-preview-staging';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='preview-',dir=parent));print('Staging:',stage,flush=True)
    def save(name,value):value.detach().numpy().astype('<f4').tofile(stage/name)
    save('initial-noise.f32le',noise);save('timesteps.f32le',times)
    for i in CHECKPOINTS:
        for n,index in [('latent',0)]:
            save(f'step-{i}-{n}-reference.f32le',expected[i][index]);save(f'step-{i}-{n}-fp32-reference.f32le',gold[i][index])
    save('audio-reference.f32le',audio);save('audio-fp32-reference.f32le',gold_audio)
    # Reload only the storage16 denoiser after decoder reference computation to
    # keep host working memory bounded; no old native bundle is overwritten.
    with torch.device('meta'):root=kind(cfg).eval()
    with safe_open(str(path),framework='pt',device='cpu') as w:state={k[len('decoder.'):]:w.get_tensor(k) for k in w.keys() if k.startswith('decoder.')}
    root.load_state_dict(state,strict=True,assign=True);del state
    root.rotary_emb=rotary(cfg,device='cpu');root.to(device='cpu',dtype=torch.float32).requires_grad_(False);convert_storage16(root,window.weight_storage_dtype)
    parts=partitions(root,null,make_mask,window);state={'noise':noise,'condition':cond,'context':context,'timestep':times[:1]}
    def export(module,inputs,input_names,output_names,name):
        module.eval();print('Export',name,flush=True)
        ep=torch.export.export(module,inputs).run_decompositions(get_decomp_table())
        program=TorchConverter().add_exported_program(ep,input_names=input_names,output_names=output_names).to_coreai()
        program.save_asset(stage/name);del program,ep;gc.collect()
    with torch.no_grad():
        for part,s in zip(parts,plan):
            inputs=tuple(state[n] for n in s['input_shapes']);out=part(*inputs);out=out if isinstance(out,tuple) else (out,)
            export(part,inputs,list(s['input_shapes']),list(s['output_shapes']),s['asset_file'])
            for n,v in zip(s['output_shapes'],out):state[s['output_bindings'][n]]=v
    export(update,(state['velocity'],noise,torch.zeros_like(noise),(times[0]-times[1]).reshape(1)),list(APG_INPUTS),list(APG_OUTPUTS),'apg-euler.aimodel')
    export(decoder,(expected_latent.transpose(1,2).contiguous(),),['latent'],['audio'],'decoder.aimodel')
    ref={'schema_version':1,'profile':window.preview_bundle,'function_name':'main','stages':plan,'inference_steps':50,'guidance_scale':7.,'shift':1.,
         'method':'euler_ode_apg','apg_momentum':-0.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
         'apg_projection_precision':window.apg_projection_precision,'checkpoints':CHECKPOINTS,'latent_frames':window.frames,'condition_tokens':window.condition_tokens,
         'sample_rate':48000,'audio_seconds':window.seconds,'scalar_type':'float32','byte_order':'little','checkpoint_digest':publisher['model.safetensors']['digest'],
         'reference_conditioning_scope':'original_fp32_vs_storage16_conditioning',
         'host_trajectory_diagnostic_requested':args.diagnostic_trajectory,'host_denoiser_same_input':denoiser_checks,
         'weight_storage_dtype':window.weight_storage_dtype,
         'vae_checkpoint_digest':vae['digest'],'all_checkpoints_verified':True,**host_status,'host_decoder_folding_exact':True,
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'source_reference_sha256':sha256(args.source_bundle/'reference.json'),'condition_reference_sha256':sha256(args.condition_bundle/'reference.json'),
         'host_comparisons':comparisons,'apg_same_input_vs_publisher':apg_comparisons,'host_audio_vs_fp32':audio_metrics,'sampling_source':sampling_source,
         'apg_source_digest':publisher['apg_guidance.py']['digest'],'seed':0,'rng_scope':'Host CPU seeded fixtures; no CUDA RNG parity or on-device RNG claim',
         'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','diffusers','coreai-torch','coreai-core']},
         'artifact_sha256':{str(p.relative_to(stage)):sha256(p) for p in stage.rglob('*') if p.is_file()},
         'scope':f'Real piano + duration-matched bass preset -> 50 APG-guided Euler steps -> {window.seconds} s decoded waveform; actual new device profile execution pending'}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_bundle(stage,args.source_bundle,args.condition_bundle,window,args.diagnostic_trajectory)
    args.out.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(stage),str(args.out))
    print('Ready:',host_status['host_export_status'],'integrated preview bundle',args.out,flush=True)


if __name__=='__main__':main()

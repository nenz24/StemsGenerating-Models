#!/usr/bin/env python3
"""Audit bounded FP32 source encoding against the unchanged accepted750-frame capture.

Default is plan-only. No DiT/Qwen/native export or asset activation. The encoder
keeps only cores proved independent of artificial window padding.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from export_coreai_decoder import ROOT,sha256
from coreai_source_window_contract import PROFILE,WINDOW,CORE,HALO,RATIO,VAE_SHA,GOLD_FILES,tile_plan,dependency_bounds


def load_encoder(checkpoint):
    import torch
    from compare_coreai_drum_source_gain import load_encoder as original_encoder
    model=original_encoder(checkpoint)
    for module in model.modules():
        if hasattr(module,'weight_g') and hasattr(module,'weight_v'):torch.nn.utils.remove_weight_norm(module)
    return model


def verify_architecture(encoder):
    import torch
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckEncoder,OobleckEncoderBlock,OobleckResidualUnit,Snake1d
    allowed=(OobleckEncoder,OobleckEncoderBlock,OobleckResidualUnit,Snake1d,torch.nn.Conv1d,torch.nn.ModuleList)
    if any(not isinstance(module,allowed) for module in encoder.modules()):
        raise ValueError('Unknown/global temporal source operator invalidates proof')
    for module,kernel,padding in [(encoder.conv1,7,3),(encoder.conv2,3,1)]:
        if (module.kernel_size,module.padding,module.stride,module.dilation)!=((kernel,),(padding,),(1,),(1,)):
            raise ValueError('Source endpoint geometry changed')
    if len(encoder.block)!=5:raise ValueError('Source block count changed')
    for block,stride in zip(encoder.block,(2,4,4,6,10)):
        module=block.conv1
        if (module.kernel_size,module.stride,module.padding,module.dilation)!=((2*stride,),(stride,),(stride//2,),(1,)):
            raise ValueError('Source downsampling geometry changed')
        for unit,dilation in zip((block.res_unit1,block.res_unit2,block.res_unit3),(1,3,9)):
            a,b=unit.conv1,unit.conv2
            if (a.kernel_size,a.padding,a.stride,a.dilation)!=((7,),(3*dilation,),(1,),(dilation,)):
                raise ValueError('Source residual geometry changed')
            if (b.kernel_size,b.padding,b.stride,b.dilation)!=((1,),(0,),(1,),(1,)):
                raise ValueError('Source residual tail changed')


def encode_windows(encoder,audio,epsilon,mask,progress=lambda *args:None,observe=None,window_frames=WINDOW):
    import torch
    from export_coreai_source_context import posterior_context
    frames=epsilon.shape[-1]
    if (list(audio.shape)!=[1,2,frames*RATIO] or list(epsilon.shape)!=[1,64,frames]
            or list(mask.shape)!=[1,frames,64]
            or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in (audio,epsilon,mask))):
        raise ValueError('Invalid full-range source inputs')
    verify_architecture(encoder)
    outputs={name:torch.empty(1,64,frames) if name.startswith('posterior') else
             torch.empty(1,frames,128 if name=='context_latents' else 64) for name in GOLD_FILES}
    coverage=[0]*frames
    tiles=tile_plan(frames,window_frames)
    with torch.no_grad():
        for i,tile in enumerate(tiles):
            dependency_bounds(tile,frames)
            start=tile['window_start'];a,b=tile['core_start'],tile['core_end'];c=tile['crop_start']
            begin=time.monotonic()
            params=encoder(audio[...,start*RATIO:(start+window_frames)*RATIO].contiguous())
            if list(params.shape)!=[1,128,window_frames] or not torch.isfinite(params).all():
                raise ValueError('Invalid source window parameters')
            result=posterior_context(params,epsilon[...,start:start+window_frames].contiguous(),mask[:,start:start+window_frames,:].contiguous())
            for name,value in zip(GOLD_FILES,result):
                if not torch.isfinite(value).all():raise ValueError('Nonfinite source output')
                if name.startswith('posterior'):outputs[name][...,a:b]=value[...,c:c+b-a]
                else:outputs[name][:,a:b,:]=value[:,c:c+b-a,:]
            for frame in range(a,b):coverage[frame]+=1
            if observe is not None:observe(i,tile,outputs)
            progress(i+1,len(tiles),time.monotonic()-begin)
    if set(coverage)!={1}:raise ValueError('Source cores must cover each frame exactly once')
    return outputs


def execute(args):
    import numpy as np
    import torch
    from audit_coreai_drum_reference30 import verify_capture,read_tensor
    from compare_coreai_drum_lyrics import local_checkpoint
    from coreai_drum_reference30_window_contract import strict_metric
    from export_coreai_dit_block import precision_metrics
    if args.out.exists():raise FileExistsError('Existing source audit preserved')
    if any(args.out.resolve().is_relative_to(p) for p in (args.capture.resolve(),ROOT/'baselines',ROOT/'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Use inactive output outside captures/models/baselines')
    verify_capture(args.capture)
    record=next(r for r in json.loads((ROOT/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    if record['digest']!=VAE_SHA:raise ValueError('Pinned source VAE changed')
    checkpoint=local_checkpoint(record,args.vae_checkpoint)
    torch.set_num_threads(2);encoder=load_encoder(checkpoint)
    audio=read_tensor(args.capture,'source48.f32le');epsilon=read_tensor(args.capture,'source-epsilon.f32le');mask=read_tensor(args.capture,'source-mask.f32le')
    gold={name:read_tensor(args.capture,file) for name,file in GOLD_FILES.items()}
    enabled=torch.backends.mkldnn.enabled if args.convolution_backend=='default' else False
    report={'schema_version':1,'profile':PROFILE,'status':'running','vae_checkpoint_sha256':VAE_SHA,
            'capture_report_sha256':sha256(args.capture/'report.json'),'window_frames':args.window_frames,'core_frames':args.window_frames-2*HALO,'halo_frames':HALO,
            'window_plan':tile_plan(750,args.window_frames),'dependency_sample_bounds':[dependency_bounds(t,750) for t in tile_plan(750,args.window_frames)],
            'weight_storage_dtype':'float32','compute_dtype':'float32','epsilon_regenerated':False,
            'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
            'source_encoder_calls':0,'checks':{},'baseline_modified':False,'native_execution':'NOT EVALUATED'}
    report.update(reference_convolution_backend=args.convolution_backend,mkldnn_available=torch.backends.mkldnn.is_available(),
                  mkldnn_enabled_flag=enabled,tile_input_sha256=[])
    for tile in tile_plan(750,args.window_frames):
        start=tile['window_start'];width=args.window_frames
        tensors={'audio':audio[...,start*RATIO:(start+width)*RATIO],
                 'epsilon':epsilon[...,start:start+width],'chunk_mask':mask[:,start:start+width,:]}
        report['tile_input_sha256'].append({name:hashlib.sha256(value.contiguous().numpy().astype('<f4').tobytes()).hexdigest()
                                           for name,value in tensors.items()})
    args.out.mkdir(parents=True)
    def save():
        temp=args.out/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out/'report.json')
    def check(label,actual,expected):
        report['checks'][label]=precision_metrics(actual,expected)
        # Collect all finite strict failures; never expand gates or substitute gold.
        if not report['checks'][label]['finite']:raise ValueError('Nonfinite source comparison')
    def observe(index,tile,actual):
        a,b=tile['core_start'],tile['core_end']
        for name in GOLD_FILES:
            if name.startswith('posterior'):check(f'core-{index}-{name}',actual[name][...,a:b],gold[name][...,a:b])
            else:check(f'core-{index}-{name}',actual[name][:,a:b,:],gold[name][:,a:b,:])
        report['source_encoder_calls']+=1;save()
    def progress(i,total,seconds):print(f'Source encoder window{i}/{total}; seconds {seconds:.3f}',flush=True)
    try:
        with torch.backends.mkldnn.flags(enabled=enabled):
            actual=encode_windows(encoder,audio,epsilon,mask,progress,observe,window_frames=args.window_frames)
        for index,tile in enumerate(tile_plan(750,args.window_frames)[1:],1):
            a,b=tile['core_start']-2,min(tile['core_start']+2,750)
            for name in GOLD_FILES:
                if name.startswith('posterior'):check(f'seam-{index}-{name}',actual[name][...,a:b],gold[name][...,a:b])
                else:check(f'seam-{index}-{name}',actual[name][:,a:b,:],gold[name][:,a:b,:])
        for name in GOLD_FILES:
            check('overall-'+name,actual[name],gold[name])
            actual[name].numpy().astype('<f4').tofile(args.out/(name+'.f32le'))
        expected_checks=(len(tile_plan(750,args.window_frames))*2)*4
        if len(report['checks'])!=expected_checks:raise ValueError('Incomplete source gate collection')
        passed=all(m['parity_passed'] for m in report['checks'].values())
        report.update(status='completed',strict_verdict='PASS' if passed else 'NOT PASSED',
                      comparisons_passed=sum(m['parity_passed'] for m in report['checks'].values()),comparisons_target=expected_checks,
                      full_song_native_readiness='NOT EVALUATED; original full-trajectory gate unchanged')
        save();print('Bounded source original gates:',report['strict_verdict'],report['comparisons_passed'],'/',expected_checks,flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--window-frames',type=int,choices=[128,256],default=WINDOW)
    parser.add_argument('--convolution-backend',choices=['aten','default'],default='aten',
                        help='CPU reference convolution control only; production/native compute and tolerances unchanged')
    parser.add_argument('--capture',type=Path,default=ROOT/'.runtime-cache/drums-reference30-v3')
    parser.add_argument('--vae-checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'.runtime-cache/source-window30-host-v1')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'PLAN ONLY','window_frames':args.window_frames,'core_frames':args.window_frames-2*HALO,'halo_frames':HALO,
                          'tiles':tile_plan(750,args.window_frames),'dependency_sample_bounds':[dependency_bounds(t,750) for t in tile_plan(750,args.window_frames)],
                          'model_calls':0,'baseline_modified':False},indent=2));return
    execute(args)


if __name__=='__main__':main()

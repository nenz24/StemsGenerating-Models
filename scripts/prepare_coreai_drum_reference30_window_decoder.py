#!/usr/bin/env python3
"""Prepare one FP32 window256 decoder after full750's high device footprint.

Six host windows must match the exact accepted full30s audio at original bounds.
Reuse an intact existing256 decoder when eligible; otherwise export only VAE.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from export_coreai_decoder import ROOT, sha256
from coreai_drum_reference30_window_contract import (
    PROFILE, BUNDLE, CAPTURE_SHA, VAE_SHA, DATA_HASHES, DATA_SHAPES, NATIVE_FILES, FILES,
    tile_plan, dependency_bounds, strict_metric, validate)

DEFAULT_OUT = ROOT / 'prototypes/ios27-coreai/ModelCandidates' / BUNDLE


def verify_capture(folder):
    folder = Path(folder)
    for name, digest in DATA_HASHES.items():
        path = folder / ('report.json' if name == 'capture-report.json' else name)
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError('Accepted capture identity mismatch: ' + name)
        if name in DATA_SHAPES and path.stat().st_size != math.prod(DATA_SHAPES[name])*4:
            raise ValueError('Capture shape mismatch: ' + name)


def load_decoder(checkpoint):
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    cfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
    with torch.device('meta'):
        decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
            audio_channels=cfg['audio_channels'], upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
            channel_multiples=cfg['channel_multiples']).eval()
    with safe_open(str(checkpoint), framework='pt', device='cpu') as stream:
        weights = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(weights, strict=True, assign=True)
    del weights
    decoder.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    for m in decoder.modules():
        if hasattr(m, 'weight_g') and hasattr(m, 'weight_v'): torch.nn.utils.remove_weight_norm(m)
    return decoder


def verify_architecture(decoder):
    import torch
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder, OobleckDecoderBlock, OobleckResidualUnit, Snake1d
    allowed = (OobleckDecoder, OobleckDecoderBlock, OobleckResidualUnit, Snake1d,
               torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.ModuleList)
    if any(not isinstance(m, allowed) for m in decoder.modules()):
        raise ValueError('Unknown/global temporal operator invalidates window proof')
    for m in (decoder.conv1, decoder.conv2):
        if (m.kernel_size,m.padding,m.stride,m.dilation) != ((7,),(3,),(1,),(1,)): raise ValueError('Decoder endpoint architecture changed')
    if len(decoder.block) != 5: raise ValueError('Decoder block count changed')
    for block,s in zip(decoder.block,[10,6,4,4,2]):
        m = block.conv_t1
        if (m.kernel_size,m.padding,m.stride,m.dilation) != ((2*s,),((s+1)//2,),(s,),(1,)): raise ValueError('Upsampling architecture changed')
        for unit,d in zip([block.res_unit1,block.res_unit2,block.res_unit3],[1,3,9]):
            a,b=unit.conv1,unit.conv2
            if (a.kernel_size,a.padding,a.dilation,a.stride) != ((7,),(3*d,),(d,),(1,)): raise ValueError('Residual architecture changed')
            if (b.kernel_size,b.padding,b.dilation,b.stride) != ((1,),(0,),(1,),(1,)): raise ValueError('Residual tail architecture changed')


def host_checks(decoder, capture, progress):
    import numpy as np
    import torch
    from export_coreai_dit_block import precision_metrics
    verify_architecture(decoder)
    latent = torch.from_numpy(np.fromfile(Path(capture)/'final-latent.f32le',dtype='<f4').reshape(1,750,64)).transpose(1,2).contiguous()
    gold = torch.from_numpy(np.fromfile(Path(capture)/'final-audio.f32le',dtype='<f4').reshape(1,2,1440000))
    assembled = torch.zeros_like(gold); coverage = [0]*750
    checks = {}; hashes=[]; bounds=[]
    with torch.no_grad():
        for i,t in enumerate(tile_plan()):
            bounds.append(dependency_bounds(t))
            x = latent[...,t['window_start']:t['window_start']+256].contiguous()
            hashes.append(hashlib.sha256(x.numpy().astype('<f4').tobytes()).hexdigest())
            progress(f'host decode window{i+1}/6; compare retained core to original full30s')
            y = decoder(x)
            if list(y.shape) != [1,2,491520] or not torch.isfinite(y).all(): raise ValueError('Invalid window decoder output')
            a,b=t['core_start']*1920,t['core_end']*1920
            kept=y[...,t['crop_start']*1920:t['crop_start']*1920+b-a]
            checks[f'tile-{i}']=precision_metrics(kept,gold[...,a:b]); strict_metric(checks[f'tile-{i}'])
            assembled[...,a:b]=kept
            for f in range(t['core_start'],t['core_end']): coverage[f]+=1
            del y,kept
    if set(coverage)!={1}: raise ValueError('Output coverage must be exactly once')
    for i,t in enumerate(tile_plan()[1:],1):
        c=t['core_start']*1920
        checks[f'seam-{i}']=precision_metrics(assembled[...,c-3840:c+3840],gold[...,c-3840:c+3840]); strict_metric(checks[f'seam-{i}'])
    checks['overall']=precision_metrics(assembled,gold); strict_metric(checks['overall'])
    return checks,hashes,bounds


def reusable_decoder(folder):
    """Only a fully hashed original FP32 VAE decoder with the proven256 shape."""
    folder=Path(folder); ref=json.loads((folder/'reference.json').read_text())
    if (ref.get('profile') not in ('bass-preview-1024','drums-preview-1024','guitar-strum-preview-1024','guitar-pick-preview-1024')
            or ref.get('latent_frames')!=256 or ref.get('audio_seconds')!=10.24 or ref.get('function_name')!='main'
            or ref.get('vae_checkpoint_digest')!=VAE_SHA or ref.get('host_decoder_folding_exact') is not True
            or ref.get('all_checkpoints_verified') is not True or ref.get('scalar_type')!='float32' or ref.get('byte_order')!='little'):
        raise ValueError('Existing decoder does not meet original FP32/256 provenance contract')
    hashes={n:ref['artifact_sha256']['decoder.aimodel/'+n] for n in NATIVE_FILES}
    for name,digest in hashes.items():
        p=folder/'decoder.aimodel'/name
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha256(p)!=digest:
            raise ValueError('Existing decoder integrity mismatch; preserved')
    return {'profile':ref['profile'],'reference_sha256':sha256(folder/'reference.json'),'artifact_sha256':hashes,
            'versions':{k:v for k,v in ref.get('versions',{}).items() if k in ('torch','diffusers','coreai-torch','coreai-core')}}


def find_reusable(candidates):
    skipped=[]
    for name in ('drums-preview-1024','bass-preview-1024','guitar-strum-preview-1024','guitar-pick-preview-1024'):
        folder=Path(candidates)/name
        if not (folder/'reference.json').is_file(): continue
        try: return folder,reusable_decoder(folder),skipped
        except (OSError,ValueError,KeyError) as error: skipped.append({'profile':name,'reason':type(error).__name__})
    return None,None,skipped


def execute(capture,out,report,progress,checkpoint=None,reuse=None,host_only=False):
    import torch
    from compare_coreai_drum_lyrics import local_checkpoint
    torch.set_num_threads(2)
    record=next(r for r in json.loads((ROOT/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    if record['digest']!=VAE_SHA: raise ValueError('Pinned VAE changed')
    progress('verify existing VAE weights; no download/generation')
    checkpoint=local_checkpoint(record,checkpoint)
    decoder=load_decoder(checkpoint)
    checks,hashes,bounds=host_checks(decoder,capture,progress)
    report.update(host_checks=checks,tile_input_sha256=hashes,dependency_bounds=bounds,
        architecture_guard_passed=True,coverage_exactly_once=True,host_numerical_gate_passed=True,host_window_decodes=6)
    report['versions']={n:importlib.metadata.version(n) for n in ('torch','diffusers')}
    if host_only:
        report['status']='host_controls_complete'; return
    if reuse is None:
        reuse,info,skipped=find_reusable(ROOT/'prototypes/ios27-coreai/ModelCandidates')
        if skipped: report['ineligible_reuse_sources_preserved']=skipped
    else: info=reusable_decoder(reuse)
    if info is not None:
        progress('reuse verified existing FP32 decoder256 asset; no new native export')
        (out/'decoder.aimodel').mkdir()
        for name in NATIVE_FILES: shutil.copy2(Path(reuse)/'decoder.aimodel'/name,out/'decoder.aimodel'/name)
        report['native_reuse']=info
    else:
        if sys.platform!='darwin' or platform.machine()!='arm64': raise RuntimeError('Native export requires Apple Silicon Mac')
        from coreai_torch import TorchConverter,get_decomp_table
        import numpy as np
        x=torch.from_numpy(np.fromfile(Path(capture)/'final-latent.f32le',dtype='<f4').reshape(1,750,64)).transpose(1,2)[...,:256].contiguous()
        progress('export one FP32 decoder256; no DiT/Qwen export')
        ep=torch.export.export(decoder,(x,)).run_decompositions(get_decomp_table())
        model=TorchConverter().add_exported_program(ep,input_names=['latent'],output_names=['audio']).to_coreai()
        model.save_asset(out/'decoder.aimodel'); report['native_exports']=1
        report['versions'].update({n:importlib.metadata.version(n) for n in ('coreai-torch','coreai-core')})
        del ep,model
    del decoder;gc.collect()
    for name in DATA_HASHES:
        shutil.copy2(Path(capture)/('report.json' if name=='capture-report.json' else name),out/name)
    report['artifact_sha256']={n:sha256(out/n) for n in FILES}
    report['status']='prepared_window_decoder'


def verify_bundle(folder):
    folder=Path(folder);ref=validate(json.loads((folder/'reference.json').read_text()))
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if actual!=FILES: raise ValueError('Incomplete/unexpected bounded decoder payload')
    for n,h in ref['artifact_sha256'].items():
        p=folder/n
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha256(p)!=h: raise ValueError('Bounded decoder integrity mismatch: '+n)
    for n,shape in DATA_SHAPES.items():
        if (folder/n).stat().st_size!=math.prod(shape)*4: raise ValueError('Bounded decoder shape mismatch')
    if 'native_reuse' in ref:
        if {n:ref['artifact_sha256']['decoder.aimodel/'+n] for n in NATIVE_FILES}!=ref['native_reuse']['artifact_sha256']:
            raise ValueError('Reused decoder payload differs')
    return ref


def prepare(capture,out,checkpoint=None,reuse=None,host_only=False):
    capture,out=Path(capture),Path(out)
    if out.exists():
        verify_bundle(out);print('Existing complete bounded decoder verified; no inference/export repeated.',flush=True);return
    if (out.is_symlink() or out.resolve().is_relative_to(capture.resolve()) or out.resolve().is_relative_to(ROOT/'baselines')
            or (out.resolve().is_relative_to(ROOT/'prototypes/ios27-coreai/ModelCandidates') and out.resolve()!=DEFAULT_OUT.resolve())):
        raise ValueError('Use a new bounded-decoder output; preserve source/baselines/other models')
    verify_capture(capture)
    out.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'profile':PROFILE,'status':'preparing','asset_file':'decoder.aimodel','function_name':'main',
        'input_shapes':{'latent':[1,64,256]},'output_shapes':{'audio':[1,2,491520]},'sample_rate':48000,'audio_seconds':30.,
        'latent_frames':750,'window_frames':256,'core_frames':128,'halo_frames':64,'window_plan':tile_plan(),
        'weight_storage_dtype':'float32','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
        'vae_checkpoint_sha256':VAE_SHA,'capture_report_sha256':CAPTURE_SHA,'absolute_tolerance':1e-4,'relative_tolerance':1e-3,
        'relative_rmse_tolerance':1e-3,'crossfade':False,'extra_latent_padding':False,'native_exports':0,'host_window_decodes':0,
        'native_parity':'NOT EVALUATED','native_memory_gate':'NOT EVALUATED','full_trajectory_parity':'NOT EVALUATED','baseline_modified':False}
    def save():
        tmp=out/'reference.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'reference.json')
    def progress(phase):report['phase']=phase;save();print(phase,flush=True)
    save()
    try:
        execute(capture,out,report,progress,checkpoint,reuse,host_only)
        save()
        if not host_only: verify_bundle(out)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--vae-checkpoint',type=Path)
    parser.add_argument('--reuse-native-from',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--host-audit-only',action='store_true')
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'profile':PROFILE,'window_plan':tile_plan(),'native_assets':1,'native_calls':6,'full_trajectory':'NOT EVALUATED'},indent=2));return
    if args.capture is None:parser.error('--capture required')
    if args.activate and (args.host_audit_only or args.out.resolve()!=DEFAULT_OUT.resolve()):parser.error('--activate requires the default new ModelCandidates bundle, not host-only output')
    try:
        prepare(args.capture,args.out,args.vae_checkpoint,args.reuse_native_from,args.host_audit_only)
        if args.activate:
            from activate_coreai_drum_reference30 import select_profile
            verify_bundle(args.out);select_profile(DEFAULT_OUT.parent,PROFILE)
    finally:
        if sys.platform=='darwin' and (args.out/'reference.json').is_file():
            subprocess.run(['open','-R',str(args.out/'reference.json')],check=False)


if __name__=='__main__':main()

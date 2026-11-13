#!/usr/bin/env python3
"""Prepare inactive full-song global inputs, saved noise and per-window identities.

Reuse verified local source/conditioning/context. If context is absent, capture
it once with the proven bounded CPU source encoder. No DiT/native generation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import wave

from export_coreai_decoder import ROOT, sha256
from prepare_coreai_full_song import schedule

ROLES = ('bass','drums','guitar')
PROFILE = 'full-song-prepared-inputs-v1'


def verify_source(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    wav = folder/'source32.wav'
    if (report.get('status') != 'source_and_schedule_prepared' or report.get('source_gain') != .7
            or report.get('baseline_modified') is not False or wav.is_symlink()
            or sha256(wav) != report.get('source32_wav_sha256')):
        raise ValueError('Prepared full-song source identity differs; preserved')
    digest = hashlib.sha256()
    with wave.open(str(wav),'rb') as stream:
        if (stream.getnchannels(),stream.getsampwidth(),stream.getframerate()) != (2,2,32000):
            raise ValueError('Expected stereo32kHz PCM16 source')
        frames = stream.getnframes()
        while raw := stream.readframes(65536): digest.update(raw)
    plan = schedule(frames)
    saved_plan=json.loads((folder/'plan.json').read_text())
    # Early plan files omit this purely derived padding count. Validate every
    # other field and position; normalize only in memory, never rewrite source.
    for window in saved_plan.get('windows',[]):
        if 'right_padding_samples48' not in window:
            window['right_padding_samples48']=1440000-(window['sample_end48']-window['sample_start48'])
    if (frames != report.get('source32_frames') or digest.hexdigest() != report.get('source32_pcm_sha256')
            or saved_plan != plan or plan['latent_frames'] < 750):
        raise ValueError('Full-song source/plan mismatch or source shorter than30s')
    return report,plan


def mapped(path,shape,expected):
    import numpy as np
    if path.is_symlink() or not path.is_file() or path.stat().st_size != math.prod(shape)*4 or sha256(path) != expected:
        raise ValueError('Input shape/SHA256 mismatch: '+str(path))
    value = np.memmap(path,dtype='<f4',mode='r',shape=tuple(shape))
    if not np.isfinite(value).all(): raise ValueError('Nonfinite input: '+str(path))
    return value


def verify_context(folder,source_report,plan):
    import numpy as np
    from coreai_source_window_contract import VAE_SHA, tile_plan
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    frames = plan['latent_frames']
    expected = {'status':'host_source_context_captured','source32_wav_sha256':source_report['source32_wav_sha256'],
                'source_gain_already_applied':.7,'additional_gain':1.,'sample_rate':48000,
                'samples':plan['target_samples48'],'latent_frames':frames,'vae_checkpoint_sha256':VAE_SHA,
                'window_frames':256,'core_frames':224,'halo_frames':16,'reference_convolution_backend':'aten',
                'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
                'mask_preserved_exact':True,'baseline_modified':False,'ready_for_native_execution':False,
                'source_encoder_calls':len(tile_plan(frames)),'expected_source_encoder_calls':len(tile_plan(frames)),
                'DiT_calls':0,'Qwen_calls':0,'native_exports':0,'original_source_proof_comparisons_passed':32,
                'window_plan':tile_plan(frames),'global_right_padding_samples':frames*1920-plan['target_samples48']}
    if any(report.get(k) != v for k,v in expected.items()): raise ValueError('Existing global context contract differs; no overwrite/re-encode')
    shapes = {'audio48-model.f32le':[1,2,frames*1920],'epsilon-global.f32le':[1,64,frames],
              'source-mask.f32le':[1,frames,64],'posterior_mean.f32le':[1,64,frames],
              'posterior_std.f32le':[1,64,frames],'source_latents.f32le':[1,frames,64],'context_latents.f32le':[1,frames,128]}
    if set(report.get('artifact_sha256',{})) != set(shapes): raise ValueError('Incomplete global context')
    arrays = {name:mapped(folder/name,shape,report['artifact_sha256'][name]) for name,shape in shapes.items()}
    sample = (arrays['posterior_mean.f32le']+arrays['posterior_std.f32le']*arrays['epsilon-global.f32le']).transpose(0,2,1)
    if (not np.array_equal(sample,arrays['source_latents.f32le'])
            or not np.array_equal(arrays['context_latents.f32le'][...,:64],sample)
            or not np.array_equal(arrays['context_latents.f32le'][...,64:],arrays['source-mask.f32le'])
            or not (arrays['source-mask.f32le']==1).all()):
        raise ValueError('Saved epsilon/posterior/source/context consistency failed')
    if report['global_right_padding_samples'] and not (arrays['audio48-model.f32le'][...,plan['target_samples48']:]==0).all():
        raise ValueError('Global right padding differs')
    return report


def slice_sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def window_records(plan,context,piano,epsilon,noises,condition_hashes):
    shared = []
    for window in plan['windows']:
        a,b = window['latent_start'],window['latent_end']
        shared.append({'source_context_sha256':slice_sha(context[:,a:b,:]),
                       'source_audio_sha256':slice_sha(piano[...,a*1920:b*1920]),
                       'source_epsilon_sha256':slice_sha(epsilon[...,a:b])})
    return [{'instrument':role,'window_index':i,'latent_start':w['latent_start'],'latent_end':w['latent_end'],
             'condition_sha256':condition_hashes[role],'initial_noise_sha256':slice_sha(noises[role][:,w['latent_start']:w['latent_end'],:]),
             **shared[i]} for role in ROLES for i,w in enumerate(plan['windows'])]


def verify_job(folder):
    import numpy as np
    from coreai_instrument_preview30_contract import PRESETS
    folder = Path(folder)
    if (folder/'reference.json').is_symlink(): raise ValueError('Linked job reference rejected')
    ref = json.loads((folder/'reference.json').read_text())
    plan = schedule(ref['frames32'])
    if plan['latent_frames'] < 750:raise ValueError('Global context must contain at least750 frames')
    expected = {'schema_version':1,'profile':PROFILE,'status':'prepared_inactive_inputs',
                'instruments':list(ROLES),'window_frames':750,'condition_tokens':127,
                'windows':plan['windows'],'target_samples48':plan['target_samples48'],'latent_frames':plan['latent_frames'],
                'preset_sha256':PRESETS,'sampler_weight_storage_dtype':'bfloat16','compute_dtype':'float32','vae_dtype':'float32',
                'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
                'inference_steps':50,'guidance_scale':7.,'apg_momentum':-.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
                'baseline_modified':False,'native_exports':0,'profile_activated':False,'ready_for_native_execution':False,
                'full_song_music_acceptance':'NOT EVALUATED',
                'full_song_numerical_parity':'NOT EVALUATED; new global inputs are not the frozen30s input',
                'noise_cpu_seeds':{role:seed for seed,role in enumerate(ROLES)},
                'source_gain_already_applied':.7,'additional_source_gain':1.,
                'noise_policy':'Actual saved global CPU Float32 arrays; overlap uses identical slices; seeds informational only',
                'assembly_policy':'Float64 normalized cosine overlap-add on raw outputs; crop to exact target length'}
    if any(ref.get(k) != v for k,v in expected.items()): raise ValueError('Full-song input contract differs')
    if ref.get('native_batch_runtime') not in ('NOT IMPLEMENTED','NOT EVALUATED'):raise ValueError('Input preparation cannot accept native execution')
    frames = plan['latent_frames']
    shapes = {'piano48.f32le':[1,2,frames*1920],'source-context.f32le':[1,frames,128],
              'source-epsilon.f32le':[1,64,frames],'timesteps.f32le':[51],
              **{role+'/'+name:shape for role in ROLES for name,shape in
                 [('noise-global.f32le',[1,frames,64]),('condition.f32le',[1,127,2048]),('condition-mask.f32le',[1,127])]}}
    provenance = {'provenance/source-report.json','provenance/context-report.json','provenance/conditioning-report.json'}
    hashes = ref.get('artifact_sha256',{})
    if ref.get('tensor_shapes') != shapes or set(hashes) != set(shapes)|provenance:
        raise ValueError('Incomplete full-song input manifest')
    arrays = {}
    for name in hashes:
        path = folder/name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()): raise ValueError('Linked/external job input rejected')
        if name in shapes: arrays[name] = mapped(path,shapes[name],hashes[name])
        elif sha256(path) != hashes[name]: raise ValueError('Provenance report differs')
    for role in ROLES:
        if not (arrays[role+'/condition-mask.f32le']==1).all(): raise ValueError('Invalid condition mask')
    source=json.loads((folder/'provenance/source-report.json').read_text())
    context=json.loads((folder/'provenance/context-report.json').read_text())
    condition=json.loads((folder/'provenance/conditioning-report.json').read_text())
    if (source.get('status')!='source_and_schedule_prepared' or source.get('source32_frames')!=plan['frames32']
            or source.get('source_gain')!=.7 or source.get('baseline_modified') is not False
            or context.get('status')!='host_source_context_captured' or context.get('source32_wav_sha256')!=source.get('source32_wav_sha256')
            or condition.get('status')!='host_conditioning_captured' or condition.get('geometry')!={'text':111,'lyric':15,'packed':127}):
        raise ValueError('Source/context/conditioning provenance does not match')
    for name,original in [('piano48.f32le','audio48-model.f32le'),('source-context.f32le','context_latents.f32le'),('source-epsilon.f32le','epsilon-global.f32le')]:
        if hashes[name]!=context.get('artifact_sha256',{}).get(original):raise ValueError('Copied context differs from captured provenance')
    for role in ROLES:
        if condition.get('prompts',{}).get(role,{}).get('preset_sha256')!=PRESETS[role]:raise ValueError('Instrument preset provenance differs')
        for name in ('condition.f32le','condition-mask.f32le'):
            if hashes[role+'/'+name]!=condition.get('artifact_sha256',{}).get(role+'/'+name):raise ValueError('Copied conditioning differs from host capture')
    from coreai_drum_reference30_contract import CAPTURE_REPORT_SHA256
    original=ROOT/'reviews/coreai-drums-host30s-listening-accepted/report.json'
    if sha256(original)!=CAPTURE_REPORT_SHA256 or hashes['timesteps.f32le']!=json.loads(original.read_text())['artifact_sha256']['timesteps.f32le']:
        raise ValueError('Original captured timesteps changed')
    times = arrays['timesteps.f32le']
    if times[0] != 1 or times[-1] != 0 or not (times[:-1] > times[1:]).all(): raise ValueError('Invalid original50-step schedule')
    if not (arrays['source-context.f32le'][...,64:]==1).all(): raise ValueError('Global source mask changed')
    records = window_records(plan,arrays['source-context.f32le'],arrays['piano48.f32le'],arrays['source-epsilon.f32le'],
                             {r:arrays[r+'/noise-global.f32le'] for r in ROLES},{r:hashes[r+'/condition.f32le'] for r in ROLES})
    if ref.get('runs') != records or len(records) != plan['proposed_calls']['generation_runs']:
        raise ValueError('Window input identities differ or runs missing')
    if any(np.array_equal(arrays[a+'/noise-global.f32le'],arrays[b+'/noise-global.f32le'])
           for a,b in [('bass','drums'),('bass','guitar'),('drums','guitar')]): raise ValueError('Each instrument requires its own saved global noise')
    return ref


def build_job(prepared,context,conditioning,capture,out):
    import numpy as np
    import torch
    from prepare_coreai_instrument_preview30 import verify_host
    from coreai_instrument_preview30_contract import PRESETS
    from coreai_drum_reference30_contract import CAPTURE_REPORT_SHA256
    if out.exists() or out.is_symlink(): raise FileExistsError('Existing full-song job preserved: '+str(out))
    if any(out.resolve().is_relative_to(p.resolve()) for p in (prepared,context,conditioning,capture,ROOT/'baselines',ROOT/'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Use inactive output outside inputs/models/baselines')
    source,plan = verify_source(prepared)
    context_report = verify_context(context,source,plan)
    cond = verify_host(conditioning)
    original_report = capture/'report.json'
    if sha256(original_report) != CAPTURE_REPORT_SHA256: raise ValueError('Original timestep capture report changed')
    capture_report = json.loads(original_report.read_text())
    mapped(capture/'timesteps.f32le',[51],capture_report['artifact_sha256']['timesteps.f32le'])
    out.mkdir(parents=True)
    ref = {'schema_version':1,'profile':PROFILE,'status':'preparing','frames32':plan['frames32'],
           'instruments':list(ROLES),'window_frames':750,'condition_tokens':127,'target_samples48':plan['target_samples48'],
           'latent_frames':plan['latent_frames'],'windows':plan['windows'],'preset_sha256':PRESETS,
           'sampler_weight_storage_dtype':'bfloat16','compute_dtype':'float32','vae_dtype':'float32',
           'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
           'inference_steps':50,'guidance_scale':7.,'apg_momentum':-.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
           'baseline_modified':False,'native_exports':0,'profile_activated':False,'ready_for_native_execution':False,
           'source_gain_already_applied':.7,'additional_source_gain':1.,
           'noise_policy':'Actual saved global CPU Float32 arrays; overlap uses identical slices; seeds informational only',
           'noise_cpu_seeds':{role:seed for seed,role in enumerate(ROLES)},'noise_generator_torch_version':torch.__version__,
           'assembly_policy':'Float64 normalized cosine overlap-add on raw outputs; crop to exact target length',
           'source_frontend':'Host-prepared bounded VAE context; full-song native source encoder NOT EVALUATED',
           'full_song_numerical_parity':'NOT EVALUATED; new global inputs are not the frozen30s input',
           'full_song_music_acceptance':'NOT EVALUATED','native_batch_runtime':'NOT EVALUATED',
           'original_sampler_gate':'NOT PASSED4/6 retained independently','artifact_sha256':{},'tensor_shapes':{}}
    def save():
        temporary=out/'reference.json.tmp';temporary.write_text(json.dumps(ref,indent=2)+'\n');temporary.replace(out/'reference.json')
    def copy(source,name,shape=None):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
        ref['artifact_sha256'][name]=sha256(path)
        if shape is not None:ref['tensor_shapes'][name]=shape
    save()
    try:
        frames=plan['latent_frames']
        copy(context/'audio48-model.f32le','piano48.f32le',[1,2,frames*1920])
        copy(context/'context_latents.f32le','source-context.f32le',[1,frames,128])
        copy(context/'epsilon-global.f32le','source-epsilon.f32le',[1,64,frames])
        copy(capture/'timesteps.f32le','timesteps.f32le',[51])
        copy(prepared/'report.json','provenance/source-report.json')
        copy(context/'report.json','provenance/context-report.json')
        copy(conditioning/'report.json','provenance/conditioning-report.json')
        for seed,role in enumerate(ROLES):
            copy(conditioning/role/'condition.f32le',role+'/condition.f32le',[1,127,2048])
            copy(conditioning/role/'condition-mask.f32le',role+'/condition-mask.f32le',[1,127])
            name=role+'/noise-global.f32le'
            noise=torch.randn((1,frames,64),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
            noise.numpy().astype('<f4').tofile(out/name)
            ref['artifact_sha256'][name]=sha256(out/name);ref['tensor_shapes'][name]=[1,frames,64]
        def read(name):return np.memmap(out/name,dtype='<f4',mode='r',shape=tuple(ref['tensor_shapes'][name]))
        ref['runs']=window_records(plan,read('source-context.f32le'),read('piano48.f32le'),read('source-epsilon.f32le'),
                                   {r:read(r+'/noise-global.f32le') for r in ROLES},{r:ref['artifact_sha256'][r+'/condition.f32le'] for r in ROLES})
        ref['status']='prepared_inactive_inputs';save();verify_job(out)
    except BaseException as error:
        ref.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    print('Prepared global inputs:',len(ref['runs']),'proposed30s generations; native execution NOT EVALUATED:',out,flush=True)
    return ref


def execute(args):
    if args.out.exists():
        if not args.reuse_verified:raise FileExistsError('Existing output preserved; use --reuse-verified or a new --out')
        result=verify_job(args.out)
        if args.source is not None:
            old=json.loads((args.out/'provenance/source-report.json').read_text())
            if sha256(args.source)!=old['source_sha256']:raise ValueError('Selected source differs from saved job; preserved')
        print('Verified existing full-song inputs; no inference or RNG regeneration:',args.out);return result
    if args.prepared_source.exists():
        if not args.reuse_verified:raise FileExistsError('Prepared source exists; use --reuse-verified to verify/reuse it')
        source,plan=verify_source(args.prepared_source)
        if args.source is not None and sha256(args.source)!=source['source_sha256']:raise ValueError('Selected source differs; preserved')
    else:
        source=args.source
        if args.choose_source:
            if sys.platform!='darwin':raise ValueError('--choose-source requires macOS; use --source elsewhere')
            choice=subprocess.run(['osascript','-e','POSIX path of (choose file with prompt "Pilih audio piano asli untuk lagu penuh")'],text=True,capture_output=True)
            if choice.returncode:raise ValueError('Source selection canceled; no preparation created')
            source=Path(choice.stdout.rstrip('\r\n'))
        if source is None:raise ValueError('Select source with --source or --choose-source')
        from prepare_coreai_full_song import prepare
        prepare(source,args.prepared_source)
        verify_source(args.prepared_source)
    if not args.context.exists():
        from prepare_coreai_full_song_context import execute as encode
        encode(SimpleNamespace(out=args.context,prepared_source=args.prepared_source,source_audit=args.source_audit,
                               source_bundle=args.source_bundle,vae_checkpoint=args.vae_checkpoint))
    elif not args.reuse_verified:raise FileExistsError('Context exists; use --reuse-verified to verify/reuse it')
    return build_job(args.prepared_source,args.context,args.conditioning,args.capture,args.out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--reuse-verified',action='store_true')
    group=parser.add_mutually_exclusive_group();group.add_argument('--source',type=Path);group.add_argument('--choose-source',action='store_true')
    parser.add_argument('--prepared-source',type=Path,default=ROOT/'.runtime-cache/full-song-source-v1')
    parser.add_argument('--context',type=Path,default=ROOT/'.runtime-cache/full-song-context-v1')
    parser.add_argument('--conditioning',type=Path,default=ROOT/'.runtime-cache/full-song-conditioning-v1')
    parser.add_argument('--capture',type=Path,default=ROOT/'.runtime-cache/drums-reference30-v3')
    parser.add_argument('--source-audit',type=Path,help='Optional explicit standalone audit; default reuses the verified prepared source bundle')
    parser.add_argument('--source-bundle',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/source-reference30-window-encoder-256')
    parser.add_argument('--vae-checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'.runtime-cache/full-song-inputs-v1')
    args=parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: verified full source/context/conditioning, actual global per-role noise and window hashes; zero generation/export/activation. Native execution requires separate activation and is NOT EVALUATED.')
        return
    result=execute(args)
    print('Input windows:',len(result['windows']),'per instrument; full-song native execution NOT EVALUATED')


if __name__=='__main__':main()

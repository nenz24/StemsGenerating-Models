#!/usr/bin/env python3
"""Capture bounded host source context for a prepared full song; no DiT/Qwen/native export.

Requires strict source-window proof first. Actual epsilon arrays are saved once;
every window uses a slice of that global array. Default is plan-only.
"""
import argparse
import json
import math
from pathlib import Path
import wave

from export_coreai_decoder import ROOT,sha256


def source_proof(path, bundle=None):
    """Reuse the verified prepared source bundle if no standalone audit exists."""
    if path is not None:
        path=Path(path)
        if path.is_symlink() or (path.exists() and not path.is_file()):raise ValueError('Existing source audit must be a regular JSON file')
        if path.is_file():return json.loads(path.read_text()),path
    if bundle is None:raise FileNotFoundError('Existing source audit or prepared source window bundle required; no repeated audit')
    from prepare_coreai_source_window import verify_bundle
    from coreai_source_window_contract import PROFILE
    bundle=Path(bundle)
    ref=verify_bundle(bundle)
    proof={'profile':PROFILE,'capture_report_sha256':ref['capture_report_sha256'],
           'strict_verdict':'PASS','comparisons_passed':32,'comparisons_target':32,
           'window_frames':ref['window_frames'],'vae_checkpoint_sha256':ref['vae_checkpoint_sha256'],
           'reference_convolution_backend':ref['reference_convolution_backend'],
           'window_plan':ref['window_plan'],'checks':ref['host_checks']}
    return proof,bundle/'reference.json'


def execute(args):
    import numpy as np
    import torch
    from audit_coreai_source_window import load_encoder,encode_windows
    from compare_coreai_drum_lyrics import local_checkpoint
    from coreai_source_window_contract import VAE_SHA,PROFILE,ACCEPTED_REPORT_SHA256,tile_plan,dependency_bounds,validate_host_checks
    from generate_coreai_drum_reference import resample
    if args.out.exists():raise FileExistsError('Existing full-song context preserved')
    if any(args.out.resolve().is_relative_to(p) for p in (args.prepared_source.resolve(),ROOT/'baselines',ROOT/'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Use an inactive context folder outside source/models/baselines')
    proof,proof_path=source_proof(args.source_audit,getattr(args,'source_bundle',ROOT/'prototypes/ios27-coreai/ModelCandidates/source-reference30-window-encoder-256'))
    if (proof.get('profile')!=PROFILE or proof.get('capture_report_sha256')!=ACCEPTED_REPORT_SHA256
            or proof.get('strict_verdict')!='PASS' or proof.get('comparisons_passed')!=32
            or proof.get('comparisons_target')!=32 or proof.get('window_frames')!=256
            or proof.get('vae_checkpoint_sha256')!=VAE_SHA or proof.get('reference_convolution_backend')!='aten'
            or proof.get('window_plan')!=tile_plan(750)):
        raise ValueError('Original32 source window gates and explicit CPU ATen proof required')
    validate_host_checks(proof.get('checks',{}))
    source=args.prepared_source/'source32.wav';prepared=json.loads((args.prepared_source/'report.json').read_text())
    if prepared.get('status')!='source_and_schedule_prepared' or prepared.get('source_gain')!=.7 or sha256(source)!=prepared['source32_wav_sha256']:
        raise ValueError('Prepared full-song source identity mismatch')
    with wave.open(str(source),'rb') as wav:
        if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(2,2,32000):raise ValueError('Expected stereo32kHz PCM16')
        pcm=wav.readframes(wav.getnframes())
    import hashlib
    if hashlib.sha256(pcm).hexdigest()!=prepared['source32_pcm_sha256']:raise ValueError('Prepared PCM hash mismatch')
    record=next(r for r in json.loads((ROOT/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    if record['digest']!=VAE_SHA:raise ValueError('Pinned VAE changed')
    checkpoint=local_checkpoint(record,args.vae_checkpoint)
    torch.set_num_threads(2)
    print('Resample full PCM once on global timeline; no second source gain',flush=True)
    audio=resample(pcm);del pcm
    samples=audio.shape[-1];frames=(samples+1919)//1920
    padding=frames*1920-samples
    audio=torch.nn.functional.pad(audio,(0,padding)).contiguous()
    tiles=tile_plan(frames)
    bounds=[dependency_bounds(t,frames) for t in tiles]
    epsilon=torch.randn((1,64,frames),generator=torch.Generator(device='cpu').manual_seed(31),dtype=torch.float32)
    mask=torch.ones(1,frames,64,dtype=torch.float32)
    args.out.mkdir(parents=True)
    report={'schema_version':1,'status':'source_encoding','scope':'Host bounded source capture only; no full-song generation/native acceptance',
            'source32_wav_sha256':sha256(source),'source_audit_sha256':sha256(proof_path),'vae_checkpoint_sha256':VAE_SHA,
            'source_gain_already_applied':.7,'additional_gain':1.,'sample_rate':48000,'samples':samples,'latent_frames':frames,
            'global_right_padding_samples':padding,'epsilon_shape':[1,64,frames],
            'epsilon_provenance':'Actual saved CPU Float32 array; seed31 informational, no cross-platform RNG identity claim',
            'window_plan':tiles,'dependency_sample_bounds':bounds,'window_frames':256,'core_frames':224,'halo_frames':16,
            'source_encoder_calls':0,'expected_source_encoder_calls':len(tiles),'reference_convolution_backend':'aten',
            'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
            'DiT_calls':0,'Qwen_calls':0,'native_exports':0,'baseline_modified':False,'ready_for_native_execution':False,
            'full_song_numerical_parity':'NOT EVALUATED; no monolithic full-song or native reference run',
            'original_source_proof_comparisons_passed':32,'artifact_sha256':{}}
    def save():
        temp=args.out/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out/'report.json')
    def write(name,value):
        value.numpy().astype('<f4').tofile(args.out/name);report['artifact_sha256'][name]=sha256(args.out/name)
    write('audio48-model.f32le',audio);write('epsilon-global.f32le',epsilon);write('source-mask.f32le',mask);save()
    try:
        encoder=load_encoder(checkpoint)
        def progress(i,total,seconds):
            report['source_encoder_calls']=i;save();print(f'Full song source{i}/{total}; seconds {seconds:.3f}',flush=True)
        with torch.backends.mkldnn.flags(enabled=False):
            result=encode_windows(encoder,audio,epsilon,mask,progress)
        if not torch.equal(result['context_latents'][...,64:],mask):raise ValueError('Source mask changed')
        for name,value in result.items():write(name+'.f32le',value)
        report.update(status='host_source_context_captured',mask_preserved_exact=True)
        save();print('Full-song host context captured; no native readiness or full-song parity claim:',args.out,flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--prepared-source',type=Path,default=ROOT/'.runtime-cache/full-song-source-v1')
    parser.add_argument('--source-audit',type=Path,default=ROOT/'.runtime-cache/source-window30-host-v1/report.json')
    parser.add_argument('--source-bundle',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/source-reference30-window-encoder-256')
    parser.add_argument('--vae-checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'.runtime-cache/full-song-context-v1')
    args=parser.parse_args()
    if not args.execute:print('PLAN ONLY: verify32 source gates, global resample/epsilon capture, bounded FP32 source encoding. No DiT/Qwen/native export.');return
    execute(args)


if __name__=='__main__':main()

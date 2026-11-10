#!/usr/bin/env python3
"""Replay saved native velocity outputs through CPU APG; no DiT/VAE/weights.

Velocities remain frozen to the actual native trajectory. This isolates APG
arithmetic; it is not a new independent DiT trajectory or baseline replacement.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replay_trace(velocities, noise, times, publisher=None):
    import torch
    from export_coreai_bass_preview import apg_euler
    if (len(velocities) != 50 or list(noise.shape) != [1,750,64]
            or list(times.shape) != [51] or times[0] != 1 or times[-1] != 0
            or not (times[:-1] > times[1:]).all()
            or any(v.dtype != torch.float32 or list(v.shape) != [2,750,64] or not torch.isfinite(v).all() for v in velocities)
            or noise.dtype != torch.float32 or times.dtype != torch.float32
            or not torch.isfinite(noise).all() or not torch.isfinite(times).all()):
        raise ValueError('Invalid complete50-step fixed velocity trace or original inputs')
    latent, momentum = noise.clone(), torch.zeros_like(noise)
    buffer = publisher.MomentumBuffer() if publisher else None
    checkpoints = {}
    with torch.no_grad():
        for i, velocity in enumerate(velocities):
            if publisher is None:
                latent, momentum, guided = apg_euler(velocity, latent, momentum, times[i:i+1]-times[i+1:i+2])
            else:
                conditional, unconditional = velocity.chunk(2)
                guided = publisher.apg_forward(conditional, unconditional, 7., momentum_buffer=buffer,
                                               norm_threshold=2.5, dims=[1])
                latent = latent-guided*(times[i]-times[i+1])
                momentum = buffer.running_average
            if any(not torch.isfinite(v).all() for v in (latent, momentum, guided)):
                raise ValueError(f'Nonfinite APG trace replay at step{i+1}')
            if i+1 in (1,5,10,25,50):
                checkpoints[i+1] = latent.clone()
    return checkpoints, momentum


def audit(args):
    import torch
    from audit_coreai_drum_reference30 import verify_capture, read_tensor
    from export_coreai_decoder import sha256
    from export_coreai_dit_block import precision_metrics
    from package_coreai_drum_reference30_run import VELOCITY_SHAPES
    from replay_coreai_drum_reference30_sampler import native_capture, replay_verdicts
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Existing APG audit preserved')
    if (args.out.resolve().is_relative_to(args.capture.resolve())
            or args.out.resolve().is_relative_to(ROOT/'prototypes/ios27-coreai/ModelCandidates')
            or args.out.resolve().is_relative_to(ROOT/'baselines')):
        raise ValueError('Audit output must be outside captures/models/baselines')
    torch.set_num_threads(2)
    captured = verify_capture(args.capture)
    native, manifest = native_capture(args.archive)
    if manifest.get('velocity_trace_steps') != 50:
        raise ValueError('This run has no complete velocity trace. Historical states cannot reconstruct missing velocities.')
    publisher_path = args.publisher_dir/'apg_guidance.py'
    if not publisher_path.is_file() or publisher_path.is_symlink() or sha256(publisher_path) != captured['publisher_source_sha256']['apg_guidance.py']:
        raise ValueError('Original pinned local APG source required; no download')
    spec = importlib.util.spec_from_file_location('coreai_pinned_trace_apg', publisher_path)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    velocities = [native[name] for name in VELOCITY_SHAPES]
    noise, times = read_tensor(args.capture,'initial-noise.f32le'), read_tensor(args.capture,'timesteps.f32le')
    result = {'schema_version':1, 'profile':'fixed-native-velocity-apg-cpu-replay',
              'native_archive_sha256':sha256(args.archive), 'native_report_sha256':manifest['report_sha256'],
              'capture_report_sha256':sha256(args.capture/'report.json'), 'velocity_trace_sha256':manifest['velocity_trace_sha256'],
              'native_run_id':manifest['run_id'], 'baseline_modified':False,
              'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
              'DiT_calls':0,'decoder_calls':0,'native_calls':0,'native_exports':0,
              'scope':'50 fixed actual native velocity tensors; CPU APG carries its own latent/momentum. No independent DiT feedback trajectory.',
              'modes':{}}
    for label, original in [('exported_fp32_formula',None),('original_publisher_fp64_projection',publisher)]:
        checkpoints, momentum = replay_trace(velocities, noise, times, original)
        checks = {str(step):precision_metrics(value, native[f'native-step-{step:02d}-latent.f32le'])
                  for step,value in checkpoints.items()}
        final = precision_metrics(momentum,native['final-momentum.f32le'])
        result['modes'][label] = {'steps':50,'latent_checkpoints_vs_native':checks,'final_momentum_vs_native':final,
                                 'acceptance':replay_verdicts(checks,final)}
        print(label,json.dumps(result['modes'][label]['acceptance']),flush=True)
    result['full_native_trajectory_parity'] = 'NOT EVALUATED; original native numerical failures unchanged'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print('APG replay complete:',args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--capture',type=Path,default=ROOT/'.runtime-cache/drums-reference30-v3')
    parser.add_argument('--publisher-dir',type=Path,default=ROOT/'.runtime-cache/coreai-generator-source-publisher')
    parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args())


if __name__ == '__main__':
    main()

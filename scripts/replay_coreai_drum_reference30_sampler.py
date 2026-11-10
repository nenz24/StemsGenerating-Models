#!/usr/bin/env python3
"""Replay the frozen BF16-storage/FP32-compute sampler on CPU, without native export.

Default is a read-only plan. --execute requires existing pinned local weights,
accepted host captures and an exported native run ZIP. No source/Qwen/RNG/VAE.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEPS = (1, 5, 10, 25, 50)


def replay_verdicts(latent_checks, momentum_check):
    """Sparse latent acceptance cannot imply momentum or decoded-audio acceptance."""
    def passes(metric):
        return (metric.get('finite') is True and metric.get('parity_passed') is True
                and metric.get('elementwise_fraction_over_bound') == 0
                and isinstance(metric.get('normalized_rmse'), (int, float))
                and 0 <= metric['normalized_rmse'] <= 1e-3)
    latent_ok = set(latent_checks) == {str(s) for s in STEPS} and all(passes(m) for m in latent_checks.values())
    momentum_ok = passes(momentum_check)
    return {'latent_checkpoint_verdict': 'PASS' if latent_ok else 'NOT PASSED',
            'final_momentum_verdict': 'PASS' if momentum_ok else 'NOT PASSED',
            'sampled_state_verdict': 'PASS' if latent_ok and momentum_ok else 'NOT PASSED',
            'waveform_verdict': 'NOT EVALUATED',
            'scope': 'Five sparse latent checkpoints plus final momentum only; not every intermediate or decoded audio'}


def native_capture(archive_path):
    import numpy as np
    import torch
    from package_coreai_drum_reference30_run import SHAPES, VELOCITY_SHAPES, validate_velocity_trace
    path = Path(archive_path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('Native archive exceeds32MiB')
    with zipfile.ZipFile(path) as archive:
        for name in ('report.txt', 'run-manifest.json'):
            if archive.getinfo(name).file_size > 1024 * 1024:
                raise ValueError('Native metadata exceeds1MiB')
        manifest = json.loads(archive.read('run-manifest.json'))
        raw_report = archive.read('report.txt')
        shapes = dict(SHAPES)
        trace_steps = manifest.get('velocity_trace_steps', 0)
        if trace_steps not in (0, 50):
            raise ValueError('Incomplete native velocity trace')
        if trace_steps == 50:
            shapes.update(VELOCITY_SHAPES)
        names = set(shapes) | {'report.txt', 'run-manifest.json'}
        if trace_steps:
            names.add('velocity-trace.json')
        if set(archive.namelist()) != names or len(archive.infolist()) != len(names):
            raise ValueError('Unexpected/duplicated native archive entries')
        if (manifest.get('schema_version') != 1 or manifest.get('profile') != 'private-coreai-frozen30-live-run'
                or set(manifest.get('artifacts', {})) != set(shapes)
                or hashlib.sha256(raw_report).hexdigest() != manifest.get('report_sha256')):
            raise ValueError('Native archive manifest/report mismatch')
        import uuid
        run_id = str(uuid.UUID(manifest['run_id'])).upper()
        report = raw_report.decode('utf-8')
        for line in (f'Runtime: CoreAI frozen drums30 generation; run {run_id}',
                     'Reference30 execution: COMPLETED; steps50/50; sampling calls750/750; decoder calls6/6'):
            if line not in report:
                raise ValueError('Not a matching completed native frozen30 run')
        revision = 2 if trace_steps else 1
        if f'Revision: reference30-live50-window256-v{revision};' not in report:
            raise ValueError('Native revision/trace mismatch')
        trace = None
        if trace_steps:
            if archive.getinfo('velocity-trace.json').file_size > 1024 * 1024:
                raise ValueError('Velocity metadata exceeds1MiB')
            raw = archive.read('velocity-trace.json')
            if hashlib.sha256(raw).hexdigest() != manifest.get('velocity_trace_sha256'):
                raise ValueError('Velocity metadata checksum mismatch')
            trace = validate_velocity_trace(raw, report, run_id)
        values = {}
        for name, shape in shapes.items():
            entry = manifest['artifacts'][name]
            if (entry.get('shape') != shape or entry.get('dtype') != 'float32_le'
                    or entry.get('bytes') != math.prod(shape)*4
                    or archive.getinfo(name).file_size != entry['bytes']):
                raise ValueError('Native tensor shape/length mismatch: ' + name)
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Native tensor checksum mismatch: ' + name)
            if trace and name in VELOCITY_SHAPES and entry['sha256'] != trace['artifact_sha256'][name]:
                raise ValueError('Velocity tensor/device metadata mismatch')
            array = np.frombuffer(raw, dtype='<f4').copy().reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite native tensor: ' + name)
            values[name] = torch.from_numpy(array)
        final = manifest['artifacts']['final-latent.f32le']['sha256']
        if (f'LIVE final latent SHA256: {final}' not in report
                or not torch.equal(values['final-latent.f32le'], values['native-step-50-latent.f32le'])):
            raise ValueError('Native final/checkpoint50/report mismatch')
    return values, manifest


def live_replay(predict, noise, times, checkpoint, progress):
    """One independent50-step loop; references are never loop inputs."""
    import torch
    from export_coreai_bass_preview import apg_euler
    if (list(noise.shape) != [1, 750, 64] or noise.dtype != torch.float32
            or list(times.shape) != [51] or times.dtype != torch.float32
            or not torch.isfinite(noise).all() or not torch.isfinite(times).all()
            or times[0] != 1 or times[-1] != 0 or not (times[:-1] > times[1:]).all()):
        raise ValueError('Invalid original50-step input geometry/schedule')
    latent, momentum = noise.clone(), torch.zeros_like(noise)
    with torch.no_grad():
        for i in range(50):
            begin = time.monotonic()
            velocity = predict(latent, times[i:i+1])
            if list(velocity.shape) != [2, 750, 64] or velocity.dtype != torch.float32 or not torch.isfinite(velocity).all():
                raise ValueError(f'Invalid velocity at step{i+1}')
            latent, momentum, guided = apg_euler(velocity, latent, momentum, times[i:i+1]-times[i+1:i+2])
            if any(not torch.isfinite(v).all() for v in (latent, momentum, guided)):
                raise ValueError(f'Nonfinite live feedback at step{i+1}')
            # Detached clones prevent diagnostic observers modifying live state.
            if i+1 in STEPS:
                checkpoint(i+1, latent.detach().clone(), momentum.detach().clone())
            progress(i+1, time.monotonic()-begin)
    return latent, momentum


def execute(args):
    import torch
    from safetensors import safe_open
    from audit_coreai_drum_reference30 import verify_capture, read_tensor
    from compare_coreai_drum_lyrics import local_checkpoint
    from export_coreai_decoder import sha256
    from export_coreai_denoiser import load_full_upstream
    from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
    from prepare_coreai_drum_reference30 import WINDOW, plan, partitions, run_partitions
    import sys
    torch.set_num_threads(2)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Existing sampler replay preserved: ' + str(out))
    protected = [args.capture.resolve(), ROOT/'baselines', ROOT/'prototypes/ios27-coreai/ModelCandidates',
                 args.publisher_dir.resolve()]
    if any(out.is_relative_to(p) for p in protected):
        raise ValueError('Choose a new inactive output outside captures/models/publisher sources')
    captured = verify_capture(args.capture)
    native, native_manifest = native_capture(args.native_archive)
    publisher = records()
    base_record = publisher['model.safetensors']
    if base_record['digest'] != captured['checkpoint_sha256']['base']:
        raise ValueError('Accepted/base checkpoint identity differs')
    paths = {n: args.publisher_dir/n for n in
             ('modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json')}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError('Existing publisher source required; no download: ' + name)
        verified_source(publisher[name], args.publisher_dir)
        if sha256(path) != captured['publisher_source_sha256'][name]:
            raise ValueError('Accepted publisher identity differs: ' + name)
    base = local_checkpoint(base_record, args.base_checkpoint)
    cfg, kind, rotary = load_full_upstream(paths)
    if cfg.patch_size != 2 or cfg.num_hidden_layers != 24:
        raise ValueError('Original sampler architecture differs')
    out.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'loading', 'profile': 'frozen30-cpu-partitioned-sampler-replay',
              'base_checkpoint_sha256': base_record['digest'],
              'capture_report_sha256': sha256(args.capture/'report.json'),
              'native_archive_sha256': sha256(args.native_archive), 'native_run_id': native_manifest['run_id'],
              'native_report_sha256': native_manifest['report_sha256'],
              'weight_storage_dtype': 'bfloat16', 'compute_dtype': 'float32',
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'scope': 'Independent CPU partitioned50-step trajectory; original FP32 APG export formula; sparse comparisons only',
              'gold_feedback_resets': 0, 'source_encoder_calls': 0, 'Qwen_calls': 0, 'decoder_calls': 0,
              'native_exports': 0, 'native_calls': 0, 'baseline_modified': False,
              'steps_completed': 0, 'partition_calls': 0, 'host_vs_gold': {}, 'host_vs_native': {},
              'step_seconds': [], 'torch': torch.__version__}
    def save():
        temp = out/'report.json.tmp'
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(out/'report.json')
    save()
    try:
        print('Load verified local decoder weights; no Qwen/source/VAE/native export', flush=True)
        with torch.device('meta'):
            model = kind(cfg).eval()
        with safe_open(str(base), framework='pt', device='cpu') as stream:
            state = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
            null = stream.get_tensor('null_condition_emb').float()
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model.rotary_emb = rotary(cfg, device='cpu')
        model.to(device='cpu', dtype=torch.float32).requires_grad_(False)
        convert_storage16(model, 'bfloat16')
        parts = partitions(model, null, sys.modules[kind.__module__].create_4d_mask, WINDOW)
        stages = plan()['stages']
        condition, context = read_tensor(args.capture, 'condition.f32le'), read_tensor(args.capture, 'source-context.f32le')
        def predict(latent, timestep):
            result = run_partitions(parts, stages, {'noise': latent, 'condition': condition,
                                                   'context': context, 'timestep': timestep})
            report['partition_calls'] += len(parts)
            return result
        def checkpoint(step, latent, momentum):
            report['host_vs_gold'][str(step)] = precision_metrics(latent, read_tensor(args.capture, f'step-{step:02d}-latent.f32le'))
            report['host_vs_native'][str(step)] = precision_metrics(latent, native[f'native-step-{step:02d}-latent.f32le'])
            latent.numpy().astype('<f4').tofile(out/f'host-step-{step:02d}-latent.f32le')
            if step == 50:
                report['final_momentum_vs_gold'] = precision_metrics(momentum, read_tensor(args.capture, 'step-50-momentum.f32le'))
                report['final_momentum_vs_native'] = precision_metrics(momentum, native['final-momentum.f32le'])
                momentum.numpy().astype('<f4').tofile(out/'host-final-momentum.f32le')
        def progress(step, seconds):
            report.update(status='running', steps_completed=step)
            report['step_seconds'].append(seconds)
            if step in STEPS or step % 5 == 0:
                save()
            print(f'CPU sampler {step}/50; seconds {seconds:.3f}', flush=True)
        live_replay(predict, read_tensor(args.capture, 'initial-noise.f32le'),
                    read_tensor(args.capture, 'timesteps.f32le'), checkpoint, progress)
        if report['partition_calls'] != 700:
            raise ValueError('Incomplete14-partition50-step execution')
        report['status'] = 'completed'
        for key in ('host_vs_gold', 'host_vs_native'):
            report[key+'_strict_verdict'] = 'PASS' if all(m['parity_passed'] for m in report[key].values()) else 'NOT PASSED'
            report[key+'_acceptance'] = replay_verdicts(report[key], report['final_momentum_vs_'+key.removeprefix('host_vs_')])
        report['legacy_strict_verdict_scope'] = 'host_vs_*_strict_verdict covers only the five latent checkpoints; see *_acceptance for momentum. Waveform not evaluated.'
        report['artifact_sha256'] = {p.name: sha256(p) for p in out.glob('*.f32le')}
        report['full_native_trajectory_parity'] = 'NOT EVALUATED; CPU replay does not accept native intermediates'
        save()
        # Small private review: six host tensors + report, no model/source/audio.
        with tempfile.TemporaryFile() as temp:
            with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ['report.json', *report['artifact_sha256']]:
                    archive.write(out/name, name)
            temp.seek(0)
            with (out/'sampler-replay-review.zip').open('xb') as destination:
                import shutil
                shutil.copyfileobj(temp, destination)
        print('Complete. Sparse latent verdicts:', report['host_vs_gold_strict_verdict'],
              '(gold),', report['host_vs_native_strict_verdict'], '(native).', flush=True)
        print('Final momentum verdicts:', report['host_vs_gold_acceptance']['final_momentum_verdict'],
              '(gold),', report['host_vs_native_acceptance']['final_momentum_verdict'], '(native). No waveform test.', flush=True)
        print('Private review ZIP:', out/'sampler-replay-review.zip', flush=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--capture', type=Path, default=ROOT/'.runtime-cache/drums-reference30-v3')
    parser.add_argument('--native-archive', type=Path, default=ROOT/'.runtime-cache/drums-native30-live-review-v1.zip')
    parser.add_argument('--publisher-dir', type=Path, default=ROOT/'.runtime-cache/coreai-generator-source-publisher')
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'.runtime-cache/drums-reference30-sampler-replay-v1')
    args = parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: existing frozen inputs; CPU BF16-storage/FP32-compute14-partition sampler,50 live steps.')
        print('Compare checkpoints1/5/10/25/50 to original and saved native; original tolerances unchanged.')
        print('No downloads, source/Qwen/VAE, native export, profile activation or baseline changes. Use --execute to run.')
        return
    execute(args)


if __name__ == '__main__':
    main()

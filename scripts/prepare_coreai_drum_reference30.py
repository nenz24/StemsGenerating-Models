#!/usr/bin/env python3
"""Prepare an inactive 750-frame native diagnostic from the accepted host capture.

Exports partitioned DiT + APG + decoder only after strict same-input host gates.
Uses frozen conditioning/source/noise; not arbitrary-import or full-song support.
Default prints a plan without loading checkpoints or writing files.
"""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from audit_coreai_drum_reference30 import verify_capture, read_tensor
from export_coreai_decoder import ROOT, sha256
from export_coreai_bass_preview import stage_plan, partitions, apg_euler, bounded_comparison
from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
from package_coreai_drum_reference import ACCEPTED_REPORT_SHA256, SHAPES


class Reference30Window:
    """Separate fixed diagnostic contract; never registered with PreviewWindow."""
    frames = 750
    samples = 1440000
    seconds = 30.
    hidden_tokens = 375
    condition_tokens = 127
    weight_storage_dtype = 'bfloat16'


WINDOW = Reference30Window()
PROFILE = 'drums-reference30-frozen-diagnostic'


def plan():
    return {'schema_version': 1, 'profile': PROFILE, 'latent_frames': 750,
            'condition_tokens': 127, 'text_tokens': 111, 'lyric_tokens': 15,
            'sample_rate': 48000, 'audio_seconds': 30.,
            'weight_storage_dtype': 'bfloat16', 'compute_dtype': 'float32',
            'scalar_type': 'float32', 'byte_order': 'little', 'function_name': 'main',
            'VAE_dtype': 'float32', 'stages': stage_plan(WINDOW),
            'apg': {'asset_file': 'apg-euler.aimodel',
                    'input_shapes': {'velocity': [2, 750, 64], 'latent': [1, 750, 64],
                                     'momentum': [1, 750, 64], 'dt': [1]},
                    'output_shapes': {'latent_next': [1, 750, 64], 'momentum_next': [1, 750, 64],
                                      'guided_velocity': [1, 750, 64]}},
            'decoder': {'asset_file': 'decoder.aimodel', 'input_shapes': {'latent': [1, 64, 750]},
                        'output_shapes': {'audio': [1, 2, 1440000]}},
            'asset_count': 16, 'sampling_assets': 15, 'inference_steps': 50,
            'checkpoints': [1, 5, 10, 25, 50], 'guidance_scale': 7., 'shift': 1.,
            'method': 'euler_ode_apg', 'apg_momentum': -.75, 'apg_norm_threshold': 2.5,
            'apg_projection_dims': [1],
            'apg_projection_precision': 'float32_validated_against_original_float64',
            'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
            'activation': 'NONE; no selector, Xcode resources or baseline changed',
            'input_scope': 'Exact frozen accepted host inputs; no source/Qwen/conditioning export',
            'native_parity': 'NOT EVALUATED', 'full_trajectory_parity': 'NOT EVALUATED',
            'scope': 'Inactive native component diagnostic; not production generation or full-song support'}


def require_gate(report, name, actual, expected):
    if actual.shape != expected.shape:
        report['host_checks'][name] = {'parity_passed': False, 'error': 'shape mismatch'}
        raise ValueError('Same-input shape gate failed: ' + name)
    metrics = precision_metrics(actual, expected)
    report['host_checks'][name] = metrics
    print(name, metrics, flush=True)
    if not metrics['parity_passed']:
        raise ValueError('Original strict host gate failed: ' + name + '; no tolerance override')


def run_partitions(parts, stages, state):
    import torch
    state = dict(state)
    with torch.no_grad():
        for module, stage in zip(parts, stages):
            values = module(*(state[n] for n in stage['input_shapes']))
            values = values if isinstance(values, tuple) else (values,)
            if len(values) != len(stage['output_shapes']):
                raise ValueError('Partition output count mismatch')
            for (name, shape), value in zip(stage['output_shapes'].items(), values):
                if list(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid partition output: ' + name)
                state[stage['output_bindings'][name]] = value
    return state['velocity']


def execute(capture, out, report, progress, base_checkpoint=None, vae_checkpoint=None):
    # Fail for unavailable Apple SDK before loading the large learned model.
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Native export requires the existing Apple Silicon Mac CoreAI environment')
    import torch
    from coreai_torch import TorchConverter, get_decomp_table
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    from compare_coreai_drum_lyrics import local_checkpoint
    from export_coreai_denoiser import load_full_upstream
    torch.set_num_threads(2)
    publisher = records()
    vae = next(r for r in json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
               if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    progress('verify existing base and VAE checkpoints; no weight downloads')
    base = local_checkpoint(publisher['model.safetensors'], base_checkpoint)
    vae_path = local_checkpoint(vae, vae_checkpoint)
    captured = json.loads((capture / 'report.json').read_text())
    if (publisher['model.safetensors']['digest'] != captured['checkpoint_sha256']['base']
            or vae['digest'] != captured['checkpoint_sha256']['vae']):
        raise ValueError('Captured and export checkpoints differ')
    cache = ROOT / '.runtime-cache/coreai-generator-source-publisher'
    cache.mkdir(parents=True, exist_ok=True)
    paths = {n: verified_source(publisher[n], cache) for n in
             ('modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json')}
    for name, path in paths.items():
        if sha256(path) != captured['publisher_source_sha256'][name]:
            raise ValueError('Captured publisher source differs: ' + name)
    cfg, kind, rotary = load_full_upstream(paths)
    if cfg.patch_size != 2 or cfg.num_hidden_layers != 24:
        raise ValueError('Unsupported publisher architecture')
    make_mask = sys.modules[kind.__module__].create_4d_mask
    progress('load original host FP32 DiT for one same-input partition check')
    with torch.device('meta'):
        model = kind(cfg).eval()
    with safe_open(str(base), framework='pt', device='cpu') as stream:
        state = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
        null = stream.get_tensor('null_condition_emb').float()
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.rotary_emb = rotary(cfg, device='cpu')
    model.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    load = lambda name: read_tensor(capture, name + '.f32le')
    inputs = {'noise': load('initial-noise'), 'condition': load('condition'),
              'context': load('source-context'), 'timestep': load('timesteps')[:1]}
    expected = torch.cat([load('step-01-conditional'), load('step-01-unconditional')])
    parts = partitions(model, null, make_mask, WINDOW)
    report['DiT_partition_passes'] += 1
    require_gate(report, 'FP32_partitions_step1', run_partitions(parts, report['stages'], inputs), expected)
    # Wrappers reference the original modules; rebuild them after replacement.
    del parts
    convert_storage16(model, 'bfloat16')
    parts = partitions(model, null, make_mask, WINDOW)
    progress('strict same-input BF16-storage FP32-compute partition check')
    report['DiT_partition_passes'] += 1
    require_gate(report, 'BF16_partitions_step1', run_partitions(parts, report['stages'], inputs), expected)

    def export(module, values, input_names, output_names, filename):
        progress('export ' + filename)
        ep = torch.export.export(module.eval(), values).run_decompositions(get_decomp_table())
        converted = TorchConverter().add_exported_program(ep, input_names=input_names,
                                                         output_names=output_names).to_coreai()
        converted.save_asset(out / filename)
        report['native_exports'] += 1
        del converted, ep
        gc.collect()

    state = dict(inputs)
    report['DiT_partition_passes'] += 1
    with torch.no_grad():
        for part, stage in zip(parts, report['stages']):
            values = tuple(state[n] for n in stage['input_shapes'])
            outputs = part(*values)
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            export(part, values, list(stage['input_shapes']), list(stage['output_shapes']), stage['asset_file'])
            for name, value in zip(stage['output_shapes'], outputs):
                state[stage['output_bindings'][name]] = value
    # Release all DiT weights before full-window decoder allocation.
    del part, parts, model, state, outputs, values
    gc.collect()
    class Update(torch.nn.Module):
        def forward(self, velocity, latent, momentum, dt):
            return apg_euler(velocity, latent, momentum, dt)
    update = Update().eval()
    apg_values = (expected, inputs['noise'], torch.zeros_like(inputs['noise']),
                  load('timesteps')[:1] - load('timesteps')[1:2])
    with torch.no_grad():
        next_x, next_m, guided = update(*apg_values)
    for name, value, gold in [('APG_step1_latent', next_x, load('step-01-latent')),
                               ('APG_step1_momentum', next_m, load('step-01-momentum')),
                               ('APG_step1_guided', guided, load('step-01-guided'))]:
        require_gate(report, name, value, gold)
    export(update, apg_values, list(report['apg']['input_shapes']), list(report['apg']['output_shapes']), 'apg-euler.aimodel')
    cfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
    with torch.device('meta'):
        decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                                 audio_channels=cfg['audio_channels'], upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                                 channel_multiples=cfg['channel_multiples']).eval()
    with safe_open(str(vae_path), framework='pt', device='cpu') as stream:
        weights = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(weights, strict=True, assign=True)
    del weights
    decoder.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    for module in decoder.modules():
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            torch.nn.utils.remove_weight_norm(module)
    decoder_values = (load('final-latent').transpose(1, 2).contiguous(),)
    progress('strict folded FP32 decoder check against accepted final audio')
    with torch.no_grad():
        audio = decoder(*decoder_values)
    report['host_decoder_calls'] = 1
    require_gate(report, 'FP32_decoder_saved_final_latent', audio, load('final-audio'))
    export(decoder, decoder_values, ['latent'], ['audio'], 'decoder.aimodel')
    # Copy exact gold inputs/checkpoints for later independent native trajectory.
    target = out / 'capture'
    target.mkdir()
    for name in list(SHAPES) + ['report.json']:
        shutil.copyfile(capture / name, target / name)
    verify_capture(target)
    report['versions'] = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'diffusers', 'coreai-torch', 'coreai-core')}
    report['checkpoint_sha256'] = {'base': publisher['model.safetensors']['digest'], 'vae': vae['digest']}


def prepare(capture, out, base_checkpoint=None, vae_checkpoint=None):
    capture, out = Path(capture), Path(out)
    if out.exists():
        raise FileExistsError('Existing native diagnostic preserved: ' + str(out))
    if (out.resolve().is_relative_to(capture.resolve()) or any(out.resolve().is_relative_to(ROOT / p)
            for p in ('baselines', 'prototypes/ios27-coreai/ModelCandidates'))):
        raise ValueError('Use a new inactive cache directory outside capture/baselines/ModelCandidates')
    verify_capture(capture)
    report = {**plan(), 'status': 'preparing', 'accepted_report_sha256': ACCEPTED_REPORT_SHA256,
              'native_exports': 0, 'DiT_partition_passes': 0, 'host_decoder_calls': 0,
              'host_checks': {}, 'baseline_modified': False}
    out.mkdir(parents=True, exist_ok=False)
    def save():
        temporary = out / 'reference.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'reference.json')
    def progress(phase):
        report['phase'] = phase
        save()
        print(phase, flush=True)
    save()
    try:
        execute(capture, out, report, progress, base_checkpoint, vae_checkpoint)
        if report['native_exports'] != 16 or not report['host_checks']:
            raise ValueError('Incomplete native diagnostic export')
        report['status'] = 'prepared_inactive_component_diagnostic'
        report['artifact_sha256'] = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                     if p.is_file() and p.name != '.DS_Store' and p != out / 'reference.json'}
        save()
        verify_bundle(out)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    return report


def verify_bundle(folder):
    folder = Path(folder)
    report = json.loads((folder / 'reference.json').read_text())
    expected = plan()
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Native diagnostic contract mismatch')
    if (report.get('status') != 'prepared_inactive_component_diagnostic' or report.get('native_exports') != 16
            or report.get('accepted_report_sha256') != ACCEPTED_REPORT_SHA256 or report.get('baseline_modified') is not False
            or set(report.get('host_checks', {})) != {'FP32_partitions_step1', 'BF16_partitions_step1',
                'APG_step1_latent', 'APG_step1_momentum', 'APG_step1_guided', 'FP32_decoder_saved_final_latent'}
            or not all(bounded_comparison(v) for v in report['host_checks'].values())):
        raise ValueError('Native diagnostic incomplete or original host gates failed')
    hashes = report.get('artifact_sha256', {})
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != '.DS_Store' and p != folder / 'reference.json'}
    required = {f'{stage["asset_file"]}/{name}' for stage in report['stages'] + [report['apg'], report['decoder']]
                for name in ('main.hash', 'main.mlirb', 'metadata.json')}
    if set(hashes) != actual or not required.issubset(hashes):
        raise ValueError('Incomplete asset manifest')
    for name, digest in hashes.items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError('Native diagnostic integrity mismatch: ' + name)
    captured = verify_capture(folder / 'capture')
    if report.get('checkpoint_sha256') != {n: captured['checkpoint_sha256'][n] for n in ('base', 'vae')}:
        raise ValueError('Native/captured checkpoint provenance mismatch')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--out', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--verify-only', action='store_true')
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--vae-checkpoint', type=Path)
    args = parser.parse_args()
    if args.verify_only:
        if args.out is None: parser.error('--out required')
        report = verify_bundle(args.out)
        print('VERIFIED inactive bundle; native/full trajectory:', report['native_parity'], report['full_trajectory_parity'])
    elif args.execute:
        if args.capture is None or args.out is None: parser.error('--capture and --out required')
        existed = args.out.exists()
        try:
            prepare(args.capture, args.out, args.base_checkpoint, args.vae_checkpoint)
        finally:
            reference = args.out / 'reference.json'
            if not existed and sys.platform == 'darwin' and reference.is_file():
                subprocess.run(['open', '-R', str(reference)], check=False)
        print('Ready inactive:', args.out, '; send reference.json before device activation')
    else:
        print(json.dumps(plan(), indent=2))


if __name__ == '__main__':
    main()

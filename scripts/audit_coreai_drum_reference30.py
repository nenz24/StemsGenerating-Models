#!/usr/bin/env python3
"""Audit the accepted capture, optionally replaying VAE with saved random tensors.

No DiT generation, native export, source decoding or RNG regeneration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from package_coreai_drum_reference import ACCEPTED_REPORT_SHA256, SHAPES, checked_file
from export_coreai_decoder import ROOT, sha256, verify_checkpoint


def verify_capture(folder):
    import numpy as np
    folder = Path(folder)
    raw = (folder / 'report.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACCEPTED_REPORT_SHA256:
        raise ValueError('Accepted reference report mismatch')
    report = json.loads(raw)
    for name, shape in SHAPES.items():
        path = folder / name
        checked_file(path, math.prod(shape) * 4)
        if sha256(path) != report['artifact_sha256'][name]:
            raise ValueError(f'Capture checksum mismatch: {name}')
        if not np.isfinite(np.fromfile(path, dtype='<f4')).all():
            raise ValueError(f'Nonfinite capture: {name}')
    return report


def read_tensor(folder, name):
    import numpy as np
    import torch
    return torch.from_numpy(np.fromfile(Path(folder) / name, dtype='<f4').reshape(SHAPES[name]))


def audit(folder, vae_checkpoint=None):
    import torch
    from export_coreai_dit_block import precision_metrics
    from export_coreai_source_context import posterior_context
    from export_coreai_bass_preview import apg_euler
    torch.set_num_threads(2)
    report = verify_capture(folder)
    load = lambda name: read_tensor(folder, name + '.f32le')
    result = {'schema_version': 1, 'capture_report_sha256': ACCEPTED_REPORT_SHA256,
              'scope': 'Saved-input component audit only; no DiT/full-trajectory/native/full-song parity claim',
              'tensor_count': len(SHAPES), 'absolute_tolerance': 1e-4,
              'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'random_tensors_regenerated': 0, 'DiT_calls': 0, 'native_exports': 0,
              'native_parity': 'NOT EVALUATED', 'full_trajectory_parity': 'NOT EVALUATED',
              'component_checks': {}}
    checks = result['component_checks']
    def check(name, actual, expected):
        metrics = precision_metrics(actual, expected)
        checks[name] = metrics
        print(name, metrics, flush=True)

    # Recompute the posterior using the actual captured CPU epsilon, not seed31.
    mean, std, epsilon = load('source-mean'), load('source-std'), load('source-epsilon')
    source = (mean + std * epsilon).transpose(1, 2).contiguous()
    context = torch.cat([source, load('source-mask')], dim=-1)
    check('captured_posterior_to_source', source, load('source-latent'))
    check('captured_source_mask_to_context', context, load('source-context'))
    check('full_range_mask', load('source-mask'), torch.ones_like(load('source-mask')))
    check('final_latent_equals_step50', load('final-latent'), load('step-50-latent'))

    # At step1 previous momentum/latent are known exactly. Later prior states were
    # not captured; only check projection using each stored post-update momentum.
    velocity = torch.cat([load('step-01-conditional'), load('step-01-unconditional')])
    times = load('timesteps')
    x, momentum, guided = apg_euler(velocity, load('initial-noise'),
                                    torch.zeros_like(load('initial-noise')), times[:1] - times[1:2])
    check('native_APG_step1_momentum', momentum, load('step-01-momentum'))
    check('native_APG_step1_guided', guided, load('step-01-guided'))
    check('native_APG_step1_Euler', x, load('step-01-latent'))
    for step in (1, 5, 10, 25, 50):
        cond = load(f'step-{step:02d}-conditional')
        running = load(f'step-{step:02d}-momentum')
        diff = running * torch.minimum(torch.ones_like(running), 2.5 / running.norm(p=2, dim=1, keepdim=True))
        unit = torch.nn.functional.normalize(cond, p=2, dim=1)
        native_guided = cond + 6 * (diff - (diff * unit).sum(dim=1, keepdim=True) * unit)
        check(f'native_APG_projection_step{step}', native_guided, load(f'step-{step:02d}-guided'))
    result['source_encoder_calls'] = result['decoder_calls'] = 0
    if vae_checkpoint is not None:
        import gc
        from safetensors import safe_open
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
        from compare_coreai_drum_source_gain import load_encoder
        record = next(r for r in json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
                      if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
        verify_checkpoint(vae_checkpoint, record)
        if record['digest'] != report['checkpoint_sha256']['vae']:
            raise ValueError('Captured VAE checkpoint differs')
        result['vae_checkpoint_sha256'] = record['digest']
        print('Replay one full source encoder with saved epsilon', flush=True)
        encoder = load_encoder(Path(vae_checkpoint))
        with torch.no_grad():
            outputs = posterior_context(encoder(load('source48')), epsilon, load('source-mask'))
        result['source_encoder_calls'] = 1
        for name, actual in zip(('mean', 'std', 'latent', 'context'), outputs):
            check('VAE_source_' + name, actual, load('source-' + name))
        del encoder, outputs
        gc.collect()
        cfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
        with torch.device('meta'):
            decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                                     audio_channels=cfg['audio_channels'], upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                                     channel_multiples=cfg['channel_multiples']).eval()
        with safe_open(str(vae_checkpoint), framework='pt', device='cpu') as stream:
            state = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
        decoder.load_state_dict(state, strict=True, assign=True)
        del state
        decoder.to(device='cpu', dtype=torch.float32).requires_grad_(False)
        print('Replay one full30s decoder with saved final latent', flush=True)
        with torch.no_grad():
            decoded = decoder(load('final-latent').transpose(1, 2).contiguous())
        result['decoder_calls'] = 1
        check('VAE_final_audio', decoded, load('final-audio'))
    result['all_component_checks_passed'] = all(m['parity_passed'] for m in checks.values())
    result['vae_replayed'] = vae_checkpoint is not None
    result['versions'] = {'torch': torch.__version__}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--vae-checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New audit JSON outside the accepted capture/baselines')
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Existing audit preserved')
    if (args.out.resolve().is_relative_to(args.capture.resolve()) or any(args.out.resolve().is_relative_to(ROOT / p)
            for p in ('baselines', 'prototypes/ios27-coreai/ModelCandidates'))):
        raise ValueError('Audit output must be outside captures/baselines/native bundles')
    result = audit(args.capture, args.vae_checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print('Component checks:', result['all_component_checks_passed'], '; native/full trajectory NOT EVALUATED', flush=True)
    if not result['all_component_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

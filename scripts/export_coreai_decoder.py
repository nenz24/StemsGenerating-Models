#!/usr/bin/env python3
"""Prepare a checksum-verified pretrained ACE VAE component for an iPhone parity test.

Only the pinned VAE checkpoint is needed. No generator/LM, new training, Core ML
conversion, or musical quality claim. Output is an ignored, local app resource.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_checkpoint(path, record):
    if Path(path).stat().st_size != record['size'] or sha256(path) != record['digest']:
        raise ValueError(f'Checkpoint size/SHA-256 mismatch: {path}; not used or overwritten')


def verify_bundle(folder, checkpoint_record, component='decoder'):
    reference = json.loads((folder / 'reference.json').read_text())
    encoding = component == 'encoder'
    if component not in ('decoder', 'encoder'):
        raise ValueError('Unknown VAE component')
    input_name, output_name = ('audio', 'latent') if encoding else ('latent', 'audio')
    if (reference.get('schema_version') != 1 or reference.get('asset_file') != f'{component}.aimodel'
            or reference.get('component', 'decoder') != component):
        raise ValueError('Unexpected VAE bundle contract')
    frames = (reference['output_shape'] if encoding else reference['input_shape'])[-1]
    input_shape, output_shape = ([1, 2, frames * 1920], [1, 64, frames]) if encoding else ([1, 64, frames], [1, 2, frames * 1920])
    if (reference['input_shape'] != input_shape or not 1 <= frames <= 25
            or reference['output_shape'] != output_shape
            or (encoding and (frames != 25 or reference.get('posterior_mode') != 'mean'))
            or reference.get('input_file') != f'{input_name}.f32le'
            or reference.get('expected_file') != f'{output_name}-reference.f32le'
            or reference.get('scalar_type') != 'float32' or reference.get('byte_order') != 'little'
            or reference.get('checkpoint', {}).get('digest') != checkpoint_record['digest']
            or reference.get('checkpoint_full_sha256_verified') is not True):
        raise ValueError('VAE bundle shape/checkpoint mismatch')
    required = {reference['input_file'], reference['expected_file'], f'{component}.aimodel/main.hash',
                f'{component}.aimodel/main.mlirb', f'{component}.aimodel/metadata.json'}
    hashes = reference['artifact_sha256']
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != '.DS_Store' and p != folder / 'reference.json'}
    if not required.issubset(hashes) or actual != set(hashes):
        raise ValueError('Decoder manifest does not cover the complete prepared bundle')
    for name, digest in reference['artifact_sha256'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f'Decoder bundle integrity mismatch: {name}')
    for name, shape in [(reference['input_file'], input_shape), (reference['expected_file'], output_shape)]:
        count = shape[1] * shape[2]
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or path.stat().st_size != count * 4:
            raise ValueError('Decoder reference binary size/path mismatch')
    return reference


def checkpoint_path(record, explicit=None):
    if explicit is not None:
        verify_checkpoint(explicit, record)
        return explicit
    local = ROOT / 'vendor/ACE-Step-1.5' / record['destination']
    if local.is_file():
        verify_checkpoint(local, record)
        return local
    # Keep transport choice before importing the hub client; preserve credentials
    # and certificate settings. Reuse complete caches before downloading.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    caches = [None, ROOT / '.runtime-cache/huggingface/hub', ROOT / '.runtime-cache/huggingface/hub/http']
    for cache in caches:
        try:
            candidate = Path(hf_hub_download(record['repo_id'], record['filename'], revision=record['revision'],
                                             cache_dir=str(cache) if cache else None, local_files_only=True))
        except LocalEntryNotFoundError:
            continue
        verify_checkpoint(candidate, record)
        return candidate
    print(f'Downloading pinned checkpoint: {record["size"] / 1e6:.1f} MB', flush=True)
    candidate = Path(hf_hub_download(record['repo_id'], record['filename'], revision=record['revision'],
                                    cache_dir=str(caches[-1])))
    verify_checkpoint(candidate, record)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Optional existing exact VAE safetensors')
    parser.add_argument('--component', choices=['decoder', 'encoder'], default='decoder')
    parser.add_argument('--out', type=Path, help='New local bundle directory; existing output is preserved')
    parser.add_argument('--frames', type=int, help='Decoder: 1..25, default 5; encoder: fixed 25 (1 s)')
    parser.add_argument('--verify-only', action='store_true', help='Verify/reuse a previously prepared local bundle')
    args = parser.parse_args()
    encoding = args.component == 'encoder'
    if args.frames is None:args.frames = 25 if encoding else 5
    if args.out is None:
        args.out = ROOT / 'prototypes/ios27-coreai/ModelCandidates' / ('encoder-pretrained-1s' if encoding else 'decoder-pretrained')
    if not 1 <= args.frames <= 25:
        parser.error('--frames must be 1..25 (at most one second for the initial test)')
    if encoding and args.frames != 25:
        parser.error('Encoder probe requires --frames 25')
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())
    record = next(r for r in manifest['files'] if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    if args.verify_only:
        reference = verify_bundle(args.out, record, args.component)
        if reference['output_shape' if encoding else 'input_shape'][-1] != args.frames:
            raise ValueError('Existing bundle has a different window; use the matching --frames value')
        print(f'Existing pretrained {args.component} bundle: VERIFIED (no download/re-export)')
        return
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}; preserve it or choose --out elsewhere')
    config_record = next(r for r in manifest['files'] if r['filename'] == 'vae/config.json')
    raw_config = (ROOT / 'configs/coreai-vae-config.json').read_bytes()
    if hashlib.sha1(f'blob {len(raw_config)}\0'.encode() + raw_config).hexdigest() != config_record['digest']:
        raise ValueError('Pinned VAE configuration checksum mismatch')
    checkpoint = checkpoint_path(record, args.checkpoint)
    print('Publisher checkpoint size and SHA-256: VERIFIED', flush=True)

    import numpy as np
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder, OobleckEncoder
    from coreai_torch import TorchConverter, get_decomp_table

    torch.set_num_threads(2)
    cfg = json.loads(raw_config)
    with torch.device('meta'):
        if encoding:
            model = OobleckEncoder(encoder_hidden_size=cfg['encoder_hidden_size'], audio_channels=cfg['audio_channels'],
                                   downsampling_ratios=cfg['downsampling_ratios'], channel_multiples=cfg['channel_multiples']).eval()
        else:
            model = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                                   audio_channels=cfg['audio_channels'],
                                   upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                                   channel_multiples=cfg['channel_multiples']).eval()
    state = {}
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for key in source.keys():
            if key.startswith(args.component + '.'):
                state[key[len(args.component) + 1:]] = source.get_tensor(key)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    del state
    if encoding:
        # This is exactly the mean/mode half used by latent_dist.mode(), not a
        # stochastic posterior sample. The checkpoint's complete encoder loads strictly.
        class EncoderMean(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__();self.encoder = encoder
            def forward(self, audio):
                return self.encoder(audio).chunk(2, dim=1)[0]
        model = EncoderMean(model).eval()
        t = torch.arange(48000, dtype=torch.float64) / 48000
        envelope = torch.sin(torch.pi * t).square()
        left = envelope * (0.15 * torch.sin(2 * torch.pi * 220 * t) + 0.075 * torch.sin(2 * torch.pi * 330 * t))
        right = envelope * (0.15 * torch.sin(2 * torch.pi * 220 * t) + 0.075 * torch.sin(2 * torch.pi * 440 * t))
        x = torch.stack([left, right]).unsqueeze(0).to(torch.float32)
        provenance = 'Deterministic 1 s stereo multitone with sine-squared envelope; synthetic, not a user recording'
    else:
        generator = torch.Generator(device='cpu').manual_seed(0)
        x = torch.randn(1, cfg['decoder_input_channels'], args.frames, generator=generator, dtype=torch.float32)
        provenance = 'Seed-0 standard-normal latent for numerical decoder testing; not a recorded performance'
    input_name, output_name = ('audio', 'latent') if encoding else ('latent', 'audio')
    print(f'CPU FP32 {args.component} reference for {args.frames / 25:.2f} seconds; NOT instrument generation', flush=True)
    started = time.monotonic()
    with torch.no_grad():
        original = model(x).detach().clone()
    # Fold weight norm using the real checkpoint values, and verify this graph
    # simplification before converting. No learned weights are replaced.
    for module in model.modules():
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            torch.nn.utils.remove_weight_norm(module)
    with torch.no_grad():
        expected = model(x).detach()
    torch.testing.assert_close(expected, original, rtol=0, atol=0)
    if not torch.isfinite(expected).all():
        raise ValueError('Nonfinite PyTorch reference; export stopped')
    print('Weight-normalization folding parity: EXACT; reference seconds:', time.monotonic() - started, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage_root = ROOT / f'.runtime-cache/coreai-{args.component}-staging'
    stage_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.component + '-', dir=stage_root))
    print('Staging folder (retained on failure):', stage, flush=True)
    array = expected.numpy().astype('<f4', copy=False)
    x.numpy().astype('<f4', copy=False).tofile(stage / f'{input_name}.f32le')
    array.tofile(stage / f'{output_name}-reference.f32le')
    print(f'Exporting pretrained {args.component} through coreai-torch…', flush=True)
    exported = torch.export.export(model, (x,)).run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(exported, input_names=[input_name], output_names=[output_name]).to_coreai()
    program.save_asset(stage / f'{args.component}.aimodel')
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    reference = {
        'schema_version': 1, 'component': args.component, 'asset_file': f'{args.component}.aimodel', 'function_name': 'main',
        'input_name': input_name, 'output_name': output_name,
        'input_shape': list(x.shape), 'output_shape': list(expected.shape),
        'input_file': f'{input_name}.f32le', 'expected_file': f'{output_name}-reference.f32le',
        'scalar_type': 'float32', 'byte_order': 'little',
        'sample_rate': cfg['sampling_rate'], 'duration_seconds': args.frames / 25,
        'input_provenance': provenance,
        'checkpoint': {k: record[k] for k in ['repo_id', 'revision', 'filename', 'size', 'digest']},
        'checkpoint_full_sha256_verified': True, 'reference_backend': 'PyTorch CPU Float32',
        'weight_norm_fold_max_abs_error': 0.0,
        'reference_peak': float(np.max(np.abs(array))),
        'reference_rms': float(np.sqrt(np.mean(array.astype(np.float64) ** 2))),
        'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'diffusers', 'coreai-torch', 'coreai-core']},
        'artifact_sha256': {str(p.relative_to(stage)): sha256(p) for p in files},
        'coreai_device_inference_verified': False,
    }
    if encoding:reference['posterior_mode'] = 'mean'
    (stage / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    if args.out.exists():
        raise FileExistsError(f'Destination appeared during export; staged output preserved: {stage}')
    verify_bundle(stage, record, args.component)
    shutil.move(str(stage), str(args.out))
    print(f'Prepared pretrained {args.component} bundle:', args.out, flush=True)
    print('iPhone compilation, model parity, timing and memory: NOT YET VERIFIED', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare a checksum-verified pretrained ACE VAE decoder for an iPhone parity test.

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


def verify_bundle(folder, checkpoint_record):
    reference = json.loads((folder / 'reference.json').read_text())
    if reference.get('schema_version') != 1 or reference.get('asset_file') != 'decoder.aimodel':
        raise ValueError('Unexpected decoder bundle contract')
    frames = reference['input_shape'][-1]
    if (reference['input_shape'] != [1, 64, frames] or not 1 <= frames <= 25
            or reference['output_shape'] != [1, 2, frames * 1920]
            or reference.get('input_file') != 'latent.f32le'
            or reference.get('expected_file') != 'audio-reference.f32le'
            or reference.get('scalar_type') != 'float32' or reference.get('byte_order') != 'little'
            or reference.get('checkpoint', {}).get('digest') != checkpoint_record['digest']
            or reference.get('checkpoint_full_sha256_verified') is not True):
        raise ValueError('Decoder bundle shape/checkpoint mismatch')
    required = {'latent.f32le', 'audio-reference.f32le', 'decoder.aimodel/main.hash',
                'decoder.aimodel/main.mlirb', 'decoder.aimodel/metadata.json'}
    hashes = reference['artifact_sha256']
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != '.DS_Store' and p != folder / 'reference.json'}
    if not required.issubset(hashes) or actual != set(hashes):
        raise ValueError('Decoder manifest does not cover the complete prepared bundle')
    for name, digest in reference['artifact_sha256'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f'Decoder bundle integrity mismatch: {name}')
    for name, count in [(reference['input_file'], 64 * frames), (reference['expected_file'], 2 * frames * 1920)]:
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
    print(f'Downloading only pinned VAE: {record["size"] / 1e6:.1f} MB', flush=True)
    candidate = Path(hf_hub_download(record['repo_id'], record['filename'], revision=record['revision'],
                                    cache_dir=str(caches[-1])))
    verify_checkpoint(candidate, record)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Optional existing exact VAE safetensors')
    parser.add_argument('--out', type=Path, default=ROOT / 'prototypes/ios27-coreai/ModelCandidates/decoder-pretrained')
    parser.add_argument('--frames', type=int, default=5, help='Fixed latent frames: 5 = 0.2 seconds, max 25')
    parser.add_argument('--verify-only', action='store_true', help='Verify/reuse a previously prepared local bundle')
    args = parser.parse_args()
    if not 1 <= args.frames <= 25:
        parser.error('--frames must be 1..25 (at most one second for the initial test)')
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())
    record = next(r for r in manifest['files'] if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    if args.verify_only:
        reference = verify_bundle(args.out, record)
        if reference['input_shape'][-1] != args.frames:
            raise ValueError('Existing bundle has a different window; use the matching --frames value')
        print('Existing pretrained decoder bundle: VERIFIED (no download/re-export)')
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
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    from coreai_torch import TorchConverter, get_decomp_table

    torch.set_num_threads(2)
    cfg = json.loads(raw_config)
    with torch.device('meta'):
        model = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                               audio_channels=cfg['audio_channels'],
                               upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                               channel_multiples=cfg['channel_multiples']).eval()
    state = {}
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for key in source.keys():
            if key.startswith('decoder.'):
                state[key[len('decoder.'):]] = source.get_tensor(key)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    del state
    generator = torch.Generator(device='cpu').manual_seed(0)
    x = torch.randn(1, cfg['decoder_input_channels'], args.frames, generator=generator, dtype=torch.float32)
    print(f'CPU FP32 reference for {args.frames / 25:.2f} seconds; seeded latent, not instrument generation', flush=True)
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
    stage_root = ROOT / '.runtime-cache/coreai-decoder-staging'
    stage_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='decoder-', dir=stage_root))
    print('Staging folder (retained on failure):', stage, flush=True)
    array = expected.numpy().astype('<f4', copy=False)
    x.numpy().astype('<f4', copy=False).tofile(stage / 'latent.f32le')
    array.tofile(stage / 'audio-reference.f32le')
    print('Exporting pretrained decoder through coreai-torch…', flush=True)
    exported = torch.export.export(model, (x,)).run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(exported, input_names=['latent'], output_names=['audio']).to_coreai()
    program.save_asset(stage / 'decoder.aimodel')
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    reference = {
        'schema_version': 1, 'asset_file': 'decoder.aimodel', 'function_name': 'main',
        'input_name': 'latent', 'output_name': 'audio',
        'input_shape': list(x.shape), 'output_shape': list(expected.shape),
        'input_file': 'latent.f32le', 'expected_file': 'audio-reference.f32le',
        'scalar_type': 'float32', 'byte_order': 'little',
        'sample_rate': cfg['sampling_rate'], 'duration_seconds': expected.shape[-1] / cfg['sampling_rate'],
        'input_provenance': 'Seed-0 standard-normal latent for numerical decoder testing; not a recorded performance',
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
    (stage / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    if args.out.exists():
        raise FileExistsError(f'Destination appeared during export; staged output preserved: {stage}')
    verify_bundle(stage, record)
    shutil.move(str(stage), str(args.out))
    print('Prepared pretrained decoder bundle:', args.out, flush=True)
    print('iPhone compilation, model parity, timing and memory: NOT YET VERIFIED', flush=True)


if __name__ == '__main__':
    main()

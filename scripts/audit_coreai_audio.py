#!/usr/bin/env python3
"""Audit pinned ACE VAE shapes/weights and CoreAI lowering without loading checkpoints.

The exported blocks use synthetic random parameters: operator probes, not a
pretrained codec, audio generation, or evidence of iPhone execution.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def header_inventory(header):
    counts = Counter()
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        shape = tensor['shape']
        if any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError('Invalid checkpoint tensor shape')
        counts[tensor['dtype']] += math.prod(shape)
    return dict(counts)


def fetch_header(record):
    url = f"https://huggingface.co/{record['repo_id']}/resolve/{record['revision']}/{record['filename']}"
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-1048575'})
    with urllib.request.urlopen(request, timeout=45) as stream:
        status = stream.status
        data = stream.read(1048576)
    if len(data) < 8:
        raise ValueError('Missing safetensors header')
    length = struct.unpack('<Q', data[:8])[0]
    if length > len(data) - 8:
        raise ValueError('Header exceeds bounded read; no full checkpoint download attempted')
    header = json.loads(data[8:8+length])
    return {'http_status': status, 'header_bytes': length,
            'prefix_bytes_read': len(data),
            'scope': 'Bounded prefix may include partial tensor bytes; no full weights loaded or complete checkpoint checksum verified',
            'header_sha256': hashlib.sha256(data[:8+length]).hexdigest(),
            'parameters_by_dtype': header_inventory(header),
            'full_checkpoint_checksum_verified': False,
            'provenance': {k: record[k] for k in ['repo_id', 'revision', 'filename', 'size', 'algorithm', 'digest']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/coreai-audio-audit')
    parser.add_argument('--fetch-headers', action='store_true', help='Read bounded headers from pinned HF URLs')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite prior audit: {args.out}; use a new output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())
    config_path = ROOT / 'configs/coreai-vae-config.json'
    raw = config_path.read_bytes()
    config_record = next(r for r in manifest['files'] if r['filename'] == 'vae/config.json')
    digest = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    if len(raw) != config_record['size'] or digest != config_record['digest']:
        raise ValueError('Pinned VAE config checksum mismatch')
    cfg = json.loads(raw)

    import torch
    from diffusers.models import AutoencoderOobleck
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoderBlock
    from coreai_torch import TorchConverter, get_decomp_table

    torch.set_num_threads(2)
    with torch.device('meta'):
        vae = AutoencoderOobleck(**{k: v for k, v in cfg.items() if not k.startswith('_')}).eval()
    counts = {name: sum(p.numel() for p in module.parameters())
              for name, module in [('full', vae), ('encoder', vae.encoder), ('decoder', vae.decoder)]}
    shapes = []
    def shape_hook(name):
        def observe(module, inputs, output):
            if isinstance(output, torch.Tensor):
                shapes.append({'module': name, 'class': type(module).__name__,
                               'shape': list(output.shape), 'elements': output.numel()})
        return observe
    hooks = [module.register_forward_hook(shape_hook(name)) for name, module in vae.decoder.named_modules()]
    latent_rate = cfg['sampling_rate'] // math.prod(cfg['downsampling_ratios'])
    with torch.no_grad():
        output = vae.decoder(torch.empty(1, cfg['decoder_input_channels'], latent_rate, device='meta'))
    for hook in hooks:
        hook.remove()
    largest = max(shapes, key=lambda item: item['elements'])
    report = {
        'schema_version': 1, 'upstream_commit': 'ca1e85fe9430179831e6bc6be790c332190a3866',
        'scope': 'Meta-device architecture/shape audit and synthetic block native export only',
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'diffusers', 'coreai-torch', 'coreai-core']},
        'vae_config_verified_against_publisher_manifest': True,
        'sample_rate': cfg['sampling_rate'], 'audio_channels': cfg['audio_channels'],
        'latent_channels': cfg['decoder_input_channels'], 'latent_rate_hz': latent_rate,
        'downsampling_factor': math.prod(cfg['downsampling_ratios']),
        'vae_parameter_counts': counts,
        'raw_weight_storage_estimates': {precision: {name: count * width for name, count in counts.items()}
                                         for precision, width in [('bf16_or_fp16', 2), ('fp32', 4)]},
        'one_second_decoder_output_shape': list(output.shape),
        'largest_observed_decoder_tensor_one_second': largest,
        'single_tensor_fp32_bytes_by_seconds': {str(seconds): int(largest['elements'] * seconds * 4)
                                              for seconds in [0.2, 1, 10, 90]},
        'memory_estimate_limit': 'Logical unfused tensor storage, NOT measured device peak RAM; fusion may avoid materialization; excludes weights, simultaneous tensors and runtime buffers',
        'checkpoint_headers': {}, 'synthetic_operator_probes': [],
        'full_pretrained_checkpoints_downloaded': False, 'pretrained_weights_loaded': False,
        'coreai_device_execution_verified': False,
    }
    if args.fetch_headers:
        for label, filename, repo in [('vae', 'vae/diffusion_pytorch_model.safetensors', 'ACE-Step/Ace-Step1.5'),
                                      ('base_generator', 'model.safetensors', 'ACE-Step/acestep-v15-base')]:
            record = next(r for r in manifest['files'] if r['filename'] == filename and r['repo_id'] == repo)
            report['checkpoint_headers'][label] = fetch_header(record)
        assert sum(report['checkpoint_headers']['vae']['parameters_by_dtype'].values()) == counts['full']
    for channels, stride in [(128, 2), (16, 4), (16, 6), (16, 10)]:
        torch.manual_seed(0)
        model = OobleckDecoderBlock(channels, channels, stride=stride).eval()
        x = torch.randn(1, channels, 64)
        with torch.no_grad():
            y = model(x)
        probe = {'module': 'OobleckDecoderBlock', 'channels': channels, 'stride': stride,
                 'input_shape': list(x.shape), 'output_shape': list(y.shape),
                 'parameters': sum(p.numel() for p in model.parameters()),
                 'weight_provenance': 'Synthetic seed 0, not pretrained; reduced channels except final-stage width 128 probe',
                 'device_execution': 'not_run'}
        try:
            exported = torch.export.export(model, (x,)).run_decompositions(get_decomp_table())
            probe['operators'] = dict(Counter(str(node.target) for node in exported.graph.nodes if node.op == 'call_function'))
            program = TorchConverter().add_exported_program(exported, input_names=['x'], output_names=['y']).to_coreai()
            asset = args.out / f'block-c{channels}-s{stride}.aimodel'
            if asset.exists():
                raise FileExistsError(f'Refusing to overwrite probe {asset}; use a new output directory')
            program.save_asset(asset)
            probe['native_export'] = 'passed'
            probe['asset'] = asset.name
        except Exception as error:
            probe['native_export'] = 'failed'
            probe['error'] = repr(error)
        report['synthetic_operator_probes'].append(probe)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.out / 'report.json'), 'vae_parameter_counts': counts,
                      'native_exports': [p['native_export'] for p in report['synthetic_operator_probes']],
                      'device_execution': 'NOT RUN'}, indent=2))
    if any(p['native_export'] != 'passed' for p in report['synthetic_operator_probes']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

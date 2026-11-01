#!/usr/bin/env python3
"""Export fixed pretrained Base DiT profiles; not instrument generation."""
import argparse
import ast
import __future__
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import types
import urllib.request

from export_coreai_decoder import ROOT, checkpoint_path, sha256

INPUT_SHAPES = {
    'hidden': [1, 16, 2048], 'condition': [1, 8, 2048], 'temb': [1, 6, 2048],
    'rope_cos': [1, 16, 128], 'rope_sin': [1, 16, 128],
    'self_mask': [1, 1, 16, 16], 'cross_mask': [1, 1, 16, 8],
}
OUTPUT_SHAPE = [1, 16, 2048]
CONTEXT_INPUT_SHAPES = {
    'hidden': [1, 160, 2048], 'condition': [1, 64, 2048], 'temb': [1, 6, 2048],
    'rope_cos': [1, 160, 128], 'rope_sin': [1, 160, 128],
    'self_mask': [1, 1, 160, 160], 'cross_mask': [1, 1, 160, 64],
    'local_mask': [1, 1, 160, 160],
}


def profile_spec(profile):
    if profile == 'short':
        return INPUT_SHAPES, OUTPUT_SHAPE, 'dit-block.aimodel', [0]
    if profile == 'context':
        return CONTEXT_INPUT_SHAPES, [1, 160, 2048], 'dit-context.aimodel', [0, 1]
    raise ValueError('Unknown DiT profile')


def records():
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())
    return {r['filename']: r for r in manifest['files'] if r['repo_id'] == 'ACE-Step/acestep-v15-base'}


def verified_source(record, folder):
    path = folder / record['filename']
    if not path.exists():
        url = f'https://huggingface.co/{record["repo_id"]}/resolve/{record["revision"]}/{record["filename"]}'
        with urllib.request.urlopen(url, timeout=60) as stream:raw = stream.read(record['size'] + 1)
        digest = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if len(raw) != record['size'] or digest != record['digest']:
            raise ValueError('Publisher source/config checksum mismatch')
        path.write_bytes(raw)
    raw = path.read_bytes()
    if len(raw) != record['size'] or hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != record['digest']:
        raise ValueError(f'Existing publisher source/config checksum mismatch: {path}; preserved')
    return path


def verify_bundle(folder, record, profile='short'):
    ref = json.loads((folder / 'reference.json').read_text())
    shapes, output_shape, asset, indices = profile_spec(profile)
    if (ref.get('schema_version') != 1 or ref.get('asset_file') != asset
            or ref.get('profile', 'short') != profile
            or ref.get('checkpoint', {}).get('digest') != record['digest']
            or ref.get('checkpoint_full_sha256_verified') is not True
            or (profile == 'short' and ref.get('layer_prefix') != 'decoder.layers.0.')
            or (profile == 'context' and (ref.get('layer_prefixes') != ['decoder.layers.0.', 'decoder.layers.1.']
                                         or ref.get('sliding_window') != 128 or ref.get('mask_geometry_verified') is not True))
            or ref.get('input_shapes') != shapes or ref.get('output_shape') != output_shape
            or ref.get('function_name') != 'main' or ref.get('output_name') != 'hidden_out'
            or ref.get('scalar_type') != 'float32' or ref.get('byte_order') != 'little'):
        raise ValueError('Unexpected pretrained DiT block contract')
    hashes = ref['artifact_sha256']
    required = {f'{name}.f32le' for name in shapes} | {
        'hidden-reference.f32le', f'{asset}/main.hash',
        f'{asset}/main.mlirb', f'{asset}/metadata.json'}
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != '.DS_Store' and p != folder / 'reference.json'}
    if not required.issubset(hashes) or actual != set(hashes):
        raise ValueError('Incomplete DiT block manifest')
    for name, digest in hashes.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f'DiT block integrity mismatch: {name}')
    for name, shape in [(f'{n}.f32le', s) for n, s in shapes.items()] + [('hidden-reference.f32le', output_shape)]:
        if (folder / name).stat().st_size != math.prod(shape) * 4:
            raise ValueError('DiT block binary size mismatch')
    return ref


def load_upstream(source, config_path, config_source):
    """Use the verified publisher class bodies and actual Transformers helpers.

    Loading these class/mask AST nodes avoids importing the unrelated full generator,
    quantizer, sampler and guidance modules. No numerical layer is replaced.
    """
    import torch
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding,
        apply_rotary_pos_emb, eager_attention_forward)

    spec = importlib.util.spec_from_file_location('coreai_pinned_base_config', config_source)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg = module.AceStepConfig(**json.loads(config_path.read_text()))
    cfg._attn_implementation = 'eager'
    tree = ast.parse(source.read_text(), filename=str(source))
    names = {'AceStepAttention', 'AceStepDiTLayer', 'create_4d_mask'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(nodes) != 3:raise ValueError('Pinned publisher classes/mask missing')
    selected = ast.Module(body=nodes, type_ignores=[])
    namespace = types.ModuleType('coreai_pinned_base_layers')
    namespace.__file__ = str(source)
    namespace.__dict__.update(torch=torch, nn=torch.nn, GradientCheckpointingLayer=GradientCheckpointingLayer,
                              Qwen3MLP=Qwen3MLP, Qwen3RMSNorm=Qwen3RMSNorm,
                              apply_rotary_pos_emb=apply_rotary_pos_emb, eager_attention_forward=eager_attention_forward)
    sys.modules[namespace.__name__] = namespace
    exec(compile(selected, str(source), 'exec', flags=__future__.annotations.compiler_flag), namespace.__dict__)
    return cfg, namespace.AceStepDiTLayer, Qwen3RotaryEmbedding, namespace.create_4d_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['short', 'context'], default='short')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    context = args.profile == 'context'
    shapes, output_shape, asset, indices = profile_spec(args.profile)
    if args.out is None:
        args.out = ROOT / 'prototypes/ios27-coreai/ModelCandidates' / ('dit-context-pretrained' if context else 'dit-block-pretrained')
    publisher = records();record = publisher['model.safetensors']
    if args.verify_only:
        verify_bundle(args.out, record, args.profile);print('Existing pretrained DiT profile: VERIFIED (no download/re-export)');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    print(f'Base checkpoint required: 4.79 GB if not cached; exporting {len(indices)} block(s).', flush=True)
    checkpoint = checkpoint_path(record, args.checkpoint)
    print('Full publisher Base checkpoint SHA-256: VERIFIED', flush=True)
    sources = ROOT / '.runtime-cache/coreai-generator-source-publisher';sources.mkdir(parents=True, exist_ok=True)
    paths = {name: verified_source(publisher[name], sources) for name in [
        'modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json']}
    cfg, layer_type, rotary_type, create_mask = load_upstream(paths['modeling_acestep_v15_base.py'], paths['config.json'], paths['configuration_acestep_v15.py'])
    import numpy as np
    import torch
    from safetensors import safe_open
    from coreai_torch import TorchConverter, get_decomp_table
    torch.set_num_threads(2)
    layers = []
    prefixes = [f'decoder.layers.{index}.' for index in indices]
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for index, prefix in zip(indices, prefixes):
            with torch.device('meta'):layer = layer_type(cfg, layer_idx=index, use_cross_attention=True).eval()
            state = {key[len(prefix):]: source.get_tensor(key) for key in source.keys() if key.startswith(prefix)}
            layer.load_state_dict(state, strict=True, assign=True)
            layer.to(device='cpu', dtype=torch.float32).requires_grad_(False);del state
            layers.append(layer)

    class Block(torch.nn.Module):
        def __init__(self, layer):super().__init__();self.layer = layer
        def forward(self, hidden, condition, temb, rope_cos, rope_sin, self_mask, cross_mask):
            return self.layer(hidden_states=hidden, encoder_hidden_states=condition,
                              position_embeddings=(rope_cos, rope_sin), temb=temb,
                              attention_mask=self_mask, encoder_attention_mask=cross_mask,
                              use_cache=False, past_key_value=None)[0]

    class Pair(torch.nn.Module):
        def __init__(self, first, second):
            super().__init__();self.first = Block(first);self.second = Block(second)
        def forward(self, hidden, condition, temb, rope_cos, rope_sin, self_mask, cross_mask, local_mask):
            hidden = self.first(hidden, condition, temb, rope_cos, rope_sin, local_mask, cross_mask)
            return self.second(hidden, condition, temb, rope_cos, rope_sin, self_mask, cross_mask)

    model = (Pair(*layers) if context else Block(layers[0])).eval()
    generator = torch.Generator(device='cpu').manual_seed(0)
    values = {name: torch.randn(shape, generator=generator) * 0.1 for name, shape in list(shapes.items())[:3]}
    tokens = shapes['hidden'][1]
    with torch.no_grad():cos, sin = rotary_type(cfg)(values['hidden'], torch.arange(tokens).unsqueeze(0))
    values.update(rope_cos=cos, rope_sin=sin, self_mask=torch.zeros(shapes['self_mask']), cross_mask=torch.zeros(shapes['cross_mask']))
    # Finite additive masks exercise a masked key in both attention paths.
    values['self_mask'][..., -1] = -10000.0;values['cross_mask'][..., -1] = -10000.0
    if context:
        if cfg.sliding_window != 128 or cfg.layer_types[:2] != ['sliding_attention', 'full_attention']:
            raise ValueError('Pinned local/full attention configuration changed')
        padding = torch.ones(1, tokens);padding[..., -1] = 0
        values['self_mask'] = create_mask(tokens, torch.float32, torch.device('cpu'), attention_mask=padding, is_causal=False)
        values['local_mask'] = create_mask(tokens, torch.float32, torch.device('cpu'), attention_mask=padding,
                                            sliding_window=cfg.sliding_window, is_sliding_window=True, is_causal=False)
        values['cross_mask'][..., -1] = torch.finfo(torch.float32).min
        if not (values['local_mask'][0,0,0,128] == 0 and values['local_mask'][0,0,0,129] < -1e30
                and values['self_mask'][0,0,0,129] == 0 and values['local_mask'][0,0,159,0] < -1e30
                and values['self_mask'][0,0,159,0] == 0):
            raise ValueError('Publisher local/full mask geometry mismatch')
    inputs = tuple(values[name] for name in shapes)
    with torch.no_grad():expected = model(*inputs).detach()
    if list(expected.shape) != output_shape or not torch.isfinite(expected).all():raise ValueError('Invalid PyTorch reference')
    stage_root = ROOT / '.runtime-cache/coreai-dit-staging';stage_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.profile + '-', dir=stage_root))
    print('Exporting actual pretrained DiT block; staging retained on failure:', stage, flush=True)
    for name, value in values.items():value.numpy().astype('<f4', copy=False).tofile(stage / f'{name}.f32le')
    expected.numpy().astype('<f4', copy=False).tofile(stage / 'hidden-reference.f32le')
    exported = torch.export.export(model, inputs).run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(exported, input_names=list(shapes), output_names=['hidden_out']).to_coreai()
    program.save_asset(stage / asset)
    ref = {'schema_version': 1, 'profile': args.profile, 'asset_file': asset, 'function_name': 'main', 'output_name': 'hidden_out',
           'input_shapes': shapes, 'output_shape': output_shape, 'scalar_type': 'float32', 'byte_order': 'little',
           'checkpoint': {k:record[k] for k in ['repo_id','revision','filename','size','digest']},
           'checkpoint_full_sha256_verified': True, 'publisher_source_digests': {n:publisher[n]['digest'] for n in paths},
           'input_provenance': f'Seed-0 synthetic activations; RoPE positions 0..{tokens-1}; finite additive key masks',
           'scope': f'{len(indices)} complete pretrained block(s), self/cross-attention, RoPE, RMSNorm, AdaLN and MLP; NOT full denoiser or instrument generation',
           'parameter_count': sum(p.numel() for p in model.parameters()),
           'reference_backend': 'PyTorch CPU Float32 / publisher eager attention',
           'reference_peak': float(expected.abs().max()), 'reference_rms': float(np.sqrt(np.mean(expected.numpy().astype(np.float64)**2))),
           'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
           'versions': {n:importlib.metadata.version(n) for n in ['torch','transformers','coreai-torch','coreai-core']},
           'artifact_sha256': {str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()},
           'coreai_device_inference_verified': False}
    if context:
        ref.update(layer_prefixes=prefixes, sliding_window=cfg.sliding_window, mask_geometry_verified=True)
    else:
        ref['layer_prefix'] = prefixes[0]
    (stage / 'reference.json').write_text(json.dumps(ref, indent=2)+'\n')
    verify_bundle(stage, record, args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():raise FileExistsError(f'Destination appeared; preserved staging: {stage}')
    shutil.move(str(stage), str(args.out))
    print('Prepared pretrained DiT block:', args.out, flush=True)
    print('CoreAI iPhone build, parity, latency and memory: NOT YET VERIFIED', flush=True)


if __name__ == '__main__':main()

"""Stdlib-only contract and schedule for the fixed-input BF16 diagnostic."""
import json
import math
from pathlib import Path
from export_coreai_decoder import sha256

PROFILE = 'dit-interleaving-1024'
SIZES = (2, 4, 8, 12)
CALLS = 20
NATIVE = {'metadata.json', 'main.hash', 'main.mlirb'}
INPUTS = {'hidden': [2,128,2048], 'condition': [2,118,2048], 'temb': [2,6,2048],
          'rope_cos': [1,128,128], 'rope_sin': [1,128,128],
          'self_mask': [1,1,128,128], 'local_mask': [1,1,128,128], 'cross_mask': [1,1,128,118]}
OUTPUT = [2,128,2048]


def assets(count):
    return [f'blocks-{i:02d}-{i+1:02d}.aimodel' for i in range(0, count*2, 2)]


def schedule(mode, count):
    if count not in SIZES: raise ValueError('Unsupported pool size')
    if mode in ('single-before', 'pool-only', 'single-after'): return [0]*CALLS
    if mode == 'grouped': return [i for i in range(count) for _ in range(CALLS)]
    if mode == 'interleaved': return list(range(count))*CALLS
    raise ValueError('Unknown mode')


def payload(count):
    return ({n+'.f32le' for n in INPUTS}
            | {f'reference-{i:02d}.f32le' for i in range(count)}
            | {a+'/'+n for a in assets(count) for n in NATIVE})


def validate(ref):
    count = ref.get('pair_count')
    if type(count) is not int or count not in SIZES: raise ValueError('Invalid pair count')
    expected = {'schema_version':1, 'profile':PROFILE, 'assets':assets(count),
                'input_shapes':INPUTS, 'output_shape':OUTPUT, 'calls_per_asset':CALLS,
                'weight_storage_dtype':'bfloat16', 'compute_dtype':'float32',
                'scalar_type':'float32', 'byte_order':'little',
                'absolute_tolerance':1e-4, 'relative_tolerance':1e-3, 'relative_rmse_tolerance':1e-3,
                'checkpoint_full_sha256_verified':True, 'host_exact': [True]*count}
    for key, value in expected.items():
        if type(ref.get(key)) is not type(value) or ref[key] != value:
            raise ValueError('Interleaving contract mismatch: '+key)
    # bools must not be accepted as numerical proof flags.
    if any(type(v) is not bool for v in ref['host_exact']): raise ValueError('Invalid host proof')
    from export_coreai_dit_block import records
    if ref.get('checkpoint_digest') != records()['model.safetensors']['digest']:
        raise ValueError('Checkpoint changed')
    hashes = ref.get('artifact_sha256', {})
    source = ref.get('source_asset_sha256', {})
    import re
    if set(hashes) != payload(count): raise ValueError('Unexpected interleaving payload')
    if set(source) != {a+'/'+n for a in assets(count) for n in NATIVE}:
        raise ValueError('Missing source identity')
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest): raise ValueError('Invalid hash')
    if any(hashes[n] != value for n, value in source.items()): raise ValueError('BF16 source changed')
    if len({hashes[a+'/main.mlirb'] for a in assets(count)}) != count:
        raise ValueError('Pair assets must be distinct')
    if not re.fullmatch('[0-9a-f]{64}', ref.get('source_manifest_sha256','')): raise ValueError('Missing source manifest')
    return ref


def verify_bundle(folder):
    folder = Path(folder)
    ref = validate(json.loads((folder/'reference.json').read_text()))
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
              if p.is_file() and p.name != '.DS_Store' and p != folder/'reference.json'}
    if actual != payload(ref['pair_count']): raise ValueError('Incomplete interleaving bundle')
    for name, digest in ref['artifact_sha256'].items():
        p = folder/name
        if not p.resolve().is_relative_to(folder.resolve()) or sha256(p) != digest:
            raise ValueError('Interleaving integrity mismatch: '+name)
        shape = INPUTS.get(name.removesuffix('.f32le')) if '/' not in name else None
        if name.startswith('reference-'): shape = OUTPUT
        if shape and p.stat().st_size != math.prod(shape)*4: raise ValueError('Tensor size mismatch: '+name)
    return ref

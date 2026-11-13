#!/usr/bin/env python3
"""Verify host caches and reuse existing BF16 sampler/FP32 decoder for30s previews.

No models run/exported/downloaded. Selection changes only with --execute --activate.
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from export_coreai_decoder import ROOT, sha256
from coreai_instrument_preview30_contract import BUNDLE, PROFILE, ROLES, PRESETS, DATA_SHAPES, validate
from coreai_drum_reference30_generation_contract import CONFIG, SAMPLER_BUNDLE, DECODER_BUNDLE, configuration
from coreai_drum_reference30_contract import CAPTURE_REPORT_SHA256
from prepare_coreai_full_song_conditioning import SHAPES


def checked_tensor(path, shape, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != math.prod(shape) * 4 or sha256(path) != expected:
        raise ValueError('Conditioning size/SHA256 mismatch: ' + str(path))
    values = array.array('f', path.read_bytes())
    if sys.byteorder != 'little': values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError('Nonfinite conditioning: ' + str(path))
    if path.name == 'condition-mask.f32le' and any(v != 1 for v in values):
        raise ValueError('All127 tokens must be valid; no padding mask')
    return values


def verify_host(folder):
    folder = Path(folder)
    if (folder / 'report.json').is_symlink(): raise ValueError('Linked conditioning report rejected')
    report = json.loads((folder / 'report.json').read_text())
    expected = {'schema_version': 1, 'profile': 'full-song-conditioning-127-v1',
                'status': 'host_conditioning_captured', 'capture_report_sha256': CAPTURE_REPORT_SHA256,
                'geometry': {'text': 111, 'lyric': 15, 'packed': 127},
                'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
                'host_compute_dtype': 'float32', 'sampler_weight_storage_dtype': 'bfloat16',
                'sampler_weights_modified': False, 'baseline_modified': False, 'profile_activated': False,
                'Qwen_model_loads': 1, 'Qwen_text_calls': 3,
                'ConditionEncoder_model_loads': 1, 'ConditionEncoder_calls': 3,
                'DiT_calls': 0, 'VAE_calls': 0, 'native_exports': 0,
                'padding_tokens': 0, 'truncated_tokens': 0, 'ready_for_staging': True,
                'ready_for_native_execution': False,
                'tokenizer_sha256': 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a'}
    if any(report.get(k) != value for k, value in expected.items()):
        raise ValueError('Host conditioning contract incomplete or changed')
    from export_coreai_bass_conditioning import BASE_DIGEST, QWEN_DIGEST, SILENCE_DIGEST
    if report.get('checkpoint_sha256') != dict(base=BASE_DIGEST, qwen=QWEN_DIGEST, silence=SILENCE_DIGEST):
        raise ValueError('Conditioning checkpoint identity changed')
    baseline_path = ROOT / 'reviews/coreai-drums-host30s-listening-accepted/report.json'
    if sha256(baseline_path) != CAPTURE_REPORT_SHA256: raise ValueError('Original report identity changed')
    baseline = json.loads(baseline_path.read_text())
    from prepare_coreai_full_song_conditioning import formatted_prompts
    # Reconstruct public prompt definitions from verified cached formatters.
    # Independent token values are verified against the pinned local tokenizer
    # by the host preparation; the frozen drum arrays are checked byte-for-byte.
    if report.get('prompts') != formatted_prompts(): raise ValueError('Caption/metadata/lyric preset changed')
    shapes = {role + '/' + name + '.f32le': shape for role in ROLES for name, shape in SHAPES.items()}
    if report.get('tensor_shapes') != shapes or set(report.get('artifact_sha256', {})) != set(shapes):
        raise ValueError('Expected exactly21 conditioning tensors')
    hashes = report['artifact_sha256']
    for name, shape in shapes.items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()): raise ValueError('External conditioning file')
        values = checked_tensor(path, shape, hashes[name])
        if name.endswith('-ids.f32le') and any(v < 0 or v != int(v) for v in values):
            raise ValueError('Token IDs must be nonnegative integers')
        role, tensor = name.split('/')
        if role == 'drums' and hashes[name] != baseline['artifact_sha256'][tensor]:
            raise ValueError('Drum control differs from original captured bytes: ' + tensor)
        if tensor in ('lyric-ids.f32le', 'lyric-embeddings.f32le', 'reference-audio.f32le', 'condition-mask.f32le'):
            if hashes[name] != hashes['drums/' + tensor]: raise ValueError('Shared lyric/silence/mask changed')
    checks = report.get('drums_control_checks', {})
    if set(checks) != set(SHAPES): raise ValueError('Expected seven drum control checks')
    for name, metric in checks.items():
        if (metric.get('finite') is not True or metric.get('parity_passed') is not True
                or any(metric.get(k) != 0 for k in ('max_absolute_error', 'rmse', 'normalized_rmse', 'elementwise_fraction_over_bound'))
                or (name in ('text-ids', 'lyric-ids', 'condition-mask') and metric.get('exact_equal') is not True)):
            raise ValueError('Drum control must match all original arrays exactly')
    return report


def verify_bundle(folder):
    folder = Path(folder)
    if (folder / 'reference.json').is_symlink(): raise ValueError('Linked reference rejected')
    ref = validate(json.loads((folder / 'reference.json').read_text()))
    for name, expected in ref['artifact_sha256'].items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Prepared conditioning changed: ' + name)
        if name in DATA_SHAPES: checked_tensor(path, DATA_SHAPES[name], expected)
    return ref


def prepare(source, candidates, activate=False):
    from prepare_coreai_drum_reference30 import verify_bundle as verify_sampler
    from prepare_coreai_drum_reference30_window_decoder import verify_bundle as verify_decoder
    from activate_coreai_drum_reference30 import select_profile
    source, candidates = Path(source).resolve(), Path(candidates).resolve()
    report = verify_host(source)
    verify_sampler(candidates / SAMPLER_BUNDLE)
    verify_decoder(candidates / DECODER_BUNDLE)
    ref = {'schema_version': 1, 'profile': BUNDLE, 'status': 'prepared_instrument_preview30',
           'roles': list(ROLES), 'preset_sha256': PRESETS, 'condition_tokens': 127,
           'condition_shape': [1, 127, 2048], 'mask_shape': [1, 127],
           'capture_report_sha256': CAPTURE_REPORT_SHA256, 'host_drums_control_exact': True,
           'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
           'sampler_weight_storage_dtype': 'bfloat16', 'compute_dtype': 'float32', 'vae_dtype': 'float32',
           'native_exports': 0, 'baseline_modified': False, 'padding_tokens': 0, 'truncated_tokens': 0,
           'full_song_native_readiness': 'NOT EVALUATED', 'bass_guitar_numerical_parity': 'NOT EVALUATED',
           'artifact_sha256': {name: report['artifact_sha256'][name] for name in DATA_SHAPES}}
    ref['artifact_sha256']['conditioning-report.json'] = sha256(source / 'report.json')
    validate(ref)
    target = candidates / BUNDLE
    if target.is_symlink(): raise ValueError('Linked target preserved')
    if target.exists():
        if verify_bundle(target) != ref: raise FileExistsError('Different existing conditioning bundle preserved')
    else:
        if target.resolve().is_relative_to(source): raise ValueError('Do not write inside host capture')
        staging_root = ROOT / '.runtime-cache/instrument-preview30-staging'
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='prepare-', dir=staging_root)) / BUNDLE
        staging.mkdir()
        for name in ref['artifact_sha256']:
            path = staging / name
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(source / ('report.json' if name == 'conditioning-report.json' else name), path)
        (staging / 'reference.json').write_text(json.dumps(ref, indent=2) + '\n')
        verify_bundle(staging)
        if target.exists(): raise FileExistsError('Target appeared; preserved with staging')
        staging.rename(target)
    payload = configuration(sha256(candidates / DECODER_BUNDLE / 'reference.json'))
    config = candidates / CONFIG
    encoded = (json.dumps(payload, indent=2) + '\n').encode()
    if config.is_symlink() or (config.exists() and config.read_bytes() != encoded):
        raise FileExistsError('Different existing generation configuration preserved')
    if not config.exists():
        descriptor, temporary = tempfile.mkstemp(prefix='.instrument30-config-', dir=candidates)
        try:
            with os.fdopen(descriptor, 'wb') as stream: stream.write(encoded)
            os.link(temporary, config)
        finally: Path(temporary).unlink(missing_ok=True)
    if activate: select_profile(candidates, PROFILE)
    print('Ready: bass/drums/guitar30s previews; shared unchanged sampler and bounded decoder. Full-song NOT EVALUATED.', flush=True)
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT / '.runtime-cache/full-song-conditioning-v1')
    parser.add_argument('--candidates', type=Path, default=ROOT / 'prototypes/ios27-coreai/ModelCandidates')
    args = parser.parse_args()
    if args.activate and not args.execute: parser.error('--activate requires --execute')
    if not args.execute:
        print('PLAN ONLY: verify21 cached arrays, copy only3 conditions/3 masks/report, reuse15 sampler assets and bounded decoder; no models/downloads/selection changes.')
        return
    prepare(args.source, args.candidates, args.activate)


if __name__ == '__main__': main()

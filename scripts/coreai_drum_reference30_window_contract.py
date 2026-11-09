"""Stdlib-only bounded decoder contract; full30s gold and original gates fixed."""
import math
import re

PROFILE = 'drums-reference30-window-decoder'
BUNDLE = 'drums-reference30-window-decoder-256'
CAPTURE_SHA = 'f5f84314fa37f49f73502a77f7e3364b31450898231731d64e216bf1b606bfdf'
VAE_SHA = 'da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0'
LATENT_SHA = '54c574fbcec00d7404d5072258d738b3de5e167728e707a2b271264733d5740c'
AUDIO_SHA = '5f81f20205c39fbba6418f95d5d42d8346fe48f457c8c455c77958c98fabd615'
NATIVE_FILES = ('main.hash', 'main.mlirb', 'metadata.json')
DATA_SHAPES = {'final-latent.f32le': [1,750,64], 'final-audio.f32le': [1,2,1440000]}
DATA_HASHES = {'final-latent.f32le': LATENT_SHA, 'final-audio.f32le': AUDIO_SHA, 'capture-report.json': CAPTURE_SHA}
FILES = set(DATA_HASHES) | {'decoder.aimodel/' + n for n in NATIVE_FILES}


def tile_plan():
    rows = []
    for start in range(0, 750, 128):
        window = min(max(start - 64, 0), 494)
        rows.append({'window_start': window, 'window_frames': 256, 'core_start': start,
                     'core_end': min(start+128, 750), 'crop_start': start-window})
    return rows


def dependency_bounds(tile):
    """Check each inverse-convolution edge against genuine clip boundaries."""
    size = 256 * 1920
    lo, hi = tile['crop_start']*1920, (tile['core_end']-tile['window_start'])*1920-1
    def clip(a, b, length):
        if a < 0 and tile['window_start'] != 0:
            raise ValueError('Artificial left padding reaches retained output')
        if b >= length and tile['window_start']+256 != 750:
            raise ValueError('Artificial right padding reaches retained output')
        return max(a, 0), min(b, length-1)
    lo, hi = clip(lo-3, hi+3, size)
    for stride in (2, 4, 4, 6, 10):
        for dilation in (9, 3, 1):
            lo, hi = clip(lo-3*dilation, hi+3*dilation, size)
        padding = (stride+1)//2
        lo = (lo+padding-(2*stride-1)+stride-1)//stride
        hi = (hi+padding)//stride
        size //= stride
        lo, hi = clip(lo, hi, size)
    lo, hi = clip(lo-3, hi+3, 256)
    return [tile['window_start']+lo, tile['window_start']+hi]


def strict_metric(metric):
    fields = ('max_absolute_error', 'rmse', 'normalized_rmse', 'elementwise_fraction_over_bound')
    if (metric.get('finite') is not True or metric.get('parity_passed') is not True
            or any(type(metric.get(k)) not in (float, int) or not math.isfinite(metric[k]) or metric[k] < 0 for k in fields)
            or metric['elementwise_fraction_over_bound'] != 0 or metric['normalized_rmse'] > 1e-3):
        raise ValueError('Original strict decoder host gate failed; no tolerance override')


def validate(ref):
    expected = {'schema_version': 1, 'profile': PROFILE, 'status': 'prepared_window_decoder',
        'asset_file': 'decoder.aimodel', 'function_name': 'main', 'input_shapes': {'latent':[1,64,256]},
        'output_shapes': {'audio':[1,2,491520]}, 'sample_rate':48000, 'audio_seconds':30.,
        'latent_frames':750, 'window_frames':256, 'core_frames':128, 'halo_frames':64,
        'weight_storage_dtype':'float32', 'compute_dtype':'float32', 'scalar_type':'float32', 'byte_order':'little',
        'vae_checkpoint_sha256':VAE_SHA, 'capture_report_sha256':CAPTURE_SHA,
        'absolute_tolerance':1e-4, 'relative_tolerance':1e-3, 'relative_rmse_tolerance':1e-3,
        'window_plan':tile_plan(), 'host_numerical_gate_passed':True, 'architecture_guard_passed':True,
        'coverage_exactly_once':True, 'crossfade':False, 'extra_latent_padding':False,
        'native_parity':'NOT EVALUATED', 'native_memory_gate':'NOT EVALUATED', 'full_trajectory_parity':'NOT EVALUATED',
        'baseline_modified':False}
    if any(ref.get(k) != v for k, v in expected.items()):
        raise ValueError('Bounded decoder contract mismatch')
    if (set(ref.get('artifact_sha256', {})) != FILES or any(ref['artifact_sha256'][n] != h for n,h in DATA_HASHES.items())
            or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in ref['artifact_sha256'].values())):
        raise ValueError('Bounded decoder artifact/input identity mismatch')
    checks = ref.get('host_checks', {})
    required = {'overall'} | {f'tile-{i}' for i in range(6)} | {f'seam-{i}' for i in range(1,6)}
    if set(checks) != required:
        raise ValueError('Missing full/seam/tile host comparisons')
    for metric in checks.values(): strict_metric(metric)
    if (len(ref.get('tile_input_sha256', [])) != 6
            or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in ref['tile_input_sha256'])
            or ref.get('dependency_bounds') != [dependency_bounds(t) for t in tile_plan()]):
        raise ValueError('Missing tile identity/receptive-field proof')
    return ref

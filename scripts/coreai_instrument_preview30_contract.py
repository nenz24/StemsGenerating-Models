"""Stdlib-only transport contract for three conditioning caches, shared weights."""
from coreai_drum_reference30_contract import CAPTURE_REPORT_SHA256
import re

PROFILE = 'instrument-reference30-preview'
BUNDLE = 'instrument-reference30-conditioning-127'
ROLES = ('bass', 'drums', 'guitar')
PRESETS = {'bass': '73177b9d57ef961a66f3df375d24d2a869ad4f27fd6f007b6641d3d58c8614a1',
           'drums': 'bcfd73641fddbda1fb0975febfe28cccb3a3121ec48cec1d877933e64a6d70ca',
           'guitar': '2f49a2db3ee92b2c52418794d38e4e29312dec5219c3f4ab4c7cf571c12df389'}
DATA_SHAPES = {role + '/' + name + '.f32le': shape for role in ROLES
               for name, shape in [('condition', [1, 127, 2048]), ('condition-mask', [1, 127])]}
FILES = set(DATA_SHAPES) | {'conditioning-report.json'}


def validate(ref):
    expected = {'schema_version': 1, 'profile': BUNDLE, 'status': 'prepared_instrument_preview30',
                'roles': list(ROLES), 'preset_sha256': PRESETS, 'condition_tokens': 127,
                'condition_shape': [1, 127, 2048], 'mask_shape': [1, 127],
                'capture_report_sha256': CAPTURE_REPORT_SHA256, 'host_drums_control_exact': True,
                'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
                'sampler_weight_storage_dtype': 'bfloat16', 'compute_dtype': 'float32', 'vae_dtype': 'float32',
                'native_exports': 0, 'baseline_modified': False, 'padding_tokens': 0, 'truncated_tokens': 0,
                'full_song_native_readiness': 'NOT EVALUATED', 'bass_guitar_numerical_parity': 'NOT EVALUATED'}
    if any(ref.get(k) != value for k, value in expected.items()):
        raise ValueError('Three instrument preview contract differs; baseline and bounds preserved')
    if set(ref.get('artifact_sha256', {})) != FILES:
        raise ValueError('Conditioning bundle file set differs')
    if any(not isinstance(v, str) or not re.fullmatch('[a-f0-9]{64}', v) for v in ref['artifact_sha256'].values()):
        raise ValueError('Invalid conditioning artifact SHA256')
    return ref

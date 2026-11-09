"""Pinned, stdlib-only resource contract for the accepted native component probe."""
import hashlib
import json

PROFILE = 'drums-reference30-components'
BUNDLE = 'drums-reference30-frozen-diagnostic'
REFERENCE_SHA256 = 'f2f84e6c34b2fe42678dc38a48d9ca35b20c14ba53f10843aee8b218a24d53ed'
CAPTURE_REPORT_SHA256 = 'f5f84314fa37f49f73502a77f7e3364b31450898231731d64e216bf1b606bfdf'
ASSETS = ['front.aimodel', *[f'blocks-{i:02d}-{i+1:02d}.aimodel' for i in range(0, 24, 2)],
          'tail.aimodel', 'apg-euler.aimodel', 'decoder.aimodel']
FIXTURES = {'capture/' + name: shape for name, shape in {
    'initial-noise.f32le': [1, 750, 64], 'condition.f32le': [1, 127, 2048],
    'source-context.f32le': [1, 750, 128], 'timesteps.f32le': [51],
    'final-latent.f32le': [1, 750, 64], 'final-audio.f32le': [1, 2, 1440000],
    **{f'step-01-{kind}.f32le': [1, 750, 64] for kind in
       ('conditional', 'unconditional', 'latent', 'momentum', 'guided')},
}.items()}
KEEP = set(FIXTURES) | {'capture/report.json'} | {
    asset + '/' + name for asset in ASSETS for name in ('main.hash', 'main.mlirb', 'metadata.json')}


def validate_reference(raw):
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Native30 reference differs from the reviewed Mac export; no activation')
    ref = json.loads(raw)
    expected = {'schema_version': 1, 'profile': BUNDLE, 'status': 'prepared_inactive_component_diagnostic',
                'latent_frames': 750, 'condition_tokens': 127, 'audio_seconds': 30., 'native_exports': 16,
                'weight_storage_dtype': 'bfloat16', 'compute_dtype': 'float32', 'VAE_dtype': 'float32',
                'scalar_type': 'float32', 'byte_order': 'little', 'function_name': 'main',
                'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
                'inference_steps': 50, 'guidance_scale': 7., 'shift': 1.,
                'apg_momentum': -.75, 'apg_norm_threshold': 2.5, 'apg_projection_dims': [1],
                'baseline_modified': False, 'native_parity': 'NOT EVALUATED',
                'full_trajectory_parity': 'NOT EVALUATED', 'accepted_report_sha256': CAPTURE_REPORT_SHA256}
    if any(ref.get(k) != v for k, v in expected.items()):
        raise ValueError('Native30 reviewed export contract mismatch')
    if (len(ref['stages']) != 14 or len(ref['artifact_sha256']) != 92 or not KEEP.issubset(ref['artifact_sha256'])
            or ref['artifact_sha256']['capture/report.json'] != CAPTURE_REPORT_SHA256
            or not all(m.get('parity_passed') is True and m.get('finite') is True
                       and m.get('elementwise_fraction_over_bound') == 0 and m.get('normalized_rmse', 1) <= 1e-3
                       for m in ref['host_checks'].values())):
        raise ValueError('Native30 assets/host gates incomplete')
    return ref

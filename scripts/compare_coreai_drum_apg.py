#!/usr/bin/env python3
"""One separate host drum APG experiment; production cap/gates stay unchanged.

Keep 10.24s input/conditioning/noise/schedule fixed. Compare the stored original
FP32 control (APG cap 2.5) with one candidate whose cap matches the per-frame RMS
bound of the approved 30s reference: 2.5*sqrt(256/750). This is not audio EQ,
output normalization, a new baseline, or a demonstrated glitch fix.
"""
import argparse
import json
import math
from pathlib import Path

from coreai_preview_window import PreviewWindow
from export_coreai_decoder import ROOT, sha256
from compare_coreai_drum_lyrics import execute as execute_host

WINDOW = PreviewWindow(256, 'drums')
BASELINE_CAP = 2.5


def scaled_cap(frames, reference_frames):
    if (type(frames) is not int or type(reference_frames) is not int
            or frames <= 0 or reference_frames <= 0):
        raise ValueError('Expected positive integer frame counts')
    return BASELINE_CAP * math.sqrt(frames / reference_frames)


class NormLimitedAPG:
    """Delegate actual APG/momentum/projection to the pinned publisher implementation."""
    def __init__(self, original, cap):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError('Expected a finite positive APG cap')
        self.original = original
        self.cap = cap
        self.MomentumBuffer = original.MomentumBuffer
        self.diagnostics = []

    def apg_forward(self, conditional, unconditional, scale, momentum_buffer, dims):
        import torch
        if dims != [1] or conditional.ndim != 3:
            raise ValueError('Experiment requires original APG time dimension [1]')
        guided = self.original.apg_forward(conditional, unconditional, scale,
                   momentum_buffer=momentum_buffer, norm_threshold=self.cap, dims=dims)
        norm = torch.linalg.vector_norm(momentum_buffer.running_average, dim=1)
        if not torch.isfinite(norm).all() or not torch.isfinite(guided).all():
            raise ValueError('Nonfinite APG diagnostics')
        update_rms = (guided - conditional).square().mean(dim=1).sqrt()
        self.diagnostics.append({'step': len(self.diagnostics) + 1,
                                 'momentum_norm_max': float(norm.max()),
                                 'candidate_clip_fraction': float((norm > self.cap).float().mean()),
                                 'baseline_cap_fraction_for_same_candidate_momentum': float((norm > BASELINE_CAP).float().mean()),
                                 'guidance_update_per_channel_rms_max': float(update_rms.max())})
        return guided


def compare(candidates, out, diagnostic=False, execute_requested=False, explicit=None):
    candidates, out = Path(candidates), Path(out)
    if out.exists():
        raise FileExistsError(f'Existing comparison preserved: {out}')
    if out.resolve().is_relative_to(candidates.resolve()):
        raise ValueError('Comparison must be outside ModelCandidates')
    from export_coreai_source_context import verify_bundle as source
    from export_coreai_bass_conditioning import verify_bundle as condition
    from export_coreai_bass_preview import verify_bundle as preview
    refs = {'source': source(candidates / WINDOW.source_bundle, WINDOW),
            'condition': condition(candidates / WINDOW.condition_bundle, WINDOW)}
    refs['preview'] = preview(candidates / WINDOW.preview_bundle, candidates / WINDOW.source_bundle,
                              candidates / WINDOW.condition_bundle, WINDOW, diagnostic)
    if refs['preview']['apg_norm_threshold'] != BASELINE_CAP:
        raise ValueError('Expected original APG cap 2.5 control')
    reference_path = ROOT / 'baselines/acoustic-drums-30-v1/baseline.json'
    reference = json.loads(reference_path.read_text())
    if (reference['id'] != 'acoustic-drums-30-v1' or reference['duration_seconds'] != 30
            or reference['status'] != 'listener_approved_excerpt'):
        raise ValueError('Expected frozen approved 30-second reference')
    reference_frames = round(reference['duration_seconds'] * WINDOW.frames / WINDOW.seconds)
    if reference_frames != 750:
        raise ValueError('Expected 30s * pinned 25Hz latent geometry')
    cap = scaled_cap(WINDOW.frames, reference_frames)
    prompts = refs['condition']['prompts']
    report = {'schema_version': 1, 'experiment': 'drums-host-apg-per-frame-bound', 'status': 'planned',
              'changed_parameter': 'APG norm_threshold ONLY; numeric acceptance tolerances unchanged',
              'baseline_apg_norm_threshold': BASELINE_CAP, 'candidate_apg_norm_threshold': cap,
              'latent_frames': WINDOW.frames, 'reference_latent_frames': reference_frames,
              'reference_baseline_sha256': sha256(reference_path),
              'baseline_per_channel_momentum_rms_cap': BASELINE_CAP / math.sqrt(WINDOW.frames),
              'reference_per_channel_momentum_rms_cap': BASELINE_CAP / math.sqrt(reference_frames),
              'candidate_per_channel_momentum_rms_cap': cap / math.sqrt(WINDOW.frames),
              'control_prompts': prompts, 'candidate_prompts': dict(prompts),
              'source_pcm_sha256': refs['source']['artifact_sha256']['audio.f32le'],
              'component_reference_sha256': {key: sha256(candidates / getattr(WINDOW, f'{key}_bundle') / 'reference.json')
                                             for key in ['source', 'condition', 'preview']},
              'control_host_export_status': refs['preview'].get('host_export_status', 'strict_pass'),
              'unchanged': ['piano PCM/context', 'FP32 conditioning including current lyric text',
                            'initial noise/schedule', '10.24 seconds', '50 APG Euler steps', 'guidance 7',
                            'APG momentum/projection dims/FP64 original projection', 'checkpoint values', 'VAE decoder'],
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'scope': 'One HOST original-FP32 quality experiment; not native/production parity or a 30-second context reproduction',
              'native_exports': 0, 'baseline_modified': False, 'musical_acceptance': 'NOT EVALUATED'}
    out.mkdir(parents=True, exist_ok=False)

    def save():
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'report.json')

    def progress(phase):
        report['phase'] = phase
        save()
        print(phase, flush=True)

    save()
    if not execute_requested:
        return report
    try:
        report['status'] = 'running'
        execute_host(candidates, out, refs, report, progress, explicit or {},
                     candidate_apg_factory=lambda apg: NormLimitedAPG(apg, cap),
                     require_unchanged_condition=True,
                     control_label='01-control-original-apg', candidate_label='02-candidate-scaled-apg')
        if len(report.get('candidate_apg_diagnostics', [])) != 50:
            raise ValueError('Incomplete candidate APG diagnostics')
        for key, digest in report['component_reference_sha256'].items():
            if sha256(candidates / getattr(WINDOW, f'{key}_bundle') / 'reference.json') != digest:
                raise ValueError('Prepared component changed during comparison')
        report['candidate_clipping_active_steps'] = sum(row['candidate_clip_fraction'] > 0
                                                       for row in report['candidate_apg_diagnostics'])
        report['diagnostic_note'] = ('Baseline-cap fractions after step 1 are counterfactual on CANDIDATE momentum; '
                                     'they are not a replay of all 50 control states. No listening verdict inferred.')
        report['status'] = 'completed_quality_experiment'
        report['artifact_sha256'] = {p.name: sha256(p) for p in out.iterdir() if p.name != 'report.json'}
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic-trajectory', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Compute one candidate; otherwise write plan only')
    for name in ['base', 'qwen', 'vae']:
        parser.add_argument(f'--{name}-checkpoint', type=Path)
    args = parser.parse_args()
    report = compare(ROOT / 'prototypes/ios27-coreai/ModelCandidates', args.out,
                     args.diagnostic_trajectory, args.execute,
                     {name: getattr(args, f'{name}_checkpoint') for name in ['base', 'qwen', 'vae']})
    print(report['status'], args.out)


if __name__ == '__main__':
    main()

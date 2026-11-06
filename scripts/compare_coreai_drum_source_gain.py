#!/usr/bin/env python3
"""One host source-gain-only candidate; no native export or new weight download.

Test fixed piano gain0.7 used by accepted Colab preparation, before the VAE.
Keep saved10.24s PCM/resampler, epsilon, conditioning, noise, original APG2.5,
50steps and decoder. This does not reproduce Colab's32kHz preparation/30s context.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from compare_coreai_drum_lyrics import execute as execute_host, local_checkpoint, require_parity
from coreai_preview_window import PreviewWindow
from export_coreai_decoder import ROOT, sha256
from export_coreai_source_context import posterior_context

WINDOW = PreviewWindow(256, 'drums')
SOURCE_GAIN = 0.7


def load_encoder(checkpoint):
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckEncoder
    records = json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
    record = next(r for r in records if r['filename'] == 'vae/config.json')
    raw = (ROOT / 'configs/coreai-vae-config.json').read_bytes()
    if hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != record['digest']:
        raise ValueError('Pinned VAE config checksum mismatch')
    cfg = json.loads(raw)
    with torch.device('meta'):
        encoder = OobleckEncoder(encoder_hidden_size=cfg['encoder_hidden_size'], audio_channels=cfg['audio_channels'],
                                 downsampling_ratios=cfg['downsampling_ratios'], channel_multiples=cfg['channel_multiples'])
    with safe_open(str(checkpoint), framework='pt', device='cpu') as weights:
        state = {k[len('encoder.'):]: weights.get_tensor(k) for k in weights.keys() if k.startswith('encoder.')}
    encoder.load_state_dict(state, strict=True, assign=True)
    return encoder.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)


def encode(encoder, audio, epsilon, mask):
    import torch
    with torch.no_grad():
        result = posterior_context(encoder(audio), epsilon, mask)
    expected = [[1, 64, 256], [1, 64, 256], [1, 256, 64], [1, 256, 128]]
    if any(list(value.shape) != shape or not torch.isfinite(value).all()
           for value, shape in zip(result, expected)):
        raise ValueError('Invalid source posterior/context')
    return result


def prepare_context(candidates, out, refs, report, progress, explicit):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    folder = candidates / WINDOW.source_bundle
    record = next(r for r in json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
                  if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    progress('verify pinned local VAE and replay original source encoder')
    checkpoint = local_checkpoint(record, explicit.get('vae'))
    encoder = load_encoder(checkpoint)

    def tensor(name, shape):
        a = np.fromfile(folder / name, dtype='<f4').copy()
        if a.size != int(np.prod(shape)) or not np.isfinite(a).all():
            raise ValueError(f'Invalid saved source tensor: {name}')
        return torch.from_numpy(a).reshape(shape)

    audio = tensor('audio.f32le', [1, 2, WINDOW.samples])
    epsilon = tensor('epsilon.f32le', [1, 64, WINDOW.frames])
    mask = tensor('chunk_mask.f32le', [1, WINDOW.frames, 64])
    original = encode(encoder, audio, epsilon, mask)
    report['source_encoder_calls'] = {'control': 1, 'candidate': 0}
    for name, value in zip(['posterior_mean', 'posterior_std', 'source_latents', 'context_latents'], original):
        require_parity('source_' + name, value,
                       tensor(name + '-reference.f32le', list(value.shape)), report)
    del original
    progress('encode piano once at fixed input gain0.7; same saved epsilon and mask')
    scaled = (audio * SOURCE_GAIN).contiguous()
    mean, std, latent, context = encode(encoder, scaled, epsilon, mask)
    report['source_encoder_calls']['candidate'] = 1
    if not torch.equal(context[..., 64:], mask):
        raise ValueError('Candidate must preserve the original full-range lego mask')
    for name, value in [('candidate-source-audio', scaled), ('candidate-source-mean', mean),
                        ('candidate-source-std', std), ('candidate-source-latent', latent),
                        ('candidate-context', context)]:
        value.numpy().astype('<f4').tofile(out / (name + '.f32le'))
    report['source_checkpoint_sha256'] = record['digest']
    report['source_pcm_peaks'] = {'control': float(audio.abs().max()), 'candidate': float(scaled.abs().max())}
    report['source_epsilon_sha256'] = refs['source']['artifact_sha256']['epsilon.f32le']
    report['source_mask_unchanged'] = True
    del encoder, audio, scaled, mean, std, latent
    gc.collect()
    return context


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
    if (refs['preview']['apg_norm_threshold'] != 2.5
            or refs['source']['recording']['canonicalization'] !=
            'Host ffmpeg -> 48 kHz stereo Float32, planar tensor, no peak normalization; same PCM sent to device'):
        raise ValueError('Expected original unnormalized source and APG cap2.5 control')
    prompts = refs['condition']['prompts']
    report = {'schema_version': 1, 'status': 'planned', 'experiment': 'drums-host-source-gain-only',
              'changed_input': 'piano Float32 PCM multiplied by fixed0.7 before source VAE; nothing else',
              'candidate_source_gain': SOURCE_GAIN, 'control_source_gain': 1.,
              'control_prompts': prompts, 'candidate_prompts': dict(prompts),
              'source_pcm_sha256': refs['source']['artifact_sha256']['audio.f32le'],
              'component_reference_sha256': {key: sha256(candidates / getattr(WINDOW, f'{key}_bundle') / 'reference.json')
                                             for key in ['source', 'condition', 'preview']},
              'control_host_export_status': refs['preview'].get('host_export_status', 'strict_pass'),
              'unchanged': ['saved PCM recording/crop/resampler', 'saved source epsilon and lego mask',
                            'all text/lyric/timbre conditioning', 'duration10.24s', 'initial noise/schedule',
                            '50steps guidance7 originalAPG cap2.5/momentum/projection', 'checkpoint values', 'VAE decoder'],
              'scope': 'One original-FP32 HOST source-gain-only candidate; not full Colab32kHz/30s reproduction, native parity or an adopted fix',
              'reference': 'tools/stem_eval/audio.py canonicalize fixed gain0.7 used by accepted Colab preparation',
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
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
        context = prepare_context(candidates, out, refs, report, progress, explicit or {})
        execute_host(candidates, out, refs, report, progress, explicit or {},
                     candidate_context=context, require_unchanged_condition=True,
                     control_label='01-control-source-gain1', candidate_label='02-candidate-source-gain07')
        if report.get('candidate_steps_completed') != 50:
            raise ValueError('Incomplete source-gain candidate trajectory')
        for key, digest in report['component_reference_sha256'].items():
            if sha256(candidates / getattr(WINDOW, f'{key}_bundle') / 'reference.json') != digest:
                raise ValueError('Prepared component manifest changed during comparison')
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
    parser.add_argument('--execute', action='store_true', help='Compute one50-step host candidate; otherwise plan only')
    for name in ['base', 'qwen', 'vae']:
        parser.add_argument(f'--{name}-checkpoint', type=Path)
    args = parser.parse_args()
    result = compare(ROOT / 'prototypes/ios27-coreai/ModelCandidates', args.out,
                     args.diagnostic_trajectory, args.execute,
                     {name: getattr(args, f'{name}_checkpoint') for name in ['base', 'qwen', 'vae']})
    print(result['status'], args.out)


if __name__ == '__main__':
    main()

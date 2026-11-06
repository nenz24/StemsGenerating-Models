#!/usr/bin/env python3
"""One host FP32 drum comparison: empty lyrics vs Colab's [Instrumental].

Reuse cached control audio and prepared piano/context/noise/schedule. Recompute
conditioning and one candidate trajectory only; never export/activate models.
Weights must already be local. This is a quality experiment, not a fix claim.
"""
import argparse
import ast
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from coreai_listening_audio import headroom_gain, read_planar, write_float_wav
from coreai_preview_window import PreviewWindow
from export_coreai_decoder import ROOT, sha256, verify_checkpoint
from export_coreai_dit_block import precision_metrics, records, verified_source
from export_coreai_source_context import verified_handler_source

WINDOW = PreviewWindow(256, 'drums')


def lyric_variant(prompts):
    """Run the pinned formatter; change only the literal lyric content."""
    if prompts.get('lyrics') != '' or prompts.get('vocal_language') != 'unknown':
        raise ValueError('Expected unchanged empty-lyrics control')
    path, _ = verified_handler_source('prompt_utils.py')
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_format_lyrics')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    formatter = namespace['_format_lyrics']
    if prompts['lyric_prompt'] != formatter(None, '', 'unknown'):
        raise ValueError('Control lyric formatter mismatch')
    return {**prompts, 'lyrics': '[Instrumental]',
            'lyric_prompt': formatter(None, '[Instrumental]', 'unknown')}


def local_checkpoint(record, explicit=None):
    """Resolve/verify existing weights only; never fall back to a download."""
    if explicit is not None:
        path = Path(explicit)
        verify_checkpoint(path, record)
        return path
    vendor = ROOT / 'vendor/ACE-Step-1.5' / record['destination']
    if vendor.is_file():
        verify_checkpoint(vendor, record)
        return vendor
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    for cache in [None, ROOT / '.runtime-cache/huggingface/hub',
                  ROOT / '.runtime-cache/huggingface/hub/http']:
        try:
            path = Path(hf_hub_download(record['repo_id'], record['filename'],
                        revision=record['revision'], cache_dir=str(cache) if cache else None,
                        local_files_only=True))
        except LocalEntryNotFoundError:
            continue
        verify_checkpoint(path, record)
        return path
    raise FileNotFoundError(f'Pinned local checkpoint missing: {record["filename"]}; no download attempted')


def require_parity(label, actual, expected, report):
    if actual.shape != expected.shape:
        report.setdefault('control_replay_checks', {})[label] = {
            'parity_passed': False, 'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape)}
        raise ValueError(f'Control replay shape failed: {label}')
    stats = precision_metrics(actual, expected)
    report.setdefault('control_replay_checks', {})[label] = stats
    if not stats['parity_passed']:
        raise ValueError(f'Control replay gate failed: {label}; original tolerances unchanged')


def guided_step(predict, latent, current, following, apg, momentum):
    conditional, unconditional = predict(latent, current).chunk(2)
    velocity = apg.apg_forward(conditional, unconditional, 7.,
                               momentum_buffer=momentum, dims=[1])
    return latent - velocity * (current - following)


def candidate_trajectory(predict, noise, times, apg, progress):
    """Independent live feedback with one fresh momentum buffer, no gold resets."""
    import torch
    latent = noise.clone()
    momentum = apg.MomentumBuffer()
    for i in range(50):
        latent = guided_step(predict, latent, times[i:i+1], times[i+1:i+2], apg, momentum)
        if latent.shape != noise.shape or not torch.isfinite(latent).all():
            raise ValueError(f'Invalid candidate latent shape/values at step {i+1}')
        progress(i+1)
    return latent


def conditioned_predictor(model, condition, null, context):
    """Bind a particular source context, keeping control/candidate independent."""
    import torch
    conditions = torch.cat([condition, null.expand_as(condition)])
    def predict(latent, time):
        return model(hidden_states=torch.cat([latent, latent]), timestep=time.expand(2),
                     timestep_r=time.expand(2), attention_mask=None, encoder_hidden_states=conditions,
                     encoder_attention_mask=None, context_latents=torch.cat([context, context]), use_cache=False)[0]
    return predict


def execute(candidates, out, refs, report, progress, explicit, *, candidate_apg_factory=None,
            require_unchanged_condition=False, control_label='control-empty-lyrics',
            candidate_label='candidate-instrumental', candidate_context=None):
    import numpy as np
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    from export_coreai_bass_conditioning import load_condition_encoder
    from export_coreai_denoiser import load_full_upstream
    from prepare_coreai_denoiser_loop import verify_sampling_source

    torch.set_num_threads(2)
    report['versions'] = {n: importlib.metadata.version(n)
                          for n in ['torch', 'transformers', 'diffusers', 'tokenizers']}
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
    base = records()
    qwen = next(r for r in manifest if r['filename'] == 'Qwen3-Embedding-0.6B/model.safetensors')
    vae = next(r for r in manifest if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    weights = {}
    for name, record in [('base', base['model.safetensors']), ('qwen', qwen), ('vae', vae)]:
        progress(f'verify local {name} checkpoint')
        weights[name] = local_checkpoint(record, explicit.get(name))
    tokenizer_path = ROOT / '.runtime-cache/coreai-bass-conditioning-sources/Qwen3-Embedding-0.6B/tokenizer.json'
    tokenizer_record = next(r for r in manifest if r['filename'] == 'Qwen3-Embedding-0.6B/tokenizer.json')
    verify_checkpoint(tokenizer_path, tokenizer_record)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    condition_folder = candidates / WINDOW.condition_bundle
    preview_folder = candidates / WINDOW.preview_bundle

    def tensor(folder, name, shape):
        data = np.fromfile(folder / name, dtype='<f4').copy()
        if data.size != int(np.prod(shape)) or not np.isfinite(data).all():
            raise ValueError(f'Invalid prepared tensor: {name}')
        return torch.from_numpy(data).reshape(shape)

    old = refs['condition']['prompts']
    new = report['candidate_prompts']
    old_ids = tokenizer.encode(old['lyric_prompt']).ids
    new_ids = tokenizer.encode(new['lyric_prompt']).ids
    saved_ids = tensor(condition_folder, 'lyric_ids.f32le', [1, len(old_ids)])
    if saved_ids.tolist()[0] != old_ids:
        raise ValueError('Tokenizer does not reproduce control lyric IDs')
    report['token_geometry'] = {'text': WINDOW.text_tokens, 'control_lyric': len(old_ids),
                                'candidate_lyric': len(new_ids),
                                'candidate_condition': WINDOW.text_tokens + len(new_ids) + 1}
    source = ROOT / '.runtime-cache/coreai-generator-source-publisher'
    source.mkdir(parents=True, exist_ok=True)
    paths = {n: verified_source(base[n], source) for n in
             ['modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json', 'apg_guidance.py']}
    verify_sampling_source(paths['modeling_acestep_v15_base.py'])
    spec = importlib.util.spec_from_file_location('coreai_drum_lyrics_apg', paths['apg_guidance.py'])
    apg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(apg)
    progress('replay original conditioning and compute instrumental candidate')
    cfg, encoder, _ = load_condition_encoder(paths)
    with safe_open(str(weights['base']), framework='pt', device='cpu') as f:
        state = {k[len('encoder.'):]: f.get_tensor(k) for k in f.keys() if k.startswith('encoder.')}
    encoder.load_state_dict(state, strict=True, assign=True)
    del state
    encoder.lyric_encoder.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
    encoder.timbre_encoder.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
    encoder.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
    text = tensor(condition_folder, 'text_embeddings-fp32-reference.f32le', [1, WINDOW.text_tokens, 1024])
    reference_audio = tensor(condition_folder, 'reference_audio.f32le', [1, 750, 64])
    control = tensor(condition_folder, 'condition-fp32-reference.f32le', [1, WINDOW.condition_tokens, 2048])
    with safe_open(str(weights['qwen']), framework='pt', device='cpu') as f:
        embedding = f.get_tensor('embed_tokens.weight').float()

    def condition(ids):
        lyric = torch.nn.functional.embedding(torch.tensor([ids]), embedding)
        return encoder(text_hidden_states=text, text_attention_mask=torch.ones((1, WINDOW.text_tokens), dtype=torch.bool),
                       lyric_hidden_states=lyric, lyric_attention_mask=torch.ones((1, len(ids)), dtype=torch.bool),
                       refer_audio_acoustic_hidden_states_packed=reference_audio, refer_audio_order_mask=torch.tensor([0]))

    with torch.no_grad():
        replay, replay_mask = condition(old_ids)
        require_parity('condition_fp32', replay, control, report)
        require_parity('condition_mask', replay_mask.float(),
                       tensor(condition_folder, 'condition_mask-fp32-reference.f32le', [1, WINDOW.condition_tokens]), report)
        candidate, candidate_mask = condition(new_ids)
    candidate_tokens = WINDOW.text_tokens + len(new_ids) + 1
    if (list(candidate.shape) != [1, candidate_tokens, 2048]
            or list(candidate_mask.shape) != [1, candidate_tokens]
            or not candidate_mask.all() or not torch.isfinite(candidate).all()):
        raise ValueError('Invalid candidate conditioning shape/values/mask')
    if require_unchanged_condition:
        require_parity('candidate_unchanged_condition', candidate, control, report)
        report['candidate_condition_identical_to_control'] = torch.equal(candidate, control)
        if not report['candidate_condition_identical_to_control']:
            raise ValueError('APG-only comparison requires byte-identical FP32 conditioning')
    candidate = candidate.clone().contiguous()
    candidate.numpy().astype('<f4').tofile(out / 'candidate-condition.f32le')
    del encoder, embedding, text, reference_audio, replay, replay_mask, candidate_mask
    gc.collect()

    progress('load full FP32 DiT and replay original first step')
    cfg, kind, rotary = load_full_upstream(paths)
    with torch.device('meta'):
        model = kind(cfg).eval()
    with safe_open(str(weights['base']), framework='pt', device='cpu') as f:
        state = {k[len('decoder.'):]: f.get_tensor(k) for k in f.keys() if k.startswith('decoder.')}
        null = f.get_tensor('null_condition_emb').float()
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.rotary_emb = rotary(cfg, device='cpu')
    model.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
    context = tensor(candidates / WINDOW.source_bundle, 'context_latents-reference.f32le', [1, WINDOW.frames, 128])
    if candidate_context is not None:
        if (candidate_context.shape != context.shape or candidate_context.dtype != torch.float32
                or candidate_context.device.type != 'cpu' or not torch.isfinite(candidate_context).all()):
            raise ValueError('Invalid candidate source context')
        candidate_context = candidate_context.detach().clone().contiguous()
    noise = tensor(preview_folder, 'initial-noise.f32le', [1, WINDOW.frames, 64])
    times = tensor(preview_folder, 'timesteps.f32le', [51])

    with torch.no_grad():
        replay = guided_step(conditioned_predictor(model, control, null, context), noise, times[:1], times[1:2], apg, apg.MomentumBuffer())
        require_parity('gold_step_1', replay,
                       tensor(preview_folder, 'step-1-latent-fp32-reference.f32le', [1, WINDOW.frames, 64]), report)
        candidate_apg = candidate_apg_factory(apg) if candidate_apg_factory else apg
        if candidate_apg_factory:
            report['candidate_apg_diagnostics'] = candidate_apg.diagnostics
        active_context = context if candidate_context is None else candidate_context
        latent = candidate_trajectory(conditioned_predictor(model, candidate, null, active_context), noise, times, candidate_apg,
                                      lambda step: progress(f'candidate step {step}/50'))
    latent.numpy().astype('<f4').tofile(out / 'candidate-latent.f32le')
    del model, replay
    gc.collect()

    progress('replay cached control decoder and decode candidate')
    vcfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
    with torch.device('meta'):
        decoder = OobleckDecoder(channels=vcfg['decoder_channels'], input_channels=vcfg['decoder_input_channels'],
                                 audio_channels=vcfg['audio_channels'], upsampling_ratios=list(reversed(vcfg['downsampling_ratios'])),
                                 channel_multiples=vcfg['channel_multiples']).eval()
    with safe_open(str(weights['vae']), framework='pt', device='cpu') as f:
        state = {k[len('decoder.'):]: f.get_tensor(k) for k in f.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(state, strict=True, assign=True)
    del state
    decoder.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
    with torch.no_grad():
        replay = decoder(tensor(preview_folder, 'step-50-latent-fp32-reference.f32le', [1, WINDOW.frames, 64]).transpose(1, 2))
        gold_audio = tensor(preview_folder, 'audio-fp32-reference.f32le', [1, 2, WINDOW.samples])
        require_parity('gold_decoder', replay, gold_audio, report)
        audio = decoder(latent.transpose(1, 2)).contiguous()
    if list(audio.shape) != [1, 2, WINDOW.samples] or not torch.isfinite(audio).all():
        raise ValueError('Invalid candidate waveform')
    audio.numpy().astype('<f4').tofile(out / 'candidate-audio.f32le')
    control_audio = read_planar(preview_folder / 'audio-fp32-reference.f32le', WINDOW.samples)
    new_audio = audio.reshape(-1).tolist()
    peaks = {'control': max(abs(v) for v in control_audio), 'candidate': max(abs(v) for v in new_audio)}
    gain = headroom_gain(max(peaks.values()))
    for name, values in [(control_label, control_audio), (candidate_label, new_audio)]:
        write_float_wav(out / f'{name}-raw.wav', values, WINDOW.samples)
        write_float_wav(out / f'{name}-headroom.wav', values, WINDOW.samples, gain)
    report['common_listening_gain'] = gain
    report['raw_sample_peaks'] = peaks
    report['candidate_steps_completed'] = 50
    report['execution_counts'] = {'conditioning_replays': 1, 'candidate_conditioning': 1,
                                  'control_dit_first_step': 1, 'candidate_dit_steps': 50,
                                  'control_decodes': 1, 'candidate_decodes': 1}
    report['candidate_audio_sha256'] = sha256(out / 'candidate-audio.f32le')


def compare(candidates, out, diagnostic=False, execute_requested=False, explicit=None):
    candidates, out = Path(candidates), Path(out)
    if out.exists():
        raise FileExistsError(f'Existing experiment preserved: {out}')
    if out.resolve().is_relative_to(candidates.resolve()):
        raise ValueError('Experiment output must be outside ModelCandidates')
    from export_coreai_source_context import verify_bundle as source
    from export_coreai_bass_conditioning import verify_bundle as condition
    from export_coreai_bass_preview import verify_bundle as preview
    refs = {'source': source(candidates / WINDOW.source_bundle, WINDOW),
            'condition': condition(candidates / WINDOW.condition_bundle, WINDOW)}
    refs['preview'] = preview(candidates / WINDOW.preview_bundle, candidates / WINDOW.source_bundle,
                              candidates / WINDOW.condition_bundle, WINDOW, diagnostic)
    report = {'schema_version': 1, 'experiment': 'drums-host-instrumental-lyrics-only', 'status': 'planned',
              'changed_input': 'lyric text: empty -> [Instrumental]',
              'control_prompts': refs['condition']['prompts'],
              'candidate_prompts': lyric_variant(refs['condition']['prompts']),
              'source_pcm_sha256': refs['source']['artifact_sha256']['audio.f32le'],
              'component_reference_sha256': {k: sha256(candidates / getattr(WINDOW, f'{k}_bundle') / 'reference.json')
                                             for k in ['source', 'condition', 'preview']},
              'control_host_export_status': refs['preview'].get('host_export_status', 'strict_pass'),
              'unchanged': ['piano PCM/context', 'publisher silence timbre', 'caption/instruction/metadata',
                            'initial noise/schedule', '10.24 seconds', '50 APG Euler steps', 'guidance 7', 'checkpoint values'],
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'scope': 'Original FP32 HOST quality experiment only; no BF16/native export or on-device parity claim',
              'baseline_modified': False, 'native_exports': 0, 'musical_acceptance': 'NOT EVALUATED'}
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
        execute(candidates, out, refs, report, progress, explicit or {})
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
    parser.add_argument('--execute', action='store_true', help='Compute one new host FP32 trajectory; otherwise write plan only')
    for name in ['base', 'qwen', 'vae']:
        parser.add_argument(f'--{name}-checkpoint', type=Path)
    args = parser.parse_args()
    report = compare(ROOT / 'prototypes/ios27-coreai/ModelCandidates', args.out,
                     args.diagnostic_trajectory, args.execute,
                     {name: getattr(args, f'{name}_checkpoint') for name in ['base', 'qwen', 'vae']})
    print(report['status'], args.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One host30s reference generation with accepted Colab inputs/settings.

CPU FP32, pinned existing weights only. Capture real inputs/conditioning/noise
and live trajectory. This is not a duration-only ablation or CUDA/native parity.
"""
import argparse
import ast
import __future__
import gc
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import wave
import zipfile

from compare_coreai_drum_lyrics import local_checkpoint, lyric_variant
from compare_coreai_drum_source_gain import load_encoder
from coreai_listening_audio import headroom_gain, write_float_wav
from export_coreai_decoder import ROOT, sha256
from export_coreai_dit_block import records, verified_source
from export_coreai_source_context import posterior_context, verified_handler_source

FRAMES, SAMPLES = 750, 1440000
SOURCE_PCM_SHA = '1d57ce145dcf66d30762d81b2ca26df4798f3b0b0a5d36b303a53b7db7baaf23'
SOURCE_MP3_SHA = 'ccd94ea37159f69c272623efb319a62984564f1799d3744d096b7add49246189'
DRUM_ZIP_SHA = '70400534aac9e2ac14db1d7fc2bdaec31585cec9285e8fe5b60222fd202b8433'
DRUM_WAV_SHA = '29d409250f6104936b7fea215f5b1798c37f52bc8b91b3668003567952476fce'
CAPTURE_STEPS = {1, 5, 10, 25, 50}


def pcm16(raw):
    with wave.open(io.BytesIO(raw)) as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (2, 2, 32000, 960000):
            raise ValueError('Expected30s stereo32kHz PCM16 WAV')
        return stream.readframes(stream.getnframes())


def resample(pcm):
    import numpy as np
    import torch
    from coreai_reference_resample import _get_sinc_resample_kernel, _apply_sinc_resample_kernel
    # Matches the handler's transforms.Resample(32000,48000) default kernel
    # construction, including float64 computation then a cached float32 kernel.
    audio = torch.from_numpy(np.frombuffer(pcm, dtype='<i2').copy().reshape(-1, 2).T.copy()).float() / 32768
    kernel, width = _get_sinc_resample_kernel(32000, 48000, math.gcd(32000, 48000))
    return _apply_sinc_resample_kernel(audio, 32000, 48000, math.gcd(32000, 48000), kernel, width).clamp(-1., 1.).unsqueeze(0).contiguous()


def nearest_pcm16(decoded_float32):
    """Specify PCM rounding explicitly instead of delegating it to FFmpeg/ISA."""
    import numpy as np
    if len(decoded_float32) != 1920000 * 4:
        raise ValueError('Expected finite30s stereo32kHz Float32 decoder output')
    values = np.frombuffer(decoded_float32, dtype='<f4')
    if values.size != 1920000 or not np.isfinite(values).all():
        raise ValueError('Expected finite30s stereo32kHz Float32 decoder output')
    # Float32 -> float64 is exact. Scaling by2**15 is exact; np.rint specifies
    # nearest-even, including negative half-steps. Clipping precedes int16 cast.
    return np.rint(values.astype(np.float64) * 32768).clip(-32768, 32767).astype('<i2').tobytes()


def canonical_source32(source, target):
    decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                              '-ss', '0', '-t', '30', '-af', 'volume=0.7', '-ar', '32000', '-ac', '2',
                              '-f', 'f32le', 'pipe:1'], check=True, capture_output=True).stdout
    with (target.parent / 'source32-float.f32le').open('xb') as stream:
        stream.write(decoded)
    quantized = nearest_pcm16(decoded)
    # Standard WAV header differs from archived FFmpeg metadata. Admission is
    # against unchanged PCM bytes, not the container header.
    with target.open('xb') as stream:
        with wave.open(stream, 'wb') as audio:
            audio.setparams((2, 2, 32000, 0, 'NONE', 'not compressed'))
            audio.writeframes(quantized)
    return {'source32_float_sha256': hashlib.sha256(decoded).hexdigest(),
            'source32_quantization': 'Explicit nearest-even PCM16: np.rint(Float64(Float32)*32768), clip[-32768,32767]',
            'source_gain_before_quantization': .7}


def check_source_pcm(raw, out, input_mode, diagnostics=None):
    """Record the observed input before a strict checksum failure; never admit it."""
    pcm = pcm16(raw)
    actual = hashlib.sha256(pcm).hexdigest()
    try:
        version = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        version = 'unavailable'
    observed = {**(diagnostics or {}), 'input_mode': input_mode, 'source32_WAV_sha256': hashlib.sha256(raw).hexdigest(),
                'expected_source32_PCM_sha256': SOURCE_PCM_SHA, 'actual_source32_PCM_sha256': actual,
                'accepted_PCM_exact': actual == SOURCE_PCM_SHA, 'sample_rate': 32000,
                'channels': 2, 'sample_width_bytes': 2, 'frames': 960000,
                'ffmpeg_version': version, 'platform_system': platform.system(), 'platform_machine': platform.machine(),
                'scope': 'Input checksum diagnostic only; no numerical comparison without original PCM arrays, no permission to bypass gate'}
    with (out / 'source-preparation.json').open('x') as stream:
        json.dump(observed, stream, indent=2); stream.write('\n')
    if diagnostics is not None:
        diagnostics.update(observed)
    if actual != SOURCE_PCM_SHA:
        raise ValueError('Prepared PCM differs from accepted baseline; see source-preparation.json; no inference or reference replacement')
    return pcm


def prepare_source(out, source=None, prepared_source=None, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    target = out / 'source32.wav'
    if prepared_source is not None:
        raw = Path(prepared_source).read_bytes()
        pcm = check_source_pcm(raw, out, 'existing_prepared_WAV_no_second_gain', diagnostics)
        with target.open('xb') as stream:
            stream.write(raw)
    else:
        if source is None or sha256(source) != SOURCE_MP3_SHA:
            raise ValueError('Select original accepted piano.mp3; source hash mismatch/missing')
        diagnostics.update(canonical_source32(Path(source), target))
        pcm = check_source_pcm(target.read_bytes(), out, 'original_MP3_float32_explicit_nearest_PCM16', diagnostics)
        if sha256(source) != SOURCE_MP3_SHA:
            raise ValueError('Original source changed during preparation')
    audio = resample(pcm)
    audio.numpy().astype('<f4').tofile(out / 'source48.f32le')
    return audio, {'source32_WAV_sha256': sha256(target), 'source32_PCM_sha256': SOURCE_PCM_SHA,
                   'accepted_PCM_exact': True, 'source48_planar_sha256': sha256(out / 'source48.f32le'),
                   'source_gain_already_applied': .7, 'additional_gain': 1.,
                   'resampling': 'Pinned TorchAudio2.10 transforms.Resample defaults; CPU torch implementation; clamp[-1,1] as handler',
                   'resampler_source_sha256': sha256(ROOT / 'scripts/coreai_reference_resample.py')}


def prompts30():
    from export_coreai_bass_conditioning import format_prompts
    old = format_prompts(10.24, 'drums')
    path, _ = verified_handler_source('metadata_utils.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MetadataMixin')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'_build_metadata_dict', '_dict_to_meta_string'}]
    if len(cls.body) != 2:
        raise ValueError('Pinned metadata formatter changed')
    ns = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), ns)
    formatter = ns['MetadataMixin']()
    metadata = formatter._dict_to_meta_string(formatter._build_metadata_dict(None, '', '', 30.))
    if old['text_prompt'].count(old['metadata']) != 1:
        raise ValueError('Expected one metadata field in frozen prompt')
    return lyric_variant({**old, 'metadata': metadata, 'audio_seconds': 30.,
                          'text_prompt': old['text_prompt'].replace(old['metadata'], metadata)})


def encode_source(audio, checkpoint, out, progress):
    import torch
    progress('encode full30s source once; explicit CPU epsilon seed31')
    encoder = load_encoder(checkpoint)
    epsilon = torch.randn((1, 64, FRAMES), generator=torch.Generator(device='cpu').manual_seed(31))
    mask = torch.ones(1, FRAMES, 64)
    with torch.no_grad():
        outputs = posterior_context(encoder(audio), epsilon, mask)
    for name, value in [('source-epsilon', epsilon), ('source-mask', mask)] + list(zip(
            ['source-mean', 'source-std', 'source-latent', 'source-context'], outputs)):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite source encoding')
        value.numpy().astype('<f4').tofile(out / (name + '.f32le'))
    if [list(v.shape) for v in outputs] != [[1, 64, FRAMES], [1, 64, FRAMES], [1, FRAMES, 64], [1, FRAMES, 128]]:
        raise ValueError('Unexpected30s source geometry')
    # The pinned LEGO full-range branch preserves source and appends all-True
    # mask channels, independent of the instrument's caption.
    if not torch.equal(outputs[-1][..., :64], outputs[-2]) or not torch.equal(outputs[-1][..., 64:], mask):
        raise ValueError('Source/mask preservation failed')
    context = outputs[-1].clone()
    del encoder, outputs
    gc.collect()
    return context


def condition(prompts, weights, paths, out, progress):
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import Qwen3Config, Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from export_coreai_bass_conditioning import load_condition_encoder
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
    q = {r['filename']: r for r in manifest if r['filename'].startswith('Qwen3-Embedding-0.6B/')}
    cache = ROOT / '.runtime-cache/coreai-bass-conditioning-sources'
    (cache / 'Qwen3-Embedding-0.6B').mkdir(parents=True, exist_ok=True)
    config = verified_source(q['Qwen3-Embedding-0.6B/config.json'], cache)
    tokenizer_path = local_checkpoint(q['Qwen3-Embedding-0.6B/tokenizer.json'],
                                      cache / 'Qwen3-Embedding-0.6B/tokenizer.json')
    tok = Tokenizer.from_file(str(tokenizer_path))
    ids = [tok.encode(prompts[k]).ids for k in ['text_prompt', 'lyric_prompt']]
    if not 1 <= len(ids[0]) <= 256 or not 1 <= len(ids[1]) <= 2048:
        raise ValueError('Unexpected reference prompt lengths; no truncation')
    text_ids, lyric_ids = [torch.tensor([i]) for i in ids]
    for name, value in [('text-ids', text_ids), ('lyric-ids', lyric_ids)]:
        value.float().numpy().astype('<f4').tofile(out / (name + '.f32le'))
    progress('compute complete Qwen text and literal[Instrumental] lyric embeddings')
    cfg = Qwen3Config(**json.loads(config.read_text())); cfg._attn_implementation = 'eager'; cfg.use_cache = False
    with torch.device('meta'):
        qwen = Qwen3Model(cfg).eval()
    with safe_open(str(weights['qwen']), framework='pt', device='cpu') as stream:
        state = {k: stream.get_tensor(k) for k in stream.keys()}
    qwen.load_state_dict(state, strict=True, assign=True); del state
    qwen.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
    qwen.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    with torch.no_grad():
        text = qwen(input_ids=text_ids, attention_mask=torch.ones_like(text_ids), use_cache=False).last_hidden_state
        lyric = qwen.embed_tokens(lyric_ids)
    text.numpy().astype('<f4').tofile(out / 'text-embeddings.f32le')
    lyric.numpy().astype('<f4').tofile(out / 'lyric-embeddings.f32le')
    del qwen
    gc.collect()
    progress('compute original condition encoder with publisher silence; no kit reference')
    cfg, encoder, _ = load_condition_encoder(paths)
    with safe_open(str(weights['base']), framework='pt', device='cpu') as stream:
        state = {k[len('encoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('encoder.')}
    encoder.load_state_dict(state, strict=True, assign=True); del state
    encoder.lyric_encoder.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
    encoder.timbre_encoder.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
    encoder.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
    silence = torch.load(weights['silence'], map_location='cpu', weights_only=True).float().transpose(1, 2)[:, :750].contiguous()
    if list(silence.shape) != [1, 750, 64]:
        raise ValueError('Unexpected publisher silence reference')
    silence.numpy().astype('<f4').tofile(out / 'reference-audio.f32le')
    with torch.no_grad():
        packed, mask = encoder(text_hidden_states=text, text_attention_mask=torch.ones_like(text_ids, dtype=torch.bool),
                               lyric_hidden_states=lyric, lyric_attention_mask=torch.ones_like(lyric_ids, dtype=torch.bool),
                               refer_audio_acoustic_hidden_states_packed=silence, refer_audio_order_mask=torch.tensor([0]))
    if list(packed.shape) != [1, len(ids[0]) + len(ids[1]) + 1, 2048] or not mask.all() or not torch.isfinite(packed).all():
        raise ValueError('Invalid packed conditioning')
    packed.numpy().astype('<f4').tofile(out / 'condition.f32le'); mask.float().numpy().astype('<f4').tofile(out / 'condition-mask.f32le')
    result = packed.clone()
    del encoder
    gc.collect()
    return result, {'text': len(ids[0]), 'lyric': len(ids[1]), 'packed': packed.shape[1]}


def trajectory(predict, noise, times, apg, out, progress, report):
    import torch
    if list(noise.shape) != [1, FRAMES, 64] or list(times.shape) != [51] or times[0] != 1 or times[-1] != 0 or not (times[:-1] > times[1:]).all():
        raise ValueError('Invalid fixed50-step reference geometry/schedule')
    latent = noise.clone(); momentum = apg.MomentumBuffer()
    noise.numpy().astype('<f4').tofile(out / 'initial-noise.f32le')
    times.numpy().astype('<f4').tofile(out / 'timesteps.f32le')
    report['step_metrics'] = []
    with torch.no_grad():
        for i in range(50):
            conditional, unconditional = predict(latent, times[i:i+1]).chunk(2)
            guided = apg.apg_forward(conditional, unconditional, 7., momentum_buffer=momentum, norm_threshold=2.5, dims=[1])
            if any(t.shape != noise.shape or not torch.isfinite(t).all() for t in [conditional, unconditional, guided]):
                raise ValueError(f'Invalid velocity at step{i+1}')
            latent = latent - guided * (times[i] - times[i+1])
            if latent.shape != noise.shape or not torch.isfinite(latent).all():
                raise ValueError(f'Invalid live latent at step{i+1}')
            if i+1 in CAPTURE_STEPS:
                for name, value in [('conditional', conditional), ('unconditional', unconditional), ('guided', guided),
                                    ('momentum', momentum.running_average), ('latent', latent)]:
                    value.numpy().astype('<f4').tofile(out / f'step-{i+1:02d}-{name}.f32le')
            report['step_metrics'].append({'step': i+1, 'latent_rms': float(latent.square().mean().sqrt()),
                                           'guided_rms': float(guided.square().mean().sqrt())})
            report['steps_completed'] = i+1
            progress(f'reference step{i+1}/50')
    return latent


def cached_predictor(model, packed, null, context):
    """Match publisher cross-attention KV reuse with one fresh trajectory cache."""
    import torch
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    condition = torch.cat([packed, null.expand_as(packed)])
    source = torch.cat([context, context])
    def predict(latent, time):
        nonlocal cache
        result = model(hidden_states=torch.cat([latent, latent]), timestep=time.expand(2), timestep_r=time.expand(2),
                       attention_mask=torch.ones(2, latent.shape[1], dtype=torch.long),
                       encoder_hidden_states=condition, encoder_attention_mask=torch.ones(2, packed.shape[1], dtype=torch.long),
                       context_latents=source, use_cache=True, past_key_values=cache)
        if len(result) < 2 or result[1] is None:
            raise ValueError('Publisher reference cache was not returned')
        cache = result[1]
        return result[0]
    return predict


def execute(out, source, prepared_source, report, progress, explicit):
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    from export_coreai_denoiser import load_full_upstream
    from prepare_coreai_denoiser_loop import verify_sampling_source
    torch.set_num_threads(2)
    report['versions'] = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'diffusers', 'tokenizers']}
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())['files']; base = records()
    entries = {'base': base['model.safetensors'], 'silence': base['silence_latent.pt']}
    entries.update({label: next(r for r in manifest if r['filename'] == name) for label, name in
                   [('qwen', 'Qwen3-Embedding-0.6B/model.safetensors'), ('vae', 'vae/diffusion_pytorch_model.safetensors')]})
    weights = {}
    for label, record in entries.items():
        progress('verify existing pinned ' + label + ' checkpoint')
        weights[label] = local_checkpoint(record, explicit.get(label))
    report['checkpoint_sha256'] = {k: r['digest'] for k, r in entries.items()}
    progress('prepare30s Float32 source and explicit nearest-even PCM16; require exact reference')
    report['source_preparation_diagnostic'] = {}
    audio, metadata = prepare_source(out, source, prepared_source, report['source_preparation_diagnostic'])
    report['source_preparation'] = metadata
    context = encode_source(audio, weights['vae'], out, progress); report['source_encoder_calls'] = 1
    publisher_cache = ROOT / '.runtime-cache/coreai-generator-source-publisher'
    publisher_cache.mkdir(parents=True, exist_ok=True)
    paths = {name: verified_source(base[name], publisher_cache) for name in
             ['modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json', 'apg_guidance.py']}
    verify_sampling_source(paths['modeling_acestep_v15_base.py'])
    report['publisher_source_sha256'] = {k: sha256(v) for k, v in paths.items()}
    packed, geometry = condition(report['prompts'], weights, paths, out, progress); report['token_geometry'] = geometry
    progress('load original FP32 DiT for fixed30s independent trajectory')
    cfg, kind, rotary = load_full_upstream(paths)
    with torch.device('meta'):
        model = kind(cfg).eval()
    with safe_open(str(weights['base']), framework='pt', device='cpu') as stream:
        state = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
        null = stream.get_tensor('null_condition_emb').float()
    model.load_state_dict(state, strict=True, assign=True); del state
    model.rotary_emb = rotary(cfg, device='cpu')
    model.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    spec = importlib.util.spec_from_file_location('coreai_reference_original_apg', paths['apg_guidance.py'])
    apg = importlib.util.module_from_spec(spec); spec.loader.exec_module(apg)
    predict = cached_predictor(model, packed, null, context)
    noise = torch.randn((1, FRAMES, 64), generator=torch.Generator(device='cpu').manual_seed(0))
    times = torch.linspace(1., 0., 51)
    latent = trajectory(predict, noise, times, apg, out, progress, report)
    latent.numpy().astype('<f4').tofile(out / 'final-latent.f32le')
    del predict, model, packed, null, context
    gc.collect()
    progress('decode full30s final latent once; no song stitching')
    cfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
    with torch.device('meta'):
        decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                                 audio_channels=cfg['audio_channels'], upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                                 channel_multiples=cfg['channel_multiples']).eval()
    with safe_open(str(weights['vae']), framework='pt', device='cpu') as stream:
        state = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(state, strict=True, assign=True); del state
    decoder.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    with torch.no_grad():
        generated = decoder(latent.transpose(1, 2)).contiguous()
    if list(generated.shape) != [1, 2, SAMPLES] or not torch.isfinite(generated).all():
        raise ValueError('Invalid30s generated waveform')
    generated.numpy().astype('<f4').tofile(out / 'final-audio.f32le'); report['decoder_calls'] = 1
    archive = ROOT / 'baselines/acoustic-drums-30-v1/drum_result.zip'
    if sha256(archive) != DRUM_ZIP_SHA:
        raise ValueError('Approved drum archive mismatch')
    with zipfile.ZipFile(archive) as bundle:
        raw = bundle.read('drums.wav')
    if hashlib.sha256(raw).hexdigest() != DRUM_WAV_SHA:
        raise ValueError('Approved drum WAV mismatch')
    approved = resample(pcm16(raw))
    gain = headroom_gain(max(float(approved.abs().max()), float(generated.abs().max())))
    for name, value in [('01-colab-reference-headroom.wav', approved), ('02-host-reference-headroom.wav', generated),
                        ('03-host-piano-drums-headroom.wav', (audio + generated) * .5)]:
        write_float_wav(out / name, value.reshape(-1).numpy(), SAMPLES, gain)
    report.update(common_listening_gain=gain, raw_sample_peaks={'approved': float(approved.abs().max()), 'generated': float(generated.abs().max())},
                  reference_mix='0.5 prepared piano +0.5 generated drums; production iPhone mix unchanged')


def generate(out, source=None, prepared_source=None, execute_requested=False, explicit=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Existing reference result preserved: {out}')
    if any(out.resolve().is_relative_to((ROOT / p).resolve()) for p in ['baselines', 'prototypes/ios27-coreai/ModelCandidates']):
        raise ValueError('Reference output must be outside baseline/ModelCandidates')
    if source is not None and prepared_source is not None:
        raise ValueError('Choose original MP3 or prepared source, not both')
    report = {'schema_version': 1, 'status': 'planned', 'experiment': 'drums-host30s-reference-reproduction',
              'scope': 'One captured original-FP32 host reference with accepted input/settings. Not a duration-only ablation, exact CUDA replay, native numerical PASS or adopted fix.',
              'duration_seconds': 30, 'latent_frames': FRAMES, 'steps_requested': 50, 'guidance_scale': 7,
              'APG_norm_threshold': 2.5, 'APG_momentum': -.75, 'APG_projection_dimensions': [1],
              'cache_policy': 'Publisher EncoderDecoderCache retained across steps; layer self-attention cache disabled by original source, cross-attention reused',
              'source_epsilon_CPU_seed': 31, 'diffusion_noise_CPU_seed': 0,
              'RNG_limit': 'Actual tensors saved; same seed does not establish original CUDA RNG/source posterior parity',
              'prompts': prompts30(), 'source_encoder_calls': 0, 'decoder_calls': 0, 'steps_completed': 0,
              'numeric_parity': 'NOT EVALUATED; original CUDA intermediate tensors absent',
              'existing_preview_provenance': 'bounded_diagnostic; original full trajectory gate unchanged, no new PASS',
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'musical_acceptance': 'NOT EVALUATED', 'baseline_modified': False, 'native_exports': 0}
    out.mkdir(parents=True, exist_ok=False)
    def save():
        temporary = out / 'report.json.tmp'; temporary.write_text(json.dumps(report, indent=2) + '\n'); temporary.replace(out / 'report.json')
    def progress(phase):
        report['phase'] = phase; save(); print(phase, flush=True)
    save()
    if not execute_requested:
        return report
    try:
        report['status'] = 'running'
        execute(out, source, prepared_source, report, progress, explicit or {})
        if report['steps_completed'] != 50:
            raise ValueError('Incomplete reference trajectory')
        report['status'] = 'completed_reference_generation'
        report['artifact_sha256'] = {p.name: sha256(p) for p in out.iterdir() if p.name != 'report.json'}
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print('Ready:', out, flush=True)
    return report


def package_source_failure(out):
    """Package only preparation artifacts from this failed run, without retrying."""
    out = Path(out)
    report_path = out / 'report.json'
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'failed' or not report.get('phase', '').startswith('prepare')
            or any(report.get(key) != 0 for key in ['steps_completed', 'source_encoder_calls', 'decoder_calls'])
            or not (out / 'source32-float.f32le').is_file()):
        return None
    archive = out / 'source-preparation-review.zip'
    if archive.exists():
        return None
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ['report.json', 'source-preparation.json', 'source32.wav', 'source32-float.f32le']:
            path = out / name
            if path.is_file() and not path.is_symlink():
                bundle.write(path, name)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--source', type=Path, help='Original accepted piano.mp3')
    group.add_argument('--prepared-source', type=Path, help='Existing canonical30s stereo32kHz PCM16 input; no second gain')
    parser.add_argument('--execute', action='store_true', help='Run one full50-step30s host reference; default plan only')
    for name in ['base', 'qwen', 'vae', 'silence']:
        parser.add_argument(f'--{name}-checkpoint', type=Path)
    args = parser.parse_args()
    if args.execute and args.source is None and args.prepared_source is None:
        if args.out.exists():
            raise FileExistsError('Output already exists; preserved')
        if sys.platform != 'darwin':
            parser.error('--source or --prepared-source required outside macOS')
        args.source = Path(subprocess.check_output(['osascript', '-e',
            'POSIX path of (choose file with prompt "Pilih piano.mp3 asli untuk acuan drum30 detik")'], text=True).strip())
    preexisting = args.out.exists()
    try:
        generate(args.out, args.source, args.prepared_source, args.execute,
                 {name: getattr(args, f'{name}_checkpoint') for name in ['base', 'qwen', 'vae', 'silence']})
    except Exception:
        if not preexisting:
            try:
                archive = package_source_failure(args.out)
                if archive:
                    print('Preparation diagnostic ZIP:', archive, flush=True)
                    if sys.platform == 'darwin':
                        subprocess.run(['open', '-R', str(archive)], check=False)
            except (OSError, ValueError):
                pass  # Preserve the original preparation error if packaging fails.
        raise


if __name__ == '__main__':
    main()

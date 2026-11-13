#!/usr/bin/env python3
"""Prepare three fixed30s host conditioning caches with unchanged127-token geometry.

Plan-only by default. Execute uses existing pinned local weights, Qwen once in
memory then ConditionEncoder once; no DiT/VAE, exports or profile activation.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

from export_coreai_decoder import ROOT, sha256

ROLES = {'bass': 'bass', 'drums': 'drums', 'guitar': 'guitar-strum'}
SUFFIX = ' No percussion or vocals.'
SHAPES = {'text-ids': [1, 111], 'lyric-ids': [1, 15],
          'text-embeddings': [1, 111, 1024], 'lyric-embeddings': [1, 15, 1024],
          'reference-audio': [1, 750, 64], 'condition': [1, 127, 2048],
          'condition-mask': [1, 127]}


def role_prompts(drums, originals):
    """Keep frozen drums byte-for-byte; create separately identified30s presets."""
    if (drums.get('audio_seconds') != 30. or drums.get('lyrics') != '[Instrumental]'
            or drums.get('vocal_language') != 'unknown'):
        raise ValueError('Expected fixed30s instrumental control')
    result = {'drums': dict(drums)}
    for role in ('bass', 'guitar'):
        original = originals[role]
        caption = original['caption'] + SUFFIX
        text = drums['text_prompt']
        for old, new in [(drums['instruction'], original['instruction']), (drums['caption'], caption)]:
            if text.count(old) != 1:
                raise ValueError('Ambiguous frozen prompt substitution')
            text = text.replace(old, new, 1)
        definition = {'schema_version': 1, 'role': role, 'source_preset_sha256': original['preset_sha256'],
                      'instruction': original['instruction'], 'caption': caption,
                      'text_prompt': text, 'lyric_prompt': drums['lyric_prompt'], 'audio_seconds': 30.}
        digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        result[role] = {**drums, 'instruction': original['instruction'], 'caption': caption,
                        'text_prompt': text, 'source_preset_sha256': original['preset_sha256'],
                        'preset_sha256': digest, 'preset_definition': definition}
    return {role: result[role] for role in ROLES}


def local_file(record, path):
    """Verify cached SHA256 or Git blob identity without a downloader."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record['size']:
        raise FileNotFoundError(f'Pinned local file missing/wrong size: {path}; no download attempted')
    if len(record['digest']) == 64:
        digest = sha256(path)
    else:
        raw = path.read_bytes()
        digest = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    if digest != record['digest']:
        raise ValueError(f'Pinned local checksum mismatch: {path}; preserved')
    return path


def formatted_prompts():
    # The older shared formatter can download missing sources. Preflight every
    # required local source before invoking it; cached integrity is checked there.
    manifest = json.loads((ROOT / 'configs/coreai-lego-preprocessing.json').read_text())
    for name in ('constants.py', 'prompt_utils.py', 'metadata_utils.py'):
        record = next(r for r in manifest['files'] if r['path'].endswith('/' + name))
        candidates = [ROOT / '.runtime-cache/ace-conditioning-upstream' / record['path'],
                      ROOT / '.runtime-cache/coreai-conditioning-source' / name]
        if not any(p.is_file() for p in candidates):
            raise FileNotFoundError(f'Pinned formatter not cached: {name}; no download attempted')
    from export_coreai_bass_conditioning import format_prompts
    from generate_coreai_drum_reference import prompts30
    return role_prompts(prompts30(), {role: format_prompts(10.24, ROLES[role]) for role in ('bass', 'guitar')})


def tokenize(prompts, tokenizer):
    result = {}
    for role, prompt in prompts.items():
        text = tokenizer.encode(prompt['text_prompt']).ids
        lyric = tokenizer.encode(prompt['lyric_prompt']).ids
        if len(text) != 111 or len(lyric) != 15:
            raise ValueError(f'{role}: expected natural111/15 tokens; no padding/truncation allowed')
        result[role] = {'text': text, 'lyric': lyric}
    if any(ids['lyric'] != result['drums']['lyric'] for ids in result.values()):
        raise ValueError('Literal instrumental lyrics changed between roles')
    return result


def inspect(args):
    from tokenizers import Tokenizer
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
    record = next(r for r in manifest if r['filename'] == 'Qwen3-Embedding-0.6B/tokenizer.json')
    tokenizer_path = local_file(record, args.tokenizer or args.qwen_cache / record['filename'])
    prompts = formatted_prompts()
    ids = tokenize(prompts, Tokenizer.from_file(str(tokenizer_path)))
    return prompts, ids, record


def check_output(out, protected):
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Existing output preserved: {out}; choose a new --out')
    if any(out.resolve().is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError('Output must be outside captures, models, baselines and checkpoint/source caches')


def execute(args):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from audit_coreai_drum_reference30 import verify_capture, read_tensor
    from compare_coreai_drum_lyrics import local_checkpoint
    from export_coreai_bass_conditioning import BASE_DIGEST, QWEN_DIGEST, SILENCE_DIGEST, load_condition_encoder
    from export_coreai_dit_block import precision_metrics
    from package_coreai_drum_reference import ACCEPTED_REPORT_SHA256

    protected = [args.capture, args.publisher_dir, args.qwen_cache, ROOT / 'baselines',
                 ROOT / 'vendor', ROOT / 'prototypes/ios27-coreai/ModelCandidates',
                 ROOT / '.runtime-cache/coreai-conditioning-source']
    protected += [p.parent for p in (args.base_checkpoint, args.qwen_checkpoint, args.silence_latent) if p]
    check_output(args.out, protected)
    capture = verify_capture(args.capture)
    prompts, ids, tokenizer_record = inspect(args)
    if prompts['drums'] != capture['prompts']:
        raise ValueError('Frozen drum prompt differs; no control replacement')
    manifest = {r['filename']: r for r in json.loads((ROOT / 'configs/assets-base.json').read_text())['files']}
    entries = {'base': ('model.safetensors', BASE_DIGEST, args.base_checkpoint),
               'qwen': ('Qwen3-Embedding-0.6B/model.safetensors', QWEN_DIGEST, args.qwen_checkpoint),
               'silence': ('silence_latent.pt', SILENCE_DIGEST, args.silence_latent)}
    weights = {}
    for label, (name, digest, override) in entries.items():
        if manifest[name]['digest'] != digest or capture['checkpoint_sha256'][label] != digest:
            raise ValueError('Original checkpoint pins differ')
        print('Verify existing local checkpoint:', label, flush=True)
        weights[label] = local_checkpoint(manifest[name], override)
    paths = {name: local_file(manifest[name], args.publisher_dir / name) for name in
             ('modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json')}
    config = local_file(manifest['Qwen3-Embedding-0.6B/config.json'], args.qwen_cache / 'Qwen3-Embedding-0.6B/config.json')
    args.out.mkdir(parents=True)
    report = {'schema_version': 1, 'profile': 'full-song-conditioning-127-v1', 'status': 'running',
              'scope': 'Host fixed30s conditioning only; no native/full-song numerical or musical acceptance',
              'capture_report_sha256': ACCEPTED_REPORT_SHA256, 'tokenizer_sha256': tokenizer_record['digest'],
              'checkpoint_sha256': {k: v[1] for k, v in entries.items()},
              'publisher_source_sha256': {k: sha256(v) for k, v in paths.items()},
              'prompts': prompts, 'geometry': {'text': 111, 'lyric': 15, 'packed': 127},
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'host_compute_dtype': 'float32', 'sampler_weight_storage_dtype': 'bfloat16',
              'sampler_weights_modified': False,
              'Qwen_model_loads': 0, 'Qwen_text_calls': 0, 'ConditionEncoder_model_loads': 0,
              'ConditionEncoder_calls': 0, 'DiT_calls': 0, 'VAE_calls': 0, 'native_exports': 0,
              'baseline_modified': False, 'profile_activated': False, 'padding_tokens': 0, 'truncated_tokens': 0,
              'ready_for_staging': False, 'ready_for_native_execution': False,
              'original_full_sampler_gate': 'NOT PASSED; prior4/6 retained independently',
              'drums_control_checks': {}, 'artifact_sha256': {}, 'tensor_shapes': {},
              'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers', 'safetensors')}}

    def save():
        temporary = args.out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.out / 'report.json')

    def write(role, name, value):
        if list(value.shape) != SHAPES[name] or not torch.isfinite(value).all():
            raise ValueError(f'Invalid {role}/{name} tensor')
        relative = role + '/' + name + '.f32le'
        path = args.out / relative
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:
            stream.write(value.detach().cpu().numpy().astype('<f4').tobytes())
        report['artifact_sha256'][relative] = sha256(path)
        report['tensor_shapes'][relative] = SHAPES[name]
        if role == 'drums':
            reference = read_tensor(args.capture, name + '.f32le')
            metric = precision_metrics(value, reference)
            # IDs and the all-valid mask require exact equality as well as parity.
            if name in ('text-ids', 'lyric-ids', 'condition-mask'):
                metric['exact_equal'] = bool(torch.equal(value, reference))
                metric['parity_passed'] &= metric['exact_equal']
            report['drums_control_checks'][name] = metric
        save()

    torch.set_num_threads(2)
    save()
    try:
        text_ids = {role: torch.tensor([value['text']]) for role, value in ids.items()}
        lyric_ids = torch.tensor([ids['drums']['lyric']])
        cfg = Qwen3Config(**json.loads(config.read_text()))
        cfg._attn_implementation = 'eager'
        cfg.use_cache = False
        with torch.device('meta'):
            qwen = Qwen3Model(cfg).eval()
        with safe_open(str(weights['qwen']), framework='pt', device='cpu') as stream:
            state = {k: stream.get_tensor(k) for k in stream.keys()}
        qwen.load_state_dict(state, strict=True, assign=True)
        del state
        qwen.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
        qwen.to(device='cpu', dtype=torch.float32).requires_grad_(False)
        report['Qwen_model_loads'] = 1
        texts = {}
        with torch.no_grad():
            lyric = qwen.embed_tokens(lyric_ids)
            for role in ROLES:
                print('Qwen text:', role, flush=True)
                texts[role] = qwen(input_ids=text_ids[role], attention_mask=torch.ones_like(text_ids[role]), use_cache=False).last_hidden_state
                report['Qwen_text_calls'] += 1
                write(role, 'text-ids', text_ids[role].float())
                write(role, 'lyric-ids', lyric_ids.float())
                write(role, 'text-embeddings', texts[role])
                write(role, 'lyric-embeddings', lyric)
        del qwen
        gc.collect()
        cfg, encoder, _ = load_condition_encoder(paths)
        with safe_open(str(weights['base']), framework='pt', device='cpu') as stream:
            state = {k[len('encoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('encoder.')}
        encoder.load_state_dict(state, strict=True, assign=True)
        del state
        encoder.lyric_encoder.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
        encoder.timbre_encoder.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device='cpu')
        encoder.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
        report['ConditionEncoder_model_loads'] = 1
        silence = torch.load(weights['silence'], map_location='cpu', weights_only=True).float().transpose(1, 2)[:, :750].contiguous()
        with torch.no_grad():
            for role in ROLES:
                print('ConditionEncoder:', role, flush=True)
                write(role, 'reference-audio', silence)
                packed, mask = encoder(text_hidden_states=texts[role], text_attention_mask=torch.ones_like(text_ids[role], dtype=torch.bool),
                                       lyric_hidden_states=lyric, lyric_attention_mask=torch.ones_like(lyric_ids, dtype=torch.bool),
                                       refer_audio_acoustic_hidden_states_packed=silence, refer_audio_order_mask=torch.tensor([0]))
                report['ConditionEncoder_calls'] += 1
                if list(mask.shape) != [1, 127] or not mask.all():
                    raise ValueError('Expected127 valid conditioning tokens')
                write(role, 'condition', packed)
                write(role, 'condition-mask', mask.float())
        del encoder
        gc.collect()
        checks = report['drums_control_checks']
        if set(checks) != set(SHAPES) or not all(m['parity_passed'] for m in checks.values()):
            raise ValueError('Drum conditioning replay NOT PASSED; original gates unchanged, outputs retained')
        report.update(status='host_conditioning_captured', ready_for_staging=True)
        save()
        # Small private review package: no weights, source PCM, latent trajectory
        # or decoder audio. Report remains outside the archive too.
        with zipfile.ZipFile(args.out / 'review.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.out / 'report.json', 'report.json')
            for name in sorted(report['artifact_sha256']):
                archive.write(args.out / name, name)
        print('Three host caches captured; native/full-song NOT EVALUATED:', args.out / 'review.zip', flush=True)
    except BaseException as error:
        report.update(status='failed', ready_for_staging=False, error_type=type(error).__name__, error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--inspect-tokens', action='store_true', help='Verify cached tokenizer/prompts only; no neural models or output writes')
    parser.add_argument('--capture', type=Path, default=ROOT / '.runtime-cache/drums-reference30-v3')
    parser.add_argument('--out', type=Path, default=ROOT / '.runtime-cache/full-song-conditioning-v1')
    parser.add_argument('--publisher-dir', type=Path, default=ROOT / '.runtime-cache/coreai-generator-source-publisher')
    parser.add_argument('--qwen-cache', type=Path, default=ROOT / '.runtime-cache/coreai-bass-conditioning-sources')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--qwen-checkpoint', type=Path)
    parser.add_argument('--silence-latent', type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args)
    elif args.inspect_tokens:
        prompts, ids, _ = inspect(args)
        print(json.dumps({role: {'text': len(v['text']), 'lyric': len(v['lyric']), 'packed': len(v['text']) + len(v['lyric']) + 1,
                                'preset_sha256': prompts[role]['preset_sha256']} for role, v in ids.items()}, indent=2))
    else:
        print(json.dumps({'status': 'PLAN ONLY; no model execution', 'roles': list(ROLES),
                          'geometry': {'text': 111, 'lyric': 15, 'packed': 127}, 'neural_calls': 0,
                          'ready_for_native_execution': False, 'baseline_modified': False}, indent=2))


if __name__ == '__main__':
    main()

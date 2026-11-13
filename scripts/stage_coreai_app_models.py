#!/usr/bin/env python3
"""Stage only selected CoreAI resources into an Xcode-built .app (stdlib only).

Source models/caches are read-only. Sync/pruning is confined to the generated
.app/ModelCandidates directory. Model integrity remains checked by the runtime;
this incremental copier does not rehash gigabytes of weights on every build.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
from coreai_preview_window import SHORT, LONG
from coreai_drum_reference30_contract import PROFILE as DRUM30_PROFILE, BUNDLE as DRUM30_BUNDLE, FIXTURES as DRUM30_FIXTURES, KEEP as DRUM30_KEEP, validate_reference as validate_drum30
from coreai_drum_reference30_contract import DECODER_PROFILE as DRUM30_DECODER_PROFILE, DECODER_KEEP as DRUM30_DECODER_KEEP
from coreai_drum_reference30_window_contract import PROFILE as WINDOW_PROFILE, BUNDLE as WINDOW_BUNDLE, DATA_SHAPES as WINDOW_SHAPES, validate as validate_window
from coreai_drum_reference30_generation_contract import PROFILE as GENERATION_PROFILE, CONFIG as GENERATION_CONFIG, KEEP as GENERATION_KEEP, SHAPES as GENERATION_SHAPES, validate as validate_generation
from coreai_source_window_contract import PROFILE as SOURCE_WINDOW_PROFILE, BUNDLE as SOURCE_WINDOW_BUNDLE, DATA_SHAPES as SOURCE_WINDOW_SHAPES, validate_reference as validate_source_window
from coreai_instrument_preview30_contract import PROFILE as INSTRUMENT30_PROFILE, BUNDLE as INSTRUMENT30_BUNDLE, DATA_SHAPES as INSTRUMENT30_SHAPES, validate as validate_instrument30

PROFILES = ('auto', 'bass-aot', 'bass-source', 'bass-source-1024', 'dit-precision-1024', 'dit-interleaving-1024', 'runtime', 'all')
INSTRUMENT_PROFILES = tuple(i+'-source-1024' for i in ('drums','guitar-strum','guitar-pick'))
PROFILES += INSTRUMENT_PROFILES
PROFILES += (DRUM30_PROFILE, DRUM30_DECODER_PROFILE)
PROFILES += (WINDOW_PROFILE,)
PROFILES += (GENERATION_PROFILE,)
PROFILES += (SOURCE_WINDOW_PROFILE,)
PROFILES += (INSTRUMENT30_PROFILE,)
STAMP = 'app-resource-manifest.json'
PREVIEW_FIXTURES = {'initial-noise.f32le', 'timesteps.f32le',
                    'audio-reference.f32le', 'audio-fp32-reference.f32le'} | {
    f'step-{step}-latent-{kind}.f32le'
    for step in (1, 10, 25, 50) for kind in ('reference', 'fp32-reference')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()  # Small JSON only.


def relative_file(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts
            or str(relative) != name or '\\' in name):
        raise ValueError(f'Unsafe resource path: {name}')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or external resource: {path}')
    return path


def load_reference(root, bundle):
    path = relative_file(root, bundle + '/reference.json')
    reference = json.loads(path.read_text())
    hashes = reference.get('artifact_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f'Missing artifact manifest: {path}')
    for name, value in hashes.items():
        # Validate paths even if the selected view omits some original weights.
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or str(relative) != name
                or '\\' in name or name == 'reference.json'
                or not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)):
            raise ValueError(f'Invalid artifact manifest entry: {bundle}/{name}')
    return reference


def plan_resources(source, profile='auto'):
    """Validate the complete selection before any build output can be pruned."""
    source = Path(source).resolve()
    if profile not in PROFILES:
        raise ValueError(f'Unknown resource profile: {profile}')
    ready = all((source / name / 'reference.json').is_file()
                for name in ('source-context', 'bass-conditioning', 'bass-preview-512'))
    diagnostic_trajectory = False
    if profile == 'auto':
        # The supplied h18p/iOS 27.0.1 backtrace crashes in the Metal/MPS
        # compiler for AOT/default. Keep CPU-only source as the default;
        # compiled AOT remains an explicit diagnostic profile on the Mac.
        selector = source / 'active-preview.json'
        if selector.exists():
            if selector.is_symlink():raise ValueError('Active preview selector must not be a symlink')
            chosen=json.loads(selector.read_text())
            if not isinstance(chosen,dict):raise ValueError('Invalid active preview selector')
            expected={'schema_version':1,'resource_profile':chosen.get('resource_profile'),'audio_seconds':chosen.get('audio_seconds')}
            diagnostic_trajectory=chosen.get('diagnostic_trajectory',False)
            if (type(diagnostic_trajectory) is not bool or not isinstance(chosen,dict)
                    or set(chosen)-{'schema_version','resource_profile','audio_seconds','diagnostic_trajectory'}
                    or expected not in ({'schema_version':1,'resource_profile':'bass-source','audio_seconds':5.12},
                                        {'schema_version':1,'resource_profile':'bass-source-1024','audio_seconds':10.24},
                                        {'schema_version':1,'resource_profile':'dit-precision-1024','audio_seconds':10.24},
                                        {'schema_version':1,'resource_profile':'dit-interleaving-1024','audio_seconds':10.24},
                                        {'schema_version':1,'resource_profile':DRUM30_PROFILE,'audio_seconds':30.},
                                        {'schema_version':1,'resource_profile':DRUM30_DECODER_PROFILE,'audio_seconds':30.},
                                        {'schema_version':1,'resource_profile':WINDOW_PROFILE,'audio_seconds':30.},
                                        {'schema_version':1,'resource_profile':GENERATION_PROFILE,'audio_seconds':30.},
                                        {'schema_version':1,'resource_profile':SOURCE_WINDOW_PROFILE,'audio_seconds':30.},
                                        {'schema_version':1,'resource_profile':INSTRUMENT30_PROFILE,'audio_seconds':30.},
                                        *({'schema_version':1,'resource_profile':p,'audio_seconds':10.24} for p in INSTRUMENT_PROFILES))
                    or chosen.get('schema_version')!=1
                    or (diagnostic_trajectory and chosen.get('resource_profile') not in ('bass-source-1024',*INSTRUMENT_PROFILES))):
                raise ValueError('Invalid active preview selector; build output preserved')
            profile=chosen['resource_profile']
        else:
            profile = 'bass-source' if ready else 'runtime'
    window = LONG if profile == 'bass-source-1024' else SHORT
    if profile in INSTRUMENT_PROFILES:
        from coreai_preview_window import PreviewWindow
        window = PreviewWindow.for_seconds(10.24,profile.removesuffix('-source-1024'))
    planned = {}

    def add(name, expected=None):
        path = relative_file(source, name)
        stat = path.stat()
        planned[name] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': expected}

    def bundle(name, keep=None):
        ref = load_reference(source, name)
        add(name + '/reference.json', digest(source / name / 'reference.json'))
        hashes = ref['artifact_sha256']
        if keep is not None and not keep.issubset(hashes):
            raise ValueError(f'Missing required fixture hashes: {name}')
        for filename in sorted(hashes if keep is None else keep):
            add(name + '/' + filename, hashes[filename])
        return ref

    if profile == 'all':
        for path in sorted(source.rglob('*')):
            if path.is_symlink() and path.is_dir():
                raise ValueError(f'Directory symlink is unsupported: {path}')
            if path.is_file() and path.name != '.DS_Store' and path != source / STAMP:
                add(path.relative_to(source).as_posix())
    elif profile == SOURCE_WINDOW_PROFILE:
        validate_source_window(load_reference(source,SOURCE_WINDOW_BUNDLE))
        bundle(SOURCE_WINDOW_BUNDLE)
        for name,shape in SOURCE_WINDOW_SHAPES.items():
            if planned[SOURCE_WINDOW_BUNDLE+'/'+name]['size']!=math.prod(shape)*4:
                raise ValueError('Source window fixture size mismatch; build output preserved')
    elif profile in (GENERATION_PROFILE, INSTRUMENT30_PROFILE):
        validate_drum30(relative_file(source, DRUM30_BUNDLE + '/reference.json').read_bytes())
        validate_window(load_reference(source, WINDOW_BUNDLE))
        config=relative_file(source,GENERATION_CONFIG)
        validate_generation(json.loads(config.read_text()),digest(source/WINDOW_BUNDLE/'reference.json'))
        bundle(DRUM30_BUNDLE,GENERATION_KEEP)
        bundle(WINDOW_BUNDLE)
        add(GENERATION_CONFIG,digest(config))
        for b,shapes in [(DRUM30_BUNDLE,GENERATION_SHAPES),(WINDOW_BUNDLE,WINDOW_SHAPES)]:
            for name,shape in shapes.items():
                if planned[b+'/'+name]['size']!=math.prod(shape)*4:raise ValueError('Frozen generation fixture size mismatch; build output preserved')
        if profile == INSTRUMENT30_PROFILE:
            validate_instrument30(load_reference(source, INSTRUMENT30_BUNDLE))
            bundle(INSTRUMENT30_BUNDLE)
            for name, shape in INSTRUMENT30_SHAPES.items():
                if planned[INSTRUMENT30_BUNDLE+'/'+name]['size'] != math.prod(shape)*4:
                    raise ValueError('Instrument conditioning shape mismatch; build output preserved')
    elif profile == WINDOW_PROFILE:
        ref = load_reference(source, WINDOW_BUNDLE)
        validate_window(ref)
        bundle(WINDOW_BUNDLE)
        for name, shape in WINDOW_SHAPES.items():
            if planned[WINDOW_BUNDLE + '/' + name]['size'] != math.prod(shape)*4:
                raise ValueError('Window decoder fixture size mismatch; build output preserved')
    elif profile in (DRUM30_PROFILE, DRUM30_DECODER_PROFILE):
        validate_drum30(relative_file(source, DRUM30_BUNDLE + '/reference.json').read_bytes())
        keep = DRUM30_DECODER_KEEP if profile == DRUM30_DECODER_PROFILE else DRUM30_KEEP
        bundle(DRUM30_BUNDLE, keep)
        for name, shape in DRUM30_FIXTURES.items():
            if name in keep and planned[DRUM30_BUNDLE + '/' + name]['size'] != math.prod(shape) * 4:
                raise ValueError('Native30 component fixture size mismatch; build output preserved')
    elif profile == 'dit-interleaving-1024':
        from coreai_interleaving_contract import validate
        validate(load_reference(source, profile))
        bundle(profile)
    elif profile == 'dit-precision-1024':
        # Validate before adding files: this profile must never pull the full
        # generator or extra historical probes into the app.
        probe_ref = load_reference(source, profile)
        precision_files = {name + '.f32le' for name in ['hidden','condition','temb','rope_cos','rope_sin','self_mask','local_mask','cross_mask']}
        precision_files.add('hidden-reference.f32le')
        precision_files.update(asset + '/' + name for asset in ['blocks-bf16.aimodel','blocks-fp32.aimodel']
                               for name in ['main.hash','main.mlirb','metadata.json'])
        if set(probe_ref['artifact_sha256']) != precision_files:
            raise ValueError('Unexpected precision-probe payload; build output preserved')
        ref = bundle(profile)
        if (ref.get('profile') != profile or ref.get('schema_version') != 1
                or ref.get('assets') != {'bfloat16':'blocks-bf16.aimodel','float32':'blocks-fp32.aimodel'}
                or ref.get('layer_indices') != [0,1] or ref.get('output_shape') != [2,128,2048]
                or ref.get('pass_order') != ['bfloat16','float32','bfloat16']
                or ref.get('runs_per_pass') != 4 or ref.get('learned_parameter_values_identical') is not True):
            raise ValueError('Invalid precision-probe resource contract; build output preserved')
    elif profile == 'runtime':
        # A fresh clone can still run the tensor probe without prepared weights.
        model = source / 'affine.aimodel'
        if model.exists():
            for path in sorted(model.rglob('*')):
                if path.is_file():
                    add(path.relative_to(source).as_posix())
            if model.is_file():
                add(model.name)
            add('affine-native-reference.json', digest(source / 'affine-native-reference.json'))
    else:
        source_ref = bundle(window.source_bundle)
        condition_ref = bundle(window.condition_bundle)
        preview = bundle(window.preview_bundle, PREVIEW_FIXTURES if profile == 'bass-aot' else None)
        if profile in INSTRUMENT_PROFILES:
            if (condition_ref.get('prompts',{}).get('preset_sha256')!=window.preset_sha256
                    or condition_ref.get('prompts',{}).get('instruction')!=window.instruction):
                raise ValueError('Instrument preset identity changed; build output preserved')
        if preview.get('host_trajectory_diagnostic_requested',False) and not diagnostic_trajectory:
            raise ValueError('Diagnostic trajectory bundle requires explicit active selector permission')
        if diagnostic_trajectory and preview.get('host_trajectory_diagnostic_requested') is not True:
            raise ValueError('Diagnostic selector requires a component-validated diagnostic bundle')
        if window.frames == 256:
            if (source_ref.get('profile') != window.source_bundle or source_ref.get('audio_seconds') != window.seconds
                    or source_ref.get('input_shapes', {}).get('audio') != [1,2,window.samples]
                    or condition_ref.get('profile') != window.condition_bundle
                    or condition_ref.get('weight_storage_dtype') != window.weight_storage_dtype
                    or condition_ref.get('text_tokens') != window.text_tokens
                    or condition_ref.get('prompts', {}).get('audio_seconds') != window.seconds
                    or preview.get('profile') != window.preview_bundle
                    or preview.get('audio_seconds') != window.seconds
                    or preview.get('reference_conditioning_scope') != 'original_fp32_vs_storage16_conditioning'
                    or preview.get('weight_storage_dtype') != window.weight_storage_dtype
                    or preview.get('latent_frames') != window.frames
                    or preview.get('condition_tokens') != window.condition_tokens):
                raise ValueError('Mixed or invalid 10.24-second resource contracts; build output preserved')
        for key, name in [('source_reference_sha256', window.source_bundle),
                          ('condition_reference_sha256', window.condition_bundle)]:
            if preview.get(key) != digest(source / name / 'reference.json'):
                raise ValueError(f'Preview component identity changed: {name}')
        if profile == 'bass-aot':
            aot = bundle('bass-preview-512-aot')
            names = ['front', *[f'blocks-{i:02d}-{i+1:02d}' for i in range(0, 24, 2)],
                     'tail', 'apg-euler', 'decoder']
            arch = aot.get('architecture', '')
            expected = {n + '.aimodel': n + '.' + arch + '.aimodelc' for n in names}
            if (aot.get('profile') != 'bass-preview-512-aot' or aot.get('schema_version') != 1
                    or not re.fullmatch(r'[a-zA-Z0-9_-]+', arch)
                    or aot.get('platform') != 'iOS' or aot.get('runtime_policy') != 'default'
                    or aot.get('preferred_compute') != 'none' or aot.get('compiled_asset_count') != 16
                    or aot.get('assets') != expected
                    or aot.get('source_reference_sha256') != digest(source / 'bass-preview-512/reference.json')):
                raise ValueError('AOT identity/variant contract changed; source models preserved')
            hashes = aot['artifact_sha256']
            if (any(not any(k == v or k.startswith(v + '/') for v in expected.values()) for k in hashes)
                    or any(not any(k == v or k.startswith(v + '/') for k in hashes) for v in expected.values())):
                raise ValueError('AOT variant payload coverage is incomplete')
    inventory = sum(p.stat().st_size for p in source.rglob('*')
                    if p.is_file() and not p.is_symlink() and p.name != '.DS_Store')
    return {'schema_version': 1, 'selected_profile': profile, 'diagnostic_trajectory': diagnostic_trajectory, 'files': planned,
            'included_bytes': sum(item['size'] for item in planned.values()),
            'source_inventory_bytes': inventory}


def stage_resources(source, destination, profile='auto', dry_run=False):
    source = Path(source).resolve()
    destination = Path(destination)
    if (destination.name != 'ModelCandidates' or destination.parent.suffix != '.app'
            or destination.is_symlink() or destination.parent.is_symlink()):
        raise ValueError('Destination must be a generated .app/ModelCandidates directory without symlinks')
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and app resource output must be separate')
    plan = plan_resources(source, profile)
    if dry_run:
        return {**plan, 'copied_files': 0, 'removed_files': 0, 'dry_run': True}
    destination.mkdir(parents=True, exist_ok=True)
    previous = {}
    stamp = destination / STAMP
    if stamp.is_file() and not stamp.is_symlink():
        try:
            previous = json.loads(stamp.read_text()).get('files', {})
        except (ValueError, AttributeError):
            pass
    copied = removed = 0
    # Prune only generated app resources, including legacy whole-folder copies.
    for path in sorted(destination.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        name = path.relative_to(destination).as_posix()
        if path.is_symlink() or (path.is_file() and name not in plan['files'] and name != STAMP):
            path.unlink()
            removed += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    for name, metadata in plan['files'].items():
        target = destination / name
        if target.is_file():
            stat = target.stat()
            if (stat.st_size == metadata['size'] and stat.st_mtime_ns == metadata['mtime_ns']
                    and previous.get(name) == metadata):
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        copied += 1
    encoded = json.dumps(plan, indent=2, sort_keys=True) + '\n'
    if stamp.is_symlink():
        stamp.unlink()
    if not stamp.exists() or stamp.read_text() != encoded:
        stamp.write_text(encoded)
    return {**plan, 'copied_files': copied, 'removed_files': removed, 'dry_run': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='auto')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = stage_resources(args.source, args.destination, args.profile, args.dry_run)
    included = result['included_bytes'] / 1e9
    excluded = (result['source_inventory_bytes'] - result['included_bytes']) / 1e9
    print(f"CoreAI resources: {result['selected_profile']}; included {included:.3f} GB; "
          f"excluded {excluded:.3f} GB; {len(result['files'])} files; "
          f"copied {result['copied_files']}; removed stale {result['removed_files']}"
          + ('; dry-run, no files changed' if args.dry_run else ''), flush=True)


if __name__ == '__main__':
    main()

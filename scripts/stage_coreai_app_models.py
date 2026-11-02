#!/usr/bin/env python3
"""Stage only selected CoreAI resources into an Xcode-built .app (stdlib only).

Source models/caches are read-only. Sync/pruning is confined to the generated
.app/ModelCandidates directory. Model integrity remains checked by the runtime;
this incremental copier does not rehash gigabytes of weights on every build.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

PROFILES = ('auto', 'bass-aot', 'bass-source', 'runtime', 'all')
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
    if profile == 'auto':
        profile = ('bass-aot' if (source / 'bass-preview-512-aot').exists() else 'bass-source') if ready else 'runtime'
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
        bundle('source-context')
        bundle('bass-conditioning')
        preview = bundle('bass-preview-512', PREVIEW_FIXTURES if profile == 'bass-aot' else None)
        for key, name in [('source_reference_sha256', 'source-context'),
                          ('condition_reference_sha256', 'bass-conditioning')]:
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
    return {'schema_version': 1, 'selected_profile': profile, 'files': planned,
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

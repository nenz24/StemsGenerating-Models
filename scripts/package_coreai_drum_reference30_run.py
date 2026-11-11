#!/usr/bin/env python3
"""Package an existing iPhone frozen30 run for private numerical review; no inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid
import zipfile

from package_coreai_drum_reference import copy_tensor

SHAPES = {f'native-step-{step:02d}-latent.f32le': [1, 750, 64]
          for step in (1, 5, 10, 25, 50)}
SHAPES.update({'final-latent.f32le': [1, 750, 64],
               'final-momentum.f32le': [1, 750, 64],
               'raw-drums.f32le': [1, 2, 1440000]})
VELOCITY_SHAPES = {f'native-step-{step:02d}-velocity.f32le': [2, 750, 64]
                   for step in range(1, 51)}


def validate_velocity_trace(raw, report, run_id):
    if len(raw) > 1024 * 1024:
        raise ValueError('Velocity metadata exceeds1MiB')
    metadata = json.loads(raw)
    expected = {'schema_version': 1, 'profile': 'native-frozen30-velocity-trace',
                'run_id': run_id, 'steps': 50, 'shape': [2, 750, 64], 'dtype': 'float32_le'}
    digest = hashlib.sha256(raw).hexdigest()
    if (any(metadata.get(k) != v for k, v in expected.items())
            or set(metadata.get('artifact_sha256', {})) != set(VELOCITY_SHAPES)
            or f'Velocity trace SHA256: {digest}; tensors50/50;' not in report):
        raise ValueError('Velocity trace/report identity mismatch or incomplete trace')
    return metadata


def run_folder(source, run_id):
    source = Path(source).absolute()
    if run_id == 'latest':
        roots = [source, source/'Documents', source/'AppData/Documents']
        pointers = [p/'last-drums-reference30-generation.json' for p in roots
                    if (p/'last-drums-reference30-generation.json').exists()]
        if len(pointers) != 1:
            raise ValueError('Select one container/Documents directory with a latest-run pointer')
        pointer = regular_file(pointers[0])
        if pointer.stat().st_size > 4096 or any(p.is_symlink() for p in pointer.parents):
            raise ValueError('Invalid latest-run pointer')
        run_id = json.loads(pointer.read_bytes())['run_id']
    run_id = str(uuid.UUID(run_id)).upper()
    candidates = [source, source / run_id,
                  source / 'DrumReference30Generation' / run_id,
                  source / 'Documents/DrumReference30Generation' / run_id,
                  source / 'AppData/Documents/DrumReference30Generation' / run_id]
    for candidate in candidates:
        if candidate.name != run_id or not candidate.is_dir():
            continue
        if any(p.is_symlink() for p in (candidate, *candidate.parents)):
            raise ValueError('Linked source directories are not accepted')
        return candidate.resolve(), run_id
    raise ValueError('Run folder not found. Select the downloaded .xcappdata or its run folder.')


def regular_file(path, size=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked file: {path.name}')
    if size is not None and path.stat().st_size != size:
        raise ValueError(f'Wrong-size tensor: {path.name}')
    return path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package(source, out, run_id, require_velocity_trace=False):
    folder, run_id = run_folder(source, run_id)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Existing output preserved: {out}')
    if out.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('Choose an output outside the source/container')
    report_path = regular_file(folder / 'report.txt')
    if report_path.stat().st_size > 1024 * 1024:
        raise ValueError('Report exceeds 1MiB')
    raw = report_path.read_bytes()
    report = raw.decode('utf-8')
    required = [f'Runtime: CoreAI frozen drums30 generation; run {run_id}',
                'Reference30 execution: COMPLETED; steps50/50; sampling calls750/750; decoder calls6/6',
                'WAV generated on device: YES']
    revision = re.search(r'^Revision: reference30-live50-window256-v([12]);', report, re.M)
    if not revision or not all(line in report for line in required):
        raise ValueError('Not the selected completed frozen30 run; original report preserved')
    if require_velocity_trace and revision[1] != '2':
        raise ValueError(f'Selected run {run_id} is revisionv1 without velocity data. '
                         'Check the latest report on iPhone and download its new completed v2 container. '
                         'No archive created; do not repeat generation if a completed v2 run already exists.')
    parity = re.search(r'^Original checkpoint/audio strict parity: (PASS|NOT PASSED); (\d)/6 comparisons passed;', report, re.M)
    final_sha = re.search(r'^LIVE final latent SHA256: ([a-f0-9]{64})$', report, re.M)
    if not parity or not final_sha:
        raise ValueError('Missing original numerical verdict or final latent identity')
    shapes = dict(SHAPES)
    trace_raw = None
    trace = None
    if revision[1] == '2':
        trace_path = regular_file(folder/'velocity-trace.json')
        if trace_path.stat().st_size > 1024 * 1024:
            raise ValueError('Velocity metadata exceeds1MiB')
        trace_raw = trace_path.read_bytes()
        trace = validate_velocity_trace(trace_raw, report, run_id)
        shapes.update(VELOCITY_SHAPES)
    entries = {}
    for name, shape in shapes.items():
        path = regular_file(folder / name, math.prod(shape) * 4)
        entries[name] = {'shape': shape, 'dtype': 'float32_le',
                         'bytes': path.stat().st_size, 'sha256': digest(path)}
        if trace is not None and name in VELOCITY_SHAPES and entries[name]['sha256'] != trace['artifact_sha256'][name]:
            raise ValueError('Velocity tensor differs from device metadata: ' + name)
    if entries['final-latent.f32le']['sha256'] != final_sha[1]:
        raise ValueError('Final latent differs from the original report')
    if entries['native-step-50-latent.f32le']['sha256'] != final_sha[1]:
        raise ValueError('Checkpoint50 differs from the saved final latent')
    manifest = {
        'schema_version': 1, 'profile': 'private-coreai-frozen30-live-run',
        'run_id': run_id, 'report_sha256': hashlib.sha256(raw).hexdigest(),
        'original_strict_parity': parity[1], 'original_comparisons_passed': int(parity[2]),
        'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
        'new_inference_calls': 0, 'baseline_modified': False,
        'scope': 'Existing sparse checkpoints/final momentum/raw planar stereo audio; no missing intermediate states reconstructed',
        'integrity': 'Report binds final latent; other tensor hashes record these exported bytes, not independent device provenance',
        'privacy': 'User-derived tensors/audio; private review only. Never commit this ZIP or its contents.',
        'artifacts': entries,
    }
    if trace_raw is not None:
        manifest.update(velocity_trace_steps=50, velocity_trace_sha256=hashlib.sha256(trace_raw).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.drums-live-run-', suffix='.zip', dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('report.txt', raw)
            archive.writestr('run-manifest.json', json.dumps(manifest, indent=2) + '\n')
            if trace_raw is not None:
                archive.writestr('velocity-trace.json', trace_raw)
            for name, entry in entries.items():
                with archive.open(name, 'w') as destination:
                    copy_tensor(folder / name, destination, entry['sha256'], entry['bytes'])
        if Path(temporary).stat().st_size > 32 * 1024 * 1024:
            raise ValueError('Archive exceeds 32MiB')
        os.link(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Downloaded .xcappdata, Documents or exact run folder')
    parser.add_argument('--run-id', required=True, help='UUID from original report, or latest for the container pointer')
    parser.add_argument('--out', type=Path, required=True, help='New private ZIP outside the container')
    parser.add_argument('--require-velocity-trace', action='store_true',
                        help='Reject historical v1 input before writing a ZIP; require all50 report-bound velocities')
    args = parser.parse_args()
    manifest = package(args.source, args.out, args.run_id, args.require_velocity_trace)
    print(f'Run: {manifest["run_id"]}; velocity trace {manifest.get("velocity_trace_steps", 0)}/50')
    print(f'Ready: {args.out} ({args.out.stat().st_size} bytes; {len(manifest["artifacts"])} existing tensors)')
    print(f'Original strict parity retained: {manifest["original_strict_parity"]}. No generation or baseline changes.')


if __name__ == '__main__':
    main()

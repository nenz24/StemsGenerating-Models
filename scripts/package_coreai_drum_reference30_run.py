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


def run_folder(source, run_id):
    source = Path(source).absolute()
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


def package(source, out, run_id):
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
                'Revision: reference30-live50-window256-v1;',
                'Reference30 execution: COMPLETED; steps50/50; sampling calls750/750; decoder calls6/6',
                'WAV generated on device: YES']
    if not all(line in report for line in required):
        raise ValueError('Not the selected completed frozen30 run; original report preserved')
    parity = re.search(r'^Original checkpoint/audio strict parity: (PASS|NOT PASSED); (\d)/6 comparisons passed;', report, re.M)
    final_sha = re.search(r'^LIVE final latent SHA256: ([a-f0-9]{64})$', report, re.M)
    if not parity or not final_sha:
        raise ValueError('Missing original numerical verdict or final latent identity')
    entries = {}
    for name, shape in SHAPES.items():
        path = regular_file(folder / name, math.prod(shape) * 4)
        entries[name] = {'shape': shape, 'dtype': 'float32_le',
                         'bytes': path.stat().st_size, 'sha256': digest(path)}
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.drums-live-run-', suffix='.zip', dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('report.txt', raw)
            archive.writestr('run-manifest.json', json.dumps(manifest, indent=2) + '\n')
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
    parser.add_argument('--run-id', required=True, help='UUID from the original device report')
    parser.add_argument('--out', type=Path, required=True, help='New private ZIP outside the container')
    args = parser.parse_args()
    manifest = package(args.source, args.out, args.run_id)
    print(f'Ready: {args.out} ({args.out.stat().st_size} bytes; 8 existing tensors)')
    print(f'Original strict parity retained: {manifest["original_strict_parity"]}. No generation or baseline changes.')


if __name__ == '__main__':
    main()

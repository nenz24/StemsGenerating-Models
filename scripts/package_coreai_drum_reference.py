#!/usr/bin/env python3
"""Package the accepted host30s capture for native-port review; no inference."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_REPORT_SHA256 = 'f5f84314fa37f49f73502a77f7e3364b31450898231731d64e216bf1b606bfdf'
MAX_ZIP_BYTES = 32 * 1024 * 1024
SHAPES = {
    'source48.f32le': [1, 2, 1440000],
    'source-epsilon.f32le': [1, 64, 750],
    'source-mean.f32le': [1, 64, 750],
    'source-std.f32le': [1, 64, 750],
    'source-mask.f32le': [1, 750, 64],
    'source-latent.f32le': [1, 750, 64],
    'source-context.f32le': [1, 750, 128],
    'text-ids.f32le': [1, 111],
    'lyric-ids.f32le': [1, 15],
    'text-embeddings.f32le': [1, 111, 1024],
    'lyric-embeddings.f32le': [1, 15, 1024],
    'reference-audio.f32le': [1, 750, 64],
    'condition.f32le': [1, 127, 2048],
    'condition-mask.f32le': [1, 127],
    'initial-noise.f32le': [1, 750, 64],
    'timesteps.f32le': [51],
    'final-latent.f32le': [1, 750, 64],
    'final-audio.f32le': [1, 2, 1440000],
}
SHAPES.update({f'step-{step:02d}-{name}.f32le': [1, 750, 64]
               for step in (1, 5, 10, 25, 50)
               for name in ('conditional', 'unconditional', 'guided', 'momentum', 'latent')})


def checked_file(path, size):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f'Missing, linked or wrong-size capture: {path.name}')


def copy_tensor(path, destination, expected_sha256, size):
    """Hash and validate exactly the bytes written to the archive."""
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as source:
        while chunk := source.read(256 * 1024):
            count += len(chunk)
            if count > size or len(chunk) % 4:
                raise ValueError(f'Invalid capture length: {path.name}')
            values = array.array('f')
            values.frombytes(chunk)
            if sys.byteorder != 'little':
                values.byteswap()
            if not all(map(math.isfinite, values)):
                raise ValueError(f'Nonfinite capture: {path.name}')
            digest.update(chunk)
            destination.write(chunk)
    if count != size or digest.hexdigest() != expected_sha256:
        raise ValueError(f'Capture checksum mismatch: {path.name}')


def package(folder, out):
    folder, out = Path(folder).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Existing output preserved: {out}')
    resolved = out.resolve()
    if (resolved.is_relative_to(folder) or any(resolved.is_relative_to(ROOT / p)
            for p in ('baselines', 'prototypes/ios27-coreai/ModelCandidates'))):
        raise ValueError('Archive must be outside the capture, baselines and ModelCandidates')
    report_path = folder / 'report.json'
    if report_path.is_symlink() or not report_path.is_file() or report_path.stat().st_size > 1024 * 1024:
        raise ValueError('Missing or invalid reference report')
    raw = report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACCEPTED_REPORT_SHA256:
        raise ValueError('This is not the accepted completed host30s report; do not regenerate or replace it')
    report = json.loads(raw)
    entries = {}
    for name, shape in SHAPES.items():
        size = math.prod(shape) * 4
        checked_file(folder / name, size)
        entries[name] = {'shape': shape, 'dtype': 'float32_le', 'bytes': size,
                         'sha256': report['artifact_sha256'][name]}
    manifest = {
        'schema_version': 1, 'profile': 'accepted-drums-host30s-capture',
        'report_sha256': ACCEPTED_REPORT_SHA256,
        'human_listening': 'User accepted files01/02/03; original report retained verbatim',
        'native_parity': 'NOT EVALUATED', 'native_exports': 0,
        'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3,
        'relative_rmse_tolerance': 1e-3,
        'scope': 'Existing FP32 captures only; not a native asset bundle, numerical PASS or full-song acceptance',
        'layout': 'Contiguous row-major tensors with the recorded shapes; source48/final-audio are planar stereo',
        'step_layout': 'Post-update latent/momentum; conditional/unconditional/guided are velocities used at that step',
        'null_condition': 'Not captured; obtain null_condition_emb from the same pinned base checkpoint for native export',
        'privacy': 'Contains user-derived audio/tensors; private review only, never commit the ZIP or contents',
        'artifacts': entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.drums-reference-', suffix='.zip', dir=out.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('report.json', raw)
            archive.writestr('capture-manifest.json', json.dumps(manifest, indent=2) + '\n')
            for name, entry in entries.items():
                with archive.open(name, 'w') as destination:
                    copy_tensor(folder / name, destination, entry['sha256'], entry['bytes'])
        if Path(temporary).stat().st_size > MAX_ZIP_BYTES:
            raise ValueError('Archive exceeds32MiB; no final archive published')
        # Publish without replacing an existing destination, including a racing writer.
        os.link(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing completed drums-reference30-v3 folder')
    parser.add_argument('--out', type=Path, required=True, help='New private ZIP, outside source folder')
    args = parser.parse_args()
    manifest = package(args.source, args.out)
    print(f'Ready: {args.out} ({args.out.stat().st_size} bytes; {len(manifest["artifacts"])} verified tensors)')
    print('No generation, weights, native export, active profile or baseline changed.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Package existing bass/guitar30s runs for private transition review; no inference.

Bind saved raw audio to report-identified WAV bytes by exact Float32 gain replay.
Include original latent checkpoints and last8s listening clips; preserve sources.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile

from coreai_instrument_preview30_contract import PRESETS
from coreai_listening_audio import read_planar, write_float_wav
from export_coreai_decoder import ROOT, sha256
from package_coreai_drum_reference import copy_tensor
from package_coreai_drum_reference30_run import regular_file

ROLES = ('bass', 'guitar')
FRAMES = 1440000
MAX_ZIP = 32 * 1024 * 1024


def shapes(role):
    if role not in ROLES: raise ValueError('Select bass or guitar')
    return {**{f'native-step-{step:02d}-latent.f32le': [1, 750, 64] for step in (1, 5, 10, 25, 50)},
            'final-latent.f32le': [1, 750, 64], 'final-momentum.f32le': [1, 750, 64],
            'raw-' + role + '.f32le': [1, 2, FRAMES]}


def documents_roots(source):
    source = Path(source).absolute()
    return [source, source / 'Documents', source / 'AppData/Documents']


def run_folder(source, role, run_id):
    if role not in ROLES: raise ValueError('Select bass or guitar')
    roots = documents_roots(source)
    if run_id == 'latest':
        pointers = [root / f'last-instrument-reference30-{role}.json' for root in roots
                    if (root / f'last-instrument-reference30-{role}.json').exists()]
        if len(pointers) != 1: raise ValueError(f'Select one fresh CoreAIProbe container with the {role} run pointer')
        pointer = regular_file(pointers[0])
        if pointer.stat().st_size > 4096 or any(p.is_symlink() for p in pointer.parents):
            raise ValueError('Invalid latest pointer')
        run_id = json.loads(pointer.read_text())['run_id']
    run_id = str(uuid.UUID(run_id)).upper()
    candidates = [Path(source).absolute()] + [root / 'InstrumentReference30' / role / run_id for root in roots]
    matches = set()
    for path in candidates:
        if path.name == run_id and path.is_dir():
            if any(p.is_symlink() for p in (path, *path.parents)): raise ValueError('Linked run directory rejected')
            matches.add(path.resolve())
    if len(matches) != 1:
        raise ValueError(f'{role} run {run_id} not found. Download the current CoreAIProbe container; do not repeat generation.')
    return matches.pop(), run_id


def report_metadata(report, role, run_id):
    required = [f'Runtime: CoreAI instrument30 preview; instrument {role}; run {run_id}',
                'Revision: instrument30-shared-sampler-v1;',
                'BF16 learned DiT storage, FP32 compute/APG/I/O, FP32 VAE; original abs1e-4/rel1e-3/nRMSE1e-3',
                'Reference30 execution: COMPLETED; steps50/50; sampling calls750/750; decoder calls6/6',
                f'{role} numerical parity: NOT EVALUATED;', 'WAV generated on device: YES']
    if not all(value in report for value in required): raise ValueError('Not the selected completed instrument30 run')
    condition = re.search(r'^Condition preset SHA256: ([a-f0-9]{64}); condition SHA256: ([a-f0-9]{64}); all127 mask values valid$', report, re.M)
    latent = re.search(r'^LIVE final latent SHA256: ([a-f0-9]{64})$', report, re.M)
    wav = re.search(r'^Artifact SHA256: ' + role + r' ([a-f0-9]{64}); mix ([a-f0-9]{64}); piano ([a-f0-9]{64})$', report, re.M)
    gain = re.search(r'^Listening copies: shared gain([0-9.eE+-]+), diagnostic0.5piano\+0.5' + role + ';', report, re.M)
    if not all((condition, latent, wav, gain)) or condition[1] != PRESETS[role]:
        raise ValueError('Missing/mismatched instrument preset, latent, gain or WAV identity')
    value = float(gain[1])
    if not math.isfinite(value) or not 0 < value <= 1: raise ValueError('Invalid listening gain')
    # Swift Float.description is round-trip decimal; recover its exact FP32 value.
    value = struct.unpack('<f', struct.pack('<f', value))[0]
    return {'run_id': run_id, 'instrument': role, 'preset_sha256': condition[1], 'condition_sha256': condition[2],
            'final_latent_sha256': latent[1], 'solo_wav_sha256': wav[1], 'mix_wav_sha256': wav[2],
            'piano_wav_sha256': wav[3], 'listening_gain_float32': value,
            'numerical_parity': 'NOT EVALUATED', 'musical_acceptance': 'NOT ACCEPTED; transition glitches reported'}


def verify_audio_binding(raw_path, wav_path, gain, expected_sha, temporary, frames=FRAMES):
    """Recreate the exact device Float32 WAV; no inverse-gain approximation."""
    regular_file(wav_path, 58 + frames * 8)
    if sha256(wav_path) != expected_sha: raise ValueError('Solo WAV differs from device report')
    values = read_planar(raw_path, frames)
    recreated = temporary / 'recreated.wav'
    write_float_wav(recreated, values, frames, gain)
    if sha256(recreated) != expected_sha:
        raise ValueError('Raw audio does not reproduce the report-bound WAV at its original Float32 gain')
    return values


def package(source, out, run_ids):
    source, out = Path(source).absolute(), Path(out).absolute()
    if out.exists() or out.is_symlink(): raise FileExistsError(f'Existing review preserved: {out}; choose a new --out')
    if out.resolve().is_relative_to(source.resolve()) or any(out.resolve().is_relative_to(ROOT / p)
            for p in ('baselines', 'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Output must be outside container, models and baselines')
    folders, metadata, reports, entries = {}, {}, {}, {}
    for role in ROLES:
        folder, run_id = run_folder(source, role, run_ids[role])
        report_path = regular_file(folder / 'report.txt')
        if report_path.stat().st_size > 1024 * 1024: raise ValueError('Report exceeds1MiB')
        raw = report_path.read_bytes()
        metadata[role] = report_metadata(raw.decode(), role, run_id)
        metadata[role]['report_sha256'] = hashlib.sha256(raw).hexdigest()
        reports[role], folders[role] = raw, folder
        for name, shape in shapes(role).items():
            path = regular_file(folder / name, math.prod(shape) * 4)
            entries[role + '/' + name] = {'shape': shape, 'dtype': 'float32_le', 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if any(entries[role + '/' + name]['sha256'] != metadata[role]['final_latent_sha256']
               for name in ('final-latent.f32le', 'native-step-50-latent.f32le')):
            raise ValueError('Final latent/checkpoint50/report mismatch')
    if (metadata['bass']['listening_gain_float32'] == metadata['guitar']['listening_gain_float32']
            and metadata['bass']['piano_wav_sha256'] != metadata['guitar']['piano_wav_sha256']):
        raise ValueError('Select runs with identical piano listening copies/gain')
    manifest = {'schema_version': 1, 'profile': 'private-instrument30-transition-review',
                'runs': metadata, 'artifacts': entries, 'absolute_tolerance': 1e-4,
                'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
                'focus': 'User reports small chord-transition glitches, possibly near the end; exact timestamp unavailable',
                'decoder_seam_seconds': [5.12, 10.24, 15.36, 20.48, 25.6],
                'listening_tail': {'start_seconds': 22, 'end_seconds': 30, 'additional_gain': 1.,
                                   'policy': 'Slice of saved native raw audio at the original per-run listening gain; no DSP'},
                'new_inference_calls': 0, 'native_exports': 0, 'baseline_modified': False,
                'raw_audio_binding': 'Raw*original FP32 gain recreates the exact report-bound device solo WAV SHA256',
                'other_tensor_provenance': 'Final latent/checkpoint50 bound to report; other sparse tensor hashes describe exported bytes only',
                'scope': 'Existing outputs only; no numerical acceptance, root cause or repair claim',
                'privacy': 'Private review only; never commit ZIP, audio, tensors or reports'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.instrument-transition-', dir=out.parent) as folder:
        temporary = Path(folder)
        for role in ROLES:
            scratch = temporary / role
            scratch.mkdir()
            values = verify_audio_binding(folders[role] / ('raw-' + role + '.f32le'), folders[role] / (role + '.wav'),
                                          metadata[role]['listening_gain_float32'], metadata[role]['solo_wav_sha256'], scratch)
            tail = array.array('f', values[22*48000:FRAMES])
            tail.extend(values[FRAMES+22*48000:2*FRAMES])
            path = scratch / 'last8s-listening.wav'
            write_float_wav(path, tail, 8*48000, metadata[role]['listening_gain_float32'])
            manifest['runs'][role]['last8s_wav_sha256'] = sha256(path)
        archive_path = temporary / 'review.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for role in ROLES:
                archive.writestr(role + '/report.txt', reports[role])
                archive.write(temporary / role / 'last8s-listening.wav', role + '/last8s-listening.wav')
            for name, entry in entries.items():
                role, filename = name.split('/')
                with archive.open(name, 'w') as target:
                    copy_tensor(folders[role] / filename, target, entry['sha256'], entry['bytes'])
            archive.writestr('review-manifest.json', json.dumps(manifest, indent=2) + '\n')
        if archive_path.stat().st_size > MAX_ZIP: raise ValueError('Review exceeds32MiB; no final ZIP published')
        os.link(archive_path, out)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=Path, help='Fresh downloaded CoreAIProbe container, .xcappdata or Documents folder')
    group.add_argument('--choose-container', action='store_true', help='macOS folder picker for the downloaded container')
    parser.add_argument('--bass-run', default='latest')
    parser.add_argument('--guitar-run', default='latest')
    parser.add_argument('--out', type=Path, default=ROOT / '.runtime-cache/bass-guitar-transition-review-v1.zip')
    args = parser.parse_args()
    if args.choose_container:
        if sys.platform != 'darwin': parser.error('--choose-container requires macOS; use --source elsewhere')
        result = subprocess.run(['osascript', '-e', 'POSIX path of (choose folder with prompt "Pilih folder container CoreAIProbe yang baru di-download")'],
                                capture_output=True, text=True)
        if result.returncode: parser.exit(1, 'Container selection canceled; no review created.\n')
        args.source = Path(result.stdout.rstrip('\r\n'))
    manifest = package(args.source, args.out, {'bass': args.bass_run, 'guitar': args.guitar_run})
    print('Existing bass/guitar runs packaged; raw audio bound exactly to device WAVs. No generation or baseline changes.')
    print('Private review:', args.out, '; bytes', args.out.stat().st_size, '; tensors', len(manifest['artifacts']))


if __name__ == '__main__': main()

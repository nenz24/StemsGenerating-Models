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
MAX_SEARCH_DEPTH = 8
MAX_SEARCH_DIRECTORIES = 4096


def shapes(role):
    if role not in ROLES: raise ValueError('Select bass or guitar')
    return {**{f'native-step-{step:02d}-latent.f32le': [1, 750, 64] for step in (1, 5, 10, 25, 50)},
            'final-latent.f32le': [1, 750, 64], 'final-momentum.f32le': [1, 750, 64],
            'raw-' + role + '.f32le': [1, 2, FRAMES]}


def documents_roots(source):
    source = Path(source).absolute()
    return [source, source / 'Documents', source / 'AppData/Documents']


def discover_namespaces(source):
    """Bounded read-only discovery inside the selected folder; no linked descent."""
    source = Path(source).absolute()
    if not source.is_dir(): raise ValueError('Select an existing downloaded container or Downloads folder: ' + str(source))
    if any(p.is_symlink() for p in (source, *source.parents)): raise ValueError('Linked search directory rejected')
    found = set()
    count = 0
    errors = []
    for current, directories, _ in os.walk(source, followlinks=False, onerror=errors.append):
        count += 1
        if count > MAX_SEARCH_DIRECTORIES:
            raise ValueError('Search reached4096 directories; select a smaller folder containing the CoreAIProbe download')
        current = Path(current)
        if current.name == 'InstrumentReference30':
            found.add(current.resolve())
            directories[:] = []  # Do not walk waveform/checkpoint files.
            continue
        if len(current.relative_to(source).parts) >= MAX_SEARCH_DEPTH:
            directories[:] = []
        else:
            directories[:] = sorted(n for n in directories if n not in
                {'.git', 'node_modules', '.Trash', 'Library', 'Caches', '__pycache__'}
                and not (current/n).is_symlink())
    if errors:
        raise ValueError('Search could not read every folder; select the downloaded CoreAIProbe folder directly: ' + str(errors[0]))
    return sorted(found)


def discovery_summary(source, namespaces):
    lines = ['Searched folder: ' + str(Path(source).absolute()),
             f'InstrumentReference30 folders found: {len(namespaces)} (search depth up to8; links/cache folders skipped)']
    for namespace in namespaces[:12]:
        lines.append('  ' + str(namespace))
        for role in ROLES:
            parent = namespace/role
            runs = []
            if parent.is_dir() and not parent.is_symlink():
                for path in sorted(parent.iterdir()):
                    if path.is_symlink() or not path.is_dir(): continue
                    try: identifier = str(uuid.UUID(path.name)).upper()
                    except ValueError: continue
                    missing = [n for n in [*shapes(role), role+'.wav', 'report.txt'] if not (path/n).is_file()]
                    runs.append(identifier + (' [files present]' if not missing else ' [missing ' + ', '.join(missing[:3]) + ']'))
            lines.append('    ' + role + ': ' + ('; '.join(runs[:8]) if runs else '(no run folders)'))
            if len(runs) > 8: lines.append(f'    ... {len(runs)-8} more {role} runs')
    if len(namespaces) > 12: lines.append(f'  ... {len(namespaces)-12} more namespace folders')
    return '\n'.join(lines)


def run_fingerprint(folder, role):
    """Only auto-resolve duplicate downloads if all required bytes are identical."""
    result = []
    for name in [*shapes(role), role+'.wav', 'report.txt']:
        expected_size = math.prod(shapes(role)[name])*4 if name in shapes(role) else (58+FRAMES*8 if name.endswith('.wav') else None)
        path = regular_file(folder/name,expected_size)
        if name == 'report.txt' and path.stat().st_size > 1024*1024: raise ValueError('Report exceeds1MiB')
        if any(p.is_symlink() for p in path.parents): raise ValueError('Linked run directory rejected')
        result.append((name, path.stat().st_size, sha256(path)))
    return tuple(result)


def run_folder(source, role, run_id, namespaces=None):
    if role not in ROLES: raise ValueError('Select bass or guitar')
    source = Path(source).absolute()
    namespaces = discover_namespaces(source) if namespaces is None else namespaces
    if run_id == 'latest':
        roots = set(documents_roots(source)) | {ns.parent for ns in namespaces}
        identifiers = set()
        for root in sorted(roots):
            pointer = root / f'last-instrument-reference30-{role}.json'
            if not pointer.exists(): continue
            regular_file(pointer)
            if pointer.stat().st_size > 4096 or any(p.is_symlink() for p in pointer.parents):
                raise ValueError('Invalid latest pointer')
            identifiers.add(str(uuid.UUID(json.loads(pointer.read_text())['run_id'])).upper())
        if len(identifiers) != 1:
            raise ValueError(f'No unique latest {role} run; specify --{role}-run with the UUID from the device report.\n'
                             + discovery_summary(source,namespaces))
        run_id = identifiers.pop()
    run_id = str(uuid.UUID(run_id)).upper()
    candidates = [source] + [ns / role / run_id for ns in namespaces]
    if source.name == role: candidates.append(source/run_id)
    matches = set()
    for path in candidates:
        if path.name == run_id and path.is_dir():
            if any(p.is_symlink() for p in (path, *path.parents)): raise ValueError('Linked run directory rejected')
            matches.add(path.resolve())
    if not matches:
        raise ValueError(f'{role} run {run_id} not found in the selected search.\n' + discovery_summary(source,namespaces)
                         + '\nDownload the current CoreAIProbe container only if the requested run is absent; do not repeat generation.')
    ordered = sorted(matches)
    if len(ordered) > 1:
        try: identical = len({run_fingerprint(path,role) for path in ordered}) == 1
        except (OSError,ValueError): identical = False
        if not identical:
            raise ValueError(f'Different/incomplete copies of {role} run {run_id}; select one container with --source:\n'
                             + '\n'.join(str(path) for path in ordered))
    return ordered[0], run_id


def protected_container_roots(namespaces):
    roots = set()
    for namespace in namespaces:
        documents = namespace.parent
        if documents.name == 'Documents':
            roots.add(documents.parent.parent if documents.parent.name == 'AppData' else documents.parent)
        else:
            roots.add(documents)
    return roots


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
    namespaces = discover_namespaces(source)
    protected = protected_container_roots(namespaces)
    # Direct role/run selections are themselves read-only input folders.
    if source.name in ROLES or any(source.name == run_ids[r] for r in ROLES): protected.add(source.resolve())
    if any(out.resolve().is_relative_to(root) for root in protected) or any(out.resolve().is_relative_to(ROOT / p)
            for p in ('baselines', 'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Output must be outside container, models and baselines')
    folders, metadata, reports, entries = {}, {}, {}, {}
    for role in ROLES:
        folder, run_id = run_folder(source, role, run_ids[role], namespaces)
        print(f'Found {role} run {run_id}: {folder}', flush=True)
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
    group.add_argument('--source', type=Path, help='Downloaded container, .xcappdata, Documents, or folder containing downloads (bounded search)')
    group.add_argument('--choose-container', action='store_true', help='macOS folder picker for the downloaded container')
    parser.add_argument('--bass-run', default='latest')
    parser.add_argument('--guitar-run', default='latest')
    parser.add_argument('--list-runs', action='store_true', help='List discovered run folders without creating any review ZIP')
    parser.add_argument('--out', type=Path, default=ROOT / '.runtime-cache/bass-guitar-transition-review-v1.zip')
    args = parser.parse_args()
    if args.choose_container:
        if sys.platform != 'darwin': parser.error('--choose-container requires macOS; use --source elsewhere')
        result = subprocess.run(['osascript', '-e', 'POSIX path of (choose folder with prompt "Pilih folder container CoreAIProbe yang baru di-download")'],
                                capture_output=True, text=True)
        if result.returncode: parser.exit(1, 'Container selection canceled; no review created.\n')
        args.source = Path(result.stdout.rstrip('\r\n'))
    print('Selected search folder:', args.source, flush=True)
    try:
        if args.list_runs:
            print(discovery_summary(args.source,discover_namespaces(args.source)))
            return
        manifest = package(args.source, args.out, {'bass': args.bass_run, 'guitar': args.guitar_run})
    except (ValueError,OSError) as error:
        parser.exit(1,'Review not created: ' + str(error) + '\n')
    print('Existing bass/guitar runs packaged; raw audio bound exactly to device WAVs. No generation or baseline changes.')
    print('Private review:', args.out, '; bytes', args.out.stat().st_size, '; tensors', len(manifest['artifacts']))


if __name__ == '__main__': main()

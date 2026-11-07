#!/usr/bin/env python3
"""Reconstruct the unchanged frozen30s input from a measured local PCM cache.

The private delta is supplied separately through stdin/file, never fetched or
executed. Only sorted +/-1LSB edits are supported. Final PCM AND WAV hashes must
match the original reference before a new file is created or generation starts.
"""
import argparse
import array
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import wave
import zlib

from export_coreai_decoder import ROOT
from generate_coreai_drum_reference import SOURCE_PCM_SHA, pcm16

REFERENCE_WAV_SHA = '1f68460c7cf73acd4ad9ae7ef9632c269e18517ac0046a2f177be18ccdd66b54'
ENCODING = 'unit-delta-varint-zlib-v1'
MAX_SAMPLES = 1920000


def apply_unit_delta(pcm, compressed, count):
    if (len(pcm) % 2 or len(pcm) > 2 * MAX_SAMPLES or type(count) is not int
            or not 0 <= count <= len(pcm) // 2):
        raise ValueError('Invalid PCM/delta size')
    try:
        decoder = zlib.decompressobj()
        data = decoder.decompress(compressed, max(1, count * 4 + 1))
    except zlib.error as error:
        raise ValueError('Invalid compressed correction data') from error
    if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(data) > count * 4):
        raise ValueError('Correction stream has invalid size/trailing data')
    values = array.array('h'); values.frombytes(pcm)
    if sys.byteorder != 'little':
        values.byteswap()
    index, pos = -1, 0
    for _ in range(count):
        value, shift = 0, 0
        while True:
            if pos == len(data) or shift > 21:
                raise ValueError('Truncated or oversized correction index')
            part = data[pos]; pos += 1
            value |= (part & 127) << shift
            if not part & 128:
                break
            shift += 7
        gap = value >> 1
        if gap < 1:
            raise ValueError('Correction indices must be strictly increasing')
        index += gap
        if index >= len(values):
            raise ValueError('Correction index outside PCM')
        corrected = values[index] + (-1 if value & 1 else 1)
        if not -32768 <= corrected <= 32767:
            raise ValueError('Correction exceeds PCM16 range')
        values[index] = corrected
    if pos != len(data):
        raise ValueError('Correction count does not match stream')
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def delta_bytes(document):
    if not isinstance(document, dict):
        raise ValueError('Expected a correction JSON object')
    if (document.get('schema_version') != 1 or document.get('encoding') != ENCODING
            or document.get('target_PCM_sha256') != SOURCE_PCM_SHA
            or document.get('target_WAV_sha256') != REFERENCE_WAV_SHA):
        raise ValueError('Delta does not target the unchanged original reference')
    text = document.get('delta_b64')
    if isinstance(text, list) and all(isinstance(x, str) for x in text):
        text = ''.join(text)
    if not isinstance(text, str) or len(text) > 65536:
        raise ValueError('Invalid correction payload')
    try:
        compressed = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('Invalid base64 correction data') from error
    if hashlib.sha256(compressed).hexdigest() != document.get('compressed_delta_sha256'):
        raise ValueError('Correction payload checksum mismatch')
    return compressed


def check_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing output preserved: {path}')
    if any(path.resolve().is_relative_to((ROOT / p).resolve()) for p in
           ['baselines', 'prototypes/ios27-coreai/ModelCandidates']):
        raise ValueError('Recovery output must be outside baseline/ModelCandidates')


def restore(source, out, document):
    source, out = Path(source), Path(out)
    receipt_path = Path(str(out) + '.json')
    check_output(out); check_output(receipt_path)
    compressed = delta_bytes(document)
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Expected a bounded30s PCM16 source WAV')
    original = source.read_bytes()
    pcm = pcm16(original)
    input_digest = hashlib.sha256(pcm).hexdigest()
    if input_digest != document.get('input_PCM_sha256'):
        raise ValueError('Local input PCM differs from the measured source; no edits applied')
    count = document.get('correction_count')
    corrected = apply_unit_delta(pcm, compressed, count)
    if hashlib.sha256(corrected).hexdigest() != SOURCE_PCM_SHA:
        raise ValueError('Reconstruction failed original PCM checksum; no output created')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setparams((2, 2, 32000, 0, 'NONE', 'not compressed'))
        audio.writeframes(corrected)
    wav = buffer.getvalue()
    if hashlib.sha256(wav).hexdigest() != REFERENCE_WAV_SHA:
        raise ValueError('Reconstruction failed original WAV checksum; no output created')
    if source.read_bytes() != original:
        raise ValueError('Local source changed during recovery; no output created')
    receipt = {'schema_version': 1, 'mode': 'Exact frozen-reference input reconstruction; not output audio processing',
               'input_WAV_sha256': hashlib.sha256(original).hexdigest(), 'input_PCM_sha256': input_digest,
               'output_WAV_sha256': REFERENCE_WAV_SHA, 'output_PCM_sha256': SOURCE_PCM_SHA,
               'correction_count': count, 'correction_unit_PCM16': 1,
               'compressed_delta_sha256': document['compressed_delta_sha256'],
               'original_input_bytes_preserved': True, 'reference_hashes_unchanged': True,
               'neural_calls_during_reconstruction': 0, 'model_exports': 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        stream.write(wav)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print('Exact original WAV and PCM verified:', out, flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing v2/source32.wav, read only')
    parser.add_argument('--out', type=Path, required=True, help='New exact-reference input WAV')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--delta-file', type=Path)
    group.add_argument('--delta-stdin', action='store_true')
    parser.add_argument('--generate-out', type=Path, help='After exact verification, run one existing30s reference generation in this new folder')
    args = parser.parse_args()
    if args.generate_out:
        check_output(args.generate_out)
    if args.delta_file:
        with args.delta_file.open('rb') as stream:
            raw = stream.read(65537)
    else:
        raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError('Correction document is too large')
    document = json.loads(raw)
    restore(args.source, args.out, document)
    if args.generate_out:
        subprocess.run([sys.executable, str(ROOT / 'scripts/generate_coreai_drum_reference.py'),
                        '--execute', '--prepared-source', str(args.out), '--out', str(args.generate_out)], check=True)
        if sys.platform == 'darwin':
            subprocess.run(['open', str(args.generate_out)], check=False)


if __name__ == '__main__':
    main()

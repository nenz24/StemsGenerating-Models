"""Integer reconstruction and frozen-target guards; no model inference."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
import zlib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import recover_coreai_reference_source as M


class ReferenceSourceRecoveryTests(unittest.TestCase):
    def document(self, input_digest, delta=b'', count=0):
        compressed = zlib.compress(delta)
        return {'schema_version': 1, 'encoding': M.ENCODING, 'input_PCM_sha256': input_digest,
                'target_PCM_sha256': M.SOURCE_PCM_SHA, 'target_WAV_sha256': M.REFERENCE_WAV_SHA,
                'correction_count': count, 'compressed_delta_sha256': hashlib.sha256(compressed).hexdigest(),
                'delta_b64': base64.b64encode(compressed).decode()}

    def source(self, root):
        source = root / 'fixture.wav'; pcm = bytes(3840000)
        with wave.open(str(source), 'wb') as stream:
            stream.setparams((2, 2, 32000, 0, 'NONE', 'not compressed')); stream.writeframes(pcm)
        return source, hashlib.sha256(pcm).hexdigest()

    def test_sorted_unit_edits_match_integer_oracle_without_mutating_input(self):
        pcm = struct.pack('<4h', 5, -2, 10, 9)
        result = M.apply_unit_delta(pcm, zlib.compress(bytes([2, 5, 2])), 3)
        self.assertEqual(struct.unpack('<4h', result), (6, -2, 9, 10))
        self.assertEqual(struct.unpack('<4h', pcm), (5, -2, 10, 9))

    def test_invalid_indices_counts_range_and_compression_are_rejected(self):
        pcm = struct.pack('<2h', 32767, -32768)
        for raw, count in [(bytes([2]), 1), (bytes([0]), 1), (bytes([8]), 1), (bytes([128]), 1), (bytes([2, 2]), 1)]:
            with self.assertRaises(ValueError): M.apply_unit_delta(pcm, zlib.compress(raw), count)
        with self.assertRaises(ValueError): M.apply_unit_delta(pcm, b'bad compressed stream', 1)
        with self.assertRaises(ValueError): M.apply_unit_delta(pcm, zlib.compress(b'') + b'extra', 0)
        with self.assertRaises(ValueError): M.apply_unit_delta(b'odd', zlib.compress(b''), 0)

    def test_document_cannot_replace_frozen_reference_or_tamper_payload(self):
        doc = self.document('0' * 64)
        self.assertEqual(M.delta_bytes(doc), zlib.compress(b''))
        for key in ['target_PCM_sha256', 'target_WAV_sha256', 'compressed_delta_sha256']:
            with self.assertRaises(ValueError): M.delta_bytes({**doc, key: 'f' * 64})
        with self.assertRaises(ValueError): M.delta_bytes([])
        with self.assertRaises(ValueError): M.delta_bytes({**doc, 'delta_b64': 'not valid base64!'})

    def test_wrong_input_and_wrong_final_target_create_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, digest = self.source(root); before = source.read_bytes(); out = root / 'out.wav'
            with self.assertRaisesRegex(ValueError, 'Local input PCM differs'):
                M.restore(source, out, self.document('0' * 64))
            with self.assertRaisesRegex(ValueError, 'original PCM checksum'):
                M.restore(source, out, self.document(digest))
            self.assertFalse(out.exists()); self.assertFalse(Path(str(out) + '.json').exists())
            self.assertEqual(source.read_bytes(), before)

    def test_existing_and_protected_outputs_are_preserved_before_data_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'old.wav'; out.write_bytes(b'keep existing')
            with self.assertRaises(FileExistsError): M.restore(root / 'missing-source', out, {})
            self.assertEqual(out.read_bytes(), b'keep existing')
            for parent in [ROOT / 'baselines', ROOT / 'prototypes/ios27-coreai/ModelCandidates']:
                with self.assertRaises(ValueError): M.restore(root / 'missing', parent / 'never-create.wav', {})

    def test_generation_is_not_launched_when_reconstruction_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); delta = root / 'input.json'; delta.write_text(json.dumps(self.document('0' * 64)))
            argv = ['recover', '--source', str(root / 'source.wav'), '--out', str(root / 'exact.wav'),
                    '--delta-file', str(delta), '--generate-out', str(root / 'generation')]
            with patch.object(sys, 'argv', argv), patch.object(M, 'restore', side_effect=ValueError('checksum failed')), patch.object(M.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'checksum failed'): M.main()
                child.assert_not_called()
            self.assertFalse((root / 'generation').exists())


if __name__ == '__main__':
    unittest.main()

"""Capture transport integrity; synthetic fixtures, no inference/audio verdict."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import package_coreai_drum_reference as M


class ReferencePackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'capture'
        self.source.mkdir()
        self.out = self.root / 'review.zip'
        self.shapes = {'source48.f32le': [1, 2, 4], 'step-50-latent.f32le': [1, 2, 2]}
        report = {'artifact_sha256': {}}
        for name, shape in self.shapes.items():
            raw = struct.pack('<' + 'f' * M.math.prod(shape), *range(M.math.prod(shape)))
            (self.source / name).write_bytes(raw)
            report['artifact_sha256'][name] = hashlib.sha256(raw).hexdigest()
        self.raw = json.dumps(report).encode()
        (self.source / 'report.json').write_bytes(self.raw)
        for name, value in [('SHAPES', self.shapes), ('ACCEPTED_REPORT_SHA256', hashlib.sha256(self.raw).hexdigest())]:
            patcher = patch.object(M, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_success_preserves_report_and_selects_only_verified_captures(self):
        (self.source / 'unrelated-private-file').write_text('do not package')
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        manifest = M.package(self.source, self.out)
        with zipfile.ZipFile(self.out) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(self.shapes) | {'report.json', 'capture-manifest.json'})
            self.assertEqual(archive.read('report.json'), self.raw)
            for name in self.shapes:
                self.assertEqual(archive.read(name), before[name])
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)
        self.assertEqual(manifest['native_parity'], 'NOT EVALUATED')
        self.assertEqual(manifest['native_exports'], 0)
        self.assertEqual((manifest['absolute_tolerance'], manifest['relative_tolerance'], manifest['relative_rmse_tolerance']), (1e-4, 1e-3, 1e-3))

    def test_changed_report_and_missing_or_truncated_capture_leave_no_archive(self):
        report_path = self.source / 'report.json'
        report_path.write_bytes(self.raw + b' ')
        with self.assertRaisesRegex(ValueError, 'not the accepted'): M.package(self.source, self.out)
        report_path.write_bytes(self.raw)
        tensor = self.source / 'source48.f32le'
        tensor.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'wrong-size'): M.package(self.source, self.out)
        tensor.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): M.package(self.source, self.out)
        self.assertFalse(self.out.exists())

    def test_same_size_tamper_rejects_and_cleans_temporary_archive(self):
        tensor = self.source / 'source48.f32le'
        tensor.write_bytes(bytes(tensor.stat().st_size))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'): M.package(self.source, self.out)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.drums-reference-*')), [])

    def test_nonfinite_stream_rejected_even_if_digest_matches(self):
        raw = struct.pack('<f', float('nan'))
        tensor = self.root / 'bad.f32le'; tensor.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            M.copy_tensor(tensor, io.BytesIO(), hashlib.sha256(raw).hexdigest(), 4)

    def test_existing_protected_and_linked_files_preserved(self):
        self.out.write_bytes(b'keep')
        with self.assertRaises(FileExistsError): M.package(self.source, self.out)
        self.assertEqual(self.out.read_bytes(), b'keep')
        for target in [self.source / 'new.zip', ROOT / 'baselines/never-create-review.zip',
                       ROOT / 'prototypes/ios27-coreai/ModelCandidates/never-create-review.zip']:
            with self.assertRaises(ValueError): M.package(self.source, target)
            self.assertFalse(target.exists())
        self.out.unlink()
        tensor = self.source / 'source48.f32le'
        data = tensor.read_bytes(); tensor.unlink()
        external = self.root / 'external'; external.write_bytes(data); tensor.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'linked'): M.package(self.source, self.out)

    def test_size_limit_and_racing_output_leave_no_partial_archive(self):
        with patch.object(M, 'MAX_ZIP_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'exceeds'): M.package(self.source, self.out)
        self.assertFalse(self.out.exists())
        def race(source, target):
            Path(target).write_bytes(b'other writer')
            raise FileExistsError('racing output')
        with patch.object(M.os, 'link', side_effect=race):
            with self.assertRaises(FileExistsError): M.package(self.source, self.out)
        self.assertEqual(self.out.read_bytes(), b'other writer')
        self.assertEqual(list(self.root.glob('.drums-reference-*')), [])


if __name__ == '__main__':
    unittest.main()

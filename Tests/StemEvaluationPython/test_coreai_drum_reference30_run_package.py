"""Transport tests using synthetic zero tensors, never learned-model evidence."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import package_coreai_drum_reference30_run as module

RUN = '00000000-0000-4000-8000-000000000001'


class RunPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.container = self.root / 'saved.xcappdata'
        self.run = self.container / 'AppData/Documents/DrumReference30Generation' / RUN
        self.run.mkdir(parents=True)
        for name, shape in module.SHAPES.items():
            (self.run / name).write_bytes(bytes(module.math.prod(shape) * 4))
        sha = module.digest(self.run / 'final-latent.f32le')
        self.report = ('Runtime: CoreAI frozen drums30 generation; run ' + RUN + '\n'
                       'Revision: reference30-live50-window256-v1; frozen input\n'
                       'Reference30 execution: COMPLETED; steps50/50; sampling calls750/750; decoder calls6/6\n'
                       'Original checkpoint/audio strict parity: NOT PASSED; 4/6 comparisons passed; target6\n'
                       'LIVE final latent SHA256: ' + sha + '\nWAV generated on device: YES\n')
        (self.run / 'report.txt').write_text(self.report)
        self.out = self.root / 'review.zip'

    def test_completed_failed_parity_retained_with_exact_bytes_and_manifest(self):
        manifest = module.package(self.container, self.out, RUN)
        self.assertEqual(manifest['original_strict_parity'], 'NOT PASSED')
        self.assertEqual(manifest['new_inference_calls'], 0)
        with zipfile.ZipFile(self.out) as archive:
            self.assertEqual(archive.read('report.txt'), self.report.encode())
            self.assertEqual(len(archive.namelist()), 10)
            for name, entry in manifest['artifacts'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), entry['sha256'])
            self.assertEqual(json.loads(archive.read('run-manifest.json')), manifest)

    def test_wrong_run_or_incomplete_report_does_not_publish(self):
        with self.assertRaises(ValueError):
            module.package(self.container, self.out, '00000000-0000-0000-0000-000000000000')
        (self.run / 'report.txt').write_text(self.report.replace('COMPLETED', 'NOT COMPLETED'))
        with self.assertRaises(ValueError):
            module.package(self.container, self.out, RUN)
        self.assertFalse(self.out.exists())

    def test_report_final_hash_and_step50_must_agree(self):
        path = self.run / 'native-step-50-latent.f32le'
        with path.open('r+b') as stream:
            stream.write(struct.pack('<f', 1))
        with self.assertRaisesRegex(ValueError, 'Checkpoint50'):
            module.package(self.container, self.out, RUN)
        self.assertFalse(self.out.exists())

    def test_nonfinite_payload_removes_partial_archive(self):
        path = self.run / 'raw-drums.f32le'
        with path.open('r+b') as stream:
            stream.write(struct.pack('<f', float('nan')))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            module.package(self.container, self.out, RUN)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.drums-live-run-*')), [])

    def test_links_and_truncation_rejected(self):
        path = self.run / 'native-step-01-latent.f32le'
        path.unlink()
        path.symlink_to(self.run / 'final-latent.f32le')
        with self.assertRaisesRegex(ValueError, 'linked'):
            module.package(self.container, self.out, RUN)
        path.unlink()
        path.write_bytes(b'1234')
        with self.assertRaisesRegex(ValueError, 'Wrong-size'):
            module.package(self.container, self.out, RUN)

    def test_output_preserved_and_container_unchanged(self):
        self.out.write_bytes(b'existing result')
        with self.assertRaises(FileExistsError):
            module.package(self.container, self.out, RUN)
        self.assertEqual(self.out.read_bytes(), b'existing result')
        with self.assertRaisesRegex(ValueError, 'outside'):
            module.package(self.container, self.container / 'new.zip', RUN)
        self.assertEqual((self.run / 'report.txt').read_text(), self.report)


if __name__ == '__main__':
    unittest.main()

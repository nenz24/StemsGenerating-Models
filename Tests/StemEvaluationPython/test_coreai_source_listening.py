"""Local listening delivery guards; fixtures are nonexecutable, not VAE validation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import listen_coreai_source_roundtrip as M


class SourceListeningTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        (source / 'reference.json').write_text('{"nonexecutable_fixture":true}')
        (source / 'audio.f32le').write_bytes(bytes(8 * M.LONG.samples))
        (source / 'source_latents-reference.f32le').write_bytes(bytes(4 * M.LONG.frames * 64))
        ref = {'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source.iterdir() if p.name != 'reference.json'}}
        return source, ref

    def test_existing_output_and_bundle_child_are_preserved_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _ = self.fixture(root); out = root / 'old'; out.mkdir()
            (out / 'keep').write_text('old listening file')
            with patch.object(M, 'verify_source') as verify:
                with self.assertRaises(FileExistsError): M.reconstruct(source, out)
                with self.assertRaises(ValueError): M.reconstruct(source, source / 'new')
                verify.assert_not_called()
            self.assertEqual((out / 'keep').read_text(), 'old listening file')

    def test_source_admission_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'listening'
            with patch.object(M, 'verify_source', side_effect=ValueError('source integrity mismatch')):
                with self.assertRaisesRegex(ValueError, 'integrity'): M.reconstruct(root / 'source', out)
            self.assertFalse(out.exists())

    def test_bad_pcm_stops_before_weights_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, ref = self.fixture(root); out = root / 'listening'
            (source / 'audio.f32le').write_bytes(b'bad')
            with patch.object(M, 'verify_source', return_value=ref), patch.object(M, 'local_checkpoint') as checkpoint:
                with self.assertRaisesRegex(ValueError, 'size'): M.reconstruct(source, out)
                checkpoint.assert_not_called()
            self.assertFalse(out.exists())

    def test_missing_local_checkpoint_retains_failure_report_and_input_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, ref = self.fixture(root); out = root / 'listening'
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            with patch.object(M, 'verify_source', return_value=ref), patch.object(M, 'local_checkpoint', side_effect=FileNotFoundError('no local VAE')):
                with self.assertRaisesRegex(FileNotFoundError, 'local VAE'): M.reconstruct(source, out)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['sampling_steps'], 0)
            self.assertEqual(report['decoder_calls'], 0)
            self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}, before)
            self.assertEqual(list(out.glob('*.wav')), [])


if __name__ == '__main__':
    unittest.main()

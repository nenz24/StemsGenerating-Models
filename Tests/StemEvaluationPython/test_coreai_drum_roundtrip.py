"""Read-only admission/failure guards; real codec execution is recorded separately."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import listen_coreai_drum_roundtrip as M


class DrumRoundtripTests(unittest.TestCase):
    def test_existing_output_and_protected_folders_stop_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'old'; out.mkdir(); (out / 'keep').write_text('keep')
            with patch.object(M, 'approved_audio') as audio, patch.object(M, 'local_checkpoint') as weights:
                with self.assertRaises(FileExistsError): M.reconstruct(out)
                for parent in [M.BASELINE, M.ROOT / 'prototypes/ios27-coreai/ModelCandidates']:
                    with self.assertRaises(ValueError): M.reconstruct(parent / 'never-create-test-output')
                audio.assert_not_called(); weights.assert_not_called()
            self.assertEqual((out / 'keep').read_text(), 'keep')

    def test_bad_archive_fails_before_ffmpeg_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); inputs = folder / 'inputs'; inputs.mkdir()
            archive = inputs / 'bad.zip'; archive.write_bytes(b'not approved archive')
            with patch.object(M.subprocess, 'run') as command, patch.object(M, 'local_checkpoint') as weights:
                with self.assertRaisesRegex(ValueError, 'checksum'): M.reconstruct(folder / 'out', archive=archive)
                command.assert_not_called(); weights.assert_not_called()
            self.assertFalse((folder / 'out').exists())
            self.assertEqual(archive.read_bytes(), b'not approved archive')

    def test_real_approved_input_geometry_and_identity(self):
        import numpy as np
        before = M.sha256(M.BASELINE / 'drum_result.zip')
        audio = M.approved_audio(M.BASELINE / 'drum_result.zip')
        self.assertEqual(audio.shape, (1, 2, M.SAMPLES))
        self.assertEqual(audio.dtype, np.dtype('float32'))
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(float(np.max(np.abs(audio))), .1)
        self.assertEqual(M.sha256(M.BASELINE / 'drum_result.zip'), before)

    def test_missing_local_weights_retains_failure_report_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            with patch.object(M, 'local_checkpoint', side_effect=FileNotFoundError('no local VAE')):
                with self.assertRaisesRegex(FileNotFoundError, 'local VAE'): M.reconstruct(out)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual((report['encoder_calls'], report['decoder_calls'], report['sampling_steps']), (0, 0, 0))
            self.assertFalse(report['baseline_modified'])
            self.assertEqual(list(out.glob('*.wav')), [])


if __name__ == '__main__':
    unittest.main()

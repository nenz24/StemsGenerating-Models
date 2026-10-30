"""Real child-process output and checksum progress tests; no model downloads needed."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from stream_process import run_logged
from model_assets import digest, process


class ProgressTests(unittest.TestCase):
    def test_real_stdout_stderr_and_carriage_return_progress_reach_kernel_output(self):
        output = io.StringIO()
        code = "import sys; print('stage-start',flush=True); sys.stderr.write('10%\\r100%\\n'); sys.stderr.flush()"
        with redirect_stdout(output):
            result = run_logged([sys.executable, "-c", code])
        self.assertEqual(result.returncode, 0)
        for message in ("stage-start", "10%", "100%", "exit=0"):
            self.assertIn(message, output.getvalue())

    def test_quiet_running_child_emits_honest_heartbeat(self):
        output = io.StringIO()
        with redirect_stdout(output):
            run_logged([sys.executable, "-c", "import time; time.sleep(0.25)"], heartbeat_seconds=0.05)
        self.assertIn("tanpa output baru", output.getvalue())
        self.assertIn("bukan pengukuran byte", output.getvalue())

    def test_interrupt_stops_the_process_started_by_the_notebook(self):
        original = subprocess.Popen
        children = []
        def start(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        with redirect_stdout(io.StringIO()), patch("stream_process.subprocess.Popen", side_effect=start), patch("stream_process.queue.Queue.get", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_logged([sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_nonzero_exit_is_not_hidden_by_streaming(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError) as error:
            run_logged([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(error.exception.returncode, 7)
        self.assertIn("exit=7", output.getvalue())

    def test_child_progress_options_do_not_mutate_supplied_environment(self):
        environment = {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        output = io.StringIO()
        code = "import os; print('unbuffered='+os.environ['PYTHONUNBUFFERED']); print('bars='+os.environ['HF_HUB_DISABLE_PROGRESS_BARS']); print('tqdm-disabled='+str('TQDM_DISABLE' in os.environ))"
        with redirect_stdout(output):
            run_logged([sys.executable, "-c", code], env=environment)
        self.assertIn("unbuffered=1", output.getvalue())
        self.assertIn("bars=0", output.getvalue())
        self.assertIn("tqdm-disabled=False", output.getvalue())
        self.assertEqual(environment["TQDM_DISABLE"], "1")

    def test_checksum_progress_counts_real_bytes_without_changing_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact"
            data = b"x" * (1024 * 1024 + 17)
            source.write_bytes(data)
            counts = []
            self.assertEqual(digest(source, "sha256", counts.append), hashlib.sha256(data).hexdigest())
            self.assertEqual(counts, [1024*1024, len(data)])

    def test_cached_file_reports_verification_and_completed_asset_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights.bin"
            source.write_bytes(b"fixture")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"model": "fixture", "files": [{"destination": "weights.bin", "size": 7,
                "algorithm": "sha256", "digest": hashlib.sha256(b"fixture").hexdigest()}]}))
            output = io.StringIO()
            with redirect_stdout(output):
                process(manifest, root)
            for message in ("[1/1]", "Verifikasi file yang sudah ada", "100.0%", "Aset terverifikasi", "Verified 1"):
                self.assertIn(message, output.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Full Mac notebook contract and preflight tests; no real model execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from mac_workflow import choose_source, check_model_runtime, run_command


class FullMacWorkflowTests(unittest.TestCase):
    def test_missing_audio_stops_before_expensive_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "unduhan belum dimulai"):
                choose_source(directory)

    def test_auto_selects_one_recording_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "guitar.m4a"
            first.touch()
            self.assertEqual(choose_source(root), first)
            (root / "piano.wav").touch()
            with self.assertRaises(ValueError):
                choose_source(root)
            preferred = root / "input.wav"
            preferred.touch()
            self.assertEqual(choose_source(root), preferred)

    def test_plan_and_real_execution_share_profile_including_offload(self):
        profile = json.loads((ROOT / "configs/mac-24gb.json").read_text())
        plan = run_command(ROOT, profile, Path("prepared"), Path("plan"))
        execute = run_command(ROOT, profile, Path("prepared"), Path("result"), execute=True)
        for command in (plan, execute):
            self.assertEqual(command[command.index("--device") + 1], "mps")
            self.assertEqual(command[command.index("--seeds") + 1], "0")
            self.assertIn("--offload", command)
            self.assertEqual(command[command.index("--ace-model") + 1], "acestep-v15-base")
        self.assertNotIn("--execute", plan)
        self.assertIn("--execute", execute)
        self.assertTrue(execute[0].endswith("vendor/ACE-Step-1.5/.venv/bin/python"))
        self.assertIn("--assets-manifest", execute)

    def test_runtime_must_have_python311_and_mps_before_downloads(self):
        for version, mps in (([3, 15], True), ([3, 11], False)):
            result = type("Result", (), {"stdout": json.dumps({"python": version, "mps": mps, "torch": "fixture"})})()
            with patch("mac_workflow.subprocess.run", return_value=result):
                with self.assertRaises(RuntimeError):
                    check_model_runtime(Path("python"))
        result = type("Result", (), {"stdout": '{"python":[3,11],"mps":true,"torch":"fixture"}'})()
        with patch("mac_workflow.subprocess.run", return_value=result):
            self.assertTrue(check_model_runtime(Path("python"))["mps"])

    def test_full_notebook_defaults_enable_full_workflow_without_old_flags(self):
        book = json.loads((ROOT / "notebooks/02_mac_24gb_full.ipynb").read_text())
        code = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
        self.assertIn("DRY_RUN = False", code)
        self.assertNotIn("INSTALL_RUNTIME = False", code)
        self.assertIn('"--install"', code)
        self.assertIn('"--download"', code)
        self.assertIn("execute=True", code)
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), "mac-full-cell", "exec")


if __name__ == "__main__":
    unittest.main()

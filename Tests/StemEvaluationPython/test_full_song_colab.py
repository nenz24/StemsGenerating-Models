"""Recover Colab after kernel reset, without assuming ROOT or repeating uploads."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import venv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from electric_full_song import discover_root, model_python


class FullSongColabTests(unittest.TestCase):
    def checkout(self, path, ready=False):
        for name in ("configs/assets-base.json", "scripts/setup_runtime.py", "tools/stem_eval/run.py"):
            target = path / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text("{}")
        if ready:
            for name in ("vendor/ACE-Step-1.5/.venv/bin/python", "vendor/ACE-Step-1.5/checkpoints/acestep-v15-base/model.safetensors"):
                target = path / name; target.parent.mkdir(parents=True, exist_ok=True); target.touch()
        return path

    def test_discovery_prefers_the_unique_existing_runtime_and_never_guesses_between_two(self):
        with tempfile.TemporaryDirectory() as directory:
            content = Path(directory)
            self.assertIsNone(discover_root(content))
            empty = self.checkout(content / "songmap_cuda_empty/source")
            ready = self.checkout(content / "songmap_cuda_ready/source", ready=True)
            self.assertEqual(discover_root(content), ready)
            second = self.checkout(content / "songmap_cuda_second/source", ready=True)
            with self.assertRaisesRegex(RuntimeError, "beberapa checkout"): discover_root(content)
            self.assertEqual(discover_root(content, empty), empty)
            self.assertEqual(discover_root(content, second), second)

    def bootstrap_cell(self):
        return (ROOT / "docs/electric-full-song.md").read_text().split("```python\n", 1)[1].split("```", 1)[0]

    def test_missing_root_global_reuses_uploaded_bytes_without_opening_upload_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            content = Path(directory)
            checkout = self.checkout(content / "songmap_cuda_fixture/source", ready=True)
            uploaded = {"electric_full_song.py": (ROOT / "scripts/electric_full_song.py").read_bytes(), "piano.mp3": b"fixture audio bytes"}
            upload = Mock(side_effect=AssertionError("must reuse uploaded bytes"))
            modules = {"google": types.ModuleType("google"), "google.colab": types.SimpleNamespace(files=types.SimpleNamespace(upload=upload))}
            namespace = {"uploaded": uploaded}  # Deliberately no ROOT or notebook globals.
            with patch.dict(sys.modules, modules), patch.object(Path, "cwd", return_value=content):
                exec(compile(self.bootstrap_cell(), "full-song-bootstrap", "exec"), namespace)
            self.assertEqual(namespace["ROOT"], checkout)
            self.assertEqual(namespace["SOURCE_PATH"].read_bytes(), b"fixture audio bytes")
            self.assertTrue(callable(namespace["fullsong"].generate))
            upload.assert_not_called()

    def test_reset_kernel_recovers_uploaded_files_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            content = Path(directory)
            self.checkout(content / "songmap_cuda_fixture/source", ready=True)
            (content / "electric_full_song.py").write_bytes((ROOT / "scripts/electric_full_song.py").read_bytes())
            (content / "piano.mp3").write_bytes(b"preserve audio")
            upload = Mock(side_effect=AssertionError("files are already on disk"))
            modules = {"google": types.ModuleType("google"), "google.colab": types.SimpleNamespace(files=types.SimpleNamespace(upload=upload))}
            namespace = {}
            with patch.dict(sys.modules, modules), patch.object(Path, "cwd", return_value=content):
                exec(compile(self.bootstrap_cell(), "reset-kernel-bootstrap", "exec"), namespace)
            self.assertEqual(namespace["SOURCE_PATH"].read_bytes(), b"preserve audio")
            upload.assert_not_called()

    def test_missing_checkout_or_runtime_has_actionable_setup_message(self):
        with tempfile.TemporaryDirectory() as directory:
            content = Path(directory)
            namespace = {"uploaded": {"piano.mp3": b"keep"}}
            modules = {"google": types.ModuleType("google"), "google.colab": types.SimpleNamespace(files=types.SimpleNamespace(upload=Mock()))}
            with patch.dict(sys.modules, modules), patch.object(Path, "cwd", return_value=content):
                with self.assertRaisesRegex(RuntimeError, "10_colab_electric_full_song"):
                    exec(compile(self.bootstrap_cell(), "fresh-vm-bootstrap", "exec"), namespace)
            self.assertEqual(namespace["uploaded"], {"piano.mp3": b"keep"})
            with self.assertRaisesRegex(RuntimeError, "Runtime model belum ada"): model_python(content)

    def test_standalone_notebook_setup_handles_old_installer_without_ensurepip(self):
        book = json.loads((ROOT / "notebooks/10_colab_electric_full_song.ipynb").read_text())
        cell = next("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code" and "bootstrap_python" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logged = Mock()
            env = {"fixture": "local environment"}
            namespace = {"DRY_RUN": False, "ROOT": root, "MODEL_PYTHON": root / "vendor/.venv/bin/python",
                         "sys": sys, "subprocess": Mock(), "json": json, "run_logged": logged,
                         "runtime_environment": lambda path: env, "check_cuda": Mock(return_value={"fixture_only": True})}
            with patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("Colab ensurepip unavailable")):
                exec(compile(cell, "standalone-runtime-cell", "exec"), namespace)
            commands = [call.args[0] for call in logged.call_args_list]
            self.assertEqual(commands[0][:4], [sys.executable, "-m", "pip", "--python"])
            self.assertEqual(commands[0][4], str(root / ".bootstrap/bin/python"))
            self.assertEqual(commands[0][-3:], ["install", "pip", "uv==0.10.0"])
            self.assertEqual(commands[1], [sys.executable, str(root / "scripts/setup_runtime.py"), "--install"])

    def test_notebook_is_unexecuted_and_defines_root_before_using_it(self):
        book = json.loads((ROOT / "notebooks/10_colab_electric_full_song.ipynb").read_text())
        cells = ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]
        self.assertIn("ROOT = discover_root", cells[0])
        self.assertIn('ROOT = COLAB_BUNDLE / "source"', cells[0])
        self.assertIn("GENERATE = not DRY_RUN", "\n".join(cells))
        self.assertNotIn("run_reference(", "\n".join(cells))
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), "standalone-full-song-cell", "exec")


if __name__ == "__main__": unittest.main()

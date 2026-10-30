"""Local-notebook workflow checks that do not install ML dependencies or fetch weights."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose import diagnose
from local_clip import local_clip
from model_assets import asset_path, digest, materialize, process, verify


class AssetTests(unittest.TestCase):
    def test_hashes_use_real_sha256_and_git_blob_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test"
            path.write_bytes(b"hello\n")
            self.assertEqual(digest(path, "sha256"), hashlib.sha256(b"hello\n").hexdigest())
            self.assertEqual(digest(path, "git-blob-sha1"), "ce013625030ba8dba906f756967f9e9ca394464a")
            record = {"size": 6, "algorithm": "sha256", "digest": digest(path, "sha256")}
            verify(path, record)
            path.write_bytes(b"hellx\n")
            with self.assertRaises(ValueError):
                verify(path, record)

    def test_manifest_sizes_and_required_components(self):
        for name in ("base", "xl-base"):
            manifest = json.loads((ROOT / f"configs/assets-{name}.json").read_text())
            files = manifest["files"]
            self.assertEqual(sum(record["size"] for record in files), manifest["download_bytes"])
            self.assertEqual(len({r["destination"] for r in files}), len(files))
            for record in files:
                self.assertEqual(len(record["revision"]), 40)
                self.assertEqual(len(record["digest"]), 64 if record["algorithm"] == "sha256" else 40)
            self.assertTrue(any(f'checkpoints/acestep-v15-{name}/' in r["destination"] for r in files))

    def test_paths_cannot_escape_upstream(self):
        with self.assertRaises(ValueError):
            asset_path(ROOT, "../../outside")

    def test_runtime_model_python_is_separate_from_verified_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            source = upstream / "checkpoints/model/code.py"
            source.parent.mkdir(parents=True)
            source.write_text("original = True\n")
            record = {"destination": "checkpoints/model/code.py", "size": source.stat().st_size,
                      "algorithm": "sha256", "digest": digest(source, "sha256")}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [record], "model": "fixture"}))
            process(manifest, upstream)
            runtime = materialize(manifest, upstream, root / "runtime")
            (runtime / "model/code.py").write_text("changed = True\n")
            self.assertEqual(source.read_text(), "original = True\n")
            verify(source, record)


class NotebookTests(unittest.TestCase):
    def test_notebook_outputs_are_empty_and_heavy_steps_opt_in(self):
        book = json.loads((ROOT / "notebooks/01_quality_poc.ipynb").read_text())
        source = "".join("".join(cell["source"]) for cell in book["cells"])
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), "notebook-cell", "exec")
        for name in ("INSTALL_RUNTIME", "DOWNLOAD_WEIGHTS", "GENERATE"):
            self.assertIn(f"{name} = False", source)

    def test_no_gpu_report_is_actionable_without_printing_environment(self):
        with patch("diagnose.shutil.which", return_value=None), patch("diagnose.importlib.util.find_spec", return_value=None):
            report = diagnose(ROOT)
        self.assertFalse(report["cuda_available_to_kernel"])
        self.assertIn("No NVIDIA", report["note"])
        self.assertNotIn("environment", report)

    def test_24gb_class_gpu_suggests_xl_even_with_driver_reserved_memory(self):
        result = type("Result", (), {"returncode": 0, "stdout": "GPU 24GB, 24564, 570.0\n"})()
        with patch("diagnose.shutil.which", return_value="/mock/tool"), patch("diagnose.importlib.util.find_spec", return_value=None), patch("diagnose.subprocess.run", return_value=result):
            report = diagnose(ROOT)
        self.assertEqual(report["suggested_model"], "acestep-v15-xl-base")
        self.assertFalse(report["cuda_available_to_kernel"])

    def test_songmap_clip_cannot_be_trimmed_twice(self):
        with self.assertRaises(ValueError):
            local_clip(Path("source.wav"), Path("out"), start=4, conditioning=Path("conditioning.json"))


if __name__ == "__main__":
    unittest.main()

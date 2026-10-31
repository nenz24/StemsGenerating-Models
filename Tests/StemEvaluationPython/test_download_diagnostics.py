"""Transport and read-only diagnostics regressions; no weight transfers."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_download import paths_for_record, snapshot
import model_assets


class DownloadDiagnosticsTests(unittest.TestCase):
    def test_target_and_transport_caches_are_distinct_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"repo_id": "ACE-Step/acestep-v15-base", "digest": "a"*64,
                      "destination": "checkpoints/acestep-v15-base/model.safetensors"}
            paths = paths_for_record(root, record)
            self.assertNotEqual(paths["partial"], paths["http_partial"])
            self.assertNotEqual(paths["lock"], paths["http_lock"])
            self.assertTrue(all(not state["exists"] for state in snapshot(paths).values()))
            self.assertEqual(list(root.iterdir()), [])
            paths["partial"].parent.mkdir(parents=True)
            paths["partial"].write_bytes(b"partial")
            self.assertEqual(snapshot(paths)["partial"]["file_size_bytes"], 7)
            self.assertEqual(paths["partial"].read_bytes(), b"partial")

    def test_http_and_xet_switches_are_set_before_hub_transfer(self):
        for transport in ("http", "xet"):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "cache-file"
                source.write_bytes(b"fixture")
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"model": "fixture", "files": [{
                    "repo_id": "test/model", "filename": "model.bin", "revision": "a"*40,
                    "destination": "model.bin", "size": 7, "algorithm": "sha256",
                    "digest": hashlib.sha256(b"fixture").hexdigest()}]}))
                upstream = root / "vendor"
                upstream.mkdir()
                calls = []
                def download(*args, **kwargs):
                    calls.append((os.environ["HF_HUB_DISABLE_XET"], kwargs["cache_dir"]))
                    return str(source)
                env = {"HF_HUB_CACHE": str(root / "hub"), "HF_XET_CACHE": str(root / "xet")}
                with patch.object(model_assets, "runtime_environment", return_value=env), patch.dict(os.environ), patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=download)}):
                    model_assets.process(manifest, upstream, download=True, transport=transport)
                self.assertEqual(calls[0][0], "1" if transport == "http" else "0")
                self.assertEqual(calls[0][1], str(root / "hub/http" if transport == "http" else root / "hub"))
                self.assertEqual((upstream / "model.bin").read_bytes(), b"fixture")


if __name__ == "__main__":
    unittest.main()

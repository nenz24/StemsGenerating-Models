"""Regression tests for managed Python/cache writes escaping to an unwritable user directory."""
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
from runtime_environment import runtime_environment
import model_assets
import setup_runtime


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_managed_paths_stay_local_and_auth_trust_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inherited = {"HTTPS_PROXY": "http://proxy.invalid", "SSL_CERT_FILE": "/fixture/ca.pem",
                         "HF_TOKEN": "synthetic-test-value", "UV_PYTHON_INSTALL_DIR": "/unwritable/python"}
            environment = runtime_environment(root, inherited)
            for name in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "PIP_CACHE_DIR", "HF_HUB_CACHE", "HF_XET_CACHE"):
                path = Path(environment[name])
                self.assertTrue(path.is_relative_to(root))
                self.assertTrue(path.is_dir())
            for name in ("HTTPS_PROXY", "SSL_CERT_FILE", "HF_TOKEN"):
                self.assertEqual(environment[name], inherited[name])
            self.assertNotIn("HOME", environment)
            self.assertNotIn("HF_HOME", environment)
            self.assertEqual(inherited["UV_PYTHON_INSTALL_DIR"], "/unwritable/python")

    def test_installer_passes_local_environment_to_pip_uv_and_model_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "vendor/ACE-Step-1.5"
            python = root / ".bootstrap" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(setup_runtime, "ROOT", root), patch.object(setup_runtime, "UPSTREAM", upstream), patch.object(setup_runtime, "checkout"), patch.object(setup_runtime.subprocess, "run") as run:
                setup_runtime.install()
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                environment = call.kwargs["env"]
                self.assertEqual(environment["UV_PYTHON_INSTALL_DIR"], str(root / ".runtime-cache/uv/python"))
                self.assertEqual(environment["UV_PROJECT_ENVIRONMENT"], str(upstream / ".venv"))
            self.assertIn("--frozen", run.call_args_list[1].args[0])

    def test_model_download_has_explicit_local_hub_and_xet_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "vendor"
            upstream.mkdir()
            artifact = root / "verified.bin"
            artifact.write_bytes(b"fixture")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"model": "fixture", "files": [{
                "repo_id": "test/fixture", "filename": "model.bin", "revision": "a"*40,
                "destination": "checkpoints/model.bin", "size": 7,
                "algorithm": "sha256", "digest": hashlib.sha256(b"fixture").hexdigest()}]}))
            cache = root / ".runtime-cache/huggingface/hub"
            environment = {"HF_HUB_CACHE": str(cache), "HF_XET_CACHE": str(root / ".runtime-cache/huggingface/xet")}
            calls = []
            def download(*args, **kwargs):
                calls.append(kwargs)
                return str(artifact)
            with patch.object(model_assets, "ROOT", root), patch.object(model_assets, "runtime_environment", return_value=environment), patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=download)}), patch.dict(os.environ):
                model_assets.process(manifest, upstream, download=True)
                self.assertEqual(os.environ["HF_XET_CACHE"], environment["HF_XET_CACHE"])
            self.assertEqual(calls[0]["cache_dir"], str(cache))
            self.assertEqual((upstream / "checkpoints/model.bin").read_bytes(), b"fixture")


if __name__ == "__main__":
    unittest.main()

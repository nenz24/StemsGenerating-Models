"""Regression tests for managed Python/cache writes escaping to an unwritable user directory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import venv
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from runtime_environment import runtime_environment
import model_assets
import setup_runtime
import run as stem_runner


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_managed_paths_stay_local_and_auth_trust_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inherited = {"HTTPS_PROXY": "http://proxy.invalid", "SSL_CERT_FILE": "/fixture/ca.pem",
                         "HF_TOKEN": "synthetic-test-value", "UV_PYTHON_INSTALL_DIR": "/unwritable/python"}
            environment = runtime_environment(root, inherited)
            for name in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "PIP_CACHE_DIR", "HF_HUB_CACHE", "HF_XET_CACHE", "MPLCONFIGDIR"):
                path = Path(environment[name])
                self.assertTrue(path.is_relative_to(root))
                self.assertTrue(path.is_dir())
            for name in ("HTTPS_PROXY", "SSL_CERT_FILE", "HF_TOKEN"):
                self.assertEqual(environment[name], inherited[name])
            self.assertNotIn("HOME", environment)
            self.assertNotIn("HF_HOME", environment)
            self.assertEqual(inherited["UV_PYTHON_INSTALL_DIR"], "/unwritable/python")

    def test_jupyter_backend_is_replaced_only_for_child_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            inherited = {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}
            child = runtime_environment(directory, inherited)
            self.assertEqual(child["MPLBACKEND"], "Agg")
            self.assertEqual(child["MPLCONFIGDIR"], str(Path(directory) / ".runtime-cache/matplotlib"))
            self.assertEqual(inherited["MPLBACKEND"], "module://matplotlib_inline.backend_inline")

    def test_model_entry_applies_headless_backend_before_loading_models(self):
        environment = {"HF_HUB_CACHE": "/fixture/hub", "HF_XET_CACHE": "/fixture/xet",
                       "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/fixture/matplotlib"}
        class StopBeforeModelLoad(Exception):
            pass
        def check_device(*args):
            self.assertEqual(os.environ["MPLBACKEND"], "Agg")
            self.assertEqual(os.environ["MPLCONFIGDIR"], "/fixture/matplotlib")
            raise StopBeforeModelLoad
        args = types.SimpleNamespace(upstream=Path("/fixture/upstream"), backend="ace", device="mps")
        with patch.dict(os.environ, {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}), patch.dict(sys.modules, {"torch": types.ModuleType("torch")}), patch("runtime_environment.runtime_environment", return_value=environment), patch.object(stem_runner, "require_audio_tools"), patch.object(stem_runner, "check_upstream"), patch.object(stem_runner, "select_device", side_effect=check_device):
            with self.assertRaises(StopBeforeModelLoad):
                stem_runner.execute(args, {})

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
            self.assertEqual(run.call_args_list[0].args[0], [
                sys.executable, "-m", "pip", "--python", str(python), "install", "uv==0.10.0"])
            for call in run.call_args_list:
                environment = call.kwargs["env"]
                self.assertEqual(environment["UV_PYTHON_INSTALL_DIR"], str(root / ".runtime-cache/uv/python"))
                self.assertEqual(environment["UV_PROJECT_ENVIRONMENT"], str(upstream / ".venv"))
                self.assertEqual(environment["MPLBACKEND"], "Agg")
            self.assertIn("--frozen", run.call_args_list[1].args[0])

    def test_fresh_bootstrap_works_without_ensurepip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = runtime_environment(root)
            with patch.object(setup_runtime, "ROOT", root), patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip unavailable")), patch.object(setup_runtime.subprocess, "run") as run:
                uv = setup_runtime.bootstrap_uv(environment)
            python = root / ".bootstrap" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            self.assertTrue(python.is_file())
            self.assertEqual(uv.parent, python.parent)
            self.assertEqual(run.call_args.args[0], [
                sys.executable, "-m", "pip", "--python", str(python), "install", "uv==0.10.0"])
            self.assertEqual(run.call_args.kwargs["env"], environment)

    def test_partial_bootstrap_without_pip_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap = root / ".bootstrap"
            venv.EnvBuilder(with_pip=False).create(bootstrap)
            with patch.object(setup_runtime, "ROOT", root), patch.object(venv.EnvBuilder, "create", side_effect=AssertionError("must reuse partial bootstrap")), patch.object(setup_runtime.subprocess, "run") as run:
                setup_runtime.bootstrap_uv(runtime_environment(root))
            self.assertEqual(run.call_count, 1)
            self.assertIn("--python", run.call_args.args[0])

    def test_cuda_notebooks_resync_partial_model_runtime(self):
        for name in ("08_cuda_reference", "09_colab_reference"):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / f"notebooks/{name}.ipynb").read_text())
                cell, = ("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "code" and "scripts/setup_runtime.py" in "".join(cell["source"]))
                run_logged = Mock()
                check_cuda = Mock(return_value={"cuda_available": True})
                # The interpreter can exist even when uv sync was interrupted.
                runtime_python = Mock(return_value=Path(sys.executable))
                exec(compile(cell, name, "exec"), {
                    "DRY_RUN": False, "ROOT": ROOT, "sys": sys, "json": json,
                    "run_logged": run_logged, "check_cuda": check_cuda,
                    "runtime_python": runtime_python, "runtime_environment": lambda: {}})
                self.assertEqual(run_logged.call_args.args[0], [
                    sys.executable, str(ROOT / "scripts/setup_runtime.py"), "--install"])
                check_cuda.assert_called_once_with(Path(sys.executable))

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
            self.assertEqual(calls[0]["cache_dir"], str(cache / "http"))
            self.assertEqual((upstream / "checkpoints/model.bin").read_bytes(), b"fixture")


if __name__ == "__main__":
    unittest.main()

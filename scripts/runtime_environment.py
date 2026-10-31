"""Keep writable runtime state in the checkout while preserving authentication and trust."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_environment(root=ROOT, inherited=None):
    root = Path(root).resolve()
    cache = root / ".runtime-cache"
    paths = {
        "UV_CACHE_DIR": cache / "uv/cache",
        "UV_PYTHON_INSTALL_DIR": cache / "uv/python",
        "UV_PYTHON_BIN_DIR": cache / "uv/bin",
        "PIP_CACHE_DIR": cache / "pip",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_XET_CACHE": cache / "huggingface/xet",
        "MPLCONFIGDIR": cache / "matplotlib",
    }
    environment = dict(os.environ if inherited is None else inherited)
    for name, directory in paths.items():
        directory.mkdir(parents=True, exist_ok=True)
        environment[name] = str(directory)
    # Model subprocesses run outside the notebook and do not have matplotlib_inline.
    # Use a standard headless backend without changing the parent Jupyter kernel.
    environment["MPLBACKEND"] = "Agg"
    # Do not change HOME, HF_HOME, token bindings, proxy settings, or CA trust.
    return environment

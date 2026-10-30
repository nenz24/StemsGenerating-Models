"""Create a pinned, isolated upstream runtime without changing the Jupyter kernel."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

from runtime_environment import runtime_environment

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
UPSTREAM = ROOT / "vendor" / "ACE-Step-1.5"


def runtime_python():
    return UPSTREAM / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def checkout():
    if not UPSTREAM.exists():
        UPSTREAM.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/ace-step/ACE-Step-1.5.git", str(UPSTREAM)], check=True)
        subprocess.run(["git", "-C", str(UPSTREAM), "checkout", "--detach", COMMIT], check=True)
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != COMMIT or status:
        raise RuntimeError("Existing upstream differs from pinned clean revision; preserve your work and use a separate directory")
    return UPSTREAM


def install():
    checkout()
    environment = runtime_environment(ROOT)
    environment["UV_PROJECT_ENVIRONMENT"] = str(UPSTREAM / ".venv")
    bootstrap = ROOT / ".bootstrap"
    python = bootstrap / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(bootstrap)
    subprocess.run([str(python), "-m", "pip", "install", "uv==0.10.0"], env=environment, check=True)
    uv = bootstrap / ("Scripts/uv.exe" if sys.platform == "win32" else "bin/uv")
    subprocess.run([str(uv), "sync", "--frozen", "--python", "3.11"], cwd=UPSTREAM, env=environment, check=True)
    subprocess.run([str(runtime_python()), "-c", "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"], env=environment, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Install pinned ML dependencies, potentially several GB")
    args = parser.parse_args()
    if args.install:
        install()
    else:
        print(checkout())

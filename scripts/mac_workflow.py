"""Complete Mac 24 GB POC orchestration helpers; model execution stays in Python 3.11."""
import json
from pathlib import Path
import subprocess
import sys

from runtime_environment import runtime_environment

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".aif", ".aiff", ".ogg"}


def choose_source(directory, explicit=None):
    directory = Path(directory)
    if explicit is not None:
        source = Path(explicit).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Rekaman tidak ditemukan: {source}")
        return source
    candidates = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
    preferred = directory / "input.wav"
    if preferred in candidates:
        return preferred.resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"Letakkan satu rekaman gitar/piano di {directory}, lalu Run All lagi. "
            "Instalasi/unduhan belum dimulai. Tidak memakai fixture sintetis untuk generasi."
        )
    raise ValueError("Ada beberapa rekaman di data/. Pilih SOURCE pada notebook atau gunakan nama input.wav.")


def check_model_runtime(python):
    code = (
        "import json,sys,torch; "
        "print(json.dumps({'python': list(sys.version_info[:2]), 'torch': torch.__version__, "
        "'mps': hasattr(torch.backends,'mps') and torch.backends.mps.is_available()}))"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, env=runtime_environment(ROOT), check=True)
    report = json.loads(result.stdout.strip())
    if report["python"] != [3, 11] or not report["mps"]:
        raise RuntimeError(f"Runtime model belum siap: {report}. Perlu Python 3.11 dengan MPS aktif.")
    return report


def run_command(root, profile, prepared, output, execute=False):
    root = Path(root)
    python = (root / "vendor/ACE-Step-1.5/.venv/bin/python") if execute else Path(sys.executable)
    command = [str(python), str(root / "tools/stem_eval/run.py"),
               "--prepared", str(prepared), "--backend", "ace",
               "--instrument", profile["instrument"], "--seeds", *map(str, profile["seeds"]),
               "--ace-model", profile["model"], "--device", profile["device"], "--out", str(output)]
    if profile["offload_to_cpu"]:
        command.append("--offload")
    if execute:
        command += ["--execute", "--upstream", str(root / "vendor/ACE-Step-1.5"),
                    "--assets-manifest", str(root / "configs/assets-base.json")]
    return command

"""Safe local hardware report. Never prints process environment or credentials."""
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def diagnose(root):
    disk = shutil.disk_usage(root)
    report = {
        "os": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "disk_free_gib": round(disk.free / 1024**3, 1),
        "git": bool(shutil.which("git")), "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")), "nvidia_gpus": [],
        "torch_installed_in_kernel": importlib.util.find_spec("torch") is not None,
    }
    if shutil.which("nvidia-smi"):
        command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                name, memory, driver = (part.strip() for part in line.rsplit(",", 2))
                report["nvidia_gpus"].append({"name": name, "vram_mib": int(memory), "driver": driver})
        else:
            report["nvidia_status"] = "nvidia-smi failed; inspect the driver in your terminal"
    report["cuda_available_to_kernel"] = False
    if report["torch_installed_in_kernel"]:
        import torch
        if torch.cuda.is_available() and not report["nvidia_gpus"]:
            for index in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(index)
                report["nvidia_gpus"].append({"name": gpu.name, "vram_mib": gpu.total_memory // 1024**2, "driver": "not reported"})
        report.update(torch_version=torch.__version__, cuda_available_to_kernel=torch.cuda.is_available(),
                      mps_available_to_kernel=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    memory = max((gpu["vram_mib"] for gpu in report["nvidia_gpus"]), default=0)
    report["suggested_model"] = "acestep-v15-xl-base" if memory >= 24_000 else "acestep-v15-base"
    report["note"] = ("XL is an upstream quality candidate, not a measured winner. CUDA availability must be rechecked in the model environment."
                      if memory else "No NVIDIA GPU detected. Clip preparation works; this CUDA baseline will not run generation yet. Send this report so the runtime can be adapted.")
    return report


if __name__ == "__main__":
    print(json.dumps(diagnose(Path.cwd()), indent=2))

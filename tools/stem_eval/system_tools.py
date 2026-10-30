"""Locate audio tools, including Homebrew binaries outside a Jupyter kernel's PATH."""
from pathlib import Path
import platform
import shutil


def resolve_tool(name):
    executable = shutil.which(name)
    if executable:
        return executable
    if platform.system() == "Darwin":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            candidate = Path(prefix) / name
            if candidate.is_file():
                return str(candidate)
    return None


def require_tool(name):
    executable = resolve_tool(name)
    if executable:
        return executable
    raise RuntimeError(
        f"{name} tidak ditemukan. Di Terminal macOS jalankan `brew install ffmpeg`, "
        "lalu restart kernel Jupyter dan jalankan ulang notebook. "
        "Di Linux/Windows instal ffmpeg dan ffprobe melalui package manager OS."
    )


def require_audio_tools():
    return {name: require_tool(name) for name in ("ffmpeg", "ffprobe")}

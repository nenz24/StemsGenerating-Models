"""Download exact model revisions and verify publisher artifact hashes before use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from runtime_environment import runtime_environment
from asset_progress import FileProgress, size_text

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm, progress=None):
    if algorithm not in {"sha256", "git-blob-sha1"}:
        raise ValueError("Unsupported artifact digest algorithm")
    value = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-blob-sha1":
        value.update(f"blob {path.stat().st_size}\0".encode())
    processed = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
            processed += len(chunk)
            if progress is not None:
                progress(processed)
    return value.hexdigest()


def verify(path, record, progress=None):
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"Missing or wrong-size artifact: {path}")
    if digest(path, record["algorithm"], progress) != record["digest"]:
        raise ValueError(f"Artifact checksum mismatch: {path}. Do not use this artifact.")


def asset_path(root, destination):
    path = (root / destination).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact destination escapes the upstream directory")
    return path


def process(manifest, upstream, download=False, transport="http"):
    manifest = json.loads(manifest.read_text())
    total_bytes = sum(record["size"] for record in manifest["files"])
    completed_bytes = 0
    print(f'Model {manifest["model"]}: {len(manifest["files"])} file, total {size_text(total_bytes)}', flush=True)
    if download:
        if transport not in {"http", "xet"}:
            raise ValueError("transport must be http or xet")
        environment = runtime_environment(ROOT)
        # Supported HF transport switch: avoid Xet's native transfer path for this POC.
        # Must be set before importing huggingface_hub in this fresh download subprocess.
        os.environ["HF_HUB_DISABLE_XET"] = "1" if transport == "http" else "0"
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
        print(f"Transport: {transport}; metadata timeout={os.environ['HF_HUB_ETAG_TIMEOUT']}s, "
              f"HTTP read timeout={os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s", flush=True)
        # Set Xet's cache before importing the hub client; token/trust bindings stay inherited.
        os.environ["HF_XET_CACHE"] = environment["HF_XET_CACHE"]
        from huggingface_hub import hf_hub_download
        # Keep partial Xet blobs/locks intact and avoid reusing their preallocated files.
        download_cache = Path(environment["HF_HUB_CACHE"]) / "http" if transport == "http" else Path(environment["HF_HUB_CACHE"])
        if transport == "http":
            print("Cache HTTP terpisah; file model lengkap tetap dipakai, cache Xet tidak dihapus.", flush=True)
        # Leave room for download cache/temp files as well as installed components.
        free = shutil.disk_usage(upstream).free
        missing = sum(record["size"] for record in manifest["files"]
                      if not asset_path(upstream, record["destination"]).is_file())
        if free < missing * 2 + 5 * 1024**3:
            raise RuntimeError("Not enough disk headroom: reserve twice missing asset bytes plus 5 GiB")
    for index, record in enumerate(manifest["files"], 1):
        print(f'[{index}/{len(manifest["files"])}] {record["destination"]} | {size_text(record["size"])}', flush=True)
        target = asset_path(upstream, record["destination"])
        if target.is_file():
            # Existing artifacts are checked, never silently trusted or overwritten.
            verify(target, record, FileProgress("Verifikasi file yang sudah ada", record["size"]))
            completed_bytes += record["size"]
            print(f"  Aset terverifikasi: {size_text(completed_bytes)}/{size_text(total_bytes)}", flush=True)
            continue
        if download:
            # Download into an isolated ignored cache, then copy only after hash verification.
            print("  Mengunduh/mengambil dari cache HF; progress transfer tampil jika tersedia…", flush=True)
            path = Path(hf_hub_download(record["repo_id"], record["filename"], revision=record["revision"],
                                         cache_dir=str(download_cache)))
            verify(path, record, FileProgress("Verifikasi cache", record["size"]))
            print("  Menyalin file terverifikasi ke folder model…", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        verify(target, record, FileProgress("Verifikasi salinan model", record["size"]))
        completed_bytes += record["size"]
        print(f"  Aset terverifikasi: {size_text(completed_bytes)}/{size_text(total_bytes)}", flush=True)
    print(f'Verified {len(manifest["files"])} artifacts for {manifest["model"]}', flush=True)


def materialize(manifest_path, upstream, runtime_root):
    """Keep downloaded assets pristine when upstream synchronizes its model Python code."""
    manifest = json.loads(manifest_path.read_text())
    runtime_root.mkdir(parents=True, exist_ok=False)
    for record in manifest["files"]:
        source = asset_path(upstream, record["destination"])
        target = asset_path(runtime_root, record["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in {".safetensors", ".pt", ".bin"}:
            try:
                os.link(source, target)
            except OSError:
                if shutil.disk_usage(runtime_root).free < source.stat().st_size:
                    raise RuntimeError("Cannot hardlink weights and insufficient disk space for a runtime copy")
                shutil.copyfile(source, target)
        else:
            shutil.copyfile(source, target)
    return runtime_root / "checkpoints"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--transport", choices=("http", "xet"), default="http")
    args = parser.parse_args()
    process(args.manifest, args.upstream, args.download, args.transport)

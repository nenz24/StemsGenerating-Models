"""Download exact model revisions and verify publisher artifact hashes before use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm):
    if algorithm not in {"sha256", "git-blob-sha1"}:
        raise ValueError("Unsupported artifact digest algorithm")
    value = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-blob-sha1":
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"Missing or wrong-size artifact: {path}")
    if digest(path, record["algorithm"]) != record["digest"]:
        raise ValueError(f"Artifact checksum mismatch: {path}. Do not use this artifact.")


def asset_path(root, destination):
    path = (root / destination).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact destination escapes the upstream directory")
    return path


def process(manifest, upstream, download=False):
    manifest = json.loads(manifest.read_text())
    if download:
        from huggingface_hub import hf_hub_download
        # Leave room for download cache/temp files as well as installed components.
        free = shutil.disk_usage(upstream).free
        missing = sum(record["size"] for record in manifest["files"]
                      if not asset_path(upstream, record["destination"]).is_file())
        if free < missing * 2 + 5 * 1024**3:
            raise RuntimeError("Not enough disk headroom: reserve twice missing asset bytes plus 5 GiB")
    for record in manifest["files"]:
        target = asset_path(upstream, record["destination"])
        if target.is_file():
            # Existing artifacts are checked, never silently trusted or overwritten.
            verify(target, record)
            continue
        if download:
            # Download into an isolated ignored cache, then copy only after hash verification.
            path = Path(hf_hub_download(record["repo_id"], record["filename"], revision=record["revision"]))
            verify(path, record)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        verify(target, record)
    print(f'Verified {len(manifest["files"])} artifacts for {manifest["model"]}')


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
    args = parser.parse_args()
    process(args.manifest, args.upstream, args.download)

"""Read-only diagnostics for the specific Base weight transfer; no cache files are deleted."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def paths_for_record(root, record):
    cache = root / ".runtime-cache/huggingface/hub"
    repository = "models--" + record["repo_id"].replace("/", "--")
    blob = cache / repository / "blobs" / record["digest"]
    http_blob = cache / "http" / repository / "blobs" / record["digest"]
    return {"http_cached_blob": http_blob,
            "http_partial": http_blob.with_name(http_blob.name + ".incomplete"),
            "http_lock": cache / "http/.locks" / repository / (record["digest"] + ".lock"),
            "target": root / "vendor/ACE-Step-1.5" / record["destination"],
            "cached_blob": blob, "partial": blob.with_name(blob.name + ".incomplete"),
            "lock": cache / ".locks" / repository / (record["digest"] + ".lock")}


def snapshot(paths):
    result = {}
    for name, path in paths.items():
        try:
            stat = path.stat()
            result[name] = {"exists": True, "file_size_bytes": stat.st_size,
                            "disk_allocated_bytes": stat.st_blocks * 512 if hasattr(stat, "st_blocks") else None,
                            "modified_ns": stat.st_mtime_ns}
        except FileNotFoundError:
            result[name] = {"exists": False}
    return result


def processes_with_open_file(path):
    if not path.exists() or not shutil.which("lsof"):
        return []
    # Only process IDs and executable names for this exact model lock; no command arguments.
    try:
        response = subprocess.run(["lsof", "-n", "-Fpc", str(path)], capture_output=True,
                                  text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    processes = []
    for line in response.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            processes.append({"pid": int(line[1:])})
        elif line.startswith("c") and processes:
            processes[-1]["executable"] = line[1:]
    return processes


def diagnose(root=ROOT, interval=10):
    manifest = json.loads((root / "configs/assets-base.json").read_text())
    record = next(record for record in manifest["files"]
                  if record["destination"] == "checkpoints/acestep-v15-base/model.safetensors")
    paths = paths_for_record(root, record)
    before = snapshot(paths)
    print(f"Memeriksa metadata file selama {interval:g} detik; tidak menghapus/mengubah cache…", flush=True)
    time.sleep(interval)
    after = snapshot(paths)
    report = {"artifact": record["destination"], "expected_size_bytes": record["size"],
              "interval_seconds": interval, "lsof_available": bool(shutil.which("lsof")), "paths": {key: str(path.relative_to(root)) for key, path in paths.items()},
              "before": before, "after": after,
              "changed_entries": [name for name in paths if before[name] != after[name]],
              "processes_with_lock_file_open": {name: processes_with_open_file(paths[name]) for name in ("lock", "http_lock")},
              "notes": ["File size/allocation/mtime changes describe disk activity, not network byte progress.",
                        "A lock file existing on disk does not prove it is currently held.",
                        "An open lock file identifies candidate processes, not proof that a process is stuck.",
                        "An empty process list is inconclusive if lsof is unavailable or access is restricted.",
                        "File size matching expected size does not replace checksum verification."]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if not 0 <= args.interval <= 60:
        parser.error("interval must be between 0 and 60 seconds")
    print(json.dumps(diagnose(interval=args.interval), indent=2))

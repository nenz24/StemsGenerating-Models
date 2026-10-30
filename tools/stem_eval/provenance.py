"""Content hashes, pinned source identities and explicit experiment provenance."""
import hashlib
import json
from pathlib import Path
import subprocess

COMMITS = {
    "ace": "ca1e85fe9430179831e6bc6be790c332190a3866",
    "stage": "a015ce0bd42cc0c9789c4fdc3682609cc785ff4b",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def check_upstream(path, backend):
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMITS[backend]:
        raise ValueError(f"Expected inspected {backend} commit {COMMITS[backend]}, got {commit}")
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if status:
        raise ValueError("Upstream tracked files have changed; use the inspected source revision")
    return commit

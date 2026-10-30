"""Prepare a local guitar/piano recording, optionally using raw SongMap conditioning."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "stem_eval"))
from audio import canonicalize
from prepare import prepare


def local_clip(source, output, start=0, duration=10, conditioning=None):
    if conditioning:
        if start != 0:
            raise ValueError("SongMap source.wav is already trimmed: use start=0 with exported conditioning")
        return prepare(source, conditioning, output)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        canonicalize(source, directory / "source.wav", start=start, duration=duration, gain=1)
        metadata = {"schemaVersion": 1, "request": {
            "id": str(uuid.uuid4()), "songID": str(uuid.uuid4()), "start": start,
            "duration": duration, "instrument": "bass", "seed": 0},
            "audioFile": "source.wav", "sampleRate": 32000, "beats": [], "downbeats": [],
            "chords": [], "warnings": ["No SongMap timing/chord analysis supplied. Audio-only conditioning."]}
        metadata_path = directory / "conditioning.json"
        metadata_path.write_text(json.dumps(metadata))
        return prepare(directory / "source.wav", metadata_path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--conditioning", type=Path)
    args = parser.parse_args()
    local_clip(args.source, args.out, args.start, args.duration, args.conditioning)

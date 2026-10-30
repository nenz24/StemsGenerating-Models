"""Prepare a source plus optional STAGE click conditions from an iOS clip export."""
import argparse
import json
from pathlib import Path
import shutil

from audio import add_clicks, canonicalize, validate_times
from provenance import sha256, write_json


def prepare(source, conditioning, output):
    source, conditioning, output = Path(source), Path(conditioning), Path(output)
    metadata = json.loads(conditioning.read_text())
    if metadata.get("schemaVersion") != 1:
        raise ValueError("Expected SongMap conditioning schemaVersion 1")
    duration = metadata["request"]["duration"]
    # iOS source.wav is already trimmed. request.start is provenance, not another seek.
    beats = validate_times(metadata.get("beats", []), duration)
    downbeats = validate_times(metadata.get("downbeats", []), duration)
    if not set(downbeats).issubset(beats):
        # Apple beat/bar timestamps may differ by rounding. Match within one audio frame.
        if any(not any(abs(d - b) <= 1 / 32000 for b in beats) for d in downbeats):
            raise ValueError("Downbeats must correspond to supplied beats")
    output.mkdir(parents=True, exist_ok=False)
    try:
        canonicalize(source, output / "source.wav", duration=duration)
        shutil.copyfile(conditioning, output / "conditioning.json")
        conditions = {"A": "source.wav"}
        if beats:
            add_clicks(output / "source.wav", output / "beat.wav", beats, downbeats, False)
            conditions["B"] = "beat.wav"
            if downbeats:
                add_clicks(output / "source.wav", output / "downbeat.wav", beats, downbeats, True)
                conditions["C"] = "downbeat.wav"
        manifest = {
            "schema_version": 1, "duration": duration, "sample_rate": 32000,
            "source_sha256": sha256(source), "conditioning_sha256": sha256(conditioning),
            "fixed_source_gain": 0.7, "conditions": conditions,
            "condition_sha256": {key: sha256(output / file) for key, file in conditions.items()},
            "notes": "No beat fitting; clean source always used for listening mixes",
            "warnings": metadata.get("warnings", []),
        }
        write_json(output / "prepared.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(output)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.conditioning, args.out)

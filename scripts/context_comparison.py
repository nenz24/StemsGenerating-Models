"""Prepare controlled fresh-process/context comparisons from existing canonical song audio."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, read_pcm, write_pcm
from provenance import sha256, write_json


def create_input(directory, source, duration, label, copy_exact=False):
    directory.mkdir()
    destination = directory / "source.wav"
    if copy_exact:
        shutil.copyfile(source, destination)
    else:
        samples = read_pcm(source)
        write_pcm(destination, samples[:2*round(duration*RATE)])
    digest = sha256(destination)
    conditioning = directory / "conditioning.json"
    write_json(conditioning, {"schemaVersion": 1, "request": {"duration": duration},
                             "beats": [], "downbeats": [], "chords": [],
                             "warnings": ["Existing source reused without additional gain", label]})
    write_json(directory / "prepared.json", {
        "schema_version": 1, "duration": duration, "sample_rate": RATE,
        "source_sha256": digest, "conditioning_sha256": sha256(conditioning),
        "conditions": {"A": "source.wav"}, "condition_sha256": {"A": digest},
        "fixed_source_gain": 1.0, "notes": "Already canonicalized full-song input; no second normalization/gain"})
    return directory


def prepare_comparison(run, output, chunk_index=2):
    run, output = Path(run).resolve(), Path(output)
    identity = json.loads((run / "identity.json").read_text())
    song = json.loads((run / "prepared/song.json").read_text())
    if not 0 <= chunk_index < len(song["chunks"]):
        raise ValueError("Selected chunk is absent from this song run")
    if identity["assets_sha256"] != sha256(ROOT / "configs/assets-base.json"):
        raise ValueError("Model manifest differs from the original run; keep the original experiment separate")
    chunk = song["chunks"][chunk_index]
    source = run / "prepared" / chunk["file"]
    full_source = run / "prepared/source.wav"
    source_record = next(record for record in song["files"] if record["path"] == "source.wav")
    if sha256(source) != chunk["sha256"] or sha256(full_source) != source_record["sha256"]:
        raise ValueError("Cached source audio changed; cannot make a controlled comparison")
    profile = identity["profile"]
    if profile["model"] != "acestep-v15-base" or len(profile["seeds"]) != 1:
        raise ValueError("Expected original Base run with one seed")
    result_files = sorted(run.glob(f"{chunk_index:03d}_A_*/result.json"))
    if len(result_files) != 1:
        raise ValueError("Need one cached result for the selected chunk")
    original = json.loads(result_files[0].read_text())
    original_stem = (result_files[0].parent / original.get("stem_file", "stem.wav")).resolve()
    if original.get("status") != "generated" or not original_stem.is_relative_to(run) or sha256(original_stem) != original["stem_sha256"]:
        raise ValueError("Original chunk result is incomplete or changed")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(original_stem, output / "original_chunk_stem.wav")
    write_json(output / "original_result.json", original)
    write_json(output / "profile.json", profile)
    fresh = create_input(output / "fresh_chunk", source, chunk["frames"] / RATE,
                         f"Exact input of original chunk {chunk_index}", copy_exact=True)
    duration = min(30, song["total_frames"] / RATE)
    context = create_input(output / "context_30s", full_source, duration,
                           "One context, no joins; first up-to-30 seconds of existing source")
    report = {"original_run": str(run), "chunk_index": chunk_index,
              "original_chunk_start_seconds": chunk["start_frame"] / RATE,
              "fresh_chunk": str(fresh), "context": str(context), "context_duration": duration,
              "same_input_sha256": sha256(fresh / "source.wav") == chunk["sha256"],
              "profile": profile, "note": "New processes; same original seed/settings. MPS may not be bit-identical."}
    write_json(output / "comparison.json", report)
    return report

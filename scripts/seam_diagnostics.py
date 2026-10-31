"""Export cached seam previews and signal measurements without generating or replacing audio."""
import argparse
import array
import json
import math
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, read_pcm, write_pcm
from provenance import sha256, write_json


def signal_stats(samples):
    count = len(samples)
    peak = max((abs(value) for value in samples), default=0) / 32768
    rms = math.sqrt(sum(float(value)**2 for value in samples) / count) / 32768 if count else 0
    return {"peak_dbfs": 20*math.log10(peak) if peak else None,
            "rms_dbfs": 20*math.log10(rms) if rms else None,
            "full_scale_fraction": sum(abs(value) >= 32767 for value in samples) / count if count else 0,
            "max_adjacent_change": max((abs(samples[i]-samples[i-2]) for i in range(2,count)), default=0) / 32768}


def inspect_run(run, count=3):
    run = Path(run).resolve()
    song = json.loads((run / "prepared/song.json").read_text())
    if song["sample_rate"] != RATE:
        raise ValueError("Expected canonical 32 kHz song preparation")
    output = run / ("seam_diagnostics_" + uuid.uuid4().hex[:8])
    output.mkdir()
    records = []
    for chunk in song["chunks"][:count]:
        index = chunk["index"]
        matches = sorted(run.glob(f"{index:03d}_A_*/result.json"))
        if len(matches) != 1:
            if records:
                break
            raise ValueError(f"Need a completed result for chunk {index}")
        result = json.loads(matches[0].read_text())
        if result.get("status") != "generated" and records:
            break
        stem = (matches[0].parent / result.get("stem_file", "stem.wav")).resolve()
        if result.get("status") != "generated" or not stem.is_relative_to(run) or sha256(stem) != result["stem_sha256"]:
            raise ValueError(f"Chunk {index} is incomplete, changed or outside this run")
        samples = read_pcm(stem)
        if len(samples) != 2 * chunk["frames"]:
            raise ValueError("Cached stem frame count differs from manifest")
        exported = output / f"chunk_{index:03d}_before_stitch.wav"
        shutil.copyfile(stem, exported)
        raw_files = []
        for raw in sorted(stem.parent.glob("*.wav")):
            if raw.name in {"stem.wav", "mix.wav"}:
                continue
            copy = output / f"chunk_{index:03d}_raw_{len(raw_files)}.wav"
            shutil.copyfile(raw, copy)
            raw_files.append(copy.name)
        records.append({"index": index, "song_start_seconds": chunk["start_frame"] / RATE,
                        "duration_seconds": len(samples) / (2*RATE), "preview": exported.name,
                        "raw_model_previews": raw_files, "stats": signal_stats(samples)})
    # First two boundaries: compare source, standalone bass and final mix at identical song times.
    final = json.loads((run / "experiment.json").read_text()) if (run / "experiment.json").exists() else {}
    stem_file = Path(final["stem"]) if final.get("stem") else run / "bass_full.wav"
    if not stem_file.is_absolute():
        stem_file = (run / stem_file).resolve()
    if not stem_file.resolve().is_relative_to(run):
        raise ValueError("Final stem path escapes the selected run")
    tracks = {"source": run / "prepared/source.wav", "bass": stem_file, "mix": run / "mix_full.wav"}
    for label, file in tracks.items():
        if not file.is_file():
            continue
        if label == "bass" and final.get("stem_sha256") and sha256(file) != final["stem_sha256"]:
            raise ValueError("Final bass changed after generation")
        samples = read_pcm(file)
        if label == "bass":
            for record in records:
                index = record["index"]
                chunk = song["chunks"][index]
                previous_end = (song["chunks"][index-1]["start_frame"] + song["chunks"][index-1]["valid_frames"]) if index else 0
                next_start = song["chunks"][index+1]["start_frame"] if index+1 < len(song["chunks"]) else song["total_frames"]
                left = max(chunk["start_frame"], previous_end)
                right = min(chunk["start_frame"] + chunk["valid_frames"], next_start)
                cached = read_pcm(output / record["preview"])
                segment = samples[2*left:2*right]
                original = cached[2*(left-chunk["start_frame"]):2*(right-chunk["start_frame"])]
                record["unblended_region"] = {"song_start_seconds": left/RATE, "song_end_seconds": right/RATE,
                    "matches_cached_chunk_exactly": bool(segment == original) if right > left else None}
        start, end = 7*RATE, min(22*RATE, len(samples)//2)
        preview = output / f"{label}_song_seconds_7_to_{end/RATE:g}.wav"
        write_pcm(preview, samples[2*min(start,end):2*end])
    report = {"run": str(run), "chunks": records,
              "previews": [p.name for p in sorted(output.glob("*.wav"))],
              "note": "Signal statistics do not diagnose musical glitches. Compare chunk 2 before stitching with the final seam by listening. No inference or repair was run."}
    write_json(output / "report.json", report)
    return output, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output, report = inspect_run(args.run)
    print(json.dumps(report, indent=2))
    print("Diagnostic previews:", output)

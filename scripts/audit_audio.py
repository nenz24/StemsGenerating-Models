"""Audit cached model audio before PCM16 conversion; no model loading or new inference."""
import argparse
import array
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, canonicalize
from provenance import sha256, write_json
from run import cached_job
from system_tools import require_tool


def sample_stats(samples):
    """Count nonfinite/out-of-range samples without treating either as a musical-quality score."""
    finite_count = outside = full_scale = nonfinite = 0
    peak = square_sum = 0.0
    for value in samples:
        if not math.isfinite(value):
            nonfinite += 1
            continue
        finite_count += 1
        magnitude = abs(value)
        peak = max(peak, magnitude)
        square_sum += value*value
        outside += magnitude > 1.0
        full_scale += magnitude >= 32767/32768
    count = len(samples)
    return {"sample_count": count, "nonfinite_samples": nonfinite, "finite_peak": peak,
            "finite_rms": math.sqrt(square_sum/finite_count) if finite_count else None,
            "outside_unit_range_samples": outside,
            "outside_unit_range_fraction": outside/count if count else 0,
            "near_full_scale_fraction": full_scale/count if count else 0}


def audio_stats(path, duration, canonical=False):
    """Decode to float for measurement, optionally matching the bridge's stereo 32k resampling."""
    info = json.loads(subprocess.check_output([require_tool("ffprobe"), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_fmt,sample_rate,channels,bits_per_sample", "-of", "json", str(path)], text=True))
    command = [require_tool("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", str(duration)]
    if canonical:
        command += ["-ar", str(RATE), "-ac", "2"]
    command += ["-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"]
    samples = array.array("f")
    samples.frombytes(subprocess.check_output(command))
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError(f"No decoded audio: {path}")
    return {"metadata": info["streams"][0], **sample_stats(samples)}


def latest_replay():
    """Find the latest completed two-call experiment, rather than choosing a full-song run."""
    candidates = sorted((ROOT / "runs").glob("sequence_diagnostics_*/reversed/comparison.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if json.loads(path.read_text()).get("status") == "generated":
            return path.parent
    raise ValueError("No completed notebook 07 replay found. Supply --run /path/to/reversed")


def audit_run(run, output):
    """Inspect cached A/B outputs and make attenuation-only previews in a new directory."""
    run, output = Path(run).resolve(), Path(output).resolve()
    comparison = json.loads((run / "comparison.json").read_text())
    if comparison.get("status") != "generated" or [job.get("original_chunk_index") for job in comparison["jobs"]] != [1, 0]:
        raise ValueError("Expected completed notebook 07 replay in order [1,0]")
    inputs = []
    for position, job in enumerate(comparison["jobs"]):
        duration = job["duration"]
        if not isinstance(duration, (int,float)) or not math.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError("Expected a finite clip duration in (0,30]")
        folder = run / f'{position:03d}_A_{job["instrument"]}_seed{job["seed"]}'
        result = cached_job(folder, job, comparison.get("execution_signature"))
        source = Path(job["source"]).resolve()
        if result is None or not source.is_relative_to(run) or sha256(source) != job["source_sha256"]:
            raise ValueError("Cached source/result changed; cannot audit this comparison")
        stem = folder / result["stem_file"]
        raw_files = sorted(path for path in stem.parent.glob("*.wav") if path.name not in {"stem.wav", "mix.wav"})
        inputs.append((job, result, source, stem, raw_files))
    output.mkdir(parents=True, exist_ok=False)
    session_file = run / "session.json"
    report = {"run": str(run), "runtime": json.loads(session_file.read_text()) if session_file.exists() else {},
              "clips": [], "note": "No inference. Attenuated previews are diagnostics only, not evaluation stems. No quality or cause inferred automatically."}
    for job, result, source, stem, raw_files in inputs:
        duration, index = job["duration"], job["original_chunk_index"]
        record = {"original_chunk_index": index, "start_seconds": job.get("start_frame", 0)/RATE,
                  "settings": result.get("settings"), "source": audio_stats(source, duration),
                  "stored_stem": audio_stats(stem, duration), "raw_files": []}
        for number, path in enumerate(raw_files):
            native = audio_stats(path, duration)
            before_pcm16 = audio_stats(path, duration, canonical=True)
            entry = {"path": str(path), "sha256": sha256(path), "native": native,
                     "float_before_pcm16": before_pcm16, "attenuation_gain": None, "preview": None}
            if native["nonfinite_samples"] == 0 and before_pcm16["nonfinite_samples"] == 0:
                peak = before_pcm16["finite_peak"]
                gain = min(1.0, 10**(-3/20)/peak) if peak else 1.0
                preview = output / f"chunk_{index:03d}_raw_{number}_headroom.wav"
                canonicalize(path, preview, duration=duration, gain=gain, max_duration=30)
                entry.update(attenuation_gain=gain, preview=str(preview), preview_stats=audio_stats(preview, duration))
            else:
                entry["preview_skipped"] = "Nonfinite raw samples; attenuation cannot repair NaN/Inf"
            record["raw_files"].append(entry)
        report["clips"].append(record)
        write_json(output / "report.json", report)
    write_json(output / "report.json", report)
    return report


def main():
    """Audit the latest notebook 07 replay, or an explicitly selected local replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run if args.run is not None else latest_replay()
    output = args.out if args.out is not None else run / ("audio_audit_" + uuid.uuid4().hex[:8])
    report = audit_run(run, output)
    for clip in report["clips"]:
        print(f"Chunk asli {clip['original_chunk_index']} mulai {clip['start_seconds']:g}s:")
        if not clip["raw_files"]:
            print("  Raw file tidak ditemukan; belum bisa memeriksa sebelum ekspor PCM16")
        for raw in clip["raw_files"]:
            for label, stats in (("RAW",raw["native"]),("FLOAT sebelum PCM16",raw["float_before_pcm16"])):
                print(f"  {label}: peak={stats['finite_peak']:.6g}, di luar [-1,1]={stats['outside_unit_range_fraction']:.2%}, NaN/Inf={stats['nonfinite_samples']}")
            print("  Gain pratinjau:",raw["attenuation_gain"],"|",raw["preview"] or raw["preview_skipped"])
    print("Laporan:",output / "report.json")
    print("Tanpa inferensi/unduhan. Gain pratinjau tidak diterapkan ke hasil lagu lama.")


if __name__ == "__main__":
    main()

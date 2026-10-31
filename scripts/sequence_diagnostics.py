"""Replay the same cached inputs in reverse order to distinguish content from call-order effects."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, read_pcm
from provenance import COMMITS, sha256, write_json
from run import cached_job, execute
from seam_diagnostics import signal_stats


def prepare_sequence(run, output):
    """Validate a 30-second song cache and snapshot chunk 1 then chunk 0 without altering originals."""
    run, output = Path(run).resolve(), Path(output).resolve()
    identity = json.loads((run / "identity.json").read_text())
    song = json.loads((run / "prepared/song.json").read_text())
    previous = json.loads((run / "experiment.json").read_text())
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    profile = identity["profile"]
    if (identity["chunk_seconds"] != 30 or song["chunk_seconds"] != 30
        or identity["upstream_commit"] != COMMITS["ace"]
        or identity["assets_sha256"] != sha256(ROOT / "configs/assets-base.json")
        or previous.get("execution_signature") != signature
        or profile["model"] != "acestep-v15-base" or len(profile["seeds"]) != 1):
        raise ValueError("Expected original Base 30-second run with matching model/settings")
    if song["sample_rate"] != RATE or len(song["chunks"]) < 2 or len(previous["jobs"]) != len(song["chunks"]):
        raise ValueError("Need at least two cached 30-second chunks")
    records, jobs = [], []
    for index in (1, 0):
        chunk, old_job = song["chunks"][index], previous["jobs"][index]
        source = (run / "prepared" / chunk["file"]).resolve()
        if (not source.is_relative_to(run / "prepared") or sha256(source) != chunk["sha256"]
            or chunk["sha256"] != old_job["source_sha256"] or chunk["frames"] != 30*RATE
            or old_job["duration"] != 30 or old_job["seed"] != profile["seeds"][0]
            or old_job["instrument"] != profile["instrument"]):
            raise ValueError("Cached input/settings changed; cannot make a controlled replay")
        job_dir = run / f'{index:03d}_A_{profile["instrument"]}_seed{profile["seeds"][0]}'
        cached = cached_job(job_dir, old_job, signature)
        if cached is None:
            raise ValueError(f"Need a completed original chunk {index}")
        stem = job_dir / cached["stem_file"]
        if len(read_pcm(source)) != 60*RATE or len(read_pcm(stem)) != 60*RATE:
            raise ValueError("Original source/stem must both be exactly 30 seconds")
        records.append({"original_chunk_index": index, "source": str(source), "stem": str(stem),
                        "start_seconds": chunk["start_frame"] / RATE, "result": cached})
        jobs.append(dict(old_job, source=str(source), clean_source=str(source), original_chunk_index=index))
    # Check all inputs before making any comparison directory; never overwrite an older comparison.
    output.mkdir(parents=True, exist_ok=False)
    for position, record in enumerate(records):
        preview = output / f"cached_{record['original_chunk_index']:03d}"
        preview.mkdir()
        shutil.copyfile(record["source"], preview / "source.wav")
        shutil.copyfile(record["stem"], preview / "stem.wav")
        write_json(preview / "result.json", record["result"])
        raw = [p for p in sorted(Path(record["stem"]).parent.glob("*.wav")) if p.name not in {"stem.wav", "mix.wav"}]
        for number, path in enumerate(raw):
            shutil.copyfile(path, preview / f"raw_{number}.wav")
        record["sequence_position"] = position
        record["source_stats"] = signal_stats(read_pcm(preview / "source.wav"))
        record["cached_stem_stats"] = signal_stats(read_pcm(preview / "stem.wav"))
        # Recheck bytes after snapshotting and use snapshots for replay, with no extra gain or resampling.
        if sha256(preview / "source.wav") != jobs[position]["source_sha256"] or sha256(preview / "stem.wav") != record["result"]["stem_sha256"]:
            raise ValueError("Cache changed while snapshotting")
        jobs[position].update(source=str(preview / "source.wav"), clean_source=str(preview / "source.wav"))
    report = {"status": "planned", "backend": "ace", "jobs": jobs, "records": records,
              "original_run": str(run), "profile": profile, "max_job_seconds": 30,
              "note": "Same inputs/settings, reversed order [1,0] in one new process. No automatic audio-quality diagnosis. MPS may vary."}
    write_json(output / "comparison.json", report)
    return report


def run_sequence(output, report):
    """Run two jobs with one new model handler; preserve results and failure metadata."""
    profile = report["profile"]
    settings = SimpleNamespace(upstream=ROOT / "vendor/ACE-Step-1.5", backend="ace", checkpoint=None,
        ace_model=profile["model"], assets_manifest=ROOT / "configs/assets-base.json", device=profile["device"],
        offload=profile["offload_to_cpu"], out=output, prepared=output, max_job_seconds=30)
    try:
        execute(settings, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "comparison.json", report)


def main():
    """Prepare a cache-only comparison by default; inference requires --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = prepare_sequence(args.run, args.out)
    for record in report["records"]:
        print(f"Urutan {record['sequence_position']+1}: chunk asli {record['original_chunk_index']}, mulai {record['start_seconds']:g}s", flush=True)
    if args.execute:
        run_sequence(args.out.resolve(), report)
    else:
        print("PLAN ONLY: cache disalin; model tidak dimuat atau dijalankan", flush=True)


if __name__ == "__main__":
    main()

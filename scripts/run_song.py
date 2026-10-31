"""Generate a full bass/drum track in bounded 10/30-second contexts, then crossfade."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import listening_mix
from provenance import COMMITS, sha256, write_json
from run import cached_job, execute
from song_audio import prepare_song, stitch_song, validate_chunk_settings


def song_identity(source, profile, assets, chunk_seconds=10, overlap_seconds=1):
    validate_chunk_settings(chunk_seconds, overlap_seconds)
    overlap_seconds = int(overlap_seconds) if float(overlap_seconds).is_integer() else overlap_seconds
    return {"schema_version": 1, "source_sha256": sha256(source), "profile": profile,
                "assets_sha256": sha256(assets), "upstream_commit": COMMITS["ace"],
                "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds}


def song_plan(source, output, profile, assets, chunk_seconds=10, overlap_seconds=1):
    identity = song_identity(source, profile, assets, chunk_seconds, overlap_seconds)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    config = output / "identity.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("Run directory belongs to different audio/model/settings; choose a new output")
    write_json(config, identity)
    song = prepare_song(source, output / "prepared", chunk_seconds, overlap_seconds)
    jobs = [{"condition": "A", "instrument": profile["instrument"], "seed": profile["seeds"][0],
             "source": str((output / "prepared" / chunk["file"]).resolve()),
             "clean_source": str((output / "prepared" / chunk["file"]).resolve()),
             "source_sha256": chunk["sha256"], "duration": chunk["frames"] / song["sample_rate"],
             "start_frame": chunk["start_frame"], "valid_frames": chunk["valid_frames"]}
            for chunk in song["chunks"]]
    return {"status": "planned", "execution_signature": signature, "jobs": jobs,
            "duration": song["duration"], "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds,
            "note": "Independent chunks with crossfades; musical continuity is not guaranteed."}, song


def stem_paths(output, experiment):
    stems = []
    for index, job in enumerate(experiment["jobs"]):
        directory = output / f'{index:03d}_A_{job["instrument"]}_seed{job["seed"]}'
        report = cached_job(directory, job, experiment["execution_signature"])
        if report is None:
            return None
        stems.append(directory / report["stem_file"])
    return stems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Use same directory to resume the same source/settings")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/mac-full-song.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--chunk-seconds", type=int, choices=(10, 30), default=10)
    parser.add_argument("--overlap-seconds", type=float, default=1)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    unsupported = sorted(set(profile) & {"caption", "precision", "guidance_scale", "inference_steps"})
    if unsupported:
        parser.error("Full-song runner cannot apply these preset fields: " + ", ".join(unsupported)
                     + ". Use compare_bass_tone.py --prepared for the selected Electric Bass clip preset.")
    if profile["model"] != "acestep-v15-base" or len(profile["seeds"]) != 1:
        parser.error("Full-song POC uses Base and exactly one seed")
    assets = ROOT / "configs/assets-base.json"
    try:
        experiment, song = song_plan(args.source, args.out, profile, assets, args.chunk_seconds, args.overlap_seconds)
    except ValueError as error:
        parser.error(str(error))
    print(f"Lagu {song['duration']:.2f}s: {len(experiment['jobs'])} chunk, model dimuat sekali bila ada chunk tersisa", flush=True)
    print("Crossfade menghaluskan sambungan; harmoni/groove antar-chunk perlu dinilai", flush=True)
    if not args.execute:
        write_json(args.out / "plan.json", experiment)
        return
    import fcntl
    lock = (args.out / ".generation.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another generation process owns this song run; wait for or interrupt that process first")
    try:
        stems = stem_paths(args.out, experiment)
        if stems is None:
            settings = SimpleNamespace(upstream=ROOT / "vendor/ACE-Step-1.5", backend="ace", checkpoint=None,
                ace_model=profile["model"], assets_manifest=assets, device=profile["device"],
                offload=profile["offload_to_cpu"], out=args.out, prepared=args.out / "prepared",
                resume=True, execution_signature=experiment["execution_signature"],
                max_job_seconds=args.chunk_seconds)
            execute(settings, experiment)
            stems = stem_paths(args.out, experiment)
        stem = args.out / (profile["instrument"] + "_full.wav")
        mix = args.out / "mix_full.wav"
        previous = args.out / "experiment.json"
        if stem.exists() or mix.exists():
            report = json.loads(previous.read_text()) if previous.exists() else {}
            if (report.get("status") == "generated" and stem.is_file() and mix.is_file()
                and sha256(stem) == report.get("stem_sha256") and sha256(mix) == report.get("mix_sha256")):
                print(f"Memakai hasil lagu penuh yang lolos checksum: {stem}", flush=True)
                return
            raise ValueError("Final output exists but is not an unchanged completed result; preserve it and use a new output directory")
        temporary_stem = args.out / ("stem_" + uuid.uuid4().hex[:8] + ".tmp.wav")
        temporary_mix = args.out / ("mix_" + uuid.uuid4().hex[:8] + ".tmp.wav")
        stitch_song(song, stems, temporary_stem)
        listening_mix(args.out / "prepared/source.wav", temporary_stem, temporary_mix)
        temporary_stem.rename(stem)
        temporary_mix.rename(mix)
        experiment.update(status="generated", stem=str(stem), stem_sha256=sha256(stem), mix_sha256=sha256(mix))
        print(f"Selesai: {stem} dan mix_full.wav", flush=True)
    except BaseException as error:
        experiment.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(args.out / "experiment.json", experiment)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    write_json(args.out / "experiment.json", experiment)


if __name__ == "__main__":
    main()

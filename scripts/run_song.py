"""Generate a full bass/drum track in bounded 10-second inference chunks, then crossfade."""
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
from song_audio import prepare_song, stitch_song


def song_plan(source, output, profile, assets):
    identity = {"schema_version": 1, "source_sha256": sha256(source), "profile": profile,
                "assets_sha256": sha256(assets), "upstream_commit": COMMITS["ace"],
                "chunk_seconds": 10, "overlap_seconds": 1}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    config = output / "identity.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("Run directory belongs to different audio/model/settings; choose a new output")
    write_json(config, identity)
    song = prepare_song(source, output / "prepared")
    jobs = [{"condition": "A", "instrument": profile["instrument"], "seed": profile["seeds"][0],
             "source": str((output / "prepared" / chunk["file"]).resolve()),
             "clean_source": str((output / "prepared" / chunk["file"]).resolve()),
             "source_sha256": chunk["sha256"], "duration": 10.0,
             "start_frame": chunk["start_frame"], "valid_frames": chunk["valid_frames"]}
            for chunk in song["chunks"]]
    return {"status": "planned", "execution_signature": signature, "jobs": jobs,
            "duration": song["duration"], "chunk_seconds": 10, "overlap_seconds": 1,
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
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if profile["model"] != "acestep-v15-base" or len(profile["seeds"]) != 1:
        parser.error("Full-song POC uses Base and exactly one seed")
    assets = ROOT / "configs/assets-base.json"
    experiment, song = song_plan(args.source, args.out, profile, assets)
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
                resume=True, execution_signature=experiment["execution_signature"])
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

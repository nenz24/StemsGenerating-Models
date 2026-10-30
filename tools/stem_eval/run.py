"""Plan or execute reproducible single-stem experiments; plans never imply inference."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import time

from audio import canonicalize, listening_mix
from provenance import COMMITS, check_upstream, sha256, write_json


def plan(prepared, backend, instrument, seeds):
    manifest = json.loads((prepared / "prepared.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported preparation schema")
    if sha256(prepared / "conditioning.json") != manifest["conditioning_sha256"]:
        raise ValueError("Conditioning changed after preparation")
    if instrument not in ("bass", "drums"):
        raise ValueError("Baseline instruments are bass or drums")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("Use unique seeds between 0 and 2147483647")
    conditions = ["A"] if backend == "ace" else list(manifest["conditions"])
    jobs = []
    for condition in conditions:
        source = prepared / manifest["conditions"][condition]
        if not source.is_file() or source.resolve().parent != prepared.resolve():
            raise ValueError("Condition audio must be a file in the prepared clip directory")
        if sha256(source) != manifest["condition_sha256"][condition]:
            raise ValueError("Condition audio changed after preparation")
        for seed in seeds:
            jobs.append({"condition": condition, "instrument": instrument, "seed": seed,
                         "source": str(source.resolve()), "source_sha256": sha256(source),
                         "duration": manifest["duration"]})
    return {"schema_version": 1, "status": "planned", "backend": backend,
            "upstream_commit": COMMITS[backend], "jobs": jobs,
            "conditioning_sha256": sha256(prepared / "conditioning.json"),
            "clean_source_sha256": sha256(prepared / "source.wav"),
            "conditioning_consumed": "audio only; raw timing/chords retained for evaluation, not model inputs"}


def execute(args, experiment):
    from backends import AceBackend, StageBackend, weight_inventory
    check_upstream(args.upstream, args.backend)
    if args.backend == "stage" and (args.checkpoint is None or args.checkpoint.name != f"stage-{args.instrument}.safetensors"):
        raise ValueError("STAGE requires the matching official instrument checkpoint filename")
    # These are local experiments. Weight acquisition is explicit and separate from generation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this baseline; no CPU benchmark substitute")
    inventory = weight_inventory(args.upstream, args.backend, args.checkpoint, args.ace_model)
    if args.backend == "ace":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from model_assets import materialize, process
        process(args.assets_manifest, args.upstream)
        runtime = materialize(args.assets_manifest, args.upstream, args.out / "runtime")
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(runtime.resolve())
    session = {"weights": inventory, "model": args.ace_model if args.backend == "ace" else str(args.checkpoint), "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name(), "versions": {}}
    for package in ("torch", "torchaudio", "transformers", "lightning", "numpy"):
        try:
            session["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    started = time.perf_counter()
    backend = (AceBackend(args.upstream, model_name=args.ace_model, offload_to_cpu=args.offload)
               if args.backend == "ace" else StageBackend(args.upstream, args.checkpoint))
    torch.cuda.synchronize()
    session["model_load_seconds"] = time.perf_counter() - started
    if args.backend == "ace":
        session["assets_manifest_sha256"] = sha256(args.assets_manifest)
        session["effective_model_code"] = [
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted((runtime / args.ace_model).glob("*.py"))
        ]
    write_json(args.out / "session.json", session)
    for index, job in enumerate(experiment["jobs"]):
        directory = args.out / f'{index:03d}_{job["condition"]}_{job["instrument"]}_seed{job["seed"]}'
        directory.mkdir()
        report = dict(job, status="running")
        write_json(directory / "result.json", report)
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                raw, settings = backend.generate(Path(job["source"]), job["duration"],
                                                  job["instrument"], job["seed"], directory)
            torch.cuda.synchronize()
            report.update(inference_seconds=time.perf_counter() - started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), settings=settings)
            info = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(raw)
            ], text=True))
            if abs(float(info["format"]["duration"]) - job["duration"]) > 0.05:
                raise ValueError("Generated duration mismatch; raw audio retained for inspection")
            canonicalize(raw, directory / "stem.wav", duration=job["duration"], gain=1)
            listening_mix(args.prepared / "source.wav", directory / "stem.wav", directory / "mix.wav")
            report.update(status="generated", stem_sha256=sha256(directory / "stem.wav"))
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(directory / "result.json", report)
            raise
        write_json(directory / "result.json", report)
    experiment["status"] = "generated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backend", choices=COMMITS, required=True)
    parser.add_argument("--instrument", choices=("bass", "drums"), default="bass")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="STAGE instrument-specific safetensors file")
    parser.add_argument("--ace-model", choices=("acestep-v15-base", "acestep-v15-xl-base"), default="acestep-v15-base")
    parser.add_argument("--assets-manifest", type=Path, help="Pinned publisher artifact manifest; required for ACE execution")
    parser.add_argument("--offload", action="store_true", help="Use upstream CPU offload to reduce GPU memory")
    args = parser.parse_args()
    if args.execute and args.upstream is None:
        parser.error("--execute requires --upstream")
    if args.execute and args.backend == "ace" and args.assets_manifest is None:
        parser.error("ACE execution requires --assets-manifest for artifact verification")
    if args.assets_manifest is not None:
        manifest_model = json.loads(args.assets_manifest.read_text())["model"]
        if manifest_model != args.ace_model:
            parser.error("Asset manifest must match --ace-model")
    experiment = plan(args.prepared, args.backend, args.instrument, args.seeds)
    experiment["ace_model"] = args.ace_model
    experiment["offload_to_cpu"] = args.offload
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "experiment.json", experiment)
    if args.execute:
        try:
            execute(args, experiment)
        except Exception as error:
            experiment.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(args.out / "experiment.json", experiment)
            raise
        write_json(args.out / "experiment.json", experiment)


if __name__ == "__main__":
    main()

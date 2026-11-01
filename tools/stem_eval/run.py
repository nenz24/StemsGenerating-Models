"""Plan or execute reproducible single-stem experiments; plans never imply inference."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import subprocess
import time
import uuid

from audio import canonicalize, listening_mix
from accelerator import memory_stats, reset_memory_stats, select_device, synchronize
from system_tools import require_audio_tools, require_tool
from provenance import COMMITS, check_upstream, sha256, write_json


def plan(prepared, backend, instrument, seeds):
    manifest = json.loads((prepared / "prepared.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported preparation schema")
    if sha256(prepared / "conditioning.json") != manifest["conditioning_sha256"]:
        raise ValueError("Conditioning changed after preparation")
    if instrument not in ("bass", "drums", "guitar"):
        raise ValueError("Supported instruments are bass, drums or guitar")
    if instrument == "guitar" and backend != "ace":
        raise ValueError("Guitar currently requires the ACE-Step backend")
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


def cached_job(directory, job, signature):
    file = directory / "result.json"
    if not file.exists():
        return None
    report = json.loads(file.read_text())
    if report.get("status") != "generated":
        return None
    for name in ("source_sha256", "instrument", "seed", "duration", "condition"):
        if report.get(name) != job.get(name):
            raise ValueError("Completed chunk belongs to different inputs; use a new run")
    if "guidance_scale" in job:
        recorded = report.get("guidance_scale", report.get("settings", {}).get("params", {}).get("guidance_scale", 7.0))
        if recorded != job["guidance_scale"]:
            raise ValueError("Completed chunk belongs to different guidance; use a new run")
    if report.get("execution_signature") != signature:
        raise ValueError("Completed chunk belongs to different configuration; use a new run")
    stem = (directory / report.get("stem_file", "stem.wav")).resolve()
    if not stem.is_relative_to(directory.resolve()) or sha256(stem) != report["stem_sha256"]:
        raise ValueError("Cached stem is missing or changed; refusing to silently reuse it")
    return report


def execute(args, experiment):
    from backends import AceBackend, StageBackend, weight_inventory, validate_guidance
    guidance = validate_guidance(getattr(args, "guidance_scale", 7.0))
    require_audio_tools()
    bridge_root = Path(getattr(args, "bridge_root", Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(bridge_root / "scripts"))
    from runtime_environment import runtime_environment
    environment = runtime_environment()
    for name in ("HF_HUB_CACHE", "HF_XET_CACHE", "MPLBACKEND", "MPLCONFIGDIR"):
        os.environ[name] = environment[name]
    check_upstream(args.upstream, args.backend)
    if args.backend == "stage" and (args.checkpoint is None or args.checkpoint.name != f"stage-{args.instrument}.safetensors"):
        raise ValueError("STAGE requires the matching official instrument checkpoint filename")
    # These are local experiments. Weight acquisition is explicit and separate from generation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    device = select_device(torch, args.device)
    if args.backend == "stage" and device != "cuda":
        raise RuntimeError("The inspected STAGE adapter requires CUDA. Use ACE-Step on Apple Silicon.")
    inventory = weight_inventory(args.upstream, args.backend, args.checkpoint, args.ace_model)
    if args.backend == "ace":
        sys.path.insert(0, str(bridge_root / "scripts"))
        from model_assets import materialize, process
        process(args.assets_manifest, args.upstream)
        runtime_name = "runtime_" + uuid.uuid4().hex[:8] if getattr(args, "resume", False) else "runtime"
        runtime = materialize(args.assets_manifest, args.upstream, args.out / runtime_name)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(runtime.resolve())
    session = {"weights": inventory, "model": args.ace_model if args.backend == "ace" else str(args.checkpoint), "platform": platform.platform(),
               "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else "Apple Silicon / MPS", "versions": {}}
    for package in ("torch", "torchaudio", "transformers", "lightning", "numpy"):
        try:
            session["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    backend = None
    resume = getattr(args, "resume", False)
    signature = getattr(args, "execution_signature", None)
    for index, job in enumerate(experiment["jobs"]):
        if args.backend == "ace":
            job["guidance_scale"] = guidance
        directory = args.out / f'{index:03d}_{job["condition"]}_{job["instrument"]}_seed{job["seed"]}'
        directory.mkdir(exist_ok=resume)
        if resume and cached_job(directory, job, signature) is not None:
            print(f"[{index+1}/{len(experiment['jobs'])}] Memakai chunk selesai yang lolos checksum", flush=True)
            continue
        if backend is None:
            started = time.perf_counter()
            backend = (AceBackend(args.upstream, model_name=args.ace_model, offload_to_cpu=args.offload, device=device, guidance_scale=guidance)
                       if args.backend == "ace" else StageBackend(args.upstream, args.checkpoint))
            if getattr(args, "capture_decoder", None) is not None:
                from decoder_capture import install_capture
                install_capture(backend.handler, args.capture_decoder)
            synchronize(torch, device)
            session["model_load_seconds"] = time.perf_counter() - started
            if args.backend == "ace":
                session["assets_manifest_sha256"] = sha256(args.assets_manifest)
                session["effective_model_code"] = [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in sorted((runtime / args.ace_model).glob("*.py"))
                ]
            session_file = args.out / "session.json"
            if resume and session_file.exists():
                session_file.rename(args.out / f"session.previous_{time.time_ns()}.json")
            write_json(session_file, session)
        attempt = directory / ("attempt_" + uuid.uuid4().hex[:8]) if resume else directory
        if resume:
            attempt.mkdir()
            previous = directory / "result.json"
            if previous.exists():
                previous.rename(directory / f"result.previous_{time.time_ns()}.json")
        report = dict(job, status="running", execution_signature=signature)
        write_json(directory / "result.json", report)
        print(f"[{index+1}/{len(experiment['jobs'])}] Generate {job['instrument']} mulai {job.get('start_frame', 0)/32000:.2f}s", flush=True)
        try:
            reset_memory_stats(torch, device)
            started = time.perf_counter()
            with torch.inference_mode():
                raw, settings = backend.generate(Path(job["source"]), job["duration"],
                                                  job["instrument"], job["seed"], attempt)
            synchronize(torch, device)
            report.update(inference_seconds=time.perf_counter() - started,
                          device=device, settings=settings, **memory_stats(torch, device))
            info = json.loads(subprocess.check_output([
                require_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(raw)
            ], text=True))
            if abs(float(info["format"]["duration"]) - job["duration"]) > 0.05:
                raise ValueError("Generated duration mismatch; raw audio retained for inspection")
            canonicalize(raw, attempt / "stem.wav", duration=job["duration"], gain=1,
                         max_duration=getattr(args, "max_job_seconds", 10))
            clean_source = Path(job.get("clean_source", args.prepared / "source.wav"))
            listening_mix(clean_source, attempt / "stem.wav", attempt / "mix.wav")
            report.update(status="generated", stem_sha256=sha256(attempt / "stem.wav"),
                          stem_file=str((attempt / "stem.wav").relative_to(directory)))
            print(f"  Selesai dalam {report['inference_seconds']:.1f}s", flush=True)
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
    parser.add_argument("--instrument", choices=("bass", "drums", "guitar"), default="bass")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-job-seconds", type=int, choices=(10, 30), default=10, help="Explicitly opt into the 30-second unchunked context test")
    parser.add_argument("--capture-decoder", type=Path, help="Diagnostic: capture one ACE job's exact latents and default float decode")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="ACE Base guidance; 1 bypasses CFG amplification, default remains 7")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="STAGE instrument-specific safetensors file")
    parser.add_argument("--ace-model", choices=("acestep-v15-base", "acestep-v15-xl-base"), default="acestep-v15-base")
    parser.add_argument("--assets-manifest", type=Path, help="Pinned publisher artifact manifest; required for ACE execution")
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--offload", action="store_true", help="Use upstream CPU offload to reduce GPU memory")
    args = parser.parse_args()
    from backends import validate_guidance
    try:
        args.guidance_scale = validate_guidance(args.guidance_scale)
    except ValueError as error:
        parser.error(str(error))
    if args.backend != "ace" and args.guidance_scale != 7.0:
        parser.error("--guidance-scale is an ACE-only option")
    if args.execute and args.upstream is None:
        parser.error("--execute requires --upstream")
    if args.execute and args.backend == "ace" and args.assets_manifest is None:
        parser.error("ACE execution requires --assets-manifest for artifact verification")
    if args.assets_manifest is not None:
        manifest_model = json.loads(args.assets_manifest.read_text())["model"]
        if manifest_model != args.ace_model:
            parser.error("Asset manifest must match --ace-model")
    experiment = plan(args.prepared, args.backend, args.instrument, args.seeds)
    if args.capture_decoder is not None and (args.backend != "ace" or len(experiment["jobs"]) != 1):
        parser.error("Decoder capture requires exactly one ACE job")
    if any(not isinstance(job["duration"], (int,float)) or not math.isfinite(job["duration"])
           or not 0 < job["duration"] <= args.max_job_seconds for job in experiment["jobs"]):
        parser.error("Job duration exceeds the explicit --max-job-seconds limit")
    experiment["max_job_seconds"] = args.max_job_seconds
    experiment["device_requested"] = args.device
    experiment["ace_model"] = args.ace_model
    experiment["offload_to_cpu"] = args.offload
    if args.backend == "ace":
        experiment["guidance_scale"] = args.guidance_scale
        for job in experiment["jobs"]:
            job["guidance_scale"] = args.guidance_scale
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

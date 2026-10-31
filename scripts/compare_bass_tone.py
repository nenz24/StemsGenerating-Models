"""One natural electric-bass caption candidate against a completed CUDA FP32 baseline.

May be uploaded alone into an existing Colab session: compare(...,
float32_worker=FP32_WORKER) reuses its verified source/runtime/weights.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

ELECTRIC_BASS_CAPTION = (
    "Isolated fingerstyle electric bass guitar accompaniment. A dry, natural plucked tone "
    "with clearly articulated attacks, short note releases and audible upper harmonics. "
    "A simple musical bassline in a normal bass-guitar register following the piano's "
    "harmony, timing and dynamics."
)

ELECTRIC_BASS_PRESET = {
    "schema_version": 1, "id": "electric-bass", "model": "acestep-v15-base",
    "device": "cuda", "precision": "float32", "instrument": "bass", "seeds": [0],
    "guidance_scale": 7.0, "inference_steps": 50, "offload_to_cpu": True,
    "caption": ELECTRIC_BASS_CAPTION,
}


def selected_preset(root):
    """Use the selected versioned preset; the lone uploaded helper embeds the same values."""
    path = Path(root) / "configs/electric-bass.json"
    preset = json.loads(path.read_text()) if path.is_file() else copy.deepcopy(ELECTRIC_BASS_PRESET)
    if preset != ELECTRIC_BASS_PRESET:
        raise ValueError("Electric Bass preset differs from the selected v1 settings; preserve it and use a new version")
    return preset


def caption_worker(float32_worker, caption=ELECTRIC_BASS_CAPTION):
    """Override only the serialized caption before the existing adapter constructs params."""
    boundary = "sys.argv = sys.argv[1:]"
    if not isinstance(float32_worker, str) or float32_worker.count(boundary) != 1:
        raise ValueError("Pass the FP32_WORKER from the successful CUDA recovery cell")
    if not isinstance(caption, str) or not caption.strip() or len(caption) >= 512:
        raise ValueError("Use a nonempty caption shorter than 512 characters")
    override = f'''
import acestep.inference as ace_inference
OriginalParams = ace_inference.GenerationParams
class ElectricBassParams(OriginalParams):
    def __init__(self, *args, **kwargs):
        kwargs["caption"] = {caption!r}
        super().__init__(*args, **kwargs)
ace_inference.GenerationParams = ElectricBassParams
'''
    return float32_worker.replace(boundary, override + "\n" + boundary, 1)


def require_fp32(capture):
    calls = [call for call in capture.get("calls", []) if call.get("backend") == "pytorch_vae"]
    if (capture.get("status") != "captured" or capture.get("model_dtype") != "torch.float32"
            or capture.get("model_device") != "cuda" or not calls
            or any(call.get("dtype") != "torch.float32" or call.get("status") != "returned" for call in calls)):
        raise ValueError("Expected completed CUDA FP32 model/decoder capture")


def generate(root, prepared, output, *, execute=False, float32_worker=None):
    """Generate the selected Electric Bass directly for one prepared 10–30-second clip."""
    root, prepared, output = map(lambda p: Path(p).resolve(), (root, prepared, output))
    sys.path[:0] = [str(root / "scripts"), str(root / "tools/stem_eval")]
    from audio import canonicalize
    from provenance import COMMITS, write_json
    from run import cached_job, plan
    from mac_workflow import run_command
    from runtime_environment import runtime_environment
    from setup_runtime import runtime_python
    from cuda_reference import check_cuda
    from stream_process import run_logged

    preset = selected_preset(root)
    requested = plan(prepared, "ace", "bass", preset["seeds"])
    if requested["jobs"][0]["source_sha256"] != requested["clean_source_sha256"]:
        raise ValueError("Selected Electric Bass expects the clean audio as condition A")
    duration = requested["jobs"][0]["duration"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 10 <= duration <= 30:
        raise ValueError("Electric Bass clip preset supports prepared clips from 10 to 30 seconds")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "planned", "preset": preset, "upstream_commit": COMMITS["ace"],
              "source_sha256": requested["clean_source_sha256"], "duration": duration,
              "conditioning_sha256": requested["conditioning_sha256"],
              "note": "Selected Electric Bass preset, one new clip. Musical consistency across chunks is not guaranteed."}
    write_json(output / "generation.json", report)
    command = run_command(root, preset, prepared, output / "generation", execute=execute)
    command += ["--max-job-seconds", "30", "--guidance-scale", str(preset["guidance_scale"]),
                "--capture-decoder", str(output / "capture")]
    try:
        if execute:
            report["hardware"] = check_cuda(runtime_python())
            if float32_worker is None:
                from cuda_precision import FLOAT32_WORKER
                float32_worker = FLOAT32_WORKER
            command = [command[0], "-c", caption_worker(float32_worker, preset["caption"]), *command[1:]]
        run_logged(command, env=runtime_environment(root), check=True)
        if not execute:
            print("PLAN ONLY: selected Electric Bass, no model inference", flush=True)
            return report
        experiment = json.loads((output / "generation/experiment.json").read_text())
        if experiment.get("status") != "generated" or len(experiment["jobs"]) != 1:
            raise ValueError("Expected one completed Electric Bass clip")
        job = experiment["jobs"][0]
        folder = output / "generation/000_A_bass_seed0"
        result = cached_job(folder, job, None)
        if result is None or result.get("device") != "cuda" or job["source_sha256"] != report["source_sha256"]:
            raise ValueError("Electric Bass device/input/result verification failed")
        settings = result["settings"]
        expected = {"caption": preset["caption"], "seed": 0, "guidance_scale": 7,
                    "inference_steps": 50, "task_type": "lego", "duration": duration,
                    "instrumental": True, "enable_normalization": False, "thinking": False,
                    "use_cot_caption": False, "use_cot_metas": False, "use_cot_language": False}
        if any(settings["params"].get(key) != value for key, value in expected.items()):
            raise ValueError("Actual generation settings differ from the selected Electric Bass preset")
        config = settings["config"]
        if config.get("seeds") != [0] or config.get("batch_size") != 1 or config.get("use_random_seed") is not False:
            raise ValueError("Actual seed/batch settings differ from the selected preset")
        capture = json.loads((output / "capture/capture.json").read_text())
        require_fp32(capture)
        unchanged = plan(prepared, "ace", "bass", preset["seeds"])
        if (unchanged["clean_source_sha256"] != report["source_sha256"]
                or unchanged["jobs"][0]["source_sha256"] != report["source_sha256"]
                or unchanged["conditioning_sha256"] != report["conditioning_sha256"]):
            raise ValueError("Prepared input changed during Electric Bass generation")
        canonicalize(folder / result["stem_file"], output / "electric_listen.wav", duration=duration, gain=0.7, max_duration=30)
        shutil.copyfile(folder / "mix.wav", output / "electric_mix.wav")
        report.update(status="generated", settings=settings, decoder_capture=capture,
                      stem_sha256=result["stem_sha256"], listening_gain=0.7, mix_track_gain=0.5)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "generation.json", report)
    with zipfile.ZipFile(output / "electric_bass_result.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("generation.json", "electric_listen.wav", "electric_mix.wav", "capture/capture.json",
                     "generation/session.json", "generation/000_A_bass_seed0/result.json"):
            archive.write(output / name, name)
    return report


def compare(root, reference, baseline, output, *, execute=False, float32_worker=None):
    root, reference, baseline, output = map(lambda p: Path(p).resolve(), (root, reference, baseline, output))
    sys.path[:0] = [str(root / "scripts"), str(root / "tools/stem_eval")]
    from audio import canonicalize, listening_mix
    from provenance import COMMITS, sha256, write_json
    from run import cached_job
    from mac_workflow import run_command
    from runtime_environment import runtime_environment
    from setup_runtime import runtime_python
    from cuda_reference import check_cuda
    from stream_process import run_logged

    original = json.loads((baseline / "cuda_reference.json").read_text())
    request = original["requested"]
    if (original.get("status") != "generated" or original.get("upstream_commit") != COMMITS["ace"]
            or request != {"device": "cuda", "model": "acestep-v15-base", "seed": 0,
                           "guidance_scale": 7, "steps": 50, "duration": 30}):
        raise ValueError("Select the successful 30-second Base/seed 0/CFG 7 CUDA reference run")
    require_fp32(original["decoder_capture"])
    experiment = json.loads((baseline / "generation/experiment.json").read_text())
    if experiment.get("status") != "generated" or len(experiment["jobs"]) != 1:
        raise ValueError("Baseline must contain one completed CUDA job")
    job = experiment["jobs"][0]
    baseline_folder = baseline / "generation/000_A_bass_seed0"
    result = cached_job(baseline_folder, job, None)
    if (result is None or result.get("device") != "cuda" or result.get("instrument") != "bass"
            or result["stem_sha256"] != original["candidate_sha256"] or result["settings"] != original["settings"]):
        raise ValueError("Baseline result/settings differ from the successful CUDA report")
    source = reference / "prepared/source.wav"
    if sha256(source) != original["source_sha256"] or job["source_sha256"] != original["source_sha256"]:
        raise ValueError("B source differs from the successful CUDA reference")
    model_manifest = json.loads((baseline / "generation/session.json").read_text())["assets_manifest_sha256"]
    if model_manifest != sha256(root / "configs/assets-base.json"):
        raise ValueError("Model asset manifest changed since the CUDA baseline")
    if original["settings"]["params"]["caption"] == ELECTRIC_BASS_CAPTION:
        raise ValueError("Baseline already uses this caption; choose the original generic-bass CUDA run")
    expected_params = {"task_type": "lego", "duration": 30, "seed": 0, "guidance_scale": 7,
                       "inference_steps": 50, "instrumental": True}
    if any(original["settings"]["params"].get(key) != value for key, value in expected_params.items()):
        raise ValueError("Baseline generation settings differ from the fixed CUDA protocol")
    baseline_stem = baseline_folder / result.get("stem_file", "stem.wav")
    profile = {"device": "cuda", "model": request["model"], "instrument": "bass", "seeds": [0],
               "offload_to_cpu": experiment["offload_to_cpu"]}
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "planned", "upstream_commit": COMMITS["ace"],
              "baseline_directory": str(baseline), "source_sha256": original["source_sha256"],
              "baseline_stem_sha256": result["stem_sha256"], "assets_manifest_sha256": model_manifest,
              "requested_caption": ELECTRIC_BASS_CAPTION, "baseline_settings": original["settings"],
              "precision": "float32", "seed": 0, "guidance_scale": 7, "steps": 50, "duration": 30,
              "note": "One caption candidate; no EQ, pitch shift, metadata or seed sweep. Caption control is not a timbre guarantee or a match to chunk 0. A cached baseline is not a deterministic causal test; VAE sampling/runtime can vary."}
    write_json(output / "comparison.json", report)
    command = run_command(root, profile, reference / "prepared", output / "generation", execute=execute)
    command += ["--max-job-seconds", "30", "--guidance-scale", "7", "--capture-decoder", str(output / "capture")]
    try:
        if execute:
            report["hardware"] = check_cuda(runtime_python())
            if float32_worker is None:
                from cuda_precision import FLOAT32_WORKER
                float32_worker = FLOAT32_WORKER
            command = [command[0], "-c", caption_worker(float32_worker), *command[1:]]
        run_logged(command, env=runtime_environment(root), check=True)
        if not execute:
            print("PLAN ONLY: no tone candidate generated", flush=True)
            return report
        generated = json.loads((output / "generation/experiment.json").read_text())
        if generated.get("status") != "generated" or len(generated["jobs"]) != 1:
            raise ValueError("Expected exactly one completed tone candidate")
        candidate_job = generated["jobs"][0]
        folder = output / "generation/000_A_bass_seed0"
        candidate = cached_job(folder, candidate_job, None)
        if (candidate is None or candidate.get("device") != "cuda"
                or candidate_job["source_sha256"] != original["source_sha256"]):
            raise ValueError("Candidate device/input/result verification failed")
        before, after = original["settings"]["params"], candidate["settings"]["params"]
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if (after.get("caption") != ELECTRIC_BASS_CAPTION or changed - {"caption", "src_audio"}
                or candidate["settings"]["config"] != original["settings"]["config"]):
            raise ValueError("Generation changed parameters beyond the requested caption")
        capture = json.loads((output / "capture/capture.json").read_text())
        require_fp32(capture)
        if sha256(baseline_stem) != result["stem_sha256"] or sha256(source) != original["source_sha256"]:
            raise ValueError("Original source/baseline changed during comparison")
        for label, stem in (("baseline", baseline_stem), ("electric", folder / candidate.get("stem_file", "stem.wav"))):
            canonicalize(stem, output / f"{label}_listen.wav", duration=30, gain=0.7, max_duration=30)
            listening_mix(source, stem, output / f"{label}_mix.wav")
        report.update(status="generated", candidate_settings=candidate["settings"], decoder_capture=capture,
                      candidate_sha256=candidate["stem_sha256"], listening_gain_both=0.7, mix_track_gain_both=0.5)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "comparison.json", report)
    with zipfile.ZipFile(output / "bass_tone_result.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("comparison.json", "baseline_listen.wav", "electric_listen.wav", "baseline_mix.wav",
                     "electric_mix.wav", "capture/capture.json", "generation/session.json", "generation/000_A_bass_seed0/result.json"):
            archive.write(output / name, name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prepared", type=Path, help="Generate the selected Electric Bass directly; no A/B baseline needed")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepared:
        if args.reference or args.baseline:
            parser.error("Use --prepared for direct generation, or --reference/--baseline for comparison")
        output = args.out or args.root / "runs" / ("electric_bass_" + uuid.uuid4().hex[:12])
        report = generate(args.root, args.prepared, output, execute=args.execute)
    else:
        if not args.reference or not args.baseline:
            parser.error("Comparison requires --reference and --baseline; direct generation uses --prepared")
        output = args.out or args.root / "runs" / ("bass_tone_" + uuid.uuid4().hex[:12])
        report = compare(args.root, args.reference, args.baseline, output, execute=args.execute)
    print(json.dumps(report, indent=2))
    print("Output:", output)


if __name__ == "__main__":
    main()

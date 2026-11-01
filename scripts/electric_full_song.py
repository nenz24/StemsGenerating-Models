"""Single-pass 10–600s Electric Bass, including use as a lone uploaded Colab helper.

Uses the existing pinned runtime and engine API, without its diagnostic CLI's
10/30s cap. One source context, one model call; no song chunking or stitching.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
import wave
import zipfile

RATE = 32000
PRESET = {
    "schema_version": 1, "id": "electric-bass", "model": "acestep-v15-base",
    "device": "cuda", "precision": "float32", "instrument": "bass", "seeds": [0],
    "guidance_scale": 7.0, "inference_steps": 50, "offload_to_cpu": True,
    "caption": "Isolated fingerstyle electric bass guitar accompaniment. A dry, natural plucked tone "
               "with clearly articulated attacks, short note releases and audible upper harmonics. "
               "A simple musical bassline in a normal bass-guitar register following the piano's "
               "harmony, timing and dynamics.",
}


def discover_root(base, preferred=None):
    """Find a bridge checkout without notebook globals or recursively scanning model caches."""
    base = Path(base).resolve()
    markers = ("configs/assets-base.json", "tools/stem_eval/run.py", "scripts/setup_runtime.py")
    def valid(path):
        return all((path / name).is_file() for name in markers)
    if preferred is not None:
        path = Path(preferred).expanduser().resolve()
        if not valid(path):
            raise RuntimeError(f"ROOT yang dipilih bukan checkout bridge: {path}")
        return path
    candidates = set()
    for folder in dict.fromkeys((base, base.parent)):
        for path in (folder, folder / "source", folder / "StemsGenerating-Models",
                     *folder.glob("songmap_cuda_*/source"), *folder.glob("ace_context_source_*")):
            if valid(path): candidates.add(path.resolve())
    ready = [path for path in candidates
             if (path / "vendor/ACE-Step-1.5/.venv/bin/python").is_file()
             and (path / "vendor/ACE-Step-1.5/checkpoints/acestep-v15-base/model.safetensors").is_file()]
    choices = sorted(ready or candidates)
    if len(choices) > 1:
        raise RuntimeError("Ada beberapa checkout bridge. Isi ROOT_HINT dengan salah satu path ini: "
                           + ", ".join(map(str, choices)))
    return choices[0] if choices else None


def model_python(root):
    python = root / "vendor/ACE-Step-1.5/.venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise RuntimeError("Runtime model belum ada pada VM ini. Jalankan sel setup runtime dan bobot "
                           "di notebooks/10_colab_electric_full_song.ipynb, lalu ulangi generasi.")
    return python


def attach(root):
    sys.path[:0] = [str(root / "tools/stem_eval"), str(root / "scripts")]


def frames(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (RATE, 2, 2):
            raise ValueError("Expected stereo 32kHz PCM16")
        return audio.getnframes()


def canonicalize_long(source, target, start=0, duration=10, gain=0.7, max_duration=600):
    """Explicit long-context export used only by this worker; legacy limits stay intact."""
    from system_tools import require_tool
    values = (start, duration, gain, max_duration)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
            or start < 0 or not 0 < duration <= max_duration <= 600 or not 0 < gain <= 1):
        raise ValueError("Use finite start>=0, duration<=max_duration<=600 and gain in (0,1]")
    subprocess.run([require_tool("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", str(source),
                    "-ss", str(start), "-t", str(duration), "-af", f"volume={gain}",
                    "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(target)], check=True)
    actual = frames(target) / RATE
    if abs(actual - duration) > 1 / RATE:
        raise ValueError(f"Decoded duration mismatch: requested {duration}, got {actual}")
    return actual


def prepare(source, output, start=0, duration=None):
    from provenance import sha256, write_json
    from system_tools import require_tool
    if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
        raise ValueError("Use a finite start>=0")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                 or not math.isfinite(duration) or not 10 <= duration <= 600):
        raise ValueError("Use duration=None for the whole file, or a finite 10–600 seconds")
    original_hash = sha256(source)
    info = json.loads(subprocess.check_output([require_tool("ffprobe"), "-v", "error", "-show_entries",
                     "format=duration", "-of", "json", str(source)], text=True))
    available = float(info["format"]["duration"]) - start
    if not math.isfinite(available) or available < 10 or (duration is None and available > 600.05):
        raise ValueError("Select 10–600 seconds of audio; the full file may be at most 10 minutes")
    if duration is not None and available + 0.05 < duration:
        raise ValueError("Source is shorter than the requested excerpt")
    output.mkdir(parents=True, exist_ok=False)
    clean = output / "original.wav"
    command = [require_tool("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", str(source), "-ss", str(start)]
    if duration is not None: command += ["-t", str(duration)]
    command += ["-af", "volume=0.7", "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(clean)]
    subprocess.run(command, check=True)
    valid = frames(clean)
    if not 10 * RATE <= valid <= 600 * RATE:
        raise ValueError("Decoded audio must be 10–600 seconds")
    if duration is not None and abs(valid / RATE - duration) > 0.05:
        raise ValueError("Decoded excerpt is shorter than requested")
    model_seconds = (valid + RATE - 1) // RATE
    padding = model_seconds * RATE - valid
    source_path = output / "source.wav"
    with wave.open(str(clean), "rb") as original, wave.open(str(source_path), "wb") as padded:
        padded.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        while block := original.readframes(RATE): padded.writeframesraw(block)
        if padding: padded.writeframesraw(b"\0" * (padding * 4))
    conditioning = {"schemaVersion": 1, "request": {"duration": model_seconds, "start": start},
                    "beats": [], "downbeats": [], "chords": [],
                    "warnings": ["Audio-only conditioning; no invented timing/key/chord metadata."]}
    write_json(output / "conditioning.json", conditioning)
    write_json(output / "prepared.json", {"schema_version": 1, "duration": model_seconds,
        "conditions": {"A": "source.wav"}, "condition_sha256": {"A": sha256(source_path)},
        "conditioning_sha256": sha256(output / "conditioning.json")})
    metadata = {"original_sha256": original_hash, "clean_sha256": sha256(clean),
                "source_sha256": sha256(source_path), "sample_rate": RATE, "valid_frames": valid,
                "duration": valid / RATE, "model_duration": model_seconds, "end_padding_frames": padding,
                "start": start, "requested_duration": duration, "source_gain": 0.7}
    write_json(output / "input.json", metadata)
    return metadata


def execute_worker(root, output, *, preset=None, instrument="bass", reference_audio=None, engine=None):
    """Reuse the inspected engine directly, with process-local caption/FP32/export overrides."""
    preset = PRESET if preset is None else preset
    if instrument not in ("bass", "drums", "guitar") or preset.get("instrument") != instrument:
        raise ValueError("Worker instrument must match the requested preset")
    attach(root)
    from provenance import check_upstream, write_json
    from runtime_environment import runtime_environment
    environment = runtime_environment(root)
    for name in ("HF_HUB_CACHE", "HF_XET_CACHE", "MPLBACKEND", "MPLCONFIGDIR"):
        os.environ[name] = environment[name]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    check_upstream(root / "vendor/ACE-Step-1.5", "ace")
    sys.path.insert(2, str(root / "vendor/ACE-Step-1.5"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in model runtime; no CPU/MPS fallback")
    import acestep.handler as ace_handler
    import backends
    if engine is None:
        import run as engine

    class Float32Handler(ace_handler.AceStepHandler):
        def _load_main_model_from_checkpoint(self, *args, **kwargs):
            if kwargs.get("device") != "cuda": raise RuntimeError("Single-pass preset requires CUDA")
            self.dtype = torch.float32
            print("[Precision] Single-pass DiT/text/VAE=float32", flush=True)
            return super()._load_main_model_from_checkpoint(*args, **kwargs)
        def _get_vae_dtype(self, device=None): return torch.float32
    ace_handler.AceStepHandler = Float32Handler
    original_backend = backends.AceBackend
    class ElectricBackend(original_backend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_params = self.params_type
            def params(*args, **kwargs):
                kwargs["caption"] = preset["caption"]
                if reference_audio is not None:
                    kwargs["reference_audio"] = str(reference_audio)
                return original_params(*args, **kwargs)
            self.params_type = params
    backends.AceBackend = ElectricBackend
    # Calling execute() avoids argparse's diagnostic-only 10/30-second limit.
    engine.canonicalize = canonicalize_long
    prepared = output / "prepared"
    generation = output / "generation"
    generation.mkdir(exist_ok=False)
    experiment = engine.plan(prepared, "ace", instrument, [0])
    experiment.update(max_job_seconds=600, device_requested="cuda", ace_model=preset["model"],
                      offload_to_cpu=True, guidance_scale=7, mode="single_pass")
    write_json(generation / "experiment.json", experiment)
    args = SimpleNamespace(upstream=root / "vendor/ACE-Step-1.5", backend="ace", checkpoint=None,
        ace_model=preset["model"], assets_manifest=root / "configs/assets-base.json", device="cuda",
        offload=True, out=generation, prepared=prepared, max_job_seconds=600,
        guidance_scale=7.0, capture_decoder=output / "capture", bridge_root=root)
    try:
        engine.execute(args, experiment)
    except BaseException as error:
        experiment.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(generation / "experiment.json", experiment)


def export_audio(source, target, count, gain=1):
    from system_tools import require_tool
    subprocess.run([require_tool("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", str(source),
        "-af", f"atrim=end_sample={count},volume={gain}", "-ar", str(RATE), "-ac", "2",
        "-c:a", "pcm_s16le", str(target)], check=True)
    if frames(target) != count:
        raise ValueError("Output did not preserve the exact original frame count")


def package(output):
    from system_tools import require_tool
    for source, target in (("electric_listen.wav", "electric_preview.mp3"), ("mix_full.wav", "mix_preview.mp3")):
        subprocess.run([require_tool("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", str(output / source),
                        "-c:a", "libmp3lame", "-b:a", "192k", str(output / target)], check=True)
    reports = report_files()
    for name, audio in (("full_song_result.zip", ("electric_bass.wav", "electric_listen.wav", "mix_full.wav")),
                        ("full_song_feedback.zip", ("electric_preview.mp3", "mix_preview.mp3"))):
        with zipfile.ZipFile(output / name, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in (*reports, *audio): archive.write(output / relative, relative)


def report_files():
    return ("single_pass.json", "prepared/input.json", "generation/experiment.json",
            "generation/session.json", "generation/000_A_bass_seed0/result.json", "capture/capture.json")


def failed_feedback(output):
    """Keep a small downloadable report after OOM/failure, without masking its cause."""
    path = output / "full_song_feedback.zip"
    if not path.exists():
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in report_files():
                if (output / name).is_file(): archive.write(output / name, name)


def generate(root, source, output, *, start=0, duration=None, execute=False):
    root, source, output = map(lambda p: Path(p).resolve(), (root, source, output))
    attach(root)
    from provenance import COMMITS, sha256, write_json
    from runtime_environment import runtime_environment
    from stream_process import run_logged
    from run import cached_job
    if not source.is_file(): raise FileNotFoundError(f"Source not found: {source}")
    preset_file = root / "configs/electric-bass.json"
    if preset_file.is_file() and json.loads(preset_file.read_text()) != PRESET:
        raise ValueError("Electric Bass preset differs from the selected version")
    for required in ("configs/assets-base.json", "tools/stem_eval/run.py", "scripts/setup_runtime.py"):
        if not (root / required).is_file(): raise ValueError("ROOT must be the existing bridge source directory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preparing", "mode": "single_pass", "preset": copy.deepcopy(PRESET),
              "upstream_commit": COMMITS["ace"], "assets_manifest_sha256": sha256(root / "configs/assets-base.json"),
              "model_calls_requested": 1, "note": "One continuous DiT context, no independent song chunks/stitching. VAE may tile internally. Hardware fit and musical consistency still need actual testing."}
    write_json(output / "single_pass.json", report)
    try:
        metadata = prepare(source, output / "prepared", start, duration)
        report.update(status="planned", input=metadata)
        write_json(output / "single_pass.json", report)
        if not execute:
            print(f"PLAN: one {metadata['model_duration']}s context; no GPU inference", flush=True)
            return report
        print(f"ONE PASS: {metadata['duration']:.3f}s audio, {metadata['model_duration']}s model request; no song stitching", flush=True)
        command = [str(model_python(root)), str(Path(__file__).resolve()), "--worker", "--root", str(root), "--out", str(output)]
        started = time.perf_counter()
        run_logged(command, env=runtime_environment(root), check=True)
        report["worker_seconds"] = time.perf_counter() - started
        experiment = json.loads((output / "generation/experiment.json").read_text())
        if experiment.get("status") != "generated" or len(experiment["jobs"]) != 1:
            raise ValueError("Expected one completed single-pass model invocation")
        job = experiment["jobs"][0]
        folder = output / "generation/000_A_bass_seed0"
        result = cached_job(folder, job, None)
        if result is None or result.get("device") != "cuda" or job["source_sha256"] != metadata["source_sha256"]:
            raise ValueError("Single-pass result device/input verification failed")
        params, config = result["settings"]["params"], result["settings"]["config"]
        expected = {"caption": PRESET["caption"], "seed": 0, "guidance_scale": 7, "inference_steps": 50,
                    "duration": metadata["model_duration"], "task_type": "lego", "instrumental": True,
                    "enable_normalization": False, "thinking": False, "use_cot_caption": False,
                    "use_cot_metas": False, "use_cot_language": False}
        if any(params.get(key) != value for key, value in expected.items()):
            raise ValueError("Actual long-context parameters differ from the selected preset/request")
        if config.get("seeds") != [0] or config.get("batch_size") != 1 or config.get("use_random_seed") is not False:
            raise ValueError("Unexpected generation seed/batch settings")
        capture = json.loads((output / "capture/capture.json").read_text())
        calls = [call for call in capture.get("calls", []) if call.get("backend") == "pytorch_vae"]
        if (capture.get("status") != "captured" or capture.get("model_device") != "cuda"
            or capture.get("model_dtype") != "torch.float32" or not calls
            or any(c.get("dtype") != "torch.float32" or c.get("status") != "returned" for c in calls)):
            raise ValueError("Expected completed FP32 model/decoder capture")
        if (sha256(source) != metadata["original_sha256"]
            or sha256(output / "prepared/source.wav") != metadata["source_sha256"]
            or sha256(output / "prepared/original.wav") != metadata["clean_sha256"]):
            raise ValueError("Input changed during single-pass generation")
        export_audio(folder / result["stem_file"], output / "electric_bass.wav", metadata["valid_frames"])
        export_audio(output / "electric_bass.wav", output / "electric_listen.wav", metadata["valid_frames"], gain=0.7)
        export_audio(folder / "mix.wav", output / "mix_full.wav", metadata["valid_frames"])
        report.update(status="generated", settings=result["settings"], decoder_capture=capture,
                      inference_seconds=result.get("inference_seconds"),
                      peak_cuda_allocated_bytes=result.get("peak_cuda_allocated_bytes"),
                      peak_cuda_reserved_bytes=result.get("peak_cuda_reserved_bytes"),
                      stem_sha256=sha256(output / "electric_bass.wav"), mix_sha256=sha256(output / "mix_full.wav"),
                      listening_gain=0.7, mix_track_gain=0.5, export_frames=metadata["valid_frames"])
        write_json(output / "single_pass.json", report)
        package(output)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        failure = output / "generation/experiment.json"
        if failure.is_file(): report["worker_report"] = json.loads(failure.read_text())
        write_json(output / "single_pass.json", report)
        try: failed_feedback(output)
        except Exception: pass  # Retain the original inference/preparation exception.
        raise
    finally:
        write_json(output / "single_pass.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, help="Omit for the whole source, maximum 600s")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        execute_worker(args.root.resolve(), args.out.resolve())
    else:
        if args.source is None: parser.error("--source is required")
        report = generate(args.root, args.source, args.out, start=args.start, duration=args.duration, execute=args.execute)
        print("Status:", report["status"], "| Output:", args.out)


if __name__ == "__main__": main()

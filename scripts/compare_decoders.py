"""One cached B input, one diffusion pass, then default and CPU decoding of identical latents."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, canonicalize
from provenance import COMMITS, check_upstream, sha256, write_json
from run import cached_job
from audit_audio import latest_replay
from context_comparison import create_input
from mac_workflow import check_model_runtime, run_command
from runtime_environment import runtime_environment
from setup_runtime import runtime_python
from stream_process import run_logged


def prepare(run, output):
    """Snapshot only B with the original seed/settings; never modify the old experiment."""
    run, output = Path(run).resolve(), Path(output).resolve()
    old = json.loads((run / "comparison.json").read_text())
    job = old["jobs"][0]
    profile = old["profile"]
    if old.get("status") != "generated" or job.get("original_chunk_index") != 1 or job["duration"] != 30:
        raise ValueError("Expected completed notebook 07 replay, with 30-second B first")
    if profile["model"] != "acestep-v15-base" or profile["seeds"] != [job["seed"]] or profile["instrument"] != job["instrument"]:
        raise ValueError("Cached job/profile differ")
    folder = run / f'000_A_{job["instrument"]}_seed{job["seed"]}'
    result = cached_job(folder, job, old.get("execution_signature"))
    source = Path(job["source"]).resolve()
    if result is None or not source.is_relative_to(run) or sha256(source) != job["source_sha256"]:
        raise ValueError("Cached B source/result changed")
    output.mkdir(parents=True, exist_ok=False)
    prepared = create_input(output / "prepared", source, 30, "Exact B input for same-latent decoder comparison", copy_exact=True)
    if sha256(prepared / "source.wav") != job["source_sha256"]:
        raise ValueError("B snapshot differs")
    shutil.copyfile(folder / result["stem_file"], output / "old_B_stem.wav")
    report = {"status": "planned", "profile": profile, "original_run": str(run),
              "source_sha256": job["source_sha256"], "original_result": result,
              "upstream_commit": COMMITS["ace"], "assets_sha256": sha256(ROOT / "configs/assets-base.json"),
              "note": "One new B generation; identical saved latents used for CPU decode. CPU tiling also differs; no claim of a fix."}
    write_json(output / "comparison.json", report)
    return report


def decode_cpu(output):
    """Load only the pinned VAE in a fresh process and decode the captured latents on CPU."""
    os.environ.update({key:value for key,value in runtime_environment().items() if key in ("MPLBACKEND","MPLCONFIGDIR","HF_HUB_CACHE","HF_XET_CACHE")})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    import soundfile as sf
    from diffusers.models import AutoencoderOobleck
    from model_assets import verify
    from vae_reference import decode_tiled

    report = json.loads((output / "comparison.json").read_text())
    capture = json.loads((output / "capture/capture.json").read_text())
    if report["assets_sha256"] != sha256(ROOT / "configs/assets-base.json") or capture["status"] != "captured":
        raise ValueError("Manifest changed or default decode was not captured")
    if any((output / name).exists() for name in ("cpu_float.wav", "default_decoder_listen.wav", "cpu_decoder_listen.wav")):
        raise FileExistsError("CPU comparison already exists; preserve it and use a new experiment")
    check_upstream(ROOT / "vendor/ACE-Step-1.5", "ace")
    latent_file = output / "capture/latents.npy"
    default_file = output / "capture/default_float.wav"
    if sha256(latent_file) != capture["latent_sha256"] or sha256(default_file) != capture["default_float_sha256"]:
        raise ValueError("Captured decoder inputs/output changed")
    manifest = json.loads((ROOT / "configs/assets-base.json").read_text())
    for record in manifest["files"]:
        if record["destination"].startswith("checkpoints/vae/"):
            verify(ROOT / "vendor/ACE-Step-1.5" / record["destination"], record)
    vae_path = ROOT / "vendor/ACE-Step-1.5/checkpoints/vae"
    print("Memuat VAE saja di CPU float32; diffusion tidak dijalankan lagi", flush=True)
    vae = AutoencoderOobleck.from_pretrained(str(vae_path), local_files_only=True).to(device="cpu",dtype=torch.float32).eval()
    saved = np.load(latent_file, allow_pickle=False)
    if saved.ndim != 3 or saved.shape[0] != 1 or saved.shape[2] != vae.config.decoder_input_channels:
        raise ValueError("Captured [B,T,C] latent shape differs from VAE")
    latents = np.ascontiguousarray(saved.transpose(0,2,1),dtype=np.float32)
    started = time.perf_counter()
    window_count = math.ceil(latents.shape[-1]/64)
    decoded_windows = 0
    def decode(window):
        nonlocal decoded_windows
        decoded_windows += 1
        print(f"CPU VAE: window {decoded_windows}/{window_count}",flush=True)
        with torch.inference_mode():
            return vae.decode(torch.from_numpy(window.copy())).sample.detach().cpu().float().numpy()
    waveform = decode_tiled(latents, decode, chunk_frames=128, overlap=32)
    decode_seconds = time.perf_counter() - started
    if waveform.shape[1] != 2 or waveform.shape[-1] != capture["waveform_frames"]:
        raise ValueError("CPU/default decoded durations or channels differ; refusing to pad")
    peak = float(np.max(np.abs(waveform)))
    # Match upstream's mandatory post-decode peak guard; no loudness normalization.
    waveform /= max(1.0, peak)
    cpu_file = output / "cpu_float.wav"
    sf.write(str(cpu_file), waveform[0].T, capture["sample_rate"], format="WAV", subtype="FLOAT")
    listening_duration = min(30*RATE, math.floor(capture["waveform_frames"] * RATE / capture["sample_rate"])) / RATE
    for name, path in (("default_decoder",default_file),("cpu_decoder",cpu_file)):
        canonicalize(path, output / (name + "_listen.wav"), duration=listening_duration, gain=0.7, max_duration=30)
    report.update(status="decoded", capture=capture, same_latent_sha256=sha256(latent_file),
                  cpu={"device":"cpu","dtype":"torch.float32","chunk_frames":128,"overlap_frames":32,
                       "decode_seconds":decode_seconds,"peak_before_guard":peak,
                       "output_sha256":sha256(cpu_file),"torch":torch.__version__},
                  listening_gain_both=0.7, listening_duration=listening_duration)
    write_json(output / "comparison.json",report)
    print("Bandingkan:",output / "default_decoder_listen.wav",flush=True)
    print("Dengan:",output / "cpu_decoder_listen.wav",flush=True)


def main():
    """Plan by default; --execute runs one generation then a separate VAE-only CPU process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--decode-cpu",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.decode_cpu is not None:
        decode_cpu(args.decode_cpu.resolve())
        return
    run = args.run if args.run is not None else latest_replay()
    output = args.out.resolve() if args.out is not None else ROOT / "runs" / ("decoder_comparison_" + uuid.uuid4().hex[:12])
    report = prepare(run,output)
    print("Input B identik:",report["source_sha256"],flush=True)
    print("Folder pembanding:",output,flush=True)
    if not args.execute:
        print("PLAN ONLY. --execute: satu generasi B + decoder CPU dari latent yang sama, tanpa unduhan.",flush=True)
        return
    try:
        print(json.dumps(check_model_runtime(runtime_python()),indent=2),flush=True)
        command = run_command(ROOT,report["profile"],output / "prepared",output / "default_result",execute=True)
        command += ["--max-job-seconds","30","--capture-decoder",str(output / "capture")]
        run_logged(command,env=runtime_environment(),check=True)
        # The diffusion process exits first, releasing its model memory before CPU VAE loading.
        run_logged([str(runtime_python()),str(Path(__file__).resolve()),"--decode-cpu",str(output)],env=runtime_environment(),check=True)
    except BaseException as error:
        latest = json.loads((output / "comparison.json").read_text())
        latest.update(status="failed",error=f"{type(error).__name__}: {error}")
        write_json(output / "comparison.json",latest)
        raise


if __name__ == "__main__":
    main()

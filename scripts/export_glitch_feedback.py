"""Export ten aligned seconds of cached source/default/CPU audio for glitch review, without inference."""
import argparse
import json
import math
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, canonicalize
from provenance import sha256, write_json


PARAMETERS = ("task_type", "instruction", "caption", "instrumental", "duration", "inference_steps",
              "guidance_scale", "seed", "thinking", "use_cot_caption", "use_cot_metas", "use_cot_language")


def latest_comparison():
    """Choose a completed same-latent decoder comparison, never trigger another generation."""
    candidates = sorted((ROOT / "runs").glob("decoder_comparison_*/comparison.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if json.loads(path.read_text()).get("status") == "decoded":
            return path.parent
    raise ValueError("No completed decoder comparison found. Select --run /path/to/decoder_comparison_xxx")


def export_feedback(run, output, start=3, duration=10):
    """Verify existing audio and export only the selected interval with the same fixed gain."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if (any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            for value in (start,duration)) or start < 0 or not 0 < duration <= 10):
        raise ValueError("Use finite start >= 0 and duration in (0,10]")
    report = json.loads((run / "comparison.json").read_text())
    capture = json.loads((run / "capture/capture.json").read_text())
    prepared = json.loads((run / "prepared/prepared.json").read_text())
    if report.get("status") != "decoded" or capture.get("status") != "captured":
        raise ValueError("Need completed default and CPU decoder outputs")
    if start + duration > min(prepared["duration"], capture["waveform_frames"] / capture["sample_rate"]):
        raise ValueError("Selected interval exceeds the common audio duration")
    files = {"source.wav": (run / "prepared/source.wav",report["source_sha256"]),
             "default_decoder.wav": (run / "capture/default_float.wav",capture["default_float_sha256"]),
             "cpu_decoder.wav": (run / "cpu_float.wav",report["cpu"]["output_sha256"])}
    for path, expected in files.values():
        if sha256(path) != expected:
            raise ValueError("Cached audio changed; refusing to export a misleading comparison")
    if report["same_latent_sha256"] != capture["latent_sha256"]:
        raise ValueError("CPU and default comparison do not identify the same latent")
    output.mkdir(parents=True,exist_ok=False)
    exported = []
    for name,(path,_) in files.items():
        canonicalize(path,output / name,start=start,duration=duration,gain=0.7)
        exported.append({"file":name,"sha256":sha256(output / name)})
    original = report.get("original_result",{})
    frame = original.get("start_frame")
    song_start = frame / RATE + start if isinstance(frame,(int,float)) else None
    settings = original.get("settings",{})
    metadata = {"schema_version":1,"start_in_B_seconds":start,"duration_seconds":duration,
                "song_start_seconds":song_start,"gain_all_files":0.7,"sample_rate":RATE,
                "files":exported,"same_latent_sha256":report["same_latent_sha256"],
                "capture":{key:capture.get(key) for key in ("status","calls","model_device","model_dtype",
                    "mlx_vae_selected","mlx_vae_dtype","latent_shape","latent_std","latent_min","latent_max")},
                "params":{key:settings.get("params",{}).get(key) for key in PARAMETERS},
                "cpu":{key:report["cpu"].get(key) for key in ("device","dtype","chunk_frames","overlap_frames","torch")},
                "note":"Cached source and two decoder outputs, same interval/gain. No diffusion or training. This is listening evidence, not an automatic diagnosis."}
    write_json(output / "manifest.json",metadata)
    archive = output / "feedback.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (*files,"manifest.json"):
            bundle.write(output / name,arcname=name)
    return archive


def main():
    """Package cached audio only; upload is left to the user."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--start",type=float,default=3,help="Offset within B, not the whole song")
    parser.add_argument("--duration",type=float,default=10)
    args = parser.parse_args()
    run = args.run if args.run is not None else latest_comparison()
    output = args.out if args.out is not None else run / ("feedback_" + uuid.uuid4().hex[:8])
    archive = export_feedback(run,output,args.start,args.duration)
    print("Paket audio:",archive)
    print("Isi: potongan sumber, decoder default, decoder CPU, dan manifest.")
    print("Tidak ada inferensi, unduhan, atau unggahan otomatis. Hasil lama dipertahankan.")


if __name__ == "__main__":
    main()

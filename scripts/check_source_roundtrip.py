"""Test piano -> CPU/MLX source encoders -> common CPU decoder, without loading DiT."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "tools/stem_eval"))
from audio import canonicalize
from provenance import check_upstream,sha256,write_json
from export_glitch_feedback import latest_comparison
from runtime_environment import runtime_environment
from setup_runtime import runtime_python
from stream_process import run_logged


def prepare(run,output):
    """Take the same B interval used in the feedback bundle, without adding gain."""
    report=json.loads((run / "comparison.json").read_text())
    source=run / "prepared/source.wav"
    if report.get("status")!="decoded" or sha256(source)!=report["source_sha256"]:
        raise ValueError("Need an unchanged completed decoder comparison")
    output.mkdir(parents=True,exist_ok=False)
    canonicalize(source,output / "source.wav",start=3,duration=10,gain=1)
    metadata={"status":"prepared","source_sha256":sha256(output / "source.wav"),
              "duration":10,"start_in_B_seconds":3,
              "assets_sha256":sha256(ROOT / "configs/assets-base.json"),
              "note":"Source reconstruction only, not bass generation. CPU and MLX random samplers use different RNG streams."}
    write_json(output / "report.json",metadata)
    return metadata


def reconstruct(vae,latents):
    """Use the same CPU float32 tiled decoder for either source encoder."""
    import numpy as np
    import torch
    from vae_reference import decode_tiled
    if latents.ndim!=3 or latents.shape[0]!=1 or not np.isfinite(latents).all():
        raise ValueError("Source encoder returned invalid/nonfinite latents")
    def decode(window):
        with torch.inference_mode():
            return vae.decode(torch.from_numpy(window.copy())).sample.detach().cpu().float().numpy()
    waveform=decode_tiled(np.ascontiguousarray(latents,dtype=np.float32),decode)
    if waveform.shape[1]!=2:
        raise ValueError("Expected stereo source reconstruction")
    return waveform


def worker(output):
    """Load only VAE weights; compare source reconstructions with one common decoder."""
    os.environ.update({k:v for k,v in runtime_environment().items() if k in ("MPLBACKEND","MPLCONFIGDIR","HF_HUB_CACHE","HF_XET_CACHE")})
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    import numpy as np
    import torch
    import torchaudio
    import soundfile as sf
    from diffusers.models import AutoencoderOobleck
    from model_assets import verify

    report=json.loads((output / "report.json").read_text())
    if report.get("status")!="prepared" or sha256(output / "source.wav")!=report["source_sha256"]:
        raise ValueError("Source snapshot changed or this output was already used")
    if sha256(ROOT / "configs/assets-base.json")!=report["assets_sha256"]:
        raise ValueError("Model manifest changed")
    upstream=ROOT / "vendor/ACE-Step-1.5"
    check_upstream(upstream,"ace")
    for record in json.loads((ROOT / "configs/assets-base.json").read_text())["files"]:
        if record["destination"].startswith("checkpoints/vae/"):
            verify(upstream / record["destination"],record)
    print("Memuat VAE saja: CPU float32, tanpa DiT/LM/diffusion",flush=True)
    vae=AutoencoderOobleck.from_pretrained(str(upstream / "checkpoints/vae"),local_files_only=True).to(device="cpu",dtype=torch.float32).eval()
    data,rate=sf.read(str(output / "source.wav"),dtype="float32",always_2d=True)
    original=torch.from_numpy(np.ascontiguousarray(data.T))
    audio=torchaudio.transforms.Resample(rate,48000)(original).unsqueeze(0)
    reconstructions={}
    try:
        print("Encoder CPU -> latent -> decoder CPU",flush=True)
        torch.manual_seed(0)
        with torch.inference_mode():
            posterior=vae.encode(audio).latent_dist
            latent=posterior.sample().detach().cpu().float().numpy()
        reconstructions["cpu_encoder"]=reconstruct(vae,latent)
        report["cpu_encoder"]={"latent_shape":list(latent.shape),"latent_std":float(latent.std(dtype=np.float64)),
                               "device":"cpu","dtype":"float32"}
        sf.write(str(output / "cpu_encoder_float.wav"),reconstructions["cpu_encoder"][0].T,48000,subtype="FLOAT",format="WAV")
        write_json(output / "report.json",report)
        print("Encoder MLX upstream -> latent -> decoder CPU",flush=True)
        sys.path.insert(0,str(upstream))
        from acestep.handler import AceStepHandler
        import mlx.core as mx
        handler=AceStepHandler()
        handler.vae=vae
        handler.device="cpu"
        handler.dtype=torch.float32
        handler.disable_tqdm=False
        if not handler._init_mlx_vae():
            raise RuntimeError("Native MLX VAE unavailable; CPU roundtrip retained, no silent encoder fallback")
        mx.random.seed(0)
        # Call the actual compiled/native MLX encoder used by the original pipeline.
        with torch.inference_mode():
            latent=handler._mlx_vae_encode_sample(audio).detach().cpu().float().numpy()
        reconstructions["mlx_encoder"]=reconstruct(vae,latent)
        report["mlx_encoder"]={"latent_shape":list(latent.shape),"latent_std":float(latent.std(dtype=np.float64)),
                               "dtype":str(handler._mlx_vae_dtype),"native_upstream":True}
        sf.write(str(output / "mlx_encoder_float.wav"),reconstructions["mlx_encoder"][0].T,48000,subtype="FLOAT",format="WAV")
        peak=max(float(np.max(np.abs(data))),*(float(np.max(np.abs(x))) for x in reconstructions.values()))
        gain=min(0.7,0.7/peak) if peak else 0.7
        duration=min(10,min(w.shape[-1]/48000 for w in reconstructions.values()))
        # Use a shared duration and gain; never pad a reconstruction to appear aligned.
        duration=int(duration*32000)/32000
        canonicalize(output / "source.wav",output / "source_listen.wav",duration=duration,gain=gain)
        for name in reconstructions:
            canonicalize(output / (name+"_float.wav"),output / (name+"_listen.wav"),duration=duration,gain=gain)
        report.update(status="completed",shared_listening_gain=gain,listening_duration=duration,torch=torch.__version__)
        print("Dengarkan: source_listen.wav, cpu_encoder_listen.wav, mlx_encoder_listen.wav",flush=True)
        print("Folder:",output,flush=True)
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "report.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker is not None:
        worker(args.worker.resolve())
        return
    run=args.run.resolve() if args.run else latest_comparison()
    output=args.out.resolve() if args.out else ROOT / "runs" / ("source_roundtrip_"+uuid.uuid4().hex[:12])
    prepare(run,output)
    print("Folder:",output,flush=True)
    if args.execute:
        run_logged([str(runtime_python()),str(Path(__file__).resolve()),"--worker",str(output)],env=runtime_environment(),check=True)
    else:
        print("PLAN ONLY: source snapshot siap; --execute menjalankan VAE saja",flush=True)


if __name__=="__main__":main()

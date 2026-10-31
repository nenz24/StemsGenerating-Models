"""Opt-in capture of the exact pre-VAE latents and observed decoder calls for one job."""
from functools import wraps
from pathlib import Path
import numpy as np

from provenance import sha256, write_json


def as_numpy(tensor):
    """Copy a tensor to CPU float32 without changing the model's tensor."""
    return tensor.detach().cpu().float().numpy().copy()


def install_capture(handler, directory):
    """Wrap one handler's decode boundary; do not change its selected backend or parameters."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = handler._decode_generate_music_pred_latents
    report = {"status": "waiting", "calls": [], "model_device": str(handler.device),
              "model_dtype": str(handler.dtype), "mlx_vae_selected": bool(handler.use_mlx_vae and handler.mlx_vae is not None),
              "mlx_vae_dtype": str(getattr(handler, "_mlx_vae_dtype", None))}

    def observe(name, function):
        @wraps(function)
        def traced(*args, **kwargs):
            event = {"backend": name, "status": "running"}
            if name == "pytorch_vae":
                parameter = next(handler.vae.parameters())
                event.update(device=str(parameter.device), dtype=str(parameter.dtype))
            report["calls"].append(event)
            try:
                result = function(*args, **kwargs)
                event["status"] = "returned"
                return result
            except Exception as error:
                event.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
        return traced

    handler.vae.decode = observe("pytorch_vae", handler.vae.decode)
    handler._mlx_vae_decode = observe("mlx_vae", handler._mlx_vae_decode)

    @wraps(original)
    def captured(pred_latents, *args, **kwargs):
        if (directory / "latents.npy").exists():
            raise ValueError("Decoder capture is limited to one generation")
        latents = as_numpy(pred_latents)
        if latents.ndim != 3 or latents.shape[0] != 1 or not np.isfinite(latents).all():
            raise ValueError("Expected finite latents shaped [1, frames, channels]")
        np.save(directory / "latents.npy", latents, allow_pickle=False)
        report.update(status="decoding", latent_shape=list(latents.shape),
                      latent_sha256=sha256(directory / "latents.npy"),
                      latent_min=float(latents.min()), latent_max=float(latents.max()),
                      latent_std=float(latents.std(dtype=np.float64)))
        write_json(directory / "capture.json", report)
        try:
            result = original(pred_latents, *args, **kwargs)
            waveform = as_numpy(result[0])
            if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] != 2 or not np.isfinite(waveform).all():
                raise ValueError("Default decoder did not return finite stereo audio")
            # Save the decoder's float output before the standard export/PCM16 path.
            import soundfile as sf
            sf.write(str(directory / "default_float.wav"), waveform[0].T,
                     handler.sample_rate, format="WAV", subtype="FLOAT")
            report.update(status="captured", sample_rate=handler.sample_rate,
                          waveform_frames=waveform.shape[-1],
                          default_float_sha256=sha256(directory / "default_float.wav"))
            return result
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(directory / "capture.json", report)

    handler._decode_generate_music_pred_latents = captured

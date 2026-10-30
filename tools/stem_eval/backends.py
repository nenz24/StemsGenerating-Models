"""Adapters against the inspected upstream APIs. Real inference needs GPU and weights."""
from pathlib import Path
import sys

from provenance import sha256


def weight_inventory(upstream, backend, checkpoint, model_name="acestep-v15-base"):
    root = upstream / ("checkpoints" if backend == "ace" else "weights")
    if backend == "ace":
        # Upstream's main-model check also expects Turbo and LM folders, even with LM disabled.
        for name in (model_name, "acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"):
            if not (root / name).is_dir():
                raise ValueError(f"Missing local component {name}; runner does not download weights")
        suffixes = {".safetensors", ".bin", ".pt", ".json", ".model", ".txt"}
        files = sorted(p for p in root.rglob("*")
                       if p.is_file() and ".cache" not in p.parts and p.suffix in suffixes)
        for name in (model_name, "vae", "Qwen3-Embedding-0.6B"):
            if not any(p.suffix in {".safetensors", ".bin", ".pt"} for p in (root / name).rglob("*")):
                raise ValueError(f"No local weights in component {name}")
    else:
        if checkpoint is None or checkpoint.suffix != ".safetensors":
            raise ValueError("Supply the official STAGE .safetensors checkpoint with --checkpoint")
        files = [root / "encodec_32khz.pt", root / "lm-small-weights.pt", checkpoint]
    if not files or any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise ValueError("Required model artifacts are missing or empty")
    if backend == "ace" and not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("No ACE-Step safetensors weights found")
    return [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files]


class AceBackend:
    def __init__(self, upstream, checkpoint=None, model_name="acestep-v15-base", offload_to_cpu=False):
        sys.path.insert(0, str(upstream))
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        self.params_type, self.config_type, self.generate_music = GenerationParams, GenerationConfig, generate_music
        self.handler = AceStepHandler()
        message, success = self.handler.initialize_service(
            project_root=str(upstream), config_path=model_name, device="cuda",
            offload_to_cpu=offload_to_cpu,
            use_flash_attention=False, compile_model=False, use_mlx_dit=False,
        )
        if not success:
            raise RuntimeError(message)

    def generate(self, source, duration, instrument, seed, output):
        params = self.params_type(
            task_type="lego", src_audio=str(source),
            instruction=f"Generate the {instrument} track based on the audio context:",
            caption=f"Isolated {instrument} accompaniment following the source harmony, timing and dynamics.",
            instrumental=True, lyrics="[Instrumental]", duration=duration,
            repainting_start=0.0, repainting_end=-1,
            inference_steps=50, guidance_scale=7.0, seed=seed,
            thinking=False, use_cot_caption=False, use_cot_metas=False, use_cot_language=False,
            enable_normalization=False,
        )
        config = self.config_type(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        result = self.generate_music(self.handler, None, params, config, save_dir=str(output))
        if not result.success or len(result.audios) != 1:
            raise RuntimeError(result.error or result.status_message or "No single audio result")
        path = Path(result.audios[0]["path"])
        if not path.is_file():
            raise RuntimeError("ACE-Step reported success without an audio file")
        return path, {"params": params.to_dict(), "config": config.to_dict()}


class StageBackend:
    def __init__(self, upstream, checkpoint):
        sys.path.insert(0, str(upstream / "src"))
        from stage.loader import load_model
        self.model = load_model(checkpoint, device="cuda")

    def generate(self, source, duration, instrument, seed, output):
        import lightning as lightning
        from stage.utils.audio import load_audio, save_audio
        lightning.seed_everything(seed, workers=True)
        context = load_audio(source).to(self.model.device)
        generated = self.model.generate(
            n_samples=1, gen_seconds=duration, prompt=None, context=context,
            style=None, beat=None, description=[None], prog_bar=True,
        )
        path = output / "stage.wav"
        save_audio(generated, path)
        return path, {"gen_seconds": duration, "seed": seed, "description": None,
                      "instrument_selected_by_checkpoint": instrument}

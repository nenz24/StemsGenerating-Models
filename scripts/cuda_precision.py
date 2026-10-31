"""Process-local FP32 policy for the pinned ACE handler on pre-Ampere CUDA."""

# The pinned revision mentions ACESTEP_DTYPE in an error hint but never reads it.
# Install a handler subclass before run.py imports AceBackend. No upstream files
# or checkpoint hashes change, and the override cannot escape this worker.
FLOAT32_WORKER = r'''
import runpy, sys
from pathlib import Path
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parents[2] / "vendor/ACE-Step-1.5"))
import torch
import acestep.handler as ace_handler

class Float32Handler(ace_handler.AceStepHandler):
    def _load_main_model_from_checkpoint(self, *args, **kwargs):
        if kwargs.get("device") != "cuda":
            raise RuntimeError("FP32 CUDA worker requires device=cuda")
        self.dtype = torch.float32
        print("[Precision] Override pinned CUDA loader: DiT/text/VAE=float32", flush=True)
        return super()._load_main_model_from_checkpoint(*args, **kwargs)

    def _get_vae_dtype(self, device=None):
        return torch.float32

ace_handler.AceStepHandler = Float32Handler
sys.argv = sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
'''


def float32_command(command):
    """Keep the same interpreter, entry point and arguments in a fresh FP32 worker."""
    return [command[0], "-c", FLOAT32_WORKER, *command[1:]]

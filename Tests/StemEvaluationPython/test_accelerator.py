"""Mock backend contracts; these are not real MPS/CUDA inference tests."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/stem_eval"))
from accelerator import memory_stats, reset_memory_stats, select_device, synchronize
from backends import AceBackend
from audio import canonicalize
from system_tools import resolve_tool


class AcceleratorTests(unittest.TestCase):
    def torch(self, cuda=False, mps=False):
        torch = MagicMock()
        torch.cuda.is_available.return_value = cuda
        torch.backends.mps.is_available.return_value = mps
        return torch

    def test_auto_uses_mps_without_cuda_and_rejects_cpu_fallback(self):
        self.assertEqual(select_device(self.torch(mps=True)), "mps")
        self.assertEqual(select_device(self.torch(cuda=True, mps=True)), "cuda")
        with self.assertRaises(RuntimeError):
            select_device(self.torch())
        with self.assertRaises(RuntimeError):
            select_device(self.torch(cuda=True), "mps")

    def test_mps_metrics_never_call_cuda_or_claim_peak_memory(self):
        torch = self.torch(mps=True)
        torch.mps.current_allocated_memory.return_value = 123
        torch.mps.driver_allocated_memory.return_value = 456
        synchronize(torch, "mps")
        reset_memory_stats(torch, "mps")
        stats = memory_stats(torch, "mps")
        self.assertEqual(stats, {"mps_current_allocated_bytes": 123, "mps_driver_allocated_bytes": 456})
        self.assertFalse(torch.cuda.method_calls)
        torch.mps.synchronize.assert_called_once()

    def test_cuda_keeps_peak_metrics(self):
        torch = self.torch(cuda=True)
        synchronize(torch, "cuda")
        reset_memory_stats(torch, "cuda")
        self.assertIn("peak_cuda_allocated_bytes", memory_stats(torch, "cuda"))
        torch.cuda.reset_peak_memory_stats.assert_called_once()
        torch.cuda.synchronize.assert_called_once()

    def test_ace_receives_mps_and_rejects_silent_upstream_fallback(self):
        handler = MagicMock(device="mps")
        handler.initialize_service.return_value = ("ok", True)
        modules = {"acestep": types.ModuleType("acestep"),
                   "acestep.handler": types.SimpleNamespace(AceStepHandler=lambda: handler),
                   "acestep.inference": MagicMock()}
        with patch.dict(sys.modules, modules):
            AceBackend(Path("/upstream"), device="mps")
            self.assertEqual(handler.initialize_service.call_args.kwargs["device"], "mps")
            handler.device = "cpu"
            with self.assertRaises(RuntimeError):
                AceBackend(Path("/upstream"), device="mps")

    def test_missing_ffmpeg_gives_homebrew_action_before_subprocess(self):
        with patch("system_tools.resolve_tool", return_value=None), patch("audio.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "brew install ffmpeg"):
                canonicalize(Path("in.wav"), Path("out.wav"))
            run.assert_not_called()

    def test_homebrew_tool_found_even_if_absent_from_kernel_path(self):
        with patch("system_tools.shutil.which", return_value=None), patch("system_tools.platform.system", return_value="Darwin"), patch("system_tools.Path.is_file", return_value=True):
            self.assertEqual(resolve_tool("ffmpeg"), "/opt/homebrew/bin/ffmpeg")


if __name__ == "__main__":
    unittest.main()

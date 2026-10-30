"""Select a real accelerator and report backend-specific metrics without CUDA assumptions."""


def select_device(torch, requested="auto"):
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if cuda else "mps" if mps else "none"
    if requested == "cuda" and cuda:
        return "cuda"
    if requested == "mps" and mps:
        return "mps"
    raise RuntimeError(
        f"Accelerator {requested!r} tidak tersedia di Python model. "
        "Pada Apple Silicon pasang runtime Python 3.11 dan PyTorch MPS lewat sel instalasi. "
        "Kernel Jupyter boleh tetap memakai Python lain. Tidak beralih diam-diam ke CPU."
    )


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
    else:
        torch.mps.synchronize()


def reset_memory_stats(torch, device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def memory_stats(torch, device):
    if device == "cuda":
        return {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    # PyTorch MPS exposes current, not peak, allocator/driver values here.
    return {"mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()}

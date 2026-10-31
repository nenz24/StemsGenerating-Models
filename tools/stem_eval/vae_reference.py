"""Bounded-window CPU decoder reference with exact overlap-discard sample accounting."""
import numpy as np


def decode_tiled(latents, decode, chunk_frames=128, overlap=32):
    """Decode identical [1,C,T] latents, retaining each core once and discarding context."""
    if latents.ndim != 3 or latents.shape[0] != 1 or latents.shape[-1] < 1 or not np.isfinite(latents).all():
        raise ValueError("Expected finite nonempty [1,C,T] latents")
    if not 0 <= overlap < chunk_frames/2:
        raise ValueError("Overlap must leave a positive decode core")
    stride = chunk_frames - 2*overlap
    parts, hop, channels = [], None, None
    for start in range(0, latents.shape[-1], stride):
        end = min(start + stride, latents.shape[-1])
        left, right = max(0,start-overlap), min(latents.shape[-1],end+overlap)
        decoded = np.asarray(decode(latents[:,:,left:right]), dtype=np.float32)
        if decoded.ndim != 3 or decoded.shape[0] != 1 or not np.isfinite(decoded).all():
            raise ValueError("CPU decoder returned invalid/nonfinite audio")
        ratio, remainder = divmod(decoded.shape[-1], right-left)
        if ratio < 1 or remainder or (hop is not None and (ratio != hop or decoded.shape[1] != channels)):
            raise ValueError("Decoder changed channels or sample-to-latent ratio")
        hop, channels = ratio, decoded.shape[1]
        parts.append(decoded[:,:,(start-left)*hop:(end-left)*hop].copy())
    return np.concatenate(parts,axis=-1)

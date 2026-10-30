"""Canonical audio preparation and clean-source listening mixes; no ML dependencies."""
import array
import math
import subprocess
import sys
import wave

RATE = 32_000


def read_pcm(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 2 or audio.getframerate() != RATE:
            raise ValueError("Expected stereo 32 kHz PCM16 WAV")
        samples = array.array("h", audio.readframes(audio.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def write_pcm(path, samples):
    data = array.array("h", samples)
    if sys.byteorder != "little":
        data.byteswap()
    with wave.open(str(path), "wb") as audio:
        audio.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(data.tobytes())


def canonicalize(source, target, start=0, duration=10, gain=0.7):
    """Fixed gain, never per-condition normalization; ffmpeg errors propagate."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (start, duration, gain)) or start < 0 or not 0 < duration <= 10 or not 0 < gain <= 1:
        raise ValueError("Use a finite start >= 0 and duration in (0, 10]")
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
        "-ss", str(start), "-t", str(duration), "-af", f"volume={gain}",
        "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(target),
    ], check=True)
    actual = len(read_pcm(target)) / (2 * RATE)
    if abs(actual - duration) > 1 / RATE:
        raise ValueError(f"Source is too short: requested {duration}s, decoded {actual}s")
    return actual


def validate_times(values, duration):
    if not isinstance(values, list) or any(
        isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
        or t < 0 or t >= duration for t in values
    ) or values != sorted(set(values)):
        raise ValueError("Beat times must be finite, strictly increasing and clip-relative")
    return values


def add_clicks(source, target, beats, downbeats, accented):
    """Render actual timestamps, without fitting a constant-tempo grid."""
    samples = read_pcm(source)
    frames = len(samples) // 2
    click_length = int(RATE * 0.01)
    for time in beats:
        offset = round(time * RATE)
        is_downbeat = accented and any(abs(time - down) <= 1 / RATE for down in downbeats)
        amplitude = 0.12 if is_downbeat else 0.08
        frequency = 1600 if is_downbeat else 2200
        for frame in range(min(click_length, frames - offset)):
            click = round(32767 * amplitude * math.sin(2 * math.pi * frequency * frame / RATE)
                          * math.exp(-6 * frame / click_length))
            for channel in (0, 1):
                index = 2 * (offset + frame) + channel
                samples[index] = min(32767, max(-32768, samples[index] + click))
    write_pcm(target, samples)


def listening_mix(source, stem, target):
    """Always use clean source; equal 0.5 gains keep two full-scale tracks in range."""
    original, generated = read_pcm(source), read_pcm(stem)
    if len(original) != len(generated):
        raise ValueError("Generated audio duration differs; inspect alignment instead of padding")
    write_pcm(target, (round((a + b) * 0.5) for a, b in zip(original, generated)))

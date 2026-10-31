"""Sample-aligned full-song preparation and overlap stitching for the 10-second POC."""
import array
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, read_pcm, write_pcm
from provenance import sha256, write_json
from system_tools import require_tool


def chunk_starts(frame_count, chunk_frames=10 * RATE, overlap_frames=RATE):
    if frame_count <= 0 or not 0 < overlap_frames < chunk_frames:
        raise ValueError("Need nonempty audio and an overlap smaller than the chunk")
    starts = [0]
    while starts[-1] + chunk_frames < frame_count:
        starts.append(min(starts[-1] + chunk_frames - overlap_frames, frame_count - chunk_frames))
    return starts


def prepare_song(source, output, chunk_seconds=10, overlap_seconds=1):
    source, output = Path(source).resolve(), Path(output)
    if chunk_seconds != 10 or not 0 < overlap_seconds < chunk_seconds:
        raise ValueError("This POC uses 10-second chunks with overlap in (0, 10)")
    source_hash = sha256(source)
    if output.exists():
        manifest = json.loads((output / "song.json").read_text())
        if (manifest["original_sha256"], manifest["chunk_seconds"], manifest["overlap_seconds"]) != (source_hash, chunk_seconds, overlap_seconds):
            raise ValueError("Existing preparation belongs to different audio/settings; use a new directory")
        for record in manifest["files"]:
            if sha256(output / record["path"]) != record["sha256"]:
                raise ValueError("Prepared audio changed; refusing to resume")
        return manifest
    output.mkdir(parents=True)
    try:
        info = json.loads(subprocess.check_output([require_tool("ffprobe"), "-v", "error", "-show_entries",
            "format=duration", "-of", "json", str(source)], text=True))
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or not 0 < duration <= 600.05:
            raise ValueError("Full-song POC supports nonempty recordings up to 10 minutes")
        clean = output / "source.wav"
        subprocess.run([require_tool("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", str(source),
                        "-af", "volume=0.7", "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(clean)], check=True)
        samples = read_pcm(clean)
        frames = len(samples) // 2
        if frames <= 0 or frames > 600 * RATE:
            raise ValueError("Decoded recording is empty or exceeds 10 minutes")
        chunk_frames = round(chunk_seconds * RATE)
        starts = chunk_starts(frames, chunk_frames, round(overlap_seconds * RATE))
        files = [{"path": "source.wav", "sha256": sha256(clean)}]
        chunks = []
        for index, start in enumerate(starts):
            name = f"chunk_{index:03d}.wav"
            segment = samples[2*start:2*min(start + chunk_frames, frames)]
            valid_frames = len(segment) // 2
            if valid_frames < chunk_frames:
                segment.extend(array.array("h", [0]) * (2*(chunk_frames - valid_frames)))
            write_pcm(output / name, segment)
            digest = sha256(output / name)
            files.append({"path": name, "sha256": digest})
            chunks.append({"index": index, "start_frame": start, "valid_frames": valid_frames,
                           "frames": chunk_frames, "file": name, "sha256": digest})
        manifest = {"schema_version": 1, "original_sha256": source_hash, "sample_rate": RATE,
                    "total_frames": frames, "duration": frames / RATE, "chunk_seconds": chunk_seconds,
                    "overlap_seconds": overlap_seconds, "fixed_source_gain": 0.7, "chunks": chunks, "files": files}
        write_json(output / "song.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(output)
        raise


def stitch_song(manifest, stems, target):
    if len(stems) != len(manifest["chunks"]):
        raise ValueError("Every chunk must have a generated stem before stitching")
    output = array.array("h", [0]) * (2 * manifest["total_frames"])
    written_end = 0
    for chunk, stem in zip(manifest["chunks"], stems):
        samples = read_pcm(stem)
        if len(samples) != 2 * chunk["frames"]:
            raise ValueError("Stem chunk duration differs from the prepared input")
        start, count = chunk["start_frame"], chunk["valid_frames"]
        if start > written_end:
            raise ValueError("Gap in chunk timeline")
        overlap = max(0, written_end - start)
        for frame in range(count):
            weight = (frame / (overlap - 1) if overlap > 1 else 0.5) if frame < overlap else 1
            for channel in (0, 1):
                index = 2*(start + frame) + channel
                output[index] = round(output[index]*(1-weight) + samples[2*frame + channel]*weight)
        written_end = max(written_end, start + count)
    if written_end != manifest["total_frames"]:
        raise ValueError("Stitched audio does not cover the entire song")
    write_pcm(target, output)

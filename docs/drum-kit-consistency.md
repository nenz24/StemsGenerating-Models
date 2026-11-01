# Drum-kit identity across independently generated contexts

The uploaded 90-second chunk run is useful to the listener but changes timbre.
The preferred target is the lighter overall kit of chunk 1, with kick character
from chunks 0/2. The listener confirmed that "bass" means kick/bass drum.
Evidence and original uploaded feedback are preserved in
`baselines/acoustic-drums-context90-v1/` with zero-based chunk indices.

All three reports use the same source file, preset, caption, seed, CFG, steps
and FP32; each chunk's model-input WAV hash can be reproduced exactly. Source
contexts differ and all actual `reference_audio` values are null. Same seed is
not an instrument-identity lock. The beat-level crossfade cannot enforce kit
identity over a whole independently generated segment.

## Pinned upstream source inspection

At commit `ca1e85fe9430179831e6bc6be790c332190a3866`:

- [inference.py](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/inference.py)
  includes `GenerationParams.reference_audio` and forwards it separately from
  `src_audio` to the handler. LEGO uses direct conditioning with the supplied
  caption/lyrics and source audio duration.
- [generate_music_request.py](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/core/generation/handler/generate_music_request.py)
  processes reference audio independently of source audio. The reference
  preparation branch is not restricted to a different task; LEGO also requires
  a source waveform.
- [conditioning_batch.py](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/core/generation/handler/conditioning_batch.py)
  carries prepared references into the conditioning batch.

These are static API/preparation-path observations, not a completed GPU test or
proof of timbre locking for LEGO. The local bridge has not yet implemented or
validated a reference-conditioned kit experiment. Do not present a caption-only
change, reference parameter, or seed as a guaranteed identity constraint.

## Controlled next experiment

1. Preserve this candidate, the 30-second baseline and the failed full-song run.
2. On the existing VM, select a clean drum-only WAV excerpt from chunk 1 as a
   candidate reference; keep the piano as source, not the reference drum mix.
   Prefer the actual generated chunk WAV, not the lossy feedback preview.
3. Re-run one comparison chunk with the same piano interval, caption, seed,
   CFG, steps and precision; change only the reference conditioning and record
   its actual parameters/hash. Do not regenerate the full song first.
4. Evaluate lighter-kit consistency and whether the reference changes groove or
   weakens the desired kick. A shared reference may not preserve both properties.
5. Only after this result is assessed, choose how to specify the kick. Kick,
   snare and hats are currently mixed together; replacing a kick is not the same
   as copying a low-frequency band, and EQ does not reproduce a different kit.

No audio processing, kick separation, sample replacement, re-timing or new GPU
inference was run in this review. Existing baselines/presets remain unchanged.

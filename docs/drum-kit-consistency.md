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
proof of timbre locking for LEGO. Notebook 16 and `scripts/drum_kit_reference.py`
now implement the controlled experiment. CPU tests cover reference injection,
input/setting checks, archives and failure handling; audio behavior has not yet
been tested on GPU. Do not present a caption-only
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

## Run the comparison

Use notebook 16 on the active VM. It embeds the helpers and finds the original
notebook-15 result by the reviewed chunk 0/1 WAV hashes. If those files are gone,
upload `drum_chunked_result.zip` and the original piano; no lossy preview fallback
is used. Reference is a 30-second central excerpt of chunk 1's trusted core.
Only chunk 0 is regenerated. Its piano interval, padded input hash, caption,
seed, CFG, steps and precision must match the control. A different source-file
path is allowed only with identical model-input bytes. The actual reference
parameter/path and frozen reference hash must also be verified.

Save `drum_kit_result.zip` (WAVs) and upload `drum_kit_feedback.zip` (control,
reference and candidate previews, comparison metadata and model report). On
OOM/failure the available comparison reports are retained. `DRY_RUN=True`
prepares inputs and metadata without launching the model. Source/preset changes
or picking different baseline WAV hashes fail instead of comparing silently.

The reference alone may also change kick or groove. This experiment does not
combine a separated 0/2 kick with a chunk-1 kit, and a successful parameter
injection is not proof of timbre locking. No kick separation, replacement, EQ,
re-timing or new GPU inference was performed during implementation. Existing
baselines/presets remain unchanged. The isolated reference/control exports use
the same fixed .7 listening gain; mixes use the same fixed source/drum gains.

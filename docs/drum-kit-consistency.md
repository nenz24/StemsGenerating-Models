# Drum-kit identity across independently generated contexts

## Frozen full-song baseline

The listener has now approved the complete notebook-17 output: "sudah bagus,
simpan ini sebagai basline". The uploaded `drum_shared_kit_feedback.zip` is
preserved in [`acoustic-drums-shared-kit-full-v1`](../baselines/acoustic-drums-shared-kit-full-v1/)
with source, reference, per-chunk and export hashes. It covers 243.330625 seconds,
reuses approved chunk 0 and reports the same reference for all three chunks.
The original seam flags remain recorded; approval is the listener's assessment
of this full song. Complete lossless exports/reference still require the Colab
`drum_shared_kit_result.zip`, which has not been uploaded here.

No additional drum tuning blocks a clean electric guitar probe. Other songs,
the combined arrangement and offline iPhone/CoreAI remain separate validation.

## Approved comparison and full-song continuation

The listener approved the chunk-0 reference-conditioned result as sufficiently
good and requested the full song in that style. Its uploaded report changes only
`reference_audio` and the path to byte-identical piano input relative to the
unreferenced control. Reported candidate WAV SHA-256:
`a881a18704c322e8325a933345870f08174baa6b55bf4e5d942368fea4d20b8b`.
Reference WAV SHA-256:
`275afb8a23a7fbcc1c06a8e72baa47a77149a2d6cb0f3a82214ebfe2e07b9c53`.

Notebook 17 continues from that approved result. It creates a separate 90-second
context plan, freezes the exact same reference bytes, verifies/reuses approved
chunk 0 and generates only chunks 1/2 before stitching. Every cached/new chunk
must carry the same reference hash and actual reference parameter. Old
unreferenced plans cannot reuse referenced chunks or vice versa. The initial
piano input is re-created and checked against the approved model-input hash.
Incomplete approval preparation cannot be run as a completed plan.

Use the active runtime/weights. If the notebook-16 files are gone, upload
`drum_kit_result.zip` and original piano; the feedback MP3 ZIP is insufficient.
No old notebook-15 result ZIP is required. Save `drum_shared_kit_result.zip` and
upload `drum_shared_kit_feedback.zip`. The latter includes actual per-chunk
reports, before-stitch previews and seam excerpts. Consistent reference bytes
do not by themselves prove consistent musical results; assess the finished song.

Both the initial comparison and the full-song continuation now have listener
approval. GPU inference was performed in Colab; uploaded reports and previews
are preserved here. No independent kick replacement or EQ is part of this run.

## Original candidate

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

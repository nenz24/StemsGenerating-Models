# Acoustic drum probe

## 90-second contexts followed by stitching

Notebook 15 is the listener-requested controlled alternative to the rejected
full-song single context. It uses the same margin/float equal-power stitching
method as the bass experiment, with the acoustic drum preset unchanged. A
90-second cap includes margins/padding; this piano produces 90/89/90-second model
contexts and two seams. Drum chunking at this cap has not yet been evaluated on
GPU. The noisy bass baseline is a method reference, not proof this will be clean.

Use the active runtime/weights. Copy notebook 15 cells into the active Colab
session if necessary; original piano is reused or uploaded when requested. No
bass generation, model installation or checkpoint download is performed. Completed
drum chunks resume only after source/preset/frame/hash verification. Old bass
plans/results remain supported and are never reused as drum results.

`drum_chunked_feedback.zip` includes full drum/mix MP3, separate pre-stitch chunk
previews, actual per-chunk reports and seam excerpts. `drum_chunked_result.zip`
also preserves generated chunk WAVs and final WAVs. Pitch-continuity scoring is
not applied to drums; level/correlation flags do not certify groove or timbre.

```bash
python scripts/ace_context_chunks.py plan --instrument drums --source /path/to/piano.mp3 --out runs/drums90 --context-seconds 90
python scripts/ace_context_chunks.py run --plan runs/drums90/plan.json --execute
python scripts/ace_context_chunks.py stitch --plan runs/drums90/plan.json
```

## Full-song single-pass experiment

**Rejected by the listener:** the full 243.33-second output lost clear groove,
timbre and timing. `docs/drum-full-song-review.json` records the evidence and
limits. Input/preset consistency and the initial 30-second PCM prefix were
verified; the lossy preview has the correct length and no full-scale samples.
There was no song stitching. These checks do not isolate a unique cause in the
long-context generation/decoder pipeline. `generated` is not a quality verdict.
Notebook 14 is retained for reproduction with DRY_RUN=True by default; it is
not the next recommended quality experiment. Do not repeat duration/seed sweeps
without a testable hypothesis. Preserve the approved opening 30-second baseline.

The uploaded START=90 probe is reported glitchy; START=210 is reported clean.
Both reports have the same source, drum preset and CUDA FP32 settings. At the
listener's request, notebook 14 now tests the entire original piano in one model
invocation. It uses the existing runtime/weights and keeps the same drum caption,
seed, CFG and steps. No song chunking, stitching, automatic short-context retry,
precision change or bass regeneration is performed.

Upload the full original piano, not the exported 30-second prepared source. The
helper accepts 10–600 seconds and pads a fractional final second for the model,
then trims the export to the original sample count. The legacy engine CLI limits
stay unchanged; longer contexts call the engine API in a fresh process. Memory
fit and musical quality of the full song require actual CUDA execution.

```bash
python scripts/drum_probe.py --source /path/to/piano.mp3 --out runs/drum_full_plan --full-song
python scripts/drum_probe.py --source /path/to/piano.mp3 --out runs/drum_full_real --full-song --execute
```

`--duration 60` is an explicit optional excerpt experiment; it is not run by the
full-song notebook. The default CLI and notebook 13 remain 30 seconds. Existing
baselines are preserved. On failure, download the available `drum_feedback.zip`.

## Original 30-second probe

Bass context 90 is frozen, with its noisy chunks documented in
`baselines/electric-bass-context90/`. Do not overwrite those archives or regenerate
bass while starting a new instrument.

Use notebook 13 on the active Colab VM with the existing ACE-Step runtime/weights.
Its embedded helper discovers the source folder created by notebook 12, including
when notebook globals are missing. If needed, copy its cells into the active
notebook. Existing uploaded piano is reused; otherwise upload the original piano.
No installer/download is called by the drum probe. An empty VM needs notebook 12's
source/runtime/weights cells first; stop before its bass generation cells.

One 30-second Base/LEGO drum generation uses CUDA FP32, seed 0, CFG 7, 50 steps,
offload, and a restrained acoustic-kit caption saved in `configs/acoustic-drums.json`.
It uses the original piano directly with gain .7, no invented BPM/meter, and no
bass mix as the condition. The bass 90-second cap is not assumed suitable for drums.

Listen for kick/snare/hi-hat, groove/timing, fills and noise. Upload
`drum_feedback.zip` for review. Save `drum_result.zip` for the WAVs. Parameter,
input, seed, drum instruction, FP32 capture and output-length checks must pass
before reporting success. Failure ZIPs contain reports, not fabricated audio.

The initial 0:00–0:30 probe has now been uploaded, technically checked, and
approved by the listener ("Sudah cocok"). Original WAV/feedback archives are
preserved in `baselines/acoustic-drums-30-v1/`. Next check START=90 and START=210,
each still 30 seconds, with every other parameter fixed. This checks different
song sections without assuming that opening-excerpt success applies to the
entire song. Use a new output directory for each attempt.

CLI on a complete checkout:

```bash
python scripts/drum_probe.py --source /path/to/piano.mp3 --out runs/drums_plan
python scripts/drum_probe.py --source /path/to/piano.mp3 --out runs/drums_real --execute
```

`--start` selects a different 30-second excerpt. Every output directory must be
new; the previous bass/drum runs are preserved. CPU tests validate plan, targeting,
export and failure handling with mocked inference. Actual GPU execution was
performed by the listener in Colab, not on the chat machine. Approval does not
establish quality for other sections or a full-song drum track.

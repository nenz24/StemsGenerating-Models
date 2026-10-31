# ACE-Step: bounded contexts and measured seams

Use ACE-Step for bass performance/timbre. The MIDI experiment remains a separate
prototype; its rendered sound was not preferred by the listener. This helper
keeps the selected Electric Bass preset and the existing single-pass worker.
It does not modify the pinned upstream source, the legacy 10/30-second runner,
or its linear stitch function.

## What the evidence supports

The listener selected 90 seconds after reporting good 60/90-second output and
substantial noise at 120 seconds. The uploaded 60/90-second reports both show
the same piano file, Base/LEGO, CUDA FP32, seed 0, CFG 7, 50 steps and caption.
Their model-input hashes can be reproduced from the original piano. The 120-second
assessment is listener feedback; its ZIP has not been inspected. This selects a
working cap for this song, not a universal model limit. The selection is recorded
in `configs/electric-context.json`.

Neither full-song GPU chunk generation nor musical seam quality has been
validated on this CPU machine. MIDI chord estimates are not ground truth for
evaluating ACE-Step harmony.

The uploaded piano decodes to 243.330625 seconds at 32 kHz. The default plan has
three contexts (two seams), each no longer than 90 model seconds, including
six-second margins where available and integer end padding. Core audio is shorter
than the model context. A 90-second core with two six-second margins would need
102 seconds and is not the selected mode.

~103 BPM is an estimate from the earlier piano analysis. It does not establish
3/4 meter, downbeats or bar positions. Without verified cues, planning chooses
low-energy candidates near balanced time targets. A .58-second fade is a user
adjustable starting point, not a verified beat grid.

## Local/Colab use

Run orchestration in a kernel/environment with NumPy, SciPy and SoundFile, plus
ffmpeg/ffprobe. The already locked ACE-Step Python environment is used separately
for inference. Do not upgrade or replace its pinned dependencies.

Install missing orchestration packages with
`python -m pip install -r requirements-context-chunks.txt` in the notebook/kernel
environment, not the locked ACE-Step environment.

On Colab, reuse the active runtime that already has the bridge checkout, CUDA
runtime and verified weights from notebook 10. A new notebook may attach to a
different VM; copy the companion notebook cells into the active session if needed.
Uploading this tool ZIP alone does not install ACE-Step, supply model weights or
provide a GPU. No new checkpoint download is required on a complete existing VM.

From the bridge root:

```bash
# Plan only; no GPU, no weights, no inference.
python scripts/ace_context_chunks.py plan --source /path/to/piano.mp3 \
  --out runs/ace_context_90 --context-seconds 90 --margin-seconds 6 --fade-seconds .58

# Run one duration at a time. Omit --execute to prepare a dry-run only.
python scripts/ace_context_chunks.py probe --plan runs/ace_context_90/plan.json \
  --duration 90 --execute
# Probes are optional: 90s is already listener-approved; do not repeat by default.
# A 60s control or explicit 120s experiment remains available if needed.
```

Each probe uses the preserved original audio, source offset (default zero) and
Electric Bass preset: Base / CUDA FP32 / seed 0 / CFG 7 / 50 steps / same caption.
Compare the common audio interval and inspect memory/precision reports; do not
select the longest duration solely because it completed. Each attempt is retained
in `probe_090/attempt_000`, etc. Its `full_song_feedback.zip` can be uploaded for
review. Old experiments using a different source or preset are not assumed equal.

After choosing a tested context cap, make a new plan with that cap. Then:

```bash
python scripts/ace_context_chunks.py run --plan runs/ace_context_90/plan.json --execute
python scripts/ace_context_chunks.py stitch --plan runs/ace_context_90/plan.json
```

Use `--root /path/to/bridge` on probe/run if the two helper files were uploaded
outside the checkout. The companion notebook discovers this root and passes it.
Generation is sequential: every independent chunk currently launches a fresh
worker/model load. Completed chunks resume only after preset/input/frame/hash
checks; failed attempts remain on disk. Same seed/caption does not guarantee
the same performance, register, or timbre across contexts. Runtime may exceed
a single long invocation because margins, reloads and packaging add overhead.

New schema-2 plans preserve a byte-identical copy of the original input and use
it directly for probes and generation, matching the approved single-pass
preprocessing. Canonical PCM16 is retained only for boundary planning/mixing.
Schema-1 plans remain readable and keep their original canonical-input behavior;
make a new plan rather than editing old plans to change preprocessing. The old
canonical-input path differed by up to one PCM16 step on the supplied piano.

To supply real SongMap boundary cues, pass `--cues /path/to/cues.json` at planning:

```json
{"source_sha256": "SHA256_OF_ORIGINAL_PIANO_FILE", "downbeats": [51.2, 97.4], "sections": [192.5]}
```

Times must be source-relative seconds, finite and strictly inside the audio.
This explicit interchange format is not an automatic SongMap parser. Source hash
must match; only cues within the feasible context constraints are used. A used
cue remains the fade center; it is not shifted by signal-based seam search.

## Stitching and diagnostics

The new stitcher crops model padding/edges, retains the full source timeline,
and performs a short vectorized equal-power crossfade in float. With no supplied
cue, it searches at most ±1.5 seconds around the planned cut while retaining at
least half the margin outside the fade at both model edges. It favors low energy,
correlation and smaller predicted level changes. It does not time-shift the bass,
invert polarity, retune notes, or fill missing output with silence.

Equal-power means the squared fade weights sum to one. It preserves power for
uncorrelated signals, not all signals: correlated bass can gain level and opposite
phase can still cancel. The tests demonstrate that failure explicitly. A single
global attenuation provides headroom; it does not repair cancellation or noise.

`stitched/` contains the bass, piano mix (fixed .5 per track, canonical piano
gain .7), MP3 previews, six-second seam excerpts and `seams.json`. The report
includes overlap correlation, predicted level change, blend/reference RMS,
periodicity-gated pitch estimates before/after, and manual-review flags. Pitch
estimates can be octave-wrong; level/pitch changes may be musically intentional.
These are signal diagnostics, not calibrated measures of noise, chord accuracy
or musical continuity. Listen to flagged and unflagged seams. Whole-track noise
already present inside a chunk cannot be fixed with crossfades.

Repaint/continuation, BPM/key conditioning and automatic duration selection remain
unimplemented experiments. No API support or quality benefit is assumed.

## Validation

```bash
python -m unittest discover -s Tests/StemEvaluationPython -p test_context_chunks.py -v
```

Tests use PCM fixtures and mocked inference: coverage including fractional song
length, total model-duration caps, real provided cuts, equal-power cancellation,
unchanged interiors, source identity, failure/resume/tamper checks, and a real
ffmpeg stitch/export. Passing these checks does not establish GPU memory fit or
musical quality.

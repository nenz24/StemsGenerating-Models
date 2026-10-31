# Piano → MIDI → electric bass (CPU prototype)

Alternative to ACE-Step: analyze the piano, estimate harmony/timing, write an
editable monophonic bass MIDI, then render GM Electric Bass (finger). No GPU or
model-weight download is required. This prototype does not use ACE-Step.

From the repository root, install `requirements-piano-bass.txt` into `.venv`.
System prerequisites: ffmpeg/ffprobe, libfluidsynth3, and a General MIDI SoundFont.
The current environment includes `/usr/share/sounds/sf2/TimGM6mb.sf2`.

```bash
UV_CACHE_DIR="$PWD/.runtime-cache/uv/cache" uv pip install --python .venv/bin/python -r requirements-piano-bass.txt
NUMBA_CACHE_DIR="$PWD/.runtime-cache/numba" OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/piano_bass.py --source /path/to/piano.mp3 --out runs/piano_bass_new
.venv/bin/python -m unittest discover -s Tests/PianoBassPython -v
```

Use a new output directory for each run. The script refuses existing directories.
Outputs: `bass.mid`, isolated `bass.wav` / `bass.mp3`, `piano_with_bass.wav` /
`piano_with_bass.mp3`, `analysis.json`, and mono analysis audio. Source audio is
preserved. MIDI event times use a fixed 120 BPM clock to preserve detected absolute
times; the estimated performance BPM is reported separately in the JSON.

Analysis uses beat tracking and octave-aware constant-Q note energy, with partial
harmonic suppression. Major/minor triad templates are smoothed over time.
Extended/suspended voicings are approximated, so labels may be ambiguous.
Bass uses roots with occasional evidence-supported fifths,
limited to MIDI 28–48, with rests in quiet regions and dynamics from piano energy.
Beat phase is not a verified bar/downbeat grid. There is no explicit key prior.

Chord names and timing are estimates, particularly for sparse voicings, inversions,
suspended chords and rubato. Score margins are not confidence probabilities.
This is a conservative first arrangement and a small sampled GM instrument;
passing numerical checks does not establish musical correctness or realism.
Listen to the mix and edit the MIDI/chord decisions before judging it final.

## Phrasing and timbre revision

`scripts/bass_performance.py` reuses the saved harmony analysis and verifies the
source hash. It introduces space, varied note lengths, accents, quiet offbeats and
small timing changes, then renders a dry finger-bass patch with mild amp-style
EQ/compression. It uses a repeatable heuristic eight-beat phrase, not inferred
meter or a learned performer. It cannot supply true slides, hammer-ons or
round-robin sample variation from a General MIDI patch.

```bash
.venv/bin/python scripts/bass_performance.py \
  --analysis runs/piano_bass_v3/analysis.json \
  --source /path/to/piano.mp3 --out runs/piano_bass_performance_new \
  --soundfont /path/to/FluidR3_GM.sf2
```

The FluidR3 GM SoundFont can be extracted locally from the authenticated Debian
`fluid-soundfont-gm` package. Retain the package copyright/license with the local
instrument. The output includes a 30-second preview starting at 00:24, matching
the original demo's excerpt. Isolated bass RMS is matched approximately for
comparison; subjective realism still needs listener feedback.

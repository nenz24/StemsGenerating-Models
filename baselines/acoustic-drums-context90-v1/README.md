# Drum 90-second chunk candidate: timbre preference

The listener finds the result reasonably good, with changing drum sounds across
chunks. Indices are **0, 1, 2**, matching the exported file names.

- Overall kit character: **chunk 1**, lighter.
- Kick/bass-drum character: **chunks 0 and 2**.
- "Bass" here was confirmed to mean **kick**, not bass guitar.

Target: retain the performance/timing while obtaining a coherent kit with the
lighter character of chunk 1 and the preferred kick character of 0/2. This is
a recorded target, not an already generated composite or a fully approved kit.

`drum_chunked_feedback.zip` preserves the uploaded mix/drum MP3s, pre-stitch
previews, actual per-chunk generation reports and seam excerpts. It does not
contain full generated chunk WAVs. `baseline.json` records archive hash and
preferences; `timbre_measurements.json` holds descriptive band/level measurements.
Different musical sections and performance density affect these measurements;
they do not by themselves identify drum-kit pieces or measure noise.

All three reports have the same caption, seed 0, CFG 7, 50 steps and CUDA FP32.
Their piano model-input hashes were reproduced exactly. All references were
unset. A common seed/caption is not a shared kit constraint when source/context
changes. Crossfade is only a local blend, not a timbre-identity controller.

The next proposed test is a shared chunk-1 reference on one comparison chunk,
with other settings fixed. Validate its effect before regenerating the whole
song. The kick is not isolated in the uploaded stereo drum mix; independent
kick replacement or stronger tonal shaping has not been performed.

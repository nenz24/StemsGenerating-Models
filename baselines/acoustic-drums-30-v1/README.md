# Acoustic drum baseline: opening 30 seconds

Listener assessment: **Sudah cocok**. The uploaded reports confirm Base / LEGO /
CUDA FP32, seed 0, CFG 7, 50 steps, the acoustic-kit caption, and original piano
conditioning at gain .7. Source/model-input and exported WAV hashes were checked.

- `drum_result.zip`: original exported drum/mix WAVs and generation report.
- `drum_feedback.zip`: MP3 previews, actual generation settings and FP32 capture.
- `baseline.json`: archive hashes, selected preset and approval scope.
- `measurement.json`: exported-WAV, input/mix consistency and onset diagnostics.

The WAVs are stereo 32 kHz PCM16 and exactly 30 seconds, with no nonfinite or
full-scale samples. Upstream float audio before PCM export is not included.
The onset measurements are signal descriptors, not a calibrated groove score;
no retiming or processing was applied to the listener-approved files.

Approval is limited to 0:00–0:30 of this piano. Before a long-song run, test
1:30–2:00 and 3:30–4:00 with the same preset, seed and 30-second duration.
Notebook 13 supports this by changing START to 90, then 210. Use the active
runtime/weights. Preserve this baseline rather than replacing its audio.

The earlier bass baseline and its known noisy later chunks remain separate in
`../electric-bass-context90/`. The bass 90-second cap is not adopted for drums.

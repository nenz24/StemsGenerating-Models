# Electric Bass 90-second baseline v1

Frozen at the listener's request before moving to another instrument. This is a
reference with known defects, not an approved clean full-song result.

- `single_pass_90_feedback.zip`: listener-preferred 90-second single-pass result.
- `chunked_feedback.zip`: the 243.33-second song, generated as 90/89/90-second
  contexts with two stitched transitions. Contains bass/mix MP3, plan and seams.
- `baseline.json`: archive hashes, preset, timing and the listener's assessment.

The second full-song chunk reasonably matches the first but has more noise; the
third has substantial noise. The later 60-second retry was reported as full
noise and is rejected. The 120-second probe was also rejected. Those two audio
files have not been uploaded here and are not preserved in this baseline.

Keep Base / LEGO / CUDA FP32 / seed 0 / CFG 7 / 50 steps / selected Electric Bass
caption, 90-second context cap, six-second margins and .58-second fade unchanged
when reproducing this reference. Do not overwrite its archives with future runs.

Only the uploaded preview/report ZIPs are preserved. The original generated
full-precision WAVs and model weights are not in this directory. The source piano
is identified by hash; the preview mix naturally includes that piano audio.

Bass tuning is paused. A new instrument needs its own output directory/preset;
its performance and audio quality must be evaluated independently.

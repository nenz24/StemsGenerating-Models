# Mac30s reference stopped before inference at source PCM checksum

**Subsequent uploaded PCM review:** the [two-file ZIP](../coreai-drums-reference30-pcm-review/README.md)
has arrived. Input differs by at most1LSB and passes original input bounds,
but a same-epsilon VAE probe amplifies that perturbation and fails component
comparison. No source-hash exception is added. Explicit nearest-even PCM16
preparation reproduces the exact frozen WAV/PCM/input48 bytes here; the Mac
retry uses a new folder and retains strict checks. Pending-file requests below
describe the earlier log-only stage and should not be repeated.

[User Terminal log](terminal.txt),5050bytes, is preserved unchanged. The first
Git fetch timed out on GitHub443; retry succeeded, and the project updater
fast-forwarded3ca8ca7→ff54b24 with its normal project/signing backup. That network
issue is not the final failure. The new script verifies all four existing
checkpoints, starts source preparation and raises **Prepared PCM differs from
accepted baseline**. [Summary](summary.json) separates direct log observations
from inference based on the executed source version.

The original MP3 hash check and stereo32kHz PCM16/30s format checks precede this
error and therefore passed. The failure is at the PCM-data hash, not merely WAV
header metadata. The source32WAV was written by canonicalization before the
gate. No source encoder, Qwen, DiT or decoder calls have begun. No new30s drum
waveform, numerical PASS or musical verdict is established. Checkpoints/cache,
BF16/app selector and old failed output remain untouched.

Actual Mac PCM bytes and failed `report.json` have not been uploaded. The old
error did not record the actual digest before raising, so the difference's
size/timing/cause cannot be measured from this log. Platform/FFmpeg/rounding
differences are hypotheses; they are not accepted as explanation or grounds
for bypassing the gate. Original expected PCM SHA remains
`1d57ce145dcf66d30762d81b2ca26df4798f3b0b0a5d36b303a53b7db7baaf23`.

## Concrete change

`check_source_pcm` now records actual and expected PCM digests, full-WAV digest,
format, FFmpeg version and platform in `source-preparation.json` before the
same exact hash check. `execute` attaches that diagnostic to the failure report.
It still rejects mismatches before resampling/source encoding and does not
accept numerically close inputs without evidence, override a checksum, change
reference/tolerances, or invent original CUDA intermediate tensors.

Two new tests reproduce a shaped but wrong prepared PCM, verify retained
digest diagnostics and failure report, and assert that resampling/encoding do
not run. Original input bytes remain unchanged. A nonexecutable checkpoint
fixture tests the early-rejection boundary; it is not reported as successful
model inference. The other reference tests still cover strict input/output
admission, fixed plan parameters, capture/live feedback and cache lifetime.

[Actual positive input check](exact-input-check.json) uses the original archived
30s source in this workspace. It still passes exact admission, and the resulting
source48bytes match the previously verified750-frame encoder input exactly.
This run performs only input preparation/resampling, with no new neural calls.
It does not compare or validate the unseen Mac PCM.

## Required next evidence

The [collection command](../../docs/coreai-drums-quality.md) packages only two
existing files: failed `report.json` and `source32.wav`. It does not require a
fetch, generation restart, cache deletion, native export, source re-selection
or another download. Keep the failed folder; there is no overwrite/resume flag.
Once actual bytes arrive, measure differences against the frozen reference
under original abs1e-4/rel1e-3/normalizedRMSE1e-3 bounds before selecting recovery.
No numerical acceptance or specific recovery route is established by this
checksum error alone. No audio or weights are pushed to Git; only the supplied log,
code/tests/docs and measured metadata are archived here.

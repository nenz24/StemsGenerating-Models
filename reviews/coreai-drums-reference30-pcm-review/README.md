# Measured PCM rounding difference; recover exact reference preparation

**Subsequent Mac v2 result:** Float32/explicit-nearest preparation still fails
the unchanged source checksum before neural execution. The [new actual-data
review and exact recovery](../coreai-drums-reference30-exact-recovery/README.md)
measures1785 remaining1LSB differences and reconstructs the original reference
bit-exact from the measured cache. Its private correction data is not committed.
Do not repeat the v2 decode command; the new wrapper restores input and starts
one v3 reference run. Earlier Mac-execution/pending statements below are historical.

The requested two-file ZIP is received and verified. [Archive review](archive-review.json)
records ZIP/member hashes and format. The3147-byte [failed report](failed-report.json)
is preserved unchanged: no encoder, decoder or diffusion steps ran on Mac.
All four pinned checkpoints had verified before preparation stopped. Only
report/metadata/source code are committed; uploaded audio/tensors stay local.

## Actual input difference and downstream sensitivity

[Full PCM comparison](pcm-comparison.json) covers all1920000 interleaved samples
of stereo32kHz/30s PCM16. Reference SHA `1d57ce14…7baaf23` stays unchanged; Mac
SHA is `88b0464a…9fe9ac7`.939512 samples differ:939505 by−1LSB and7 by+1LSB.
Maximum normalized absolute error3.0517578125e-5, normalizedRMSE0.000300635,
zero pointwise violations: input passes the original abs1e-4+rel1e-3 and
nRMSE1e-3 bounds using the actual shared `precision_metrics` implementation.
This is **not bit-exact**, and the input measurement is not full pipeline parity.

An [offline source-sensitivity probe](source-sensitivity.json) executes one
pinned CPU FP32 VAE encoder on this changed input. Original CPU reference
posterior/latents are retained, and source epsilon bytes are identical. Resampled
48kHz inputs still pass original input bounds, but output comparisons **fail**:
source latent nRMSE0.021366, maximum difference1.342956; mean/std/context also
fail the original component bounds. These are responses to **different input
bytes**, not an implementation regression on identical inputs. They demonstrate
that input closeness does not guarantee comparable encoded conditioning.

No known-hash exception, numeric-input allowlist or relaxed threshold is
introduced. The original frozen PCM checksum remains required before generation.
No DiT/decoder generation or musical acceptance resulted from this probe.
[Reproduction source](source_sensitivity.py) is offline analysis only; it reads
the preserved uploaded PCM and original750-frame CPU fixtures in ignored work
folders and checks their hashes before comparing. It does not invoke generation
or alter source admission. Its one encoder call is separate from the0-call
failed Mac report.

## Explicit conversion, unchanged reference

[Quantization investigation](quantization-check.json) shows Linux FFmpeg
Float32 output at the original fixed gain0.7, followed by nearest-even PCM16,
reproduces the frozen PCM hash exactly. The old Mac PCM has a negative1LSB bias
relative to that reference; it is mostly close to floor quantization but differs
at7090 samples from floor of the Linux Float32 stream. Thus a precise cause in
platform/format negotiation/decoder arithmetic is not established from these
data alone. It is not claimed as the proven cause of earlier audible glitches.

The reference helper now decodes original MP3 to interleaved32kHz Float32 using
FFmpeg and gain0.7 once, saves that raw stream, then scales exactly by32768 in
Float64, applies `np.rint` nearest-even, clips PCM16 bounds and writes a standard
WAV. Explicit rounding includes negative ties and avoids delegating integer
quantization to platform-specific FFmpeg conversion. Recording gain, duration,
source resampling and all learned parameters/sampler settings stay fixed.
No production audio-preparation helper or iPhone baseline is edited.

[Actual original-source recovery](nearest-recovery-audit.json) confirms this
path matches **the entire accepted WAV SHA** `1f68460c…d66b54`, frozen PCM SHA,
and original resampled48kHz encoder-input bytes `9d62d4ec…07f4a8`. There are no
new neural calls in that recovery check; the earlier source input validation
applies to the identical bytes. The Mac Float32 path has **not** run here, so
successful Mac admission is not assumed: the exact hash gate remains enforced.
Original numerical bounds, earlier failed/diagnostic statuses and absent CUDA
trajectory reference remain unchanged. No new model or musical PASS is claimed.

## Retry and automatic evidence

Use [the new Mac command](../../docs/coreai-drums-quality.md), choose original
`piano.mp3` again and write `.runtime-cache/drums-reference30-v2`. Cached PCM16
has already lost fractional information; original Float32 decoding is needed.
The old failed folder is preserved. Once exact admission passes, one30s/50-step
CPU host trajectory runs. No native export, checkpoint download or app rebuild.

If preparation still fails before neural execution, a bounded ZIP of the new
report/diagnostic/WAV/Float32 stream is created automatically and selected in
Finder. Only those four named files are eligible; no model/checkpoint or unrelated
cache files are collected. Existing ZIPs/results are never overwritten. There
is no automatic retry or checksum override. Send that ZIP if the Mac Float32
path still differs, or the generated report/listening result when it completes.

[Validation](validation.json):181 host tests,179PASS,two optional skips;12
reference tests. Two new tests specify signed nearest-even/ties/clipping,
invalid Float32 rejection and failure-only ZIP selection/excluded weights/no
overwrite. The real input recovery and the separate offline VAE sensitivity
probe are distinguished from analytic fixtures. No Apple build/device execution
or full host50-step generation was performed here.

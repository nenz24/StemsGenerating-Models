# Native DiT/APG pass; full 750 decoder specialization remains blocked

**Isolation result received:** [the unchanged 750 decoder passes alone with high sampled memory](../coreai-drums-reference30-window-decoder/README.md).
The next device action measures the bounded 256 decoder; do not repeat the
original 16-call test or standalone 750 run. The investigation below is historical.

[Original device report](device-report.txt), run `6F079DC7`, completes 15/16 calls.
All five completed strict comparisons pass at original abs 1e-4 / rel 1e-3 / nRMSE 1e-3:
conditional/unconditional step 1 velocities and three isolated APG outputs.
Each has zero pointwise violations; DiT nRMSE is about 1.7e-6. The missing sixth
comparison is the decoder waveform, which was never produced. The original
component collection remains **NOT COMPLETED / NOT PASSED**, and decoder numeric
parity is **NOT EVALUATED**, not a numerical mismatch. [Audit](device-audit.json).

The decoder first opened but failed initial binding with BNNS `graphLoadFailed`.
One narrowly targeted repair was attempted as already permitted. Its retry
then failed at model-open/specialization with `AIModelError.failedToSpecialize`.
Repair budget 1/1 is exhausted for that run; no successful retry or decoder
inference occurred. Recorded opens 16 / binds 15 count completed operations.

Sampled process maximum is 954.867 MiB (10 ms sampler), thermal nominal at both
ends and low-power off. No explicit 2000 MiB guard triggered. Sampling does not
establish the transient peak or diagnose the compiler failure. The long decoder
shape/compiler, native cache state and preceding 15 model calls are not isolated
by this run. No cause is claimed from the footprint or error message alone.
The 14 DiT backend-await rows sum 2085.504 ms for one prediction, not a warm-loop
benchmark or evidence for 50-step timing/quality.

## Next controlled device test: the same decoder alone

The new `drums-reference30-decoder` resource profile includes **seven files**:
the exact original reviewed manifest, unchanged full 750 decoder's three payload
files, saved final latent, saved raw audio and captured reference JSON. It
excludes all DiT/APG assets from generated .app only; original exports and
ModelCandidates baselines remain on Mac. No re-export, fresh random tensor,
precision change, output filter or new learned weight is involved.

The [current Mac command](../../docs/coreai-drums-quality.md) selects it through
`activate_coreai_drum_reference30.py --decoder-only`. Existing source/copy
integrity verification and byte-preserved selector backup are reused. A fresh
app process exposes **Uji decoder 30 detik saja**, one model call and one
comparison against the same saved full 30s audio. The VAE remains FP32 and
input [1,64,750] / output [1,2,1440000] are unchanged. No DiT/APG model runs first.

This isolation has **no retry or cache deletion**, retains original CPU-only
request and process guards, and records sampled footprint plus native operation
breadcrumbs. Native cache state is not controlled; this is not a cold-cache
benchmark. A completed finite mismatch remains NOT PASSED, while specialization
failure leaves numerical parity NOT EVALUATED. Reports/breadcrumbs are separate
from the previous five-pass component report and survive process interruption.

Success would establish standalone decoder readiness, not combined-lifecycle or
full-trajectory acceptance. Failure would justify a smaller decoder graph or
more specific compiler investigation. No automatic 50-step sampling is triggered.
The original full component button remains available under its previous profile;
it need not be repeated for this check.

## Prepared fallback: actual host window 256 feasibility

[`window_audit.py`](window_audit.py) uses the uploaded accepted final latent and
pinned original VAE checkpoint. It executes six 256-frame decoder windows with
64-frame interior context and 128-frame retained cores; first/last windows align
to the recording boundaries. There is no added latent zero padding, crossfade,
audio smoothing, DiT chunking or new generation.

An architecture guard checks the actual decoder's local Snake/convolution
operators and stride/kernel/padding/dilation parameters. Conservative backward
dependency cones are checked at every stage to exclude artificial padding from
retained outputs. Each recording frame is retained exactly once. This handles
true clip boundaries separately from interior windows.

[Actual CPU measurements](host-window-audit.json) pass all 12 checks: six retained
regions, five seam neighborhoods and the assembled full 30s audio against the
accepted original full decoder output. Overall nRMSE `1.9987732e-6`, max abs
`2.9712915e-5`, zero pointwise violations at unchanged bounds. No native export,
device window execution or new listening verdict is claimed. This is a tested
candidate, **not selected in the current device build**: first isolate the
unchanged 750-frame decoder to distinguish it from the preceding-model context.
No extra host trial is requested from the user for this already completed audit.

Reproduce using existing exact captures and weights:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  reviews/coreai-drums-reference30-decoder-specialization/window_audit.py \
  --capture .runtime-cache/drums-reference30-v3 \
  --vae-checkpoint /path/to/existing/diffusion_pytorch_model.safetensors \
  --out .runtime-cache/drums-reference30-window-audit-new.json
```

[Validation](validation.json): 209 Python tests, 207 PASS, two optional skips;
decoder-only selection/restore, seven-file packaging, removal of 54 unneeded
generated .app files, source preservation and stdlib-only build CLI are covered.
[Swift grammar audit](swift-syntax-audit.json) passes three edited files; it is
not Apple SDK typechecking/build or device execution. No private audio, vectors,
weights, ZIP or native assets are committed. Baseline BF16, FP32 VAE, original
tolerances and accepted host listening result remain unchanged. Native full 50
and full-song bass/drums/guitar remain unevaluated.

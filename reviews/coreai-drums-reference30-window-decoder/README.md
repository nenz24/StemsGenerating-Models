# Standalone full30s decoder passes, with a high sampled footprint

**Device window test completed:** [12 strict comparisons and sampled-memory gate pass](../coreai-drums-reference30-live-generation/README.md).
The next action is one independent 50-step native reference generation; do not
repeat standalone decoder/component tests or model export. Details below describe
the completed decoder preparation stage.

[Device report](decoder-isolation-report.txt), run `22991DB1`, successfully opens,
binds and runs the original 750-frame decoder alone. All 2,880,000 waveform values
pass original abs 1e-4 / rel 1e-3 / nRMSE 1e-3 bounds: max abs `5.453825e-5`, nRMSE
`2.1635352e-6`, zero pointwise violations. No retry or cache deletion occurred.
Backend-await 7,869.882 ms, total 8,889.973 ms; thermal fair at both ends, low-power
off. [Structured audit](device-audit.json) normalizes the pasted Markdown log.

This demonstrates standalone native decoder support for the existing asset.
It does not establish combined-lifecycle readiness or the cause of the
[preceding specialization failure](../coreai-drums-reference30-decoder-specialization/README.md).
Native cache state and thermal conditions were not controlled between runs.

Sampled process maximum is **2696.770 MiB (~2.63 GiB)**, above the existing 2000 MiB
process budget. The pre-call/post-bind checks do not cap allocation during
backend execution. The numerical PASS is retained; it is not a memory-readiness
PASS. Sampled peak is not exact transient/native peak. Peaks from independent
runs cannot be added to estimate combined sampler/decoder residency. Full 50-step
sampling and native music acceptance remain unevaluated.

## Next: bounded working set with unchanged FP32 VAE and gold audio

[`prepare_coreai_drum_reference30_window_decoder.py`](../../scripts/prepare_coreai_drum_reference30_window_decoder.py)
prepares one 256-frame FP32 decoder in a new ignored ModelCandidates folder.
The default plan is read-only. Execution checks only existing local VAE weights
and the exact accepted saved final latent/raw waveform/reference report.
No DiT/Qwen/source regeneration or weight download occurs.

The production code runs **six actual host decoder windows**, with 256-frame
inputs, 128-frame retained cores and 64-frame interior context. First and last
windows align to true recording boundaries. Every output frame is retained
exactly once. An actual architecture guard plus inverse-convolution dependency
bounds excludes artificial padding from retained output regions. No extra latent
padding, crossfade, EQ, smoothing or time-chunked DiT sampling is introduced.

[Actual production host audit](production-host-audit.json) passes 12 strict checks:
six cores, five seam neighborhoods and the full assembled 30s waveform against
the accepted original full 750 output. Full waveform max abs `2.9712915e-5`,
nRMSE `1.9987732e-6`, zero pointwise violations. This runs the folded FP32 decoder
that the preparation/export path will use, not synthetic/music acceptance data.
Random tensors are not regenerated. Only metadata is committed.

When eligible, the Mac script copies the unchanged decoder256 payload from an
existing 10.24s instrument/bass bundle after verifying pinned VAE provenance,
original folding/shape contract and every payload hash. Ineligible candidates
remain intact. An explicitly requested invalid reuse source fails. If no
eligible asset exists, the script exports **only one small-shape VAE graph**
from the already verified weights; native export requires the existing Apple
Silicon/CoreAI environment. Native model bytes/export are unavailable in cloud.

All 12 host gates must pass before preparation completes. Failed/interrupt
reports are preserved; existing complete output is verified/reused without
repeat inference/export. `--activate` selects only this new seven-file decoder
build after full integrity verification, backing up the prior selector. Original
750-frame exports, existing 256 assets, baseline BF16, FP32 VAE and tolerances are
preserved. Generated model/vector folders are ignored by Git.

## Device measurement

Use the [current Mac command](../../docs/coreai-drums-quality.md), then press
**Uji decoder berjendela 30 detik** once on a fresh iPhone app process.
The device verifies the build reference identity, all six payload/input hashes,
fixed architecture/data/checkpoint contract, host gates and exact per-window
input hashes. It retains **one decoder model/function** for six calls. Expected
accounting: 1 open / 1 bind / 5 reuse hits / 6 calls. No retry or cache deletion is used.

Retained cores are copied into a planar 30s array without overlap blending;
six core, five seam and full-output comparisons use the original tolerances.
The existing pre-call/post-bind 2000 MiB guards remain and current footprint is
checked after each window. Numerical parity and sampled-memory readiness are
reported separately: readiness requires all 12 strict checks and sampled peak
below the same 2000 MiB budget. A numerical PASS alone does not adopt the candidate.
Sampling remains 10 ms, not an exact peak measurement or combined-pipeline proof.

Reports/breadcrumbs are per-run and separate from the earlier standalone/full
component reports. Live progress is bounded; restored reports include matching
interrupted native phases. No WAV/playback or native 50-step trajectory is run by
this button. Full generation and device listening follow after this resource
check; the final goal remains full-song bass/drums/guitar on iPhone.

## Validation

[Validation](validation.json): 220 Python tests, 218 PASS, two optional skips.
Eleven new tests cover edge/context coverage, malformed/false host gates,
protected/existing outputs, retained failures, strict eligible asset reuse,
metadata transport integrity, seven-file stdlib-only build staging and pruning
only generated .app resources. File fixtures are explicitly NONEXECUTABLE,
with synthetic transport identities; they are not neural evidence. Actual
learned-model evidence is the six-call host audit above.

[Swift grammar syntax](swift-syntax-audit.json) passes all three edited Swift
files. No Apple SDK typecheck/build/device execution was performed here. Native
256 memory/numerical readiness is therefore still NOT EVALUATED. No private
audio/vectors, native binaries or learned checkpoints are committed.

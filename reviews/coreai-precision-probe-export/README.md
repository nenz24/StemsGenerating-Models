# Host preparation: long-profile BF16 / expanded-FP32 pair experiment

Actual pretrained layers 0+1 were loaded from the pinned full-SHA256-verified
Base checkpoint, using the publisher's verified class bodies. The pair forward
implementation is shared with the generation shard exporter. Eight fixed
synthetic tensors use the actual long CFG geometry: batch 2, 128 hidden tokens,
118 condition tokens. This is not a full denoiser or generated audio.

The BF16 native asset is copied unchanged from `bass-preview-1024` and checked
against its artifact hashes. Only the expanded-FP32 pair is newly exported with
`coreai-torch`. Its learned parameter values match the original FP32 checkpoint
values; both host comparisons have zero error:

| PyTorch CPU comparison | Max abs / RMSE / normalized RMSE |
| --- | --- |
| BF16 stored, FP32 calculations vs original FP32 | 0 / 0 / 0 |
| Expanded FP32 constants vs BF16 stored | 0 / 0 / 0 |

The local native bundle and every artifact hash were verified after export.
The [small committed receipt](export-summary.json) records actual hashes,
precision checks and staged resource size. Native weights/tensors remain
ignored. A stage plan selects 16 files totaling 761,909,636 bytes; source
bundles for full generation are retained on disk and excluded from this app
profile. Existing complete probe bundles are verified, never overwritten.

The iPhone caller runs BF16-A / FP32-B / BF16-A, four calls per pass with a
single retained model AND function within each pass. It checks strict parity
against the original FP32 host output on all 12 calls. After the
[first device pointwise failure](../coreai-iphone17pro-precision-pointwise-failure/README.md),
finite parity failures are preserved while scheduled timing collection continues;
overall strict PASS still requires all 12 comparisons. There is no extra
warmup, cache deletion/retry or full-loop claim. Swift scopes release between
passes; physical native eviction is unmeasured. It records API timing, thermal
state, low-power mode and sampled process footprint.

Initial host preparation passed 98 regression tests, including seven new integrity/selection tests.
These tests validate contracts, corruption rejection, staging and existing host
gates; they do not emulate CoreAI device execution. No Apple SDK or Swift
compiler is available in this workspace. The first device call built and ran
but failed strict parity. The [subsequent A/B/A run](../coreai-iphone17pro-precision-aba/README.md)
completed all timing calls, with no consistent FP32 speed advantage; strict
parity remained NOT PASSED (0/12). Full-loop behavior and total native memory
remain unverified. The accepted 10.24-second baseline is not replaced.
See the [run instructions](../../docs/coreai-precision-probe.md).

# iPhone four-step feedback Euler loop: PASS

The user uploaded [report.txt](report.txt) after the self-contained fixture fix.
Its bytes are preserved unchanged. [report.json](report.json) records its SHA-256,
all 56 stage timings/footprints and all 16 numerical comparisons. This is
user-reported physical-device evidence, not an independent Linux execution.

The caller reports successful sharded-model validation and SHA-256 validation
of all 16 references and three fixture inputs. Every step executes all 24 layers
plus front/tail, and feeds its actual updated latent into the next step.
All velocity and latent comparisons pass against both the partition and original
FP32 references, with the unchanged combined elementwise and normalized bounds.

| Step | Current → next timestep | Total step | Updated latent normalized RMSE vs original FP32 |
| --- | --- | --- | --- |
| 1 | 1 → 0.75 | 13.406 s | 3.79009e-6 |
| 2 | 0.75 → 0.5 | 13.360 s | 4.92319e-6 |
| 3 | 0.5 → 0.25 | 13.113 s | 7.05395e-6 |
| 4 | 0.25 → 0 | 13.031 s | 7.01433e-6 |

The complete probe takes **53.001 s**. Final latent max absolute error versus
original FP32 is **3.933906555175781e-5**; normalized RMSE is
**7.014327054948485e-6**. The velocity maximum absolute error at step 3 is
0.00017407536506652832; it still passes the combined absolute-plus-relative
bound. This result does not imply a pure absolute error cap of 1e-4.

Performance remains the main unresolved issue in this case: all 48 layer-pair
run+copy intervals are about 1.010–1.144 s and sum to 51.316 s. Model-constructor
timings are mostly about 1.6–1.9 ms. However, `runStateless` also loads/binds the
function before running it, and the enclosing timer includes validation and
CPU output copies. The report does not separate those operations, cache/page-in
work or actual backend execution. Do not describe these intervals as kernel
time or conclude that recompilation, cache eviction or copying is the cause.
The next diagnostic gate is to separate constructor, function binding, backend
execution and detached-output-copy timings before choosing an optimization.
That instrumentation was not implemented by this evidence-only update.
A [subsequent timing caller](../../docs/coreai-loop-timing.md) now exposes these
API-boundary intervals; [device profiling now also passes](../coreai-iphone17pro-denoiser-loop-timing/README.md)
and identifies backend-await as 99.36% of completed-stage time. The work inside
that interval is not yet isolated. This baseline report and its raw timings
are unchanged.

Sampled app process maximum is **74.01760864257812 MiB**, baseline
19.173629760742188 MiB. This is 10 ms sampled `phys_footprint`, not exact peak,
isolated model allocation or total model/device RAM. Whole-device pressure,
shared/file-backed/runtime memory and a long-run plateau remain uncharacterized.
Neither a production memory budget nor a 90-second-song generation time follows
from this report.

This verifies **one four-step synthetic case** under requested `cpuOnly` policy,
guidance=1, shift=1, timestep_r=timestep, without KV cache or CFG. Actual accelerator
placement, cold-launch reliability, conditioning from recordings, waveform
generation and musical quality remain unverified. There is no need to repeat
this same test to record its result.

The [initial manifest failure](../coreai-iphone17pro-denoiser-loop-manifest-failure/README.md)
remains preserved. The corrected package now completes validation and inference;
the exact field that failed in the original generic report is still unknown.
The host manifest's export-time device flag remains unchanged; this separate
device report records the later result.

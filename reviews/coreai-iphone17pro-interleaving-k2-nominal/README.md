# Nominal-thermal K=2: control drift persists

User-supplied [report](report.txt), run `C5E95D16-7F37-48F7-AAC1-D2C79DF82A07`,
started **3 November 2026, 01:54:13 WIB**. iOS 27.0.1 (24A446), h18p,
Release / `-O`, cpuOnly requested; actual placement unmeasured. Grouped first.
Same source manifest, fixed inputs and asset hashes as the two previous runs.

All 140 calls completed: 8 opens, 8 binds, 140 prebound hits, total 13.580 s.
Sampled maximum app footprint 396.830 MiB, not exact peak/total RAM.
Low-power false. **Thermal before and after every call was nominal (0)**.
This addresses the previous fair-thermal condition but does not establish
constant clocks, power, cache or scheduling state.

| Mode | Pair 00–01 mean, calls 2–20 | Pair 02–03 mean, calls 2–20 |
| --- | ---: | ---: |
| Single-before | 54.080 ms | — |
| Pool-only | 54.527 ms | — |
| Grouped | 55.382 ms | 57.436 ms |
| Interleaved | 59.923 ms | 59.948 ms |
| Single-after | 61.384 ms | — |

| Comparison | First, grouped first / fair | Second, interleaved first / fair | Third, grouped first / nominal |
| --- | ---: | ---: | ---: |
| A interleaved/grouped | +6.81% | −4.95% | +8.20% |
| B interleaved/grouped | +3.18% | −6.82% | +4.37% |
| A pool-only/single-before | −1.89% | +2.16% | +0.83% |
| A single-after/single-before | +13.38% | +15.06% | +13.51% |

The earlier mode is faster in all three runs; single-asset control drift
persists at nominal thermal and is larger than either switching comparison.
There is still no isolated, consistent K=2 switching penalty, pool-only
slowdown or explanation of the 772-second full-generation run. Nominal thermal
does not disprove thermal effects, but the category alone cannot explain drift.
Do not average these ordered ratios into a causal effect or infer full-loop
speedup. A larger working set/native residency remains an untested hypothesis.

**Strict parity NOT PASSED: 40/140 pass.** Pair 00–01 fails 100/100 at the same
previous pointwise violation; pair 02–03 passes 40/40. Tolerances/reference are
unchanged. The reported full-output hash per asset matches all modes and all
three runs. Full tensors were not supplied for independent verification.

[Summary](summary.json) records the report SHA-256 and independently recomputed
42 timing statistics, five mode accountings, call positions, parity, thermal
and reported output-hash consistency. No native replay was performed in Linux.

## Next experiment: remove growing live-report work

Source inspection identified a measurement confound: before each call and
native phase, the diagnostic joined and rewrote its entire growing report;
the UI also received that report every call. Report length and corresponding
work increased with call position. Although the file writes are outside the
backend timer, UI rendering/other work can overlap the asynchronous API wait.
This is a hypothesis for control drift, not a diagnosed cause.

The caller now uses short live progress and a separate bounded native-phase
breadcrumb (`last-interleaving-phase.txt`). Detailed call results remain in
memory and the complete report is checkpointed before inference, between
modes, and on normal completion/caught error. Abrupt process termination may
lose current-mode details; the prior mode checkpoint and last phase remain.
The generation caller is untouched. Hashes, copies, parity, guard, lifecycle,
inputs, 140-call schedule and numerical limits remain unchanged.

Update code/build only; reuse the existing K=2 bundle. Run grouped-first once
from nominal thermal and share the diagnostic report. Its header must contain
`Instrumentation revision: bounded-live-progress-v1`. Treat this as a new
instrumentation cohort, not an automatic speedup claim. Inspect whether control
drift changes before moving to K=4; no native weight export is needed for this
calibration. The BF16/audio baseline remains retained.

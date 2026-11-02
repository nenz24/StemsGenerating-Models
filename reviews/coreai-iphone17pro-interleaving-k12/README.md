# Twelve BF16 pairs complete: no large steady-call slowdown reproduced

User-supplied [report](report.txt), run `C5D92375-4DD8-4848-B43F-8F0AD72FB3F6`,
started **3 November 2026, 02:22:53 WIB**. `bounded-live-progress-v1`, grouped
before interleaved, Release / `-O`, iOS 27.0.1 (24A446), h18p, cpuOnly
requested; actual accelerator placement unmeasured. All twelve layer pairs
are present, but front/tail/APG and actual input/feedback are absent.

All **540 calls** complete: 38 opens, 38 binds, 540 prebound hits, total
60.750 s. Low-power false; thermal before/after every call **nominal (0)**.
Sampled maximum app process **153.756 MiB** is not exact peak/total RAM and
does not establish native weight residency or scalability to a full song.

| Pair | Grouped mean, calls 2–20 | Interleaved mean, calls 2–20 | Interleaved/grouped |
| --- | ---: | ---: | ---: |
| 00–01 | 51.886 ms | 55.022 ms | +6.04% |
| 02–03 | 54.070 ms | 54.328 ms | +0.48% |
| 04–05 | 53.688 ms | 54.698 ms | +1.88% |
| 06–07 | 53.607 ms | 54.714 ms | +2.06% |
| 08–09 | 53.728 ms | 55.602 ms | +3.49% |
| 10–11 | 52.814 ms | 55.273 ms | +4.66% |
| 12–13 | 53.526 ms | 55.242 ms | +3.21% |
| 14–15 | 54.406 ms | 54.730 ms | +0.60% |
| 16–17 | 53.798 ms | 54.846 ms | +1.95% |
| 18–19 | 54.222 ms | 55.625 ms | +2.59% |
| 20–21 | 54.268 ms | 56.341 ms | +3.82% |
| 22–23 | 54.332 ms | 56.397 ms | +3.80% |

Single-before A 53.016 ms, pool-only 51.287 ms (−3.26%), single-after
56.009 ms (+5.64%). Switching ratios are small compared with the unexplained
full-generation behavior and overlap the scale of control drift. This single
ordered run does not isolate their causes or prove switching has no effect.

First interleaved A/B calls are 1178.650/1280.894 ms and C is 212.914 ms,
despite earlier grouped execution; later first calls are about 77–86 ms.
This mode-boundary observation must not be hidden in steady-call means, but
does not prove compiler-cache eviction or a native residency mechanism.
Mode total times include first-call/setup effects. backend-await is API wall
time, not isolated kernel time or a full-loop speedup measurement.

**Strict parity NOT PASSED: 400/540 pass, 140 fail.** Pair 00–01 fails 100
calls at the known one-point violation; 14–15 fails 40 calls at the known
six-point violation. The other ten pairs pass 40/40 each, including all four
newly added pairs. No threshold/reference changed. Reported per-asset output
hashes stay consistent across modes, and shared hashes match K=8. Full tensors
were not supplied for independent verification.

[Summary](summary.json) records report SHA-256, 162 independently recomputed
statistics across 27 asset/mode groups, five accountings, exact 540-call order,
14 ratios, strict outcomes per pair, thermal and reported hash consistency.
No native replay was performed in Linux.

## Decision: stop the pair-only pool sweep

K=2/4/8/12 have not reproduced the large repeated-call slowdown of the accepted
772-second generation run. Retain BF16; do not export a full FP32 sampler,
change the baseline, or infer native working-set behavior from process memory.

Next use the [three-step actual-input diagnostic](../../docs/coreai-actual-prefix-probe.md).
It uses the existing generation assets, actual imported piano and complete
source/conditioning, then front + twelve pairs + tail + APG with actual
latent/momentum feedback. Each asset's API phases are recorded per step.
Three scheduled prefix steps mean 45 sampling calls, not a shortened three-step
song-generation algorithm. No decoder, WAV or musical acceptance results from it.
Build/device validation for this new mode remains pending.

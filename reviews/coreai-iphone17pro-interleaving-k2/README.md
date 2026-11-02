# Two distinct BF16 pairs: small switching penalty, larger control drift

User-supplied [report](report.txt), run `F6F60C4F-80F9-41C2-833B-EACBC4B782ED`,
started **3 November 2026, 01:44:59 WIB**. iOS 27.0.1 (24A446), h18p,
Release / `-O`, cpuOnly requested; actual accelerator placement unmeasured.
Fixed synthetic inputs, long CFG geometry; no feedback, denoiser trajectory
or waveform. This is the first K=2 run, with grouped before interleaved.

All 140 calls completed: 8 opens, 8 binds, 140 prebound pool hits, 15.840 s
total. Sampled process maximum 407.049 MiB, not exact peak or total device RAM.
Low-power mode false. Thermal was **fair (1)** before and after all 140 calls,
not nominal. A fixed thermal category does not establish constant clocks.

| Mode | Pair 00–01 mean, calls 2–20 | Pair 02–03 mean, calls 2–20 |
| --- | ---: | ---: |
| Single-before | 56.343 ms | — |
| Pool-only | 55.279 ms | — |
| Grouped | 58.161 ms | 60.557 ms |
| Interleaved | 62.125 ms | 62.486 ms |
| Single-after | 63.884 ms | — |

Interleaved/grouped ratios are **1.06814** and **1.03184**: 6.81% and 3.18%
slower. Pool-only/single-before is 0.98110; no observed pool-only slowdown in
this run. Single-after/single-before is **1.13383**, a 13.38% control drift,
larger than either switching penalty. One ordered run at fair thermal does
not isolate switching, native weight movement, cache/residency or causality.
Grouped and interleaved have the same calls/pool; total wall times still
contain first-call/setup effects. First calls are not guaranteed cold cache.

**Strict parity remains NOT PASSED, 40/140 pass.** Pair 00–01 fails all 100
calls at the same previously reported point `[1,118,1958]`: absolute error
0.00012302398681640625 exceeds bound 0.00011857471466064454. Pair 02–03 passes
all 40 calls. No tolerance/reference changed; finite failures remain failures.
This neither resolves the earlier pointwise discrepancy nor grants numerical
acceptance to the generation baseline.

Unlike the earlier precision report's scalar summaries, this report includes
full-output hashes: one consistent reported SHA-256 per asset across every
mode/call (100 A calls, 40 B calls). This supports output stability under the
tested ordering. The tensors themselves were not uploaded, so hash calculation
and tensor identity were not independently reproduced outside the device.

The [parsed summary](summary.json) records the report hash, independently
recomputed first/mean/median/p95/min/max for seven asset/mode groups (42
statistics), all five mode accountings, 140 call positions, strict outcomes,
thermal states and output-hash consistency. Native execution was not reproduced
in Linux. backend-await is API wall time, not isolated kernel time.

## Decision and next short run

Keep BF16 and the accepted audio baseline. This two-pair component case does
not reproduce or explain the 772-second full-generation case and does not
justify an FP32 sampler export or a full-loop speedup claim.

Use the **existing bundle/build**, with no preparation or model export:

1. Let the iPhone return to nominal thermal; keep the same power conditions.
2. Run **Uji 2 aset BF16 — bergantian dahulu** once.
3. Share the diagnostic report; retain failures and thermal observations.
4. Complete a third K=2 run with grouped first from comparable conditions.

After reviewing the repeated/order-reversed results, decide whether to test
K=4 to approach a larger working set. K=2 does not cover 12 pairs or the full
front/tail/APG pipeline. Preserve strict gates and detached CPU copies.

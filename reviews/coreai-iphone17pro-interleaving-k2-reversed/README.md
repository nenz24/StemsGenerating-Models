# Reversing K=2 mode order reverses the observed timing advantage

User-supplied [report](report.txt), run `E6E1D1F5-795D-4F28-85DE-131F861853A3`,
started **3 November 2026, 01:48:13 WIB**, 3 min 14 s after the first run began.
iOS 27.0.1 (24A446), h18p, Release / `-O`, cpuOnly requested; actual placement
unmeasured. Same source manifest, fixed synthetic inputs and asset hashes as
the [first K=2 run](../coreai-iphone17pro-interleaving-k2/README.md).

All 140 calls completed, 8 opens, 8 binds, 140 prebound hits, total 13.789 s.
Sampled maximum app footprint 391.674 MiB, not exact peak or total device RAM.
Low-power false. Thermal before and after all 140 calls was **fair (1)**;
this was not the requested nominal-thermal control, and does not prove clocks
or cache state were constant.

| Mode | Pair 00–01 mean, calls 2–20 | Pair 02–03 mean, calls 2–20 |
| --- | ---: | ---: |
| Single-before | 54.299 ms | — |
| Pool-only | 55.474 ms | — |
| Interleaved (earlier) | 57.587 ms | 56.958 ms |
| Grouped (later) | 60.589 ms | 61.128 ms |
| Single-after | 62.474 ms | — |

| Repeated mean comparison | Grouped-first run | Interleaved-first run |
| --- | ---: | ---: |
| A interleaved/grouped | 1.06814 (+6.81%) | 0.95046 (−4.95%) |
| B interleaved/grouped | 1.03184 (+3.18%) | 0.93179 (−6.82%) |
| A pool-only/single-before | 0.98110 (−1.89%) | 1.02164 (+2.16%) |
| A single-after/single-before | 1.13383 (+13.38%) | 1.15056 (+15.06%) |

The earlier of grouped/interleaved is faster in both runs. Reversing order
reverses the measured advantage, while single-asset controls drift 13–15%.
These two runs do **not** establish a consistent K=2 switching penalty or a
pool-only penalty. Do not average their ratios into a causal effect or infer
native residency, weight movement, cache eviction or a thermal mechanism.
The data also do not rule out a problem at larger pool sizes/full-loop inputs.
Backend-await is API wall time, not isolated kernel time.

**Strict parity remains NOT PASSED, 40/140 pass.** Pair 00–01 fails 100/100 at
the same pointwise bound violation; pair 02–03 passes 40/40. Tolerances and
references are unchanged. The two reported full-output hashes match across
all modes and across both runs, supporting reported output stability under
ordering changes. Full tensors were not supplied for independent verification.

[Summary](summary.json) records the report SHA-256, independently recomputed
42 timing summary statistics, mode counts/accounting, call positions, parity,
thermal and reported hash consistency. No native replay was performed in Linux.

## Decision

Keep BF16, detached CPU copies and the existing generation baseline. No FP32
sampler export, numerical acceptance or full-loop speedup follows from this.

Complete the planned third K=2 run with **grouped first**, using the same
bundle/build after the iPhone returns to nominal thermal (0), with comparable
power conditions. Share the diagnostic report; a fair-thermal run remains
usable evidence but must be labelled as such. No preparation/export/update
is needed. Review those controls before deciding on K=4; the 772-second full
generation remains unexplained by these short two-pair measurements.

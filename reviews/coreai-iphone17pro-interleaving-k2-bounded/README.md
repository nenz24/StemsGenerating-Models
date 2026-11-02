# Bounded live progress: K=2 control drift falls to 0.92%

User-supplied [report](report.txt), run `24B8C737-E5F4-43AA-8E9C-82F660E5A536`,
started **3 November 2026, 02:03:37 WIB**. Instrumentation
`bounded-live-progress-v1`, Release / `-O`, iOS 27.0.1 (24A446), h18p,
cpuOnly requested; actual accelerator placement unmeasured.

All 140 calls complete: 8 opens, 8 binds, 140 prebound hits, 12.755 s total.
Thermal starts and remains **nominal (0)** for all calls; low-power false.
Sampled maximum app footprint **206.721 MiB**, compared with 396.830 MiB in
the preceding nominal run. These are sampled process observations, not exact
peaks/total device RAM or a controlled memory attribution experiment.

| Mode | Pair 00–01 mean, calls 2–20 | Pair 02–03 mean, calls 2–20 |
| --- | ---: | ---: |
| Single-before | 52.621 ms | — |
| Pool-only | 51.992 ms | — |
| Grouped | 51.818 ms | 53.562 ms |
| Interleaved | 52.482 ms | 52.856 ms |
| Single-after | 53.107 ms | — |

| Repeated mean ratio | Earlier nominal / growing report | Bounded live progress |
| --- | ---: | ---: |
| A interleaved/grouped | 1.08200 (+8.20%) | 1.01280 (+1.28%) |
| B interleaved/grouped | 1.04373 (+4.37%) | 0.98681 (−1.32%) |
| A pool-only/single-before | 1.00826 (+0.83%) | 0.98806 (−1.19%) |
| A single-after/single-before | 1.13506 (+13.51%) | 1.00923 (+0.92%) |

The prior 13–15% end-control drift is no longer observed in this single run.
This supports the reporting-work confound hypothesis and gives a better K=2
calibration. It does not independently prove the cause: run order, clocks,
cache and other process conditions are not fully controlled. First A call is
759.660 ms here versus 83.223 ms in the prior nominal run; first B call is
762.158 ms. Neither run is a controlled cold-cache measurement. Do not infer
full-loop speedup from total time, or pool-size scalability from two pairs.

**Strict parity remains NOT PASSED, 40/140 pass.** Pair 00–01 fails 100/100
at the same previously reported point; pair 02–03 passes 40/40. Both reported
full-output hashes match all previous K=2 runs. References, bounds, assets,
copies and sampling baseline are unchanged. Full tensors were not supplied
for independent verification; native execution was not replayed in Linux.

The [summary](summary.json) records report SHA-256, 42 independently recomputed
timing statistics, all five mode accountings, 140 call positions, four ratios,
thermal states and reported output-hash consistency. backend-await is API
wall time, not kernel time. Timings remain diagnostic given strict failures.

## Decision: next short experiment K=4

Retain BF16. Proceed to four distinct native pairs, layers 00–07, with the
same bounded-progress instrumentation. This tests a larger working set without
exporting new native weights. It is not the full 12-pair/front/tail/APG loop.

The preparation driver now accepts `--pairs 4 --expand-existing`. It fully
prepares/verifies a new bundle first, requires unchanged hashes for all shared
assets/inputs/references and the same source manifest, then archives K=2 under
`.runtime-cache/coreai-interleaving-archives/`. Installation failure restores
the old bundle when the destination remains free. Source generation bundles,
checkpoint and caches are retained; no bundle deletion or overwrite in place.
Re-running the command with K=4 already installed verifies/reuses it.

Close Xcode, update, and prepare:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_interleaving_probe.sh --pairs 4 --expand-existing &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Wait for Ready. Build Release Reuse on iPhone, start from nominal thermal, then
run **Uji 4 aset BF16 — berkelompok dahulu** once and share the report.
Expected 220 calls, 14 opens/14 binds, 220 prebound pool hits. No warmup, purge,
retry or waveform generation. A guard/native failure is a result to retain,
not permission for repeated retries. Do not jump to K=8/12 yet.

Eleven focused host tests pass, including archive preservation, install-failure
rollback, preparation-failure preservation, changed shared-payload rejection
and explicit-flag enforcement. Python/shell syntax and whitespace checks pass.
Actual K=4 host preparation and device timings remain pending.

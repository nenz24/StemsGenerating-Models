# Eight BF16 pairs: repeated calls remain short; another pair fails strict parity

User-supplied [report](report.txt), run `37A44012-1237-47A5-840A-F86913776337`,
started **3 November 2026, 02:16:54 WIB**. `bounded-live-progress-v1`, grouped
before interleaved, Release / `-O`, iOS 27.0.1 (24A446), h18p. cpuOnly
requested; actual accelerator placement unmeasured. Eight distinct native
pairs cover layers 00–15, fixed synthetic long-CFG input without feedback.

All **380 calls** completed: 26 opens, 26 binds, 380 prebound pool hits,
39.123 s total. Thermal before and after all calls **nominal (0)**, low-power
false. Sampled maximum app process **97.677 MiB** is not exact peak/total RAM.
Its lower value than K=4 cannot establish less model memory, residency or
better scalability; retained Swift handles do not prove resident native weights.

| Pair | Grouped mean, calls 2–20 | Interleaved mean, calls 2–20 | Interleaved/grouped |
| --- | ---: | ---: | ---: |
| 00–01 | 52.421 ms | 53.116 ms | +1.32% |
| 02–03 | 53.114 ms | 54.794 ms | +3.16% |
| 04–05 | 52.707 ms | 54.361 ms | +3.14% |
| 06–07 | 53.416 ms | 54.571 ms | +2.16% |
| 08–09 | 53.697 ms | 55.582 ms | +3.51% |
| 10–11 | 53.398 ms | 54.890 ms | +2.79% |
| 12–13 | 53.934 ms | 54.434 ms | +0.93% |
| 14–15 | 53.960 ms | 54.854 ms | +1.66% |

Single-before A 52.931 ms; pool-only 51.620 ms (−2.48%); single-after
55.298 ms (+4.47%). All switching ratios are modestly positive, but smaller
than end-control drift in this one ordered run. This is not an isolated
switching effect or an explanation of the 772-second full-generation case.
The K=2/K=4/K=8 component cases have not reproduced a large repeated-call
slowdown. Larger pool, different actual inputs and front/tail/APG remain open.

First grouped calls for B…H range 699–1237 ms; interleaved first calls are
73–85 ms. Mode wall-time differences include these effects, not just
switching. Fresh launch/first inference is not guaranteed cold compiler cache.
backend-await is API wall time, not kernel time or full-loop throughput.

## Numerical result remains NOT PASSED

**240/380 calls pass; 140 fail.** This is diagnostic timing, not acceptance.

| Pair | Passing calls | Failing calls | Pointwise violations per failing output |
| --- | ---: | ---: | ---: |
| 00–01 | 0 | 100 | 1 / 524,288, unchanged previous point |
| 02–03 through 12–13 (six pairs) | 240 | 0 | — |
| 14–15 | 0 | 40 | 6 / 524,288 |

Newly tested pair 14–15 fails in both grouped and interleaved, with one
reported output hash throughout. Its worst bound ratio is 1.4996526526 at
`[0,96,1227]`, flat index 197835: actual −0.026519060134887695, reference
−0.02670907974243164, absolute error 0.0001900196075439453 versus bound
0.00012670907974243165. This newly observed component failure is not evidence
of a regression caused by switching or pool expansion. Do not relax the
unchanged abs/rel/RMSE limits or replace the original host reference.

Each asset has one consistent reported full-output SHA-256 across all modes;
the four shared hashes match K=4. Full tensors were not supplied for independent
verification. [Summary](summary.json) records the report hash, 114 independently
recomputed timing statistics across 19 asset/mode groups, five accountings,
exact 380-call order/positions, ten ratios, per-asset strict outcomes, thermal
and reported hash consistency. No native replay was performed in Linux.

## Decision: finish the pair-pool sweep at K=12

Retain BF16 and the listening-approved generation baseline. Use the existing
driver and bounded progress to test all twelve pairs, layers 00–23. This still
excludes front, tail and APG and does not validate a full denoiser trajectory.
No new caller, precision change or native model export is needed.

Close Xcode, then:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_interleaving_probe.sh --pairs 12 --expand-existing &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

K=8 is archived only after the larger bundle is verified with unchanged
shared payload/source hashes. Wait for Ready, build Release Reuse, then run
**Uji 12 aset BF16 — berkelompok dahulu** once from nominal thermal and share
the report. Target **540 calls, 38 opens/38 binds, 540 prebound pool hits**.
Retain any failure report; no automatic retry/purge. Actual K=12 preparation
and device timing remain pending. After this sweep, choose targeted full-loop
stage/input diagnosis rather than infer a speedup from component results.

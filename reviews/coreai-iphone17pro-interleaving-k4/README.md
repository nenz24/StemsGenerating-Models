# Four BF16 pairs: no large switching penalty in this run

User-supplied [report](report.txt), run `FC2792D8-6C75-4E82-B3BD-2FEA71D36B4A`,
started **3 November 2026, 02:11:27 WIB**. Instrumentation
`bounded-live-progress-v1`, grouped before interleaved, Release / `-O`,
iOS 27.0.1 (24A446), h18p. cpuOnly requested; actual placement unmeasured.
Four distinct native pairs cover layers 00–07 at the long CFG geometry,
fixed synthetic input, no output feedback or waveform generation.

All **220 calls** completed: 14 opens, 14 binds, 220 prebound pool hits,
20.593 s total. Thermal before/after every call **nominal (0)**, low-power
false. Sampled maximum app process **202.502 MiB**, versus 206.721 MiB in the
bounded K=2 run. This is not exact peak/total RAM or evidence that retaining
more handles has no native-memory cost; clocks/cache/residency are unmeasured.

| Pair | Grouped mean, calls 2–20 | Interleaved mean, calls 2–20 | Interleaved/grouped |
| --- | ---: | ---: | ---: |
| 00–01 | 51.432 ms | 52.058 ms | 1.01216 (+1.22%) |
| 02–03 | 53.153 ms | 53.107 ms | 0.99914 (−0.09%) |
| 04–05 | 53.194 ms | 52.053 ms | 0.97855 (−2.15%) |
| 06–07 | 53.794 ms | 52.286 ms | 0.97196 (−2.80%) |

Single-before A mean is 52.795 ms, pool-only 51.852 ms (−1.79%), and
single-after 53.683 ms (+1.68%). Four-pair repeated API waits remain about
51–54 ms, comparable in scale to bounded K=2; no large switching slowdown
or pool-only slowdown is observed here. These small differences from one
ordered run are not proof of an optimization or no effect at larger K.

First calls differ substantially: single-before A 751.339 ms; grouped B/C/D
760.750/715.043/747.258 ms; interleaved first calls about 75–78 ms. Therefore
the lower total interleaved mode wall time is not an isolated switching
speedup. Neither first call nor fresh launch guarantees a cold compiler cache.
backend-await is API wall time, not isolated kernel time.

**Strict parity remains NOT PASSED, 120/220 pass.** Pair 00–01 fails all 100
calls with the unchanged previously observed pointwise discrepancy. Each of
02–03, 04–05 and 06–07 passes 40/40. Every asset has one consistent reported
full-output hash across modes/calls; the shared A/B hashes match bounded K=2.
Full tensors were not supplied for independent verification. No tolerance,
reference or accepted generation baseline was replaced.

[Summary](summary.json) records report SHA-256, 66 independently recomputed
timing statistics across eleven asset/mode groups, five accountings, exact
220-call order/positions, six ratios, thermal states and reported output-hash
consistency. Native execution was not reproduced in Linux.

## Decision: K=8 with the existing driver

Keep BF16 and bounded live progress. K=4 does not reproduce the 772-second
full-generation behavior; a larger working set remains an untested hypothesis.
Proceed to K=8 (layers 00–15), not the full 12-pair/front/tail/APG pipeline.
No caller/precision change or native weight export is needed.

Close Xcode and run:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_interleaving_probe.sh --pairs 8 --expand-existing &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

The driver verifies the new bundle before archiving K=4, requires unchanged
shared payload/source hashes, and retains generation/checkpoint/cache data.
Wait for Ready, build Release Reuse, then run **Uji 8 aset BF16 — berkelompok
dahulu** once from nominal thermal. Share the report, including a failure if
one occurs. Target **380 calls, 26 opens/26 binds, 380 prebound pool hits**.
Do not jump to K=12 or retry native failures automatically. Actual K=8 host
preparation and device timings remain pending; this is no full-loop speedup
or numerical-acceptance claim.

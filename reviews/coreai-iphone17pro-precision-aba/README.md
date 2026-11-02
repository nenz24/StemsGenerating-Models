# BF16 / expanded-FP32 A/B/A: timing collected, strict parity fails

User-supplied [report](report.txt), run `EDC1C13C-300F-4E20-B5D6-98D07A736C52`,
started **3 November 2026, 00:24:28 WIB** (2 November 17:24:28 UTC).
iOS 27.0.1 (24A446), h18p, Release / Swift `-O`; cpuOnly requested, actual
accelerator placement unmeasured. The report identifies fixed synthetic inputs
at the long CFG geometry, not a generated waveform or complete denoiser.

| Pass | First backend-await | Mean backend-await, calls 2–4 |
| --- | ---: | ---: |
| BF16 A1 | 752.458 ms | 55.964 ms |
| Expanded FP32 B | 72.518 ms | 53.533 ms |
| BF16 A2 | 80.794 ms | 53.336 ms |

All twelve scheduled calls completed: 3 opens, 3 binds, 9 retained-handle hits.
Total probe time 2.849699 seconds; sampled app process maximum 222.565 MiB.
Low-power mode was reported false and every after-call thermal state nominal.
These are API wall intervals and sampled process observations, not isolated
kernel time, exact peak/total RAM or controlled cold-cache measurements.

FP32 has no consistent repeated-call advantage in this experiment: A1/B ratio
1.0454, A2/B ratio 0.99631. With only three repeated observations per pass,
order/cache state and other runtime factors are not isolated. Do not expand
the complete sampler to FP32 based on these measurements. Preserve the accepted
BF16 generation baseline while investigating the multi-asset case.

**Strict parity remains NOT PASSED, 0/12 calls pass.** All twelve calls report
the same summary metrics and single violating point:

| Pointwise observation | Value |
| --- | --- |
| Violations per output | 1 / 524,288 (0.000190735%) |
| Index / flat index | `[1,118,1958]` / 505766 |
| Actual | −0.018451690673828125 |
| Original FP32 host reference | −0.01857471466064453 |
| Absolute error | 0.00012302398681640625 |
| Allowed `1e-4 + 1e-3*abs(reference)` | 0.00011857471466064454 |
| Error above bound | 0.00000444927215576171 (3.7523%) |
| Global normalized RMSE | 0.000002143414100761529 |

The worst relative-to-bound point differs from the location of the global
maximum absolute error, which is 0.00311279296875. A small normalized RMSE does
not override the failed pointwise gate. FP32 storage does not remove this
reported violation. No tolerance, reference, input or output was modified.
Full output tensors/hashes were not supplied, so identical reported metrics
and a matching scalar are not asserted to prove bit-identical full tensors.
The numerical cause is not established by this report.

The [parsed summary](summary.json) checks all reported means and independently
recomputes the worst scalar's bound, excess and index. The report's source
hash is recorded. No native execution is reproduced in the Linux workspace.

The component result does not reproduce the earlier 772-second full-generation
conditions: one pair is active per pass, fixed input tensors are reused, and
cache/residency state differs. A full-loop speedup cannot be inferred. The next
useful comparison is fixed-input execution with different retained-handle pool
sizes and interleaved distinct pair assets. A larger working set, native weight
movement and switching assets are hypotheses, not diagnosed causes. The current
experiment does not justify another full FP32 model export or threshold change.

The accepted 10.24-second audio baseline is retained in its existing listening
scope; this report neither replaces it nor grants strict numerical acceptance.

# Actual-input prefix: step 3 becomes short after two expensive passes

User-supplied [report](report.txt), run `4914BAE7-AD6D-48CE-BD4E-7BA49B291480`,
started **3 November 2026, 02:36:24 WIB**. `actual-prefix-bounded-v1`, Release
`-O` / DEBUG OFF, iOS 27.0.1 (24A446), h18p, cpuOnly requested; actual placement
unmeasured. Both recording and imported PCM hashes match the accepted
[772-second baseline](../coreai-iphone17pro-audio-import-1024/README.md).

Source encoding completes in 2.369 s, conditioning in 6.944 s with six preset
checks PASS. Actual feedback runs for the first three positions of the original
50-step schedule. **45 calls, 30 opens/binds, 15 hits**, targeted repair 0/1.
All 45 after-call thermal states are nominal; initial nominal, low-power false.

| Step | Wall time | Sum backend-await | Sum output-copy | Pair backend range |
| --- | ---: | ---: | ---: | ---: |
| 1, single-stage scopes | 14.858 s | 14.482 s | 238.868 ms | 1141.805–1246.428 ms |
| 2, first retained calls | 14.931 s | 14.596 s | 238.810 ms | 1128.460–1246.361 ms |
| 3, reused handles | **1.069 s** | **760.580 ms** | 247.421 ms | **54.295–78.764 ms** |

Step 3 front backend-await is 15.939 ms; tail 1.726 ms, APG 0.196 ms. Tail/APG
are not the large third-step cost here. The accepted run's step 3 was 14.998 s;
later steps remained around 15.19 s. This one short actual-input retained step
does not establish stability through step 50 or prove logging caused the change.
Cache/compiler state, order and runtime conditions remain uncontrolled.

Total prefix pipeline 40.240 s; sampled maximum app process 775.909 MiB, also
observed in source encoding. This is not exact peak/total model RAM or full-song
scaling evidence. backend-await is API wall time, not kernel time. First calls
are not guaranteed cold; do not extrapolate throughput from one warm step.

**Imported-output parity NOT EVALUATED, 0/0 checks.** Conditioning 6/6 PASS has
its separate scope. Host bundle remains `trajectory_drift_diagnostic`, full
numerical gate NOT PASSED. Pair sweep failures remain failures. No decoder/WAV,
threshold/reference/weight change or accepted-audio replacement occurred.

[Summary](summary.json) records report hash, exact 45-call order, 90 per-asset
and 18 aggregate phase-sum checks, accounting, repair and both baseline input
hashes independently checked. No output tensors or native replay in Linux.

## Next: complete generation with bounded progress

Production still rendered/wrote its growing report per sampling call/native
phase, while the diagnostic used short live progress. Apply bounded sampling
and decoder progress to the long production profile too. Complete reports are
checkpointed per step; a separate native breadcrumb keeps the last phase.
Abrupt termination may lose current-step details. Each step now records API
backend/copy deltas, thermal and sampled footprint. Short-profile reporting
is unchanged. Header: `Instrumentation revision: generation-bounded-v1`.

All **50 steps, 750 sampling calls + one decoder**, first-step single scopes
then retention, BF16/FP32 contract, copies, schedule/guidance/APG, gates and
targeted repair budget remain. This is observer work, not a new sampler.

Close Xcode, update code/build only, retaining active generation resources:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Build CoreAIProbe Reuse, import the same piano, then press **Generate bass dari
rekaman** once from nominal thermal. Use normal generation, not the prefix
button. Collect 50/50, report and piano/bass/mix WAVs. Check step 3–50 behavior,
750 calls, separate decoder, input identities and output hashes. No-repair
sampling counts remain 30 opens/30 binds/720 hits.

Baseline bass WAV: `8f423449cb287daeaa053e77272184fcf501b61b53e633321545d97b108f2315`.
Baseline mix WAV: `cb6209981c6c02521bf9ae22e9b3071ad8dfb10a0956e77f039553d93a8bb271`.
These are comparison evidence, not numerical-acceptance overrides.

17 host regressions and static branch/accounting/whitespace checks pass. New
Apple build, full-generation performance and WAV identity remain pending;
no production latency improvement is claimed yet.

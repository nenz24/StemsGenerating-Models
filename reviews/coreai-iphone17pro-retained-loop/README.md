# Retained full loop: both planned stage-1 experiments PASS

The user uploaded [report.txt](report.txt) from **Uji loop 24 layer dengan reuse**.
Its bytes are preserved unchanged. [report.json](report.json) records its SHA-256,
all 56 stage profiles, retention events and 16 numerical comparisons.

All four feedback steps pass both references, with exactly **14 model opens,
14 function binds, 42 reuse hits and 56 completed calls**. The footprint guard
did not trigger. Final latent normalized RMSE against original FP32 remains
7.014327054948485e-6, max absolute error 3.933906555175781e-5. These summaries
match the nonretained baseline; output hashes were not emitted, so byte-identical
outputs are not established.

| Step | Wall time | Backend-await total | Opens / binds |
| --- | --- | --- | --- |
| 1 | 13.428 s | 13.223 s | 14 / 14 |
| 2 | 3.931 s | 3.836 s | 0 / 0 |
| 3 | 0.503 s | 0.165 s | 0 / 0 |
| 4 | 0.526 s | 0.185 s | 0 / 0 |

Total probe time is **18.521768875 s**, versus 51.880772958 s in the separate
nonretained profile: an observed ratio of 2.80107 for these reports. Separate
device runs are not a controlled repeatability study. The earlier pair's 122×
first/repeated ratio did not become a 122× whole-loop speedup.

Step 2 still has long backend-await intervals in front and the first three
layer pairs despite no repeated open/bind; their causes are not isolated.
Retained handles do not prove physical weight residency, a cold first call or
isolated kernel timing. Steps 3–4 establish faster forwards for this case,
not a long-run steady-state plateau.

Phase sums reconcile with stage envelopes, step and loop totals. Completed-stage
measured time is 17.653890209 s, outside-stage time 0.867878666 s, including
reporting and Swift pool release. Native cache eviction is not measured.
Sampled process maximum is **174.27179718017578 MiB**, baseline
23.689231872558594 MiB. These 10 ms footprint samples are not exact peaks or
total model/device RAM. Larger contexts, additional conditioning/codec models,
full-song timing and thermal/battery budgets remain unvalidated.

**Decision: close stage 1 for prototype execution-route selection.** Both planned
main experiments passed. Use separate assets, retain model/function handles
lazily during sampling, copy outputs to fresh CPU NDArrays and release the Swift
pool before later components. Continue to recording/target conditioning; no
additional synthetic stage-1 benchmark is required for this prototype decision.
Production/full-song feasibility remains open.

This is user-reported physical iPhone execution under requested `cpuOnly` policy,
not independent Linux execution or measured accelerator placement. Inputs are
synthetic, guidance=1, shift=1, without CFG/KV cache or waveform output. The
[roadmap](../../docs/iphone-generation-roadmap.md) now enters stage 2.

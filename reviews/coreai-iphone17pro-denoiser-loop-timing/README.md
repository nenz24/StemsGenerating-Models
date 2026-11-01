# iPhone loop phase timing: backend-await dominates, dual parity PASS

The user supplied [report.txt](report.txt) from **Profil waktu loop 4 langkah**.
Its bytes are preserved unchanged. [report.json](report.json) records its SHA-256,
56 stage profiles, step totals and all 16 numerical comparisons. This is
user-reported device evidence, not independently executed on Linux.

All four feedback steps, 56 asset stages and 16 velocity/latent comparisons pass.
Final latent normalized RMSE versus original FP32 is 7.014327054948485e-6;
maximum absolute error is 3.933906555175781e-5. The reported numerical summaries
match the earlier unprofiled loop; output hashes were not emitted, so this does
not establish byte-identical outputs.

| Interval | Total |
| --- | --- |
| Model asset validation/open | 0.10017 s |
| Function binding | 0.09917 s |
| Input validation | 0.00054 s |
| Await backend run | **50.74635 s** |
| Output validation/read-back/copy | 0.12350 s |
| Other work inside run envelope | 0.00237 s |
| Outside completed-stage intervals | 0.80868 s |
| Entire loop probe | **51.88077 s** |

`backend-await` accounts for **99.36218% of completed-stage measured time**,
or **97.81340% of total loop wall time**. All per-stage phase sums reconcile
with their original load/run envelopes, step totals reconcile with stage sums,
and the completed-stage total plus outside-stage time reconciles with wall time.

At these API boundaries, model opening, binding and output copying are small
contributors. The priority is understanding the `function.run` interval. It can
include backend initialization, deferred work, paging and async scheduling as
well as execution. This report does not separate these from steady-state
computation and does not measure accelerator placement or cache residency.
Do not label 50.75 seconds as pure kernel time or diagnose a particular backend
mechanism from the timer alone.

The next bounded experiment is first versus repeated execution of **one layer
pair**, keeping both its model and loaded function alive, with fixed inputs and
numerical checks. This will test whether repeated calls amortize first-use work
without holding all 14 models at once. That experiment is not implemented by
this report. The existing short/two-block warm probes are historical component
evidence with different cases, not direct controls for this sharded loop.

The earlier loop took 53.001 seconds; this one takes 51.881 seconds. There was
no deliberate execution optimization, and device/cache/thermal conditions were
not controlled. The difference is not a verified speedup. Sampled process
maximum is 127.54878997802734 MiB from baseline 38.18925476074219 MiB, neither
isolated model allocation nor total model/device RAM. No production memory,
cold-launch reliability, full-song timing or instrument-generation claim follows.

The report supplies user evidence that the new timing code compiled and ran on
iPhone. The [53-second baseline](../coreai-iphone17pro-denoiser-loop/README.md)
and all previous failures remain preserved; model assets and numerical fixtures
were not changed in this evidence-only update.

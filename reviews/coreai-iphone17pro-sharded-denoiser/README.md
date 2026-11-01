# iPhone: complete sharded denoiser, failure followed by PASS

Subsequent evidence: [the three-attempt wrapper passed 3/3 on iPhone](../coreai-iphone17pro-sharded-denoiser-stability/README.md).
The reports below preserve the earlier failure and retry as originally supplied.

The user supplied both reports and confirmed that the second attempt came from
pressing the button again while the **same app stayed open**, without a relaunch.
These are user-reported physical-device results, not Linux device emulation.

| Attempt | Completed | Dual parity | Total probe | Sampled process maximum |
| --- | --- | --- | --- | --- |
| [First](attempt-1.txt) | 12/14 | Not passed: BNNS graph load failure at stage 13 | 9625.73 ms | 742.55 MiB |
| [Second](attempt-2.txt) | 14/14 | PASS against both references | 3785.89 ms | 40.22 MiB |

[report.json](report.json) retains per-stage timings and the scope of each result.
The second attempt executes all 24 layers, front and tail, on synthetic inputs.
Its normalized RMSE against the original FP32 reference is 3.3040618e-6,
within the unchanged bounds. This confirms **one complete velocity forward**.
It does not establish repeatability, cold-launch reliability or music generation.

The large change in constructor timings (hundreds of milliseconds to about
1–3 ms for most stages) is consistent with cache/initialization effects. However,
the second attempt also spends roughly 900 ms in execution/copy for some early
pairs. Binding, deferred initialization and paging can occur outside the measured
constructor phase; these timings do not isolate compilation or kernel time.
Neither report measures cache residency or identifies the first failure's root
cause. A successful unchanged rerun does not support declaring the last pair's
weights invalid or a deterministic unsupported operator. No cache was deleted
as a purported fix. The original failure remains part of the evidence.

The 40.22 MiB figure is sampled app `phys_footprint`, **not model RAM usage**.
It does not account for all shared, file-backed or runtime-managed resources,
does not measure whole-device pressure, and can miss peaks between samples.
The model assets still total about 3.15 GB. Do not infer a production memory
budget or eliminate the previous memory concern from this number.

The new **Uji kestabilan 3x** button repeats the exact existing forward three
times within one process, with fresh input tensors and no cache clearing. All
attempts and their errors are retained in the displayed/persisted report; a
failed preparation/manifest or footprint guard stops remaining attempts. Three
dual-parity passes are required for its PASS. This is a same-process repetition
test, not three cold starts. The reports above predate that wrapper; its later
device execution is recorded in the subsequent evidence linked above.

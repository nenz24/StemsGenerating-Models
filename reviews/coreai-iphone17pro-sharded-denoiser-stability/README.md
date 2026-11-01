# iPhone: sharded denoiser same-process repeatability PASS 3/3

The user supplied [report.txt](report.txt) from **Uji kestabilan 3x**. Its bytes
are preserved unchanged; [report.json](report.json) records its SHA-256, all 42
stage timings, per-attempt comparisons and measurement limits. This is device
evidence reported by the user, not an independent execution on Linux.

| Attempt | Stages | Dual parity | Total probe | Sampled process maximum |
| --- | --- | --- | --- | --- |
| 1 | 14/14 | PASS | 4681.29 ms | 35.72 MiB |
| 2 | 14/14 | PASS | 3463.26 ms | 60.35 MiB |
| 3 | 14/14 | PASS | 2044.49 ms | 68.35 MiB |

All three forwards execute the full 24 layers plus front/tail on the same
synthetic inputs with fresh input tensors. They pass both numerical references;
the reported original-FP32 normalized RMSE is 3.3040617978036044e-6 each time.
Identical summary metrics do not prove byte-identical outputs: output hashes
are not emitted. This verifies repetition of this case in one app process,
without cache clearing or stage retries. It also supplies user evidence that the
new wrapper built and ran on iPhone. No model export or runtime change was
needed for this result.

Latency is still variable. Stage run+copy durations exceed 500 ms in eight
places, with the affected stages changing between attempts. Most other pairs
take about 34–58 ms. Constructor timings remain about 1–2 ms after the first
front stage. The execution interval includes function binding and any deferred
backend work as well as output copies. Cache effects, paging or deferred work
are possible explanations, not measured causes. Total probe time also includes
reference reading, bookkeeping and progress reporting. Do not extrapolate this
three-run speedup into a production generation-time guarantee.

Sampled process maxima rise from 35.72 to 60.35 to 68.35 MiB; baselines also rise
from 22.78 to 27.80 to 41.35 MiB. Three observations do not establish a memory
leak or a stable plateau. These are 10 ms samples of app `phys_footprint`, not an
exact peak or total model/device memory. Shared, file-backed and runtime-owned
resources are not fully characterized. The earlier monolithic memory failure
and first sharded BNNS failure remain relevant historical evidence.

This result does **not** establish cold-launch reliability, different-input or
different-timestep parity, a diffusion loop, conditioning from recordings or
instrument generation. The earlier failed attempt and successful retry remain
in [the original report](../coreai-iphone17pro-sharded-denoiser/README.md);
their failure cause is still unconfirmed.

The next numerical gate is a short multi-timestep denoising loop: feed each
updated latent into the next full forward, use the pinned ACE timestep/update
semantics and compare intermediate/final latents against host references. Record
per-step timing and process footprint to assess drift and accumulation. This
requires new loop fixtures/caller work; it is not implemented by this report.
Conditioning from piano/text and waveform generation remain subsequent work.

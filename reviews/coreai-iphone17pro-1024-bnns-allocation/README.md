# 10.24-second source CPU sampling: compiler allocation failure

User-provided device console: iPhone 17 Pro, iOS 27.0.1, CoreAI h18p.
The log explicitly requests cpuOnly; actual accelerator placement is unmeasured.

| Last native operation | App footprint before operation, MiB |
| --- | ---: |
| `front.aimodel` output readback | 243.2847671508789 |
| `blocks-00-01.aimodel` output readback | 305.8323974609375 |
| `blocks-02-03.aimodel` model-open/specialization | 305.8323974609375 |

At the last operation BNNS reported:

```text
BNNS Graph Compile: mmap of bnns ir file for writing failed with error: Cannot allocate memory
.../Library/Caches/coreai-cache/tmp/FBE4DA2E02A839400E03CD36556D0FF7B0070ED6392741EC9BE550BC869B0758/model.aimodelx/main-this-delegates/BNNS/bnns.ir
```

The user reports the app stopped advancing at 0/50 sampling steps. Two of
fourteen denoiser assets completed in step 1; the remaining denoiser stages and
APG had not completed. No waveform or complete forward is established by this
log. The footprint observations do not bound compiler transient allocations,
native residency or total device memory. PointerUI/LaunchServices messages and
the AVAudioSession main-thread warning do not establish the BNNS cause.

The new mitigation uses one Swift model/function scope per asset for the first
actual 10.24-second sampling step, then retains sampling handles from step 2.
It changes ownership and open/bind accounting, not weights, inputs, detached CPU
copies, 50-step feedback, tolerances or numerical status. It adds no warmup or
automatic retry and does not delete caches. Expected sampling accounting:
30 opens, 30 binds, 720 hits, 750 calls; decoder is a separate call afterward.

This is a pending hardware experiment. Release of Swift ownership does not
guarantee native cache eviction or prevent another compiler allocation failure.
No Apple build or device execution is available in the Linux workspace.

Validation: seven existing window/selection/precision integrity tests and six
trajectory-diagnostic gate tests pass. The patch changes only Swift lifecycle
selection and reporting plus documentation; native assets are reused. These
host checks do not exercise the Swift lifecycle or the BNNS compiler.

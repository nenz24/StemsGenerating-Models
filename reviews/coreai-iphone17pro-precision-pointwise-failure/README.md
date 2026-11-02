# Precision experiment: first BF16 call fails the pointwise gate

User-provided run `070EE3E4-4DFF-4E8A-9F15-8A7EA352C60F`, started
2026-11-02 17:05:29 UTC / **2026-11-03 00:05:29 WIB**. The inline report is
transcribed in [report.txt](report.txt); it was not supplied as an original
file blob. Release / Swift `-O`, iOS 27.0.1 (24A446), h18p, cpuOnly requested.

Both assets, fixed tensors and original FP32 host reference passed reported
hash validation. The first BF16 model opened, bound and completed inference:

| Observation | Value |
| --- | ---: |
| Model-open / function-bind | 1.316792 / 1.369458 ms |
| First backend-await | 69.589667 ms |
| Output-copy | 18.572334 ms |
| Max absolute error | 0.00311279296875 |
| Normalized RMSE | 0.000002143414100761529 |
| Strict comparison | NOT PASSED |
| Completed calls / opens / binds / hits | 1 / 1 / 1 / 0 |
| Sampled process maximum | 58.17401885986328 MiB |
| Total probe time | 485.386667 ms |
| Low-power mode / thermal after call | false / 0 (nominal) |

The normalized-RMSE bound of 0.001 passed, so the existing combined comparator
failed its per-element bound `1e-4 + 1e-3*abs(reference)`. The number, location
and values of violating elements were not logged; neither the cause nor a
particular near-zero/cancellation explanation is established. Maximum absolute
error alone cannot identify the element with the worst relative-to-bound error.

The caller aborted on parity failure, so no FP32 result or repeated-call timing
was collected. This is not a specialization, memory or file-loading failure.
The first call's 69.59 ms cannot be treated as a full-loop speedup relative to
the earlier 772-second generation: input contents, retained-model scope and
cache state differ. Thermal and low-power observations apply to this short
attempt, not to the earlier full generation.

The accompanying console contains two BF16 open/bind/run/readback sequences;
only one is identified by the report. They are not merged into one probe run.
PointerUI/LaunchServices messages do not establish the numerical cause.

The corrected diagnostic caller keeps the same native assets, fixed inputs,
host reference and strict tolerances. It records every strict failure and
continues the twelve scheduled timing calls only for finite, correctly shaped
outputs. It logs violation count/fraction and the worst bound ratio, index,
actual/reference values, absolute error and bound. Timing completion is
separate from numerical PASS; any failed check keeps overall strict parity
NOT PASSED and prevents a sampler-adoption claim. Hash/type/shape/nonfinite,
native errors and memory guard failures still stop execution. No retry,
warmup, cache deletion, re-export, waveform generation or baseline replacement.

The existing 10.24-second listening baseline remains accepted in its original
scope. This synthetic component failure is retained as failed evidence.
The modified Swift caller still requires verification on the user's iPhone.

Follow-up validation: 21 focused host contract/gate tests pass, including a
numeric regression where normalized RMSE passes but a small error at a zero
reference fails the unchanged pointwise bound. The existing native bundle and
every artifact hash verify without re-export. These checks do not execute the
modified Swift diagnostic loop; Apple build/device verification remains pending.

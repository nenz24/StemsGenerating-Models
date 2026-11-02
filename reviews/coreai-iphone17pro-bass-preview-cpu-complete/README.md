# First observed complete iPhone bass preview execution

The supplied [console](console.txt) records exactly one ordered sequence of
**50 rounds of 14 denoiser shards + one APG update**, followed by **one VAE
decoder**. Each of the 751 calls reaches model-open, bind/validation, memory
guard, backend-await, readback and Swift scope release: **4506 phase markers**.
All 3755 markers that log a compute request say `cpuOnly requested: true` and
name source `.aimodel` resources. This is an API request, not an independent
accelerator-placement measurement. See the [derived evidence summary](summary.json).

The console then references the output `Documents/bass-preview-…/bass.wav`
in share-metadata requests. In response to the follow-up questions, the user
confirmed **“Aplikasi masih terbuka”** and **“Berhasil terdengar”** for the
**Dengarkan bass** button. Together with the complete ordered model calls and
the implemented WAV writer/return path, this supports a first successful
**on-device generation and playback execution milestone** for the 5.12-second
preview. WAV bytes were subsequently supplied and [audited separately](../coreai-iphone17pro-bass-preview-waveforms/README.md).
The next pasted report initially covered conditioning only. The final
integrated report was subsequently supplied; its measurements are recorded below.

The intended pipeline in this build uses actual source-context and conditioning
outputs as sampling inputs. Host arrays remain comparisons and explicit
PCM/token/noise fixtures; they do not replace denoising with reference audio.
The earlier [AOT/default attempt](../coreai-iphone17pro-aot-metal-crash/README.md)
crashed inside Metal/MPS compilation. This run uses the source CPU-request
route and gets past every previously failing sampling stage and the decoder.
No new model preparation or AOT compilation was needed for that route change.

## Limits of the original console evidence

The console contains phase progress, **not** the ten numerical comparison
results, final host/device parity status, pipeline elapsed time, or the 10 ms
sampled process maximum. The last previously supplied Mac host status was
`bounded_diagnostic`; it must not be promoted to strict PASS merely because
execution and playback succeeded. The maximum **phase-boundary** footprint
logged is 864.0502319335938 MiB, not an exact/sampled peak or total RAM figure.

Share metadata/LaunchServices errors and a trailing debug-session signal 9
are retained unchanged. They do not establish that generation failed or that
the app was killed for memory: the user explicitly reported the app remained
open and playback worked. Actual share/export success, or the reason that
debug session ended, is not established by this console.

This records one execution, not musical approval, cold-cache repeatability,
full-song feasibility, or an accepted 90-second bass baseline. The [subsequent WAV audit](../coreai-iphone17pro-bass-preview-waveforms/README.md)
finds close Linux-reference agreement with a few strict pointwise violations.
The user accepted the supplied short preview, and the complete generation
report is now available below. Its numerical status is separate from audition
approval. The next run is a targeted same-case Release performance measurement.

## Full generation report subsequently supplied

The unchanged [per-output report](generation-report.txt), summarized in
[generation-report.json](generation-report.json), identifies run
`654E35CC-BD44-442A-A911-8AAD8616F204`, started **2 November 2026, 16:20:14 WIB**.
It reports 50/50 steps, 750 sampling calls, decoder/WAV completion, and all ten
output comparisons evaluated. Eight comparisons pass; the two final-step latent
comparisons fail the unchanged pointwise gate. **Both decoded waveform checks
against the Mac references pass.** Host preparation is still
`bounded_diagnostic`, so integrated numerical parity correctly remains
**NOT PASSED**. This is compatible with user listening approval and the separate
Linux-reference comparison finding four waveform pointwise violations: these
are different references/gates, not inconsistent results.

Total reported pipeline time is **85953.766125 ms (85.95 s)** for **5.12 s** of
audio, a wall/audio ratio of **16.7878**. Source context takes 1475.75125 ms,
conditioning 10303.626667 ms, and reported sampling steps sum to 71789.16317 ms.
The remaining 2385.225038 ms includes decoder, reference verification, WAV work
and other outside-component work; it must not be labeled decoder time alone.
Step 1 takes 14715.578167 ms; steps 2–50 average 1164.76704088 ms. First-step
asset open/bind intervals are milliseconds while initial backend-await
intervals dominate. The report alone does not isolate compilation, kernel time,
thermal effects or cache state.

The report's 10 ms sampled app-process maximum is **417.18978118896484 MiB**,
baseline 41.314186096191406 MiB. This is not total device/model RAM or an exact
peak. Do not substitute the earlier console's 864 MiB phase-boundary maximum
for this report value or combine them into one guaranteed memory envelope.
The report lacks WAV hashes/build flags, so exact run-to-upload identity and
Debug/Release configuration are not cryptographically established from its
contents. Previous supplied build logs used Debug/Onone; cold-cache state and
repeatability remain unverified. Future reports record build configuration,
Swift optimization, WAV hashes and aggregate asset timings.

The old report's sentence about a "musical/device baseline NOT accepted" was
a programmatic numerical-gate warning, not an assessment of listener approval.
Future report wording names the **numerical** baseline and keeps subjective
review separate. Original supplied evidence is not rewritten.

The next targeted measurement is the **same 5.12-second case in the shared
CoreAIProbe Performance scheme (Release)**. Model weights, 50 steps, guidance,
precision, inputs, references and tolerances stay unchanged. Release may reduce
Swift copy/check overhead; it does not establish faster native inference or fix
the AOT Metal compiler crash. No speedup is claimed before device measurement.

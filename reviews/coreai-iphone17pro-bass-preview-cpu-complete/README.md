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
preview. WAV bytes and the final in-app report have not yet been supplied.

The intended pipeline in this build uses actual source-context and conditioning
outputs as sampling inputs. Host arrays remain comparisons and explicit
PCM/token/noise fixtures; they do not replace denoising with reference audio.
The earlier [AOT/default attempt](../coreai-iphone17pro-aot-metal-crash/README.md)
crashed inside Metal/MPS compilation. This run uses the source CPU-request
route and gets past every previously failing sampling stage and the decoder.
No new model preparation or AOT compilation was needed for that route change.

## Limits of the evidence

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
full-song feasibility, or an accepted 90-second bass baseline. Next obtain the
**existing** final in-app report and WAV stem/listening mix to assess parity and
sound; no regeneration is required merely to collect these existing outputs.

# Real-recording audio-context: device PASS

The user supplied the physical iPhone [report](report.txt); the [structured
record](report.json) includes its SHA-256 and all twelve output comparisons.
This is user-reported CoreAI execution under requested `cpuOnly`, not an
independently reproduced Linux inference or measured accelerator placement.

The first 5.12 seconds of the same `piano.mp3` yield 128 latent frames.
Posterior mean, posterior standard deviation, sampled source latents and
full-range `lego` context all pass their host references in **three runs**,
using one retained model/function and the same explicit epsilon. Inputs and
references are hash-verified. Canonical PCM decoding/resampling occurred on
the Mac; raw recording import on iPhone remains untested.

| Run | Backend-await | Readback/checks | Passing outputs |
| --- | --- | --- | --- |
| 1 | 1051.218042 ms | 1.592500 ms | 4/4 |
| 2 | 1017.439167 ms | 1.442208 ms | 4/4 |
| 3 | 1041.502042 ms | 1.472125 ms | 4/4 |

Mean backend-await is **1036.7197503333334 ms**. Model open took
1011.133833 ms, function bind 76.392667 ms and total probe 4338.059625 ms.
These wall intervals do not establish isolated kernel timing or cold-cache
performance. Repeated numerical summaries match, without claiming bit-exact
outputs. Maximum absolute mean/source/context error is 0.0001207590103149414;
acceptance uses the combined absolute-plus-relative bound and normalized RMSE,
so an error above the absolute term alone does not imply failure.

Sampled app-process maximum is **398.20570373535156 MiB**, baseline
20.236129760742188 MiB. Sampling every 10 ms is neither an exact peak nor total
model/device RAM. No combined conditioning/denoiser/decoder pipeline was measured.

**Decision: the audio-side gate in stage 2 is complete for this recording and
fixture.** No repeat of this gate is required now. Stage 2 continues with
text/instrument conditioning and ConditionEncoder following the pinned CUDA
path. The probe executes no denoiser and generates no bass or waveform;
audio-context alone does not establish instrument selection or harmony.

User audio, derived tensors and native weights remain local and ignored.
See the [guide](../../docs/coreai-source-context.md) and
[generation roadmap](../../docs/iphone-generation-roadmap.md).

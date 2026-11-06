# Local source reconstruction is clean; test one APG norm-cap hypothesis

**Subsequent result:** [the APG candidate completed](../coreai-drums-scaled-apg-partial-result/README.md).
User reports clearer drums but hit-transition glitches throughout. Five control
checks are exact-zero, clipping active48/50. Keep this partial candidate; obtain
its existing waveform/latent next, without another generation or cap sweep.
The preparation/testing state below is historical, not a request to rerun it.

The user supplied the [local source-listening report](source-listening-report.json)
and confirmed **both files are clean and contain piano only**. Piano-only output
is expected: this test reconstructs the source, not accompaniment. One decoder,
zero encoder/DiT/native-export calls, common gain 1.0, 10.24 seconds stereo/48kHz.
Source reference/PCM/latent and VAE identities match the preceding verified ZIP.
[Summary](summary.json) retains report SHA-256 and the listening decision.
New WAV bytes were not uploaded; their hashes remain reported by the Mac.

This resolves the earlier access confusion: the first affirmative reply was
withdrawn because cloud links could not be downloaded; the local Mac pair now
has actual clean listening feedback. Source/codec is retained for this input.
It does not prove all possible generated latents must decode cleanly. Drum
control, FP32/BF16-partition references and the lyric-only candidate still have
glitches. The accepted groove and original baseline stay preserved.

## Why this single APG candidate

Pinned APG clips the momentum difference with L2 norm **over time, dim 1**,
independently per latent channel. If the cap is `c` and the window is `T` frames,
the clipped vector's per-channel RMS bound is `c/sqrt(T)`. Orthogonal projection
does not increase that norm; guidance 7 gives an added-guidance RMS bound of
`6*c/sqrt(T)`. This is an internal bound, not a waveform gain or a prediction of
audibility. Actual impact depends on clipping being active and trajectory state.

| Setting | Frames | APG cap | Momentum RMS bound |
| --- | ---: | ---: | ---: |
| Current 10.24s control | 256 | 2.5 | 0.156250 |
| Approved 30s reference | 750 | 2.5 | 0.091287 |
| Isolated 10.24s candidate | 256 | 1.46059349 | 0.091287 |

The one candidate is derived as `2.5*sqrt(256/750)`, anchored to the frozen
approved 30-second excerpt. It is **not** a claim that published APG must be
duration-normalized, that the 30-second context has been reproduced, or that
this causes/fixes the glitches. No arbitrary cap grid, seed or duration sweep.
The actual drum clipping/update telemetry has not yet been measured; the new
experiment records it for all 50 candidate steps.

[Pinned APG math validation](apg-math-validation.json) uses synthetic velocity
tensors, not neural/music output. At cap 2.5 the adapter is bit-exact to original
APG. With clipping active, the measured guidance-update norm ratio 0.58423746
agrees with the expected 0.58423739, and the per-channel RMS bound holds.

## Baseline and acceptance gates remain separate

[`compare_coreai_drum_apg.py`](../../scripts/compare_coreai_drum_apg.py) validates
the same prepared source/conditioning/preview bundles and their original
diagnostic selection. It reuses original FP32 control audio and changes **only
candidate APG norm_threshold**. Original control/production cap stays **2.5**.
BF16 app assets, existing APG exporter, learned checkpoint values and resource
selector are unchanged. This is a host original-FP32 quality test, not a new
on-device precision policy or a numerical PASS for a modified candidate.

The candidate's conditioning must be byte-identical to original FP32 control;
all prompt/lyric/timbre, source PCM/context, initial noise/schedule, duration,
50 Euler steps, guidance 7, momentum, projection and decoder stay fixed.
Control conditioning/mask, first gold step and final decoder replay retain
original abs `1e-4`, rel `1e-3`, normalized RMSE `1e-3` gates. **Numerical
acceptance tolerances do not change when the candidate norm cap changes.**
Control provenance remains `bounded_diagnostic`; failed old gates stay failed.

The shared host executor gains optional candidate APG/label/unchanged-condition
hooks. Existing lyric comparison defaults still use original APG and original
filenames. The candidate runs one independent 50-step trajectory; no teacher
forcing. Two headroom copies share one attenuation, with raw WAVs/latent/audio
also retained in a new output folder. No EQ, limiter, gain boost, native export,
model activation or checkpoint download fallback. Learned weights remain cached.

Telemetry describes candidate momentum. The baseline-cap fraction after step 1
is a counterfactual on candidate states, not a replay of 50 original-control
states. A successful computation or reduced waveform level is not musical
acceptance. Assess both glitch and groove. If ineffective, retain it as a failed
candidate and stop; do not automatically sweep other caps. If useful, native
port/parity and device listening remain separate work before adoption.

[Mac command](../../docs/coreai-drums-quality.md) computes one new candidate;
this is more work than the preceding VAE-only listening step, and no app build
is needed. Full candidate execution/listening is **NOT TESTED here** because
base/Qwen weights and complete prepared bundles remain on the user's Mac.

Validation: 159 host tests run, 157 PASS and two optional native/export tests
skipped; five new APG-policy/admission/delegation tests. Python syntax/help,
JSON/docs/whitespace and production-source isolation checks pass. Real pinned
APG math was exercised; no neural generation, native execution or repair is
claimed for this candidate.

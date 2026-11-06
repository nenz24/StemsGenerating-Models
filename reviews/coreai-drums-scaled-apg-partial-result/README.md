# Scaled APG candidate: clearer drums, residual hit-transition glitches

User-supplied [report](report.json), reviewed 6 November 2026 WIB, and listening
feedback: **drums are more clearly audible, but glitches remain at changes
between hits throughout the clip**. They are not confined to the file edges.
This is partial improvement, **not finished drum acceptance**. New candidate
groove was not separately reconfirmed; retain the earlier accepted groove as a
requirement. No candidate is promoted to the BF16 app baseline.

[Summary](summary.json) records report SHA-256 and validated accounting.
The run completes 50/50; candidate cap is `2.5*sqrt(256/750)=1.46059349`, control
cap 2.5. Source PCM and three component-reference identities match the previous
host experiment. Prompts match, candidate conditioning is reported byte-identical
to control. Five conditioning/mask/unchanged-condition/first-step/decoder checks
PASS with **all reported errors exactly 0**. The original abs `1e-4`, rel `1e-3`,
normalized RMSE `1e-3` limits stay fixed; control provenance remains
`bounded_diagnostic`, not a new full numerical PASS.

## Parameter participated; telemetry cannot locate waveform clicks

All 50 telemetry rows have ordered step indices, finite values and clip fractions
quantized in units of 1/64 latent channels. Candidate clipping is active on
**48/50 steps**. Candidate clip fraction min/median/max is **0 / 0.8515625 / 1**.
Maximum per-channel added-guidance RMS is 0.547722578, consistent with the
0.547722558 theoretical bound within FP32 rounding. The first state has 70.3125%
channels above the candidate cap vs 31.25% above 2.5. Subsequent reference-cap
fractions are counterfactual on candidate momentum, not full control replay.

These steps are **diffusion iterations over the entire audio**, not physical
audio timestamps. A step index cannot identify where a drum transition crackles.
Clip activity proves the candidate parameter was used; it does not prove the
remaining artifacts are caused by clipping or identify their audio location.

Reported raw sample peaks: control 1.3228401, candidate 1.6377499; common
headroom gain 0.4322673321 matches the −3 dBFS policy. They are not perceived
loudness/quality measurements. The reported control raw-WAV hash matches the
previous control. New waveform/latent bytes have **not** been uploaded yet;
new artifact hashes and the actual sample transform still need independent
verification. The previous lyric candidate's audio is not this APG candidate.

## Attack-local investigation before another model change

This experiment emits **one continuous 10.24-second waveform** from a full-window
candidate decoder. There is no chunk stitch or song crossfade. The user's
description concerns changes between drum hits, so do not apply a seam fix or
assume the artifacts occur at model-asset boundaries.

[Collection command](../../docs/coreai-drums-quality.md) packages five existing
files from this run: report, both headroom WAVs, candidate raw Float32 audio and
latent. Source metadata/PCM/latent need not be collected again because their
identities match the verified earlier ZIP. No new generation or re-export.

Once bytes arrive, validate hashes and common-gain transform first, then inspect
candidate/control attacks across the whole clip. Sharp adjacent-sample changes,
high-frequency energy and noise are also legitimate drum transients; they are
not automatic click detectors. Do not add a de-clicker, filter, limiter or fade
that can erase desired hits. Retain this partial candidate, original baseline,
source/codec, BF16, checkpoints/cache and all numeric tolerances. No further cap,
seed, step-count or duration sweep is supported by the current evidence alone.

Only report/data/docs verification was performed here for the new candidate;
no inference, audio repair, new model, native export or additional test-suite
run is claimed. Actual input waveforms are the next missing evidence.

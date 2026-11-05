# Instrumental tag candidate completes; both variants still glitch

The user supplied [report.json](report.json) and listening feedback: **both
variants are glitchy**. The raw uploaded JSON is retained byte-for-byte as
evidence, not as instructions. [Summary](summary.json) records its SHA-256 and
separates execution checks from the user's musical assessment.

Status `completed_quality_experiment`, 50 candidate steps. Control has empty
lyrics; candidate changes only lyric content/formatted lyric text to
`[Instrumental]`. Text remains 111 tokens; lyric 11→15, packed condition 123→127.
Caption/instruction, metadata, prepared source/context, noise/schedule, timbre,
50 APG Euler steps and guidance 7 stay fixed within this experiment.

All four control checks—conditioning FP32, condition mask, first gold step,
and decoder of cached gold latent—PASS with **max error, RMSE, normalized RMSE,
and fraction over bounds all exactly 0**. This confirms those replay checks,
not full-trajectory parity or musical quality. Original abs `1e-4`, rel `1e-3`,
normalized RMSE `1e-3`; control provenance remains `bounded_diagnostic`.

Reported raw sample peaks: control 1.3228401, candidate 1.9132220. One shared
gain **0.37002804741144624** matches the −3 dBFS policy calculated from the larger
peak. Peak/gain are not a quality verdict. Execution counts: one control and
one candidate conditioning, one control first-step replay, 50 candidate DiT
steps, and two decoder calls (control/candidate). No native exports, baseline
replacement or tolerance changes. Torch 2.8.0, Transformers 4.57.3, Diffusers
0.37.0, Tokenizers 0.22.2.

The lyric-only change **did not resolve the complaint** and is not adopted.
Do not repeat that candidate, change gain or infer that native FP32 would fix
the issue. The earlier iPhone groove was accepted; candidate groove was not
reported separately. No unique glitch cause is established.

## Current input evidence is still needed

Current source PCM `836c4509…6356018` differs from the earlier iPhone import
`c79163af…d57278` and the repo host snapshot `9517cc69…10bb945`. This can reflect
decode/preparation differences; it does not prove another recording or a bad
input. It prevents exact waveform comparisons across those runs. The report
contains a source reference SHA-256 but not that reference's original recording
metadata. Its artifact hashes cannot be independently checked without bytes.

[Collection command](../../docs/coreai-drums-quality.md) packages only existing
control/candidate headroom WAVs, raw candidate audio and latent, exact source
PCM/reference metadata and source latent, plus the experiment report. No new
inference, native re-export or checkpoint upload is needed. Verify each hash
against the report/source manifest before analysis. The accepted 30-second
Colab result ZIP was inspected: it has drums/mix/report, but **no captured
latent**, so its accepted CUDA decoder input cannot be reconstructed from that
archive. Do not fabricate or substitute that reference.

All report accounting, unchanged prompt fields, zero-error checks, fixed bounds,
gain formula and candidate-audio hash consistency were validated here. Actual
waveform/latent hashes await upload. JSON, links and whitespace checks pass;
this change adds evidence/docs only, so no new inference or test suite run is
claimed. BF16 app/code, original audio, checkpoint/cache and baseline remain.

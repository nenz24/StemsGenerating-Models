# Glitches reproduced in both host references; isolate lyric conditioning

The user hears glitches in **both** `drums-original-fp32-headroom.wav` and
`drums-partition-host-headroom.wav`, following an unchanged-glitch drum headroom
check on iPhone. Groove fits the piano; glitches mask the drum sound. Musical
audio acceptance stays pending. This reproduces the complaint outside iPhone
and without the BF16 partition path; it does not establish all playback paths
as fault-free or prove a unique generation defect.

## Confirmed difference from the accepted Colab pipeline

The approved [30-second baseline](../../baselines/acoustic-drums-30-v1/baseline.json)
uses the same original piano hash, drum caption/instruction, 50 steps, guidance
7, shift 1 and Euler/APG selection. Its actual archived settings specify
`instrumental=true, lyrics="[Instrumental]"`; current preview prompts specify
empty lyrics. The archive SHA-256 is verified before reading these settings.

Pinned [LEGO inference](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/inference.py#L830)
forwards `params.lyrics` for direct-conditioning tasks.
[Text conditioning](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/core/generation/handler/conditioning_text.py#L137)
and the pinned formatter retain the literal lyric content. It is not erased
because `instrumental=true` was provided. The preview formatter intentionally
retained the empty-lyrics bass baseline; that is not reproduction of Colab's
instrumental lyric input.

[Actual pinned tokenization](tokenization.json), SHA-256 verified:

| Variant | Text | Lyric | Packed condition |
| --- | ---: | ---: | ---: |
| Existing empty-lyric drum control | 111 | 11 | 123 |
| Only add `[Instrumental]` | 111 | 15 | 127 |

No padding or caption edits force the old shape. The pinned base config does
not enable `is_lego_sft`; the separate Global/Local/Mask Control caption wrapper
is therefore not added. [Comparison metadata](comparison.json) also records
other differences that this experiment deliberately leaves fixed: duration
10.24 vs 30/90 seconds, source preparation/gain, CPU/CUDA RNG, and the full-song
baseline's kit reference. The approved 30-second excerpt had no kit reference;
the full-song lossless kit WAV is not present in the uploaded feedback archive.
Do not infer that tag, context length or timbre is the proven glitch cause.

## One host-only lyric candidate

[`compare_coreai_drum_lyrics.py`](../../scripts/compare_coreai_drum_lyrics.py)
validates existing source/condition/preview bundles using their original gates.
It reuses cached original FP32 control audio, exact prepared piano/context,
initial noise and 51 timesteps. Only the literal lyric input changes. It reuses
original text hidden states and publisher silence reference, loads the pinned
Qwen embedding/ConditionEncoder, then computes one independent original-FP32
50-step APG trajectory and decodes it. No teacher forcing, seed sweep, duration
change, new weights, training, native export or app selector activation.

Strict replay gates for original conditioning, mask, first gold Euler step and
cached-control final-latent decoding must pass original `1e-4`/`1e-3` pointwise
and `1e-3` normalized RMSE bounds. Gates fail closed and preserve reports. A
successful replay is not full-trajectory parity, nor a new PASS for the existing
bounded diagnostic provenance. Candidate vs control uses intentionally different
conditioning, so no candidate output parity or musical PASS is inferred.

Weights must already exist in the same local vendor/HF caches or be supplied
explicitly. Resolution is local-files-only with full pinned size/SHA-256
verification; missing/bad weights stop. Small pinned publisher source/config
files may be retrieved if absent from cache. No checkpoint download fallback.
Control/candidate raw and common-attenuation headroom WAVs, candidate condition,
latent/audio and report are written to a new directory outside ModelCandidates.
The accepted groove, BF16 production baseline, checkpoint/cache and original
audio stay preserved. A candidate that loses the groove is not accepted simply
because its artifacts improve.

[Mac command and listening instructions](../../docs/coreai-drums-quality.md)
require one new host CPU candidate; they do not rebuild the app. This workspace
does not contain the Mac prepared tensors or pinned model weights, so **real
candidate execution and quality remain NOT TESTED**. Tokenization and archived
settings were audited here; no audio was fabricated or claimed repaired.

Seven new host tests cover literal-only prompt changes, gate/admission failure,
read-only plan and selector preservation, retained failure reports, strict replay
metrics, 50-step live feedback/fresh momentum, and cache-only checkpoint lookup.
Model admission fixtures are explicitly nonexecutable; the constant-velocity
feedback test is control-flow validation, not musical/native inference.

[Validation](validation.json): 150 host tests run, 148 PASS and two optional
export/native tests skipped. Seven focused tests PASS again after shape guards;
Python syntax, whitespace, JSON, document links and production-source isolation
checks PASS. No real candidate audio or Apple execution is claimed.

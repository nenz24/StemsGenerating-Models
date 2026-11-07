# Accepted listening result: captured host FP32 drums 30s

The user heard all three files and said **“Saya dengarkan 1, 2, dan 3 semuanya
bagus”**. This accepts the Colab control 01, new host drum 02 and piano/drum mix 03
for this listening review. [Original report](report.json) is preserved byte for
byte; [summary](summary.json) attaches the human verdict separately. Its original
automatic `musical_acceptance: NOT EVALUATED` predates the feedback and is not
rewritten into an automated PASS. No new WAV/tensor bytes were uploaded with
this report; their listed digests remain reported, not independently verified.

The report confirms completed 30s/750-frame generation, 50/50 independent live
Euler/APG updates, one source encoder and one full VAE decode without stitching.
Prepared source32WAV/PCM match the frozen originals exactly. Source 48 input hash
also matches the previously verified frontend input. Recovery succeeded; do not
repeat failed v1/v2 source preparation, recovery or accepted v3 generation.

This is the original FP32 **host** reference. It does not accept existing glitchy
10.24s native output or establish CUDA replay, native BF16 parity, full-song
quality, bass/guitar integration or a unique cause of the earlier glitches.
Lyrics/metadata, source preparation, duration and reference settings were
restored together; this was not a duration-only ablation. Original failed full
trajectory gates and `bounded_diagnostic` preview provenance remain unchanged.
Abs1e-4/rel1e-3/nRMSE1e-3, BF16 baseline, checkpoint values, sampler and production
mix are preserved. The diagnostic listening mix is 0.5 piano + 0.5 drum, not a new
iPhone mix setting.

## Next: transfer the existing numerical reference, then port in partitions

**Capture received and independently checked:** [transport, actual component replay
and inactive native preparation](../coreai-drums-reference30-native-preparation/README.md).
Do not repeat packaging; the next action is the current Mac export command.
The transport instructions below describe the already completed transfer.

[`package_coreai_drum_reference.py`](../../scripts/package_coreai_drum_reference.py)
requires the exact uploaded completed report and packages only its 43 existing
Float32 captures. Hashes, byte sizes/shapes and finite values are checked on the
same bytes written into a temporary ZIP. It publishes the new archive only after
all checks and the 32 MiB cap pass; existing outputs and source remain untouched.
Raw tensor payload is 31,317,696 bytes (29.87 MiB); WAV listening copies, MP3,
checkpoints and unrelated cache files are excluded. The private ZIP is returned
in chat, never committed. No generation, download, native export or activation
occurs. [Current Mac command](../../docs/coreai-drums-quality.md).

The transfer preserves source48, posterior mean/std/epsilon/mask/latent/context,
token IDs/embeddings, silence reference, packed condition/mask, explicit initial
noise/timesteps, steps 1/5/10/25/50 (conditional/unconditional/guided/momentum and
post-update latent), final latent and raw final audio. Shapes/layout are recorded
in the generated manifest. Conditional/unconditional arrays are velocities for
each captured step, not branch latents. The uncaptured `null_condition_emb` must
come from the same checksum-pinned base checkpoint during export, never zeros.
Matching source/noise/conditioning tensors, rather than merely seeds, are the
comparison inputs.

Native preparation is a separate next implementation: the current contracts
allow only 128/256 latent frames and old 111/11/123 drum token geometry. The accepted
host requires 750 frames and 111/15/127 tokens. Do not rename this ZIP as a native
bundle, relax the allowlist, overwrite old shapes or load monolithic DiT on
iPhone. First verify received snapshots, then prepare a distinct partitioned
profile and assess memory. Check components and one-step outputs at original
numerical gates before live 50-step trajectory/device listening. Per-step checks
cannot substitute for the independent feedback trajectory; do not reset to gold
latents or loosen bounds if BF16 fails. A host listening acceptance alone cannot
authorize an automatic native numerical PASS. Retain the existing partitioned
BF16 baseline throughout.

No Apple export/build/device inference has been run in this review. Full-song
bass/drums/guitar on iPhone remains the final objective after native drum
quality and resource use are established.

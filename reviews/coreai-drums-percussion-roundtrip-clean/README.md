# Percussion VAE control heard without glitches

The user supplied [report.json](report.json) and said **"sudah lebih baik dan
glitch tidak terdengar"** on 6 November2026, Asia/Jakarta. This is the latest
local approved-percussion roundtrip listening pair. Per-file01/02 verdicts were
not separately specified; do not fabricate them. The original1854-byte report
is preserved unchanged, including its unfilled automatic `NOT EVALUATED`
listening field. [summary.json](summary.json) records the later human feedback.

The report identifies `approved-percussion-fp32-vae-roundtrip`: one encoder,
one decoder, zero diffusion steps and zero native exports. Approved source
archive/member and pinned VAE hashes match the previously executed cloud test.
Duration10.24s, stereo48kHz, latent256frames, explicit CPU posterior seed31 and
shared attenuation policy also match. Gain0.7182118391 is consistent with the
reported reconstruction peak0.98570609 and the unchanged−3dBFS policy.
No new WAV/latent was uploaded here: output hashes are recorded, not independently
recomputed. Mac ffmpeg9.0.2 vs cloud7.1.5, prepared/output hashes and reported
metrics differ; exact cross-platform waveform parity is not claimed.

## Decision

The percussion codec listening request is complete; do not ask to repeat it.
Retain the pinned FP32 VAE, production BF16 and APG cap2.5, saved baselines,
checkpoints/cache and abs1e-4/rel1e-3/normalizedRMSE1e-3 limits. A clean lossy
roundtrip of the existing approved Colab drum supports handling this percussion
input; it is not a new generated drum, a VAE version change, proof that every
generated latent will decode cleanly, or confirmation of all playback paths.

Prior APG feedback remains **clearer drums with residual hit-transition glitches**.
There is no new listening result for that generated candidate and no new iPhone
musical acceptance. Focus remaining work on source/conditioning and independent
live diffusion trajectories compared with the accepted Colab reference. No
filter, cap sweep, native activation or chunk/stitch change is justified by this
negative control. Full-song three-stem iPhone acceptance remains unfinished.

## Exact accepted input recovered for generation investigation

[Actual source preparation audit](source-preparation-audit.json) verifies the
original MP3 SHA `ccd94ea3…246189` and recovers the already existing30s source
WAV in ignored workspace data. Its full SHA is exactly the accepted model-input
SHA `1f68460c…d66b54`, and stereo32kHz PCM16 content SHA is
`1d57ce14…7baaf23`. The current preparation helper was also run once with fixed
gain0.7: its WAV hash differs but every PCM16 sample is identical, max difference0.
Those fresh file-hash differences therefore do not establish different audio.
Use the existing exact archived input; do not rewrite baseline references.
Only metadata is committed; original/recovered/fresh audio stays local.

[Generation reference plan](generation-reference-plan.json) fixes the accepted
30s preparation/duration metadata, literal `[Instrumental]`, absent kit reference,
caption/instruction, original cap2.5, guidance7 and50 APG Euler steps. This is a
reference reproduction, not an isolated duration-only experiment, and no new
model inference ran in this follow-up. It requires real prepared conditioning,
source posterior/epsilon, diffusion noise/schedule and live latent/velocity
captures before numerical comparison. The same seed does not equate CPU/CUDA
RNG streams. Original accepted CUDA intermediate tensors are not available;
do not invent a parity reference or claim exact trajectory parity from settings.

Validation here: report identity/accounting/gain/digest syntax and byte preservation,
actual input hashing/PCM comparison, JSON and document-link checks. No code or
model changed; no host model test suite or Apple build was rerun for this
metadata-only follow-up. The preceding163-test result belongs to the previous
code change and is not a new run. No additional action is requested from the
user for the completed codec control.

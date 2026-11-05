# Uploaded drum waveforms verified; codec replay and tiling agree

**Local listening completed:** the user now reports both piano source and VAE
reconstruction clean, piano only. [Report and next isolated host experiment](../coreai-drums-source-clean-apg-plan/README.md)
resolve the earlier cloud-access confusion. Keep VAE; do not repeat source
listening. The initial pending/access state below is historical.

The user uploaded the completed Mac experiment's eight-file ZIP. All eight
member identities, source reference SHA, source PCM/latent hashes, and supplied
candidate/control artifact hashes match their reports. The raw candidate audio
reproduces every attenuated candidate WAV sample exactly using the reported
common gain. [Integrity evidence](bundle-integrity.json) records the ZIP SHA-256
and all file hashes. No raw audio/checkpoint is committed here.

The source reference reports original piano SHA `ccd94ea3…6189`, matching the
earlier device and accepted Colab reports. The full MP3 was not supplied and is
not independently rehashed here. Current PCM `836c4509…6356018` is independently
verified; it differs from previous decoder/preparation outputs. Do not confuse
that PCM difference with proof of another recording or a corrupted input.

## Actual codec execution performed here

Only the pinned VAE was downloaded (337,431,388 bytes) and verified in full
against SHA-256 `da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0`.
Its decoder was instantiated from the pinned config and real checkpoint.
CPU PyTorch 2.8.0, FP32, two threads, no native exporter. **Six real decoder
calls, zero encoder calls, zero DiT/sampling steps**:

1. Decode the supplied source posterior latent as a piano reconstruction.
2. Decode the supplied final candidate latent, compare with uploaded raw audio.
3. Decode that same candidate latent in four overlap-discard windows (96, 128,
   128, 96 frames; core stride 64, overlap 32; 1920 output samples/latent frame).

[Codec measurements](codec-audit.json):

| Same-latent comparison | Max abs error | Normalized RMSE | Original gate |
| --- | ---: | ---: | --- |
| Cloud CPU replay vs uploaded Mac candidate | 2.13534e-5 | 2.52445e-6 | PASS |
| Overlap-discard vs full cloud decode | 4.20213e-6 | 5.87340e-7 | PASS |

No element exceeds the original pointwise bound `1e-4 + 1e-3*abs(reference)`;
normalized RMSE bound stays `1e-3`. This supports reproduction of the uploaded
candidate and makes a decoder-window change unsupported as a fix for this
waveform. It does **not** prove musical quality, native iPhone decoder parity,
all possible codec behavior, or equivalence to CUDA's exact auto-tile policy.

Piano source/reconstruction correlation is 0.955728 left and 0.985069 right;
relative RMS error 0.213063, SNR 13.430 dB. This is lossy sampled-posterior
reconstruction, not an expected exact reconstruction or an acceptance test.
Both piano listening copies use the same gain 1.0 (peaks already below −3 dBFS).
Lossless Float32 WAVs were retained locally; two 128 kbps MP3 copies were sent
in chat only for convenient listening. **Access clarification:** the user could
not download/open those cloud files and had not listened to either clip. The
earlier affirmative reply is not evidence of an audible codec defect. **Piano
reconstruction listening is UNTESTED**; statistics alone do not certify that it
is free of glitches.
Do not run another 50-step generation while this source-side distinction is open.

### Local Mac delivery

[`listen_coreai_source_roundtrip.py`](../../scripts/listen_coreai_source_roundtrip.py)
creates the two WAVs directly on the user's Mac using the existing prepared
source bundle and local pinned VAE. It verifies source integrity, performs one
CPU FP32 decoder call on the saved source latent, writes original/reconstruction
copies with common attenuation to a new folder, and checks input/reference
hashes again. No encoder/DiT, download fallback, native export, app/resource
activation, original-file replacement or interpretation of listening feedback.
Missing weights/gate failures retain the report and stop. Output inside the
source/ModelCandidates or an existing destination is rejected before work.

[Mac command](../../docs/coreai-drums-quality.md) opens the local output in
Finder; cloud file download is not needed. [Delivery validation](delivery-validation.json):
four guard/failure-preservation tests pass; one **additional real decoder call**
produces both WAV files byte-for-byte identical to the pair computed earlier.
Only source admission was mocked in that execution smoke because native payload
was not uploaded; provided source PCM/latent/reference hashes were independently
verified. Decoder/checkpoint were real. The production CLI retains full source
admission. The original six-call audit above is unchanged. Apple/Mac execution
of the new CLI is pending; no listening verdict or musical fix is claimed.

## Effective Colab source audit

The accepted Colab session's three effective Python-file hashes differ from
the older publisher files used by the preview. The exact accepted runtime files
were recovered at pinned handler revision `ca1e85fe…3866` and their SHA-256 values
verified against the archived session. APG/config files are wrappers that import
shared canonical implementations; a different file hash is not a different
active formula by itself.

[Active-path audit](source-path-audit.json) finds identical method ASTs for
attention, DiT layers, timestep embeddings, lyric/timbre/ConditionEncoder and
DiT forward. APG MomentumBuffer/project/apg_forward ASTs also match exactly.
The changed condition-projection constructor still evaluates to 2048→2048 for
this config. Archived `dcw_enabled=null` resolves to **false for Base**, so do
not enable DCW on the theory that preview omitted an active Colab correction.
Euler, ADG-off and zero velocity-clamp/EMA settings retain the relevant path.
These are static observations, **not full pipeline numerical parity** across
CUDA/Mac/iPhone or a complete audit of every handler/library/backend behavior.

Archived Colab weight hashes agree with the current pinned Qwen/base/VAE
inventory. Only VAE bytes were loaded and verified in this environment; other
weight files were not downloaded. Duration, source gain/resampling, RNG,
attention/backend versions and full-song kit reference remain unmatched factors.
No one of them is established as the cause. Do not start a blind sweep.

[Waveform measurements](waveform-audit.json) are descriptive, not a glitch
detector. The included approved Colab excerpt is from a different generated
30-second context, resampled from PCM16/32 kHz for display/measurement. It is
not an exact reference waveform for current 10.24-second outputs.

## Decision

Preserve accepted groove, BF16 baseline, original audio, all bundles/checkpoints,
cache and original tolerances. No change to app, sampler, precision, prompt,
gain or decoder implementation is justified by this audit. No musical repair is
claimed. Classify the supplied piano reconstruction by listening, then choose
the next common generation/input/context investigation from that result.

This is a read-only evidence/real-codec experiment, not a software change.
Integrity assertions, output shape/finite checks, original numerical comparisons,
static AST/hash checks, JSON/docs and whitespace checks pass. No unrelated test
suite or new device execution is claimed.

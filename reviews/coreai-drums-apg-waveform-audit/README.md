# APG candidate waveform verification and percussion codec control

**Subsequent Mac listening result:** the user reports "sudah lebih baik dan
glitch tidak terdengar" on the percussion roundtrip test. [Archived report and
bounded interpretation](../coreai-drums-percussion-roundtrip-clean/README.md)
close that listening request. No repeat is needed. This clean control does not
accept a new generated APG/iPhone candidate; pending-listening statements below
describe the earlier cloud-only stage.

The user uploaded `drums-apg-review-v1.zip` after reporting clearer drums with
remaining glitches at changes between hits throughout the clip. The five-member
archive is verified in [archive-audit.json](archive-audit.json): report bytes match
the already archived report, all four artifact hashes match, source/component
identities match the earlier eight-file ZIP, and candidate headroom samples are
bit-exact `(raw Float64 * common gain).astype(Float32)`. Both listening WAVs are
finite stereo Float32, 48kHz, 491520 frames. They have no zero stereo frames or
exactly equal adjacent stereo frames; these checks do not detect all duplicated
audio segments, playback dropouts or perceptual glitches.

## Real candidate decoder replay

One CPU FP32 decoder call on the uploaded candidate latent, pinned checkpoint
SHA-256 `da17edb6…59a99b0`, reproduces the Mac raw waveform at the original gates:
maximum absolute error `1.18017e-5`, normalized RMSE `1.74564e-6`, zero elements
outside pointwise bounds. [Replay report](candidate-replay-audit.json) records
this real execution. No encoder, DiT, BF16/native export, or tolerance change.
This supports fidelity between saved latent and exported audio; it does not
certify musical quality or exclude latent-dependent codec artifacts.

## Attack inspection did not establish a repair

[Descriptive measurements](attack-audit.json) cover source piano, the archived
approved Colab drum excerpt, current APG control and candidate. The candidate's
largest adjacent change is near 3.567771s, within a drum attack present in both
control and candidate. Both vary over a multi-sample attack, with additional
low-frequency energy in the candidate; no chunk join exists in this continuous
10.24s full-window decode. Its exact location and the large delta alone cannot
identify the user's perceived glitch. The source piano is a different signal
and cannot serve as drum waveform ground truth.

Below16kHz energy fractions differ: control 5–16kHz 7.01%, candidate 2.89%,
archived Colab 2.55%. Candidate energy below150Hz is 89.38%. These are relative
band energies, not perceived loudness or proof of improvement/cause. Colab is
a different context/RNG/preparation, exported at32kHz PCM16. The analysis
resamples with SciPy and limits band comparisons below16kHz; it does not mistake
that export cutoff for a native defect. Phase-folded derivative peaks cannot
prove frame-boundary defects: natural percussion and chance can produce peaks,
and the approved excerpt has larger peak/median ratios in this diagnostic.
Do not use these descriptive measures as a click detector or erase desired hits.

No cap sweep, de-clicker, filter, limiter, fade or stitch change is adopted.
BF16 and all numerical limits stay fixed. Candidate APG remains partial progress;
the user's drum audio quality is still unaccepted. This audit cannot establish a
unique root cause from waveform statistics, and no repaired audio is claimed.

## One bounded percussion control, already executed here

Piano source/reconstruction were both heard clean. That does not test VAE handling
of percussion attacks. [`listen_coreai_drum_roundtrip.py`](../../scripts/listen_coreai_drum_roundtrip.py)
uses the existing approved `acoustic-drums-30-v1/drum_result.zip` with pinned
archive/member checksums. It resamples the full32kHz PCM16 WAV with ffmpeg to
48kHz stereo Float32, crops its first10.24s, runs one pinned FP32 VAE encoder,
samples the posterior with explicit CPU seed31 and runs one decoder. No DiT,
native export or new weights. One common attenuation preserves relative level.
This is another lossy encode/decode, not the original Colab latent or CUDA parity.

[Actual CPU report](approved-roundtrip-report.json): one encoder, one decoder,
zero diffusion steps; finite input/output. Correlations approximately0.983 and
relative RMS error0.2265 are descriptive, not a musical PASS. Listening remains
**NOT EVALUATED**. No claims of hearing the files are made from plots/metrics.

The [Mac command](../../docs/coreai-drums-quality.md) creates accessible local
`01-approved-drums.wav` and `02-approved-drums-vae.wav` because previous cloud
file links could not be downloaded. Listen to both and report whether the same
transition artifact is present before re-encoding, introduced by the roundtrip,
or absent in both. A bad01 cannot establish a newly introduced codec artifact;
a clean01/bad02 prioritizes codec investigation, while two clean files prioritize
generation latents. These directions are not automatic causal proofs.
Output must be new and outside baseline/input/model folders. Missing/bad local
weights retain a failure report; there is no checkpoint download fallback.

Only script, tests and metadata are committed. User ZIP/WAV/latent, checkpoint
and plots remain local. [Validation](validation.json) distinguishes actual
codec calls from host tests. Reproduction analysis sources are
[inspect_attacks.py](inspect_attacks.py) and [replay_candidate.py](replay_candidate.py);
they expect the verified archives extracted into the documented ignored
`work/drum-apg-waveforms` and `work/drum-waveforms` folders, run from repo root.
The former additionally requires numpy/scipy/matplotlib. No Apple build or new
iPhone execution was performed in Linux.

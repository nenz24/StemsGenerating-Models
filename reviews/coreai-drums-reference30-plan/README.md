# One captured host30s reference after failed source-gain isolation

The [source-gain-only result](../coreai-drums-source-gain-result/README.md)
completed50steps with nine exact-zero control replays; hit-transition glitches
remain. Together with the failed lyric-only test and partial APG result, this
does not support another cap/gain sweep. Piano and existing Colab percussion
codec controls were heard clean. The accepted Colab30s excerpt provides a
specific reference configuration to examine before porting longer native assets.

[`generate_coreai_drum_reference.py`](../../scripts/generate_coreai_drum_reference.py)
runs **one independent original-FP32 CPU reference** with30s of the accepted
source, fixed gain0.7 applied once during32kHz PCM16 preparation, handler
resampling/clamp to48kHz, duration metadata30s, `[Instrumental]`, publisher
silence/no kit reference, guidance7,shift1,50Euler/APG steps, cap2.5,momentum−.75,
projection over time. This matches several known preview differences together;
**it is not an isolated duration-only ablation**, exact CUDA replay or a fix.

Original source MP3 and prepared PCM are checksum-gated. Fresh WAV headers may
differ, so the accepted original stereo32kHz PCM16 content hash
`1d57ce14…7baaf23` is required; this was independently recovered and verified
against the full accepted WAV SHA earlier. A supplied prepared source is copied
without a second gain. Mismatch stops before model inference, never updates the
reference. Checkpoints resolve only from pinned existing vendor/HF caches with
full size/SHA checks. Missing weights stop; there is no large download fallback.
Small verified publisher/config sources may be fetched as in existing helpers.

## Capture and limits

The script saves source48PCM, posterior mean/std, explicit source epsilon/mask,
source latent/context, text/lyric token IDs and embeddings, packed condition/mask,
silence reference, initial noise and timesteps. A fresh momentum buffer feeds
one live50-step trajectory without gold resets. The publisher's cross-attention
KV cache is retained across steps with one fresh EncoderDecoderCache per run;
the original layer explicitly disables self-attention caching. Steps1/5/10/25/50 save actual
conditional/unconditional/guided velocities, momentum and resulting latent;
all50steps record scalar statistics. Final latent/raw audio and listening WAVs
remain in the new local result folder. Invalid shapes/nonfinite values stop.
Failures/interrupts retain the report; existing outputs are never overwritten.

Original accepted CUDA intermediate tensors were not archived. CPU source seed31
and diffusion seed0 are explicit reproducible host choices, **not asserted to
reproduce CUDA tensors**. The source-gain smoke already demonstrated a same-seed
epsilon hash mismatch across environments. Torch/transformer versions and CPU
vs CUDA execution also differ and are recorded. Numerical parity is therefore
**NOT EVALUATED**, not a new PASS; prior preview `bounded_diagnostic` status and
failed full numerical gate remain. Abs1e-4/rel1e-3/nRMSE1e-3 thresholds are fixed.

No new native shapes, exports, selector, production sampler/mix, BF16 baseline,
cache or original source are changed. This is a Mac diagnostic full-model host
run, not an iPhone monolithic load. Learned checkpoint values and FP32 compute
follow the existing original-host diagnostics. A clean result would justify
further isolation/porting checks; a bad result would prioritize detailed tensor
comparison. Neither outcome alone proves duration is the unique cause.

## Input resampler and actual cloud verification

The accepted session records TorchAudio2.10.0. The pinned handler
[`io_audio.py`](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/core/generation/handler/io_audio.py)
reads PCM16 WAV as Float32 and uses `transforms.Resample(sr,48000)` defaults,
then clamp[-1,1]. The two actual TorchAudio2.10 pure-PyTorch functions are
[vendored with BSD2-Clause notice](../../scripts/coreai_reference_resample.py),
unchanged by AST comparison. The wrapper matches the original Resample class
bit-exact on the actual source, including float64 kernel generation then cached
Float32 kernel. No compiled torchaudio dependency was added or installed.
This is input preparation, not an output filtering/de-clicking repair.

[Input operator/token audit](input-operator-audit.json) records original function
AST hashes, wrapper identity and actual checksum-verified tokenization:
111text,15lyric,127packed tokens. [Frontend audit](frontend-audit.json) is real
CPU FP32 execution of exact accepted30s source preparation and **one VAE encoder**,
producing finite context[1,750,128] with full mask unchanged. Explicit actual
epsilon/outputs stay local. No Qwen/DiT/decoder inference or musical verdict is
claimed for that frontend smoke. [Reproduction source](frontend_audit.py) expects
the previously verified local source and VAE cache; no download is performed.
The learned base/Qwen weights are absent here, so the full candidate must run
on the Mac where those verified caches already exist.

[Validation](validation.json):177 host tests,175PASS,two optional skips; eight
new guards/capture tests. Tests cover pinned metadata/lyrics, impulse time/stereo
and duration through input resampling, wrong-source rejection before inference,
read-only planning, protected/existing outputs, retained failures/incomplete
execution, live50-step analytic feedback/capture, nonfinite stop and independent
cache lifetime. A [tiny attention mechanism check](cache-mechanism-audit.json)
also compares cached/uncached queries with the pinned attention class and actual
Transformers cache API, using8dim random synthetic weights, not learned model
validation. Cached outputs match at original bounds and self-cache length stays0.
Analytic
test velocities are control-flow oracles, not fake musical/model acceptance.
No Apple build/device execution was performed here. Only source/tests/docs and
metadata are committed, excluding all audio/tensors/weights/user ZIPs.

## User action

[Mac command](../../docs/coreai-drums-quality.md) opens a file chooser for the
original accepted `piano.mp3`, computes one30s/50-step CPU candidate and opens
the local folder. Compare01approved and02new host drum WAVs at the same output
and volume;03mix helps assess groove. They share attenuation. The diagnostic
mix uses Colab's0.5piano+0.5drum; production iPhone gains stay untouched.
Return `report.json` and listening feedback for02/03. No iPhone rebuild/Generate,
codec listening repeat or second source-gain trial is requested.

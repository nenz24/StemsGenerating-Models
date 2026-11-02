# Integrated piano-to-bass preview: actual export and host references

The real pretrained pipeline was executed on host using the prepared piano and
bass-conditioning bundles, followed by 50 APG-guided Euler steps and complete
5.12-second VAE decoding. [reference.json](reference.json) records the two
independent trajectories, APG comparisons, native hashes and component identities.
User audio, resulting waveform/latents and model weights remain local and ignored.

The original complete FP32 denoiser provides the gold path, with the original
publisher APG implementation (including FP64 projection). Its condition comes
from the full FP32 Qwen/ConditionEncoder reference. The independent partition
path uses FP16-stored denoiser weights with FP32 calculations, the storage16
conditioning output and a native-exportable FP32 APG/Euler update. Both use the
same real piano context and initial noise. Each trajectory feeds back its own
latent/momentum; it is not reset to gold values between steps.

The code preserves guidance 7, momentum -0.75, norm threshold 2.5, projection
along dim 1, 50 Euler steps, shift 1 and timestep-r=timestep. Projection changes
precision from the publisher's FP64 to FP32; it is not silently called identical
math. APG is compared at **identical inputs on all 50 steps**, with max absolute
error at most 1.9073486328125e-6; every comparison passes unchanged bounds.
The full independent **latent states pass on all 50 steps**. Final state vs gold:
max error 6.651878356933594e-5, normalized RMSE 7.215695499165619e-6.

Raw guided velocity comparisons across the independent trajectories are retained
as diagnostics: **22 of 50 exceed the pointwise bound at a few elements**. These
velocities have different accumulated latent/momentum inputs. They are not an
identical-input APG comparison, and their max errors do not directly measure
Euler state errors. They do not gate publication of the preview. The gates are
identical-input APG, all integrated latent states, and the decoded waveform;
the same absolute/relative/normalized-RMSE bounds remain unchanged.

The full decoded waveform passes against original FP32 trajectory + decoder:
max absolute error **9.492039680480957e-6**, normalized RMSE
**7.78707968614298e-6**, zero elements over the combined bound. Decoder
weight-norm folding is bit-exact on its actual generated-latent input.

Sixteen native assets were really exported through coreai-torch 0.4.3/coreai-core
1.0.0b3: 14 DiT shards for batch-two guidance and the real 128-frame/117-condition
shape, APG/Euler update, and full decoder. They total **3,491,592,039 bytes**.
All learned denoiser parameters occur exactly once across its partitions; the
verified learned null-condition parameter is carried by the front asset.
Old shape-specific assets are preserved, not relabeled as compatible.

The iPhone caller executes actual source encoding and full Qwen/ConditionEncoder
first, consumes their outputs, retains 15 sampling model/function assets across
50 steps, releases that Swift pool, then decodes and writes stereo Float32 WAVs.
It compares four latent checkpoints and decoded audio to both host paths: ten
checks. Numerical mismatch is explicitly reported and can still produce a finite
diagnostic WAV; that does not mark a device gate or musical baseline as passed.
Integrity, tensor, backend and memory-guard errors stop the run.

Fifty-five related host tests pass, including independent APG formula/momentum,
null updates, both-branch shape contracts, corrupted/stale bundles, nonfinite
values, schedules, preservation and earlier component/loop/setup checks. Unit
fixture native files are labeled non-executable; no fake inference is used.
The chosen RIFF/IEEE Float32 format was validated using actual decoded audio
through ffprobe and an exact FFmpeg PCM round-trip. This does not compile the
Swift writer or validate Apple playback.

**Apple build, integrated iPhone execution, measured combined memory/latency and
musical quality are pending.** This Linux workspace has no Apple SDK. Canonical
PCM/tokenization and epsilon/noise fixtures remain host-prepared; arbitrary new
recording import, free prompts, iPhone RNG, CUDA RNG equivalence, KV-cache parity,
long songs and thermal/battery behavior are unvalidated.
See [Mac/device instructions](../../docs/coreai-bass-preview.md).

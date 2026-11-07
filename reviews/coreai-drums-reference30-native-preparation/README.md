# Verified accepted capture; isolated native30s preparation

The uploaded `drums-reference30-native-review-v1.zip` is complete: 28,297,085 bytes,
45 unique members, CRC valid and all 43 Float32 tensors have the accepted report's
hashes, expected byte sizes/shapes and finite values. [Transport receipt](transport-audit.json)
records the actual checks. Raw arrays/audio remain local, excluded from this
commit. The accepted [host listening verdict](../coreai-drums-host30s-listening-accepted/README.md)
is preserved; the original report hash is `f5f84314…606bfdf`.

## Actual same-input component replay

[`audit_coreai_drum_reference30.py`](../../scripts/audit_coreai_drum_reference30.py)
verifies the entire capture before computation. [Actual CPU audit](component-audit.json)
passes 17 checks at unchanged abs 1e-4 / rel 1e-3 / nRMSE 1e-3:

- Saved mean/std/epsilon reconstruct source latent/context exactly; masks and
  final latent versus step 50 are also exact.
- Existing native APG **host implementation** matches step 1 momentum/guidance/
  Euler update. At steps 1/5/10/25/50 its FP32 projection from the actual saved
  momentum matches captured original FP64-projection guidance. Later pre-step
  latents/momenta were not saved, so this is not five complete step replays.
- One pinned original VAE source encoder, using actual saved epsilon and source48,
  matches mean/std/latent/context across Mac/cloud CPUs. Source latent nRMSE is
  `1.1918346e-5`; zero pointwise violations. The max absolute difference
  `2.53856e-4` is within the original abs+relative bound at every element.
- One full 30s VAE decoder from the saved final latent matches saved raw final
  audio: nRMSE `2.0086179e-6`, max abs `3.3676624e-5`, zero pointwise violations.

Mac and cloud seed 31 epsilon arrays differ; no random tensors were regenerated.
No DiT inference, new music generation, native export or Apple device execution
occurred in this actual component audit. These component results do not establish
full-trajectory/native parity or the unique cause of earlier drum glitches.

Reproduce with existing local captures and the pinned VAE only:

```bash
.runtime-cache/coreai-decoder-venv/bin/python scripts/audit_coreai_drum_reference30.py \
  --capture .runtime-cache/drums-reference30-v3 \
  --vae-checkpoint /path/to/existing/diffusion_pytorch_model.safetensors \
  --out .runtime-cache/drums-reference30-component-audit-new.json
```

## Geometry control and native preparation

[Geometry control source](partition_geometry_audit.py) loads checksum-verified
publisher code and compares its full model with the existing partition helpers
on **synthetic random 16-dim weights, 24 real layers**, alternating sliding/full
attention, 750 frames / 375 hidden positions / 127 condition tokens. FP32 and BF16
same-weight comparisons pass strict bounds (nRMSE about `2.2e-7`). This checks
shape/mask/branch/timestep plumbing, not learned model accuracy, original FP32
versus BF16 equivalence, memory, CoreAI conversion or music. [Measurements](partition-geometry-audit.json).

[`prepare_coreai_drum_reference30.py`](../../scripts/prepare_coreai_drum_reference30.py)
defines a separate **inactive frozen-input diagnostic**. It never registers 750
with `PreviewWindow`, changes existing 128/256 contracts or overwrites baseline
assets. Default invocation prints a plan only. `--execute` on the existing
Apple Silicon Mac environment:

1. Verifies all 43 accepted capture hashes and existing base/VAE checkpoints.
   Learned weight downloads and Qwen/source-condition regeneration are absent;
   small publisher code follows existing checksum-verified cache rules.
2. Checks one FP32 partitioned step 1 prediction against the saved original
   conditional/unconditional velocities, then the same BF16-storage/FP32-compute
   prediction. A failed strict pointwise/RMSE gate stops and retains its report;
   no new `bounded_diagnostic` exception is introduced for this export.
3. Exports 14 shape-matched DiT shards, existing FP32 APG/Euler and one folded
   FP32 full-window decoder. APG step 1 and folded decoder versus saved raw audio
   must also pass original bounds. Actual checkpoint `null_condition_emb` is
   used, not zero or a regenerated condition. DiT weights are released before
   decoder allocation. No 50-step host trajectory is rerun.
4. Copies the exact captured inputs/checkpoints into the new ignored bundle,
   records every asset/tensor hash and verifies completeness. `reference.json`
   retains `native_parity` and `full_trajectory_parity: NOT EVALUATED`, even if
   every host component gate passes. Output is outside ModelCandidates and
   no app selector, Xcode resources or production mix is changed.

The report is retained after errors/interrupts and selected in Finder on Mac.
Existing output folders are preserved. Native execution/Apple SDK is unavailable
in this Linux environment; its platform guard was exercised on the actual
verified capture, stopping before weights/SDK/export. Thus the next user action
is the [Mac export command](../../docs/coreai-drums-quality.md), followed by review
of `reference.json` **before** device integration. No finished native assets,
build, device performance or native musical acceptance are claimed here.

Future caller work must preserve the [measured lifecycle/performance rules](../../docs/coreai-bass-preview-performance.md):
bounded progress text, serial cold preparation then at most 15 sampling handles,
release before decoder, process guards and only the existing narrowly targeted
cache-repair policy. Runtime memory/timing at 750 frames require device evidence;
small capture size does not establish native footprint. A live native 50-step
trajectory must feed back its own latents/momentum, compare saved checkpoints
and final audio at unchanged gates, and then undergo listening review.
The listening comparison must use the accepted report's common gain and
0.5piano+0.5drum diagnostic mix; raw captured peaks exceed unity. The existing
production mix remains unchanged. No gold
resets or automatic native PASS from host checks.

[Validation](validation.json): 200 tests, 198 PASS,two optional skips; seven new
native contract/strict gate/transport tests. No private audio, arrays, ZIP,
checkpoints or native assets are committed. Full-song bass/drums/guitar on
iPhone remains the final goal after the longer drum path is validated.

# Four-step Euler feedback: real pretrained host references

The pinned ACE Base checkpoint was verified by full SHA-256 and its decoder
tensors loaded strictly. `generate_audio` in the checksum-verified publisher
source establishes the unshifted linear schedule, `timestep_r=timestep` and
Euler update. The reference generator validates these source statements before
using them. The scope is four steps, synthetic conditioning, guidance=1, shift=1,
without KV cache, CFG or waveform decoding; this is not the full sampling pipeline.

Two independent host trajectories start from the same existing synthetic noise,
context and condition: original full FP32 denoiser and all 14 storage16/F32-math
partitions. Each path feeds its own updated latent back into its next forward.
The shared partition factory was extracted from the existing exporter without
changing its tensor operations. Existing native assets were not re-exported.

| Step | Current → next timestep | Velocity normalized RMSE vs FP32 | Updated latent normalized RMSE vs FP32 |
| --- | --- | --- | --- |
| 1 | 1 → 0.75 | 9.98463e-6 | 3.24986e-6 |
| 2 | 0.75 → 0.5 | 9.05099e-6 | 4.50106e-6 |
| 3 | 0.5 → 0.25 | 2.02821e-5 | 6.15209e-6 |
| 4 | 0.25 → 0 | 5.77097e-6 | 6.12753e-6 |

All velocity and latent comparisons pass the unchanged elementwise combined
absolute/relative bound and normalized RMSE bound. Final latent max absolute
error is 3.552436828613281e-5. Full comparisons, package versions, checkpoint and
source identities, input hashes and all binary hashes are in
[the committed fixture manifest](../../prototypes/ios27-coreai/ModelCandidates/denoiser-loop-4/reference.json).
Sixteen Float32 binary references total 102400 bytes. Schema v2 additionally
carries the exact original three inputs (543488 bytes), for 645888 binary bytes
in total. These are activations, not model weights. All original reference
binaries and numerical results are unchanged. Both device and host now use
these fixture inputs independently of a Mac-prepared random test case.

The iPhone caller reuses the previously tested 14 native assets at each step,
constructs fresh timestep tensors and feeds back its own Float32 update. It
checks four comparisons per step: velocity and updated latent against both
partition and original-FP32 trajectories. References are for comparison only,
never replacements for live state. SHA-256 checks cover each fixture input and
reference. Native model/checkpoint/shape validation remains strict.

Tests cover analytic feedback decay, the actual retained publisher update,
wrong-sign/timestep rejection, exact binary update consistency for both real
trajectories, corruption/tolerance rejection and output preservation. Existing
bundle integrity/coverage/setup tests remain applicable. These host checks do
not emulate CoreAI or compile Swift against an Apple SDK. Four-step on-device
parity, latency and memory remain pending. The [first device attempt](../coreai-iphone17pro-denoiser-loop-manifest-failure/README.md)
failed manifest validation before any inference; its generic error cannot locate
the failed field. Schema v2 removes the unnecessary cross-platform random-input
hash equality and reports mismatched manifest fields explicitly.
See [device instructions](../../docs/coreai-denoiser-loop.md).

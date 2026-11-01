# Retained-pair inputs and dual host references

The publisher Base checkpoint was verified by full SHA-256 and all decoder
tensors loaded strictly. A copy of original FP32 layers 0+1 was preserved before
storage conversion. Exact committed v2 loop inputs at t=1/timestep_r=1 were
passed through the storage16 front. Both original-FP32 and storage16 pairs then
received the identical fixed front outputs. Native assets were not exported.

Storage16 versus original-FP32 output has max absolute error 0.0009765625,
normalized RMSE **2.095003501410023e-7**, and zero elements outside the unchanged
combined bound `1e-4 + 1e-3 * abs(reference)`. All inputs and outputs are finite.
The maximum absolute difference is not subject to a pure 1e-4 cap.

[The manifest](../../prototypes/ios27-coreai/ModelCandidates/denoiser-pair-reuse/reference.json)
records input shapes, checkpoint identity, package versions, source loop-manifest
hash, all binary hashes and comparison metrics. Inputs are full, projected pair
inputs including timestep modulation, masks and rotary embeddings; the device
does not regenerate them from Mac RNG. The eight input files and two output
references contain activations, not learned parameters, and total about 1 MB.

The iPhone diagnostic opens `blocks-00-01.aimodel` once, binds one function once,
and invokes that same function four times. Both Swift handles are explicitly
kept alive until the measurement loop ends. Noncopyable native outputs are
consumed/read back within each helper call; only CPU Float arrays return for
comparison. Every call must pass both independent host references.

Host tests cover real binary/metric consistency, finite input shapes, loop
provenance, corruption and incompatible identity/bounds rejection, and preserving
an existing destination before optional heavy imports. Those checks do not
simulate CoreAI reuse or establish device timing. Apple compilation, first versus
repeated latency, memory and all four device parity checks remain pending.
See [device instructions](../../docs/coreai-pair-reuse.md).

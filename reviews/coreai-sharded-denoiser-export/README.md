# Complete denoiser, 14 separate assets: actual Linux export

The same full publisher checkpoint SHA-256 and pinned source/config hashes were
verified. All `decoder.*` tensors load strictly. Parameter identity coverage is
checked: each of the 1,575,458,880 learned parameters belongs to exactly one
partition, without duplication or omission.

The 14 stages are front projections/time embeddings/RoPE/masks, twelve pairs
covering layers 0→23, and final normalization/output projection/cropping.
They use FP16 parameter storage with FP32 calculations and analytical buffers.
Input noise/context/condition and timesteps reproduce the previous complete
denoiser case; the final velocity shape remains [1,25,64]. Intermediate outputs
are made contiguous and passed in order, including the projected conditioning
and separate final timestep embedding. Explicit output bindings prevent using
the raw conditioning in place of its projection or an old hidden state.

Partitioned PyTorch execution versus the complete storage16 forward produced
max absolute error 6.079673767089844e-6 and normalized RMSE
1.228056488467816e-6. Against the original FP32-weight reference, max absolute
error was 5.9604644775390625e-6 and normalized RMSE 1.219390067065476e-6.
Both pass the unchanged elementwise and normalized bounds. Memory layout changes
at boundaries are not claimed bit-identical to the monolithic forward.

All 14 native exports succeeded through coreai-torch/coreai-core. Each stage is
approximately 252 MB or less, replacing the single 3.15 GB native asset.
[reference.json](reference.json) records the ordered plan, bindings, data hashes,
native artifact hashes and numerical comparisons. Mac metadata need not be
byte-identical to this Linux manifest. Large native files are not committed.

Thirty related tests passed, including stage order/wiring/coverage, artifact
integrity, non-overwriting retirement of the old resource, and setup reuse.
Independent bundle verification covered all 14 assets. The five input hashes
and original-FP32 reference hash match the preserved monolithic bundle exactly.
The Xcode project still includes the existing caller and resource directory;
none of these checks simulates an Apple SDK build or device execution.

The device caller loads one stage in a separate async scope, copies outputs to
independent CPU-owned tensors and releases its model/function/output references
before advancing. Backend caches are not assumed to evict immediately. The
2000 MiB pre-stage process-footprint guard may stop the chain if allocations
accumulate; it cannot guarantee an upper bound for compilation peaks inside a
stage. Apple compilation, all-stage iPhone completion, dual parity, actual cache
behavior and reduced peak memory remain unverified. This is still one velocity
forward, not a full conditioning/diffusion/music pipeline.

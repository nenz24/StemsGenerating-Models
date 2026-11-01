# Complete pretrained 24-layer denoiser: actual Linux export

The complete pinned Base checkpoint was reused and its full publisher SHA-256
verified. Publisher source/config digests match the repository manifest.
Only `decoder.*` tensors were loaded strictly: 476 state tensors and
1,575,458,880 learned parameters. The original publisher `AceStepDiTModel`,
base class, timestep embeddings and projection helpers execute alongside the
already verified publisher layer/mask classes and real Transformers helpers.
Unrelated conditioning/quantizer/sampling code is not imported or substituted.

Meta initialization avoids allocating random full-size weights. After strict
checkpoint loading, the nonpersistent analytical rotary buffer is rebuilt by
the publisher's Qwen3RotaryEmbedding implementation with the pinned config.
That buffer remains FP32. Only learned parameter storage becomes FP16. Linear,
Conv1d and ConvTranspose1d projections explicitly use FP32 weights/calculations;
normalization/AdaLN weights promote with FP32 activations.

The five runtime inputs are noise [1,25,64], context [1,25,128], conditioning
[1,64,2048], timestep [1] = 0.5 and timestep_r [1] = 0.25. Noise/context/condition
are deterministic synthetic values. All 24 layers, both timestep branches,
input/condition/output projections, internal masks/RoPE and odd-length padding/
cropping execute. Output velocity is [1,25,64], not a generated waveform.
KV cache, attention outputs and early exit are disabled. This is one forward,
not a CFG or diffusion loop, and conditioning does not originate from piano/text.

The original FP32 reference was finite. Storing parameters as FP16 with FP32 math
yielded max absolute error 4.947185516357422e-6, RMSE 1.3495991560352856e-6 and
normalized RMSE 1.1518127717546934e-6 against it. Every element passed the unchanged
combined bound. The prepared reference peak is 4.5208964347839355 and RMS
1.1717174407743296.

Actual export of the COMPLETE model through coreai-torch 0.4.3/coreai-core
1.0.0b3 succeeded on Linux. Native files total 3,152,952,948 bytes; the complete
bundle with input/reference data totals 3,153,512,618 bytes. Independent
`--verify-only` rehashed all files and checked binary sizes.
[reference.json](reference.json) records the Linux identities; Mac native metadata
need not be byte-identical. Large artifacts stay ignored and are not pushed.

Twenty-six related tests passed, including the FP32 analytical-buffer invariant
and FP32 calculations through FP16-stored linear/convolution projections.
A fresh two-block storage16 export through the updated helper reproduced all
input and both reference binary hashes exactly. All six older prepared component
bundles passed integrity checks. Project source/resource membership remains valid;
these checks do not replace an Apple SDK build or device inference.

Apple compilation of the updated Swift probe, CoreAI iPhone load, both parity
gates, latency and memory are pending. FP32 dense denoiser parameters alone
total 6,301,835,520 bytes before activations/workspaces. FP16 file storage does
not establish a runtime RAM reduction or that the whole pipeline fits. See
[device steps](../../docs/coreai-full-denoiser.md) and the separate
[two-block storage16 report](../coreai-iphone17pro-pretrained-dit-storage16/README.md).

**Subsequent device evidence:** monolithic loading/specialization reached
[a reported memory-resource limit](../coreai-iphone17pro-full-denoiser-memory-failure/README.md).
No completed device inference/parity was reported. The default preparation now
uses [14 ordered native assets](../coreai-sharded-denoiser-export/README.md);
the monolithic device entry point is disabled and its old resource is preserved
outside the app's copied resource tree.

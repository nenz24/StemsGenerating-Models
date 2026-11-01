# FP16 weight storage, FP32 calculations: actual export

The same pinned Base checkpoint was reused from cache and its full size and
SHA-256 verified. Publisher source/config digests were also verified. Blocks
0→1 retain the same 125,867,008 learned parameters, eight runtime inputs, 160
hidden / 64 conditioning tokens and local/full masks as the FP32 context gate.
No training or replacement weights are used.

The original FP32 reference is computed before converting parameter storage.
The activation-FP16 audit retains FP32 masks and the publisher's normalization/
softmax behavior. It yielded maximum absolute error 1.090606689453125,
RMSE 0.01545136786236177 and normalized RMSE 0.001137622587342533 against the
original FP32 result. Fraction of elements exceeding the original combined
absolute/relative bound: 0.5083038330078125. That numerical audit did not pass;
activation FP16 is not the selected export profile. Music quality is untested.

For the selected storage-only profile, parameters are FP16. Linear projections
explicitly upcast their weights/biases to FP32; normalization weights and AdaLN
tables promote with FP32 activations. Inputs, masks, calculations and output
remain FP32. This produced maximum absolute error 0.0004425048828125,
RMSE 6.212161350838299e-6 and normalized RMSE 4.573766628225064e-7 against
the original FP32 reference; all elements passed the unchanged bound. Export
is refused if this original-reference gate fails.

coreai-torch 0.4.3/coreai-core 1.0.0b3 successfully created the native asset.
Its three native files total 251,888,567 bytes versus 503,543,338 bytes for
the preserved context asset. The complete new bundle totals 256,807,909 bytes,
including the separate original-FP32 reference binary. `--verify-only` rehashed
the complete bundle and checked sizes. [reference.json](reference.json) records
the Linux hashes; Mac metadata need not be byte-identical.

All eight input hashes match the preserved FP32 context profile. The additional
`hidden-fp32-reference.f32le` hash matches that profile's original output binary.
The iPhone harness checks each result against both references with the original
tolerances. Apple compilation and both device parity gates are pending.

Smaller serialized weight storage does not establish reduced device RAM: native
specialization may materialize FP32 constants. Hardware placement, latency,
RAM savings, full-generator feasibility and musical quality remain unverified.
Follow the [Mac and device steps](../../docs/coreai-dit-storage16.md).

**Subsequent evidence:** both numerical gates
[passed on the user's iPhone](../coreai-iphone17pro-pretrained-dit-storage16/README.md).
The sampled process footprint was higher than in the separate FP32 context
report; this does not demonstrate runtime RAM savings. Exact Mac asset hashes
and source commit were not emitted, so this manifest remains Linux export evidence.

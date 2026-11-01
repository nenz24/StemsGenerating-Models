# Pretrained DiT blocks 0→1: longer-context Linux export

The pinned Base checkpoint was reused from cache, with its complete
4,787,825,604-byte size and publisher SHA-256 verified again before loading.
The same publisher source and configuration digests were verified. Only
`decoder.layers.0.*` and `decoder.layers.1.*` were loaded, strictly, into the
original publisher layer classes. Together they contain 125,867,008 parameters,
converted to CPU Float32; no training or replacement weights were used.

Block 0 uses the local bidirectional mask and block 1 the full mask. The masks
are built by the publisher's verified `create_4d_mask` function. Geometry checks
confirm distance 128 is allowed and 129 masked locally, while the full mask
allows distance 129; both share the final-key padding mask. A reverse-edge
distance of 159 is also masked locally but allowed globally. Cross-attention
masks the final conditioning key. Values are finite Float32 additive masks.

Eight runtime inputs cover hidden [1,160,2048], conditioning [1,64,2048],
timestep modulation [1,6,2048], cosine/sine [1,160,128], full/local self masks
[1,1,160,160] and cross mask [1,1,160,64]. Output is [1,160,2048]. Activations
are synthetic; there is no audio-duration claim or source-recording conditioning.
KV cache is disabled. All input tensors remain runtime arguments.

PyTorch produced a finite reference: peak 521.6171264648438, RMS
13.582156362116457. Native export with coreai-torch 0.4.3/coreai-core 1.0.0b3
succeeded. The local bundle is 507,151,138 bytes. A separate
`--profile context --verify-only` pass rehashed all artifacts and checked binary
sizes. [reference.json](reference.json) records Linux artifact identities;
Mac exports need not have byte-identical metadata.

The old short bundle also passed verification. A fresh short export through
the shared final code reproduced all seven input and reference-output binary
hashes of the preserved short baseline exactly. Profile integrity/setup tests
cover wrong profile/layer order, missing local mask and preservation of the
short/codec bundles. None of those tests simulates device inference.

Apple compilation of the updated Swift profile, iPhone parity, latency and
memory are pending. A longer two-block pass would not establish full-denoiser
feasibility, diffusion-loop correctness, music quality or an end-to-end timing
budget. Follow the [device steps](../../docs/coreai-dit-context.md).

**Subsequent evidence:** the user reported
[longer-context parity PASS on iPhone](../coreai-iphone17pro-pretrained-dit-context/README.md).
The report does not emit the exact source commit or Mac asset hashes; this
Linux manifest remains independent export evidence. The next export experiment
uses [FP16 weight storage with FP32 calculations](../coreai-pretrained-dit-storage16-export/README.md).

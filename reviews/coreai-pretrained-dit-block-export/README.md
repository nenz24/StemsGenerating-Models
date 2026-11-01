# Pretrained Base diffusion-transformer block: actual Linux export

The complete pinned Base checkpoint (4,787,825,604 bytes) was downloaded once,
then verified against publisher SHA-256
`4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b`.
Repo `ACE-Step/acestep-v15-base`, revision
`e432212fec32b8965a14ffa57ae653438d6abd14` matches the accepted experiment manifest.
Publisher model source, configuration source and config JSON also passed their
pinned size/Git-blob digest checks.

Only `decoder.layers.0.*` was loaded: 19 state tensors, 62,933,504 parameters,
strict state-dict match, converted to CPU Float32. The exporter executes the
publisher's unmodified `AceStepAttention` and `AceStepDiTLayer` class bodies
selected from the verified source AST, together with the actual Transformers
4.57.3 Qwen3 helpers. It avoids importing unrelated quantizer/sampler code;
no numerical layer or pretrained weights are replaced with a fixture.

The complete block includes grouped-key/value self-attention, cross-attention,
RoPE, RMSNorm, adaptive timestep modulation, gated residuals and the SiLU MLP.
All seven inputs remain runtime arguments: hidden [1,16,2048], conditioning
[1,8,2048], timestep modulation [1,6,2048], cosine/sine [1,16,128], and additive
self/cross masks [1,1,16,16]/[1,1,16,8]. Cache is disabled and eager attention
is selected. These synthetic activations do not originate from piano or text.
The 16-token context does not exercise sliding-window boundaries above 128.

PyTorch produced a finite Float32 [1,16,2048] reference, peak 77.36112976074219,
RMS 7.238686164152646. Native export through coreai-torch 0.4.3/coreai-core
1.0.0b3 succeeded. The prepared bundle is 252,171,730 bytes. An independent
`--verify-only` pass rehashed all prepared files and verified binary sizes.
[reference.json](reference.json) records identities and shapes of this Linux
export. Mac metadata need not be byte-identical. Large weights remain ignored.

CoreAI iPhone compilation, numerical parity, placement, latency and memory are
pending. This is not the complete denoiser, conditioning pipeline, diffusion
sampling loop or generated accompaniment. Full-model FP32 dense storage alone
is about 9.58 GB, before VAE and activations; a successful block export does not
establish full-device feasibility or final musical quality.

Follow the [Mac and device steps](../../docs/coreai-pretrained-dit-block.md).

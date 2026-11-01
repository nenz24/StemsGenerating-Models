# Pretrained VAE decoder: actual export validation

The pinned 337,431,388-byte VAE checkpoint was downloaded once and verified
against the publisher's complete SHA-256 before any weights were loaded.
Only decoder tensors were loaded, with strict state-dict matching, into the
Diffusers 0.37.0 OobleckDecoder architecture and converted to FP32.

For a seed-0 [1, 64, 5] standard-normal latent, PyTorch produced a finite
[1, 2, 9600] stereo reference: peak 0.7392531037330627 and RMS
0.1605678831736407. Removing weight-normalization reparameterization, while
preserving the actual computed weights, produced exactly the same output.
The full pretrained decoder then successfully exported through coreai-torch
0.4.3/coreai-core 1.0.0b3, producing a genuine `decoder.aimodel` with its native
hash. A second complete export using the final preparation path also passed.

The prepared local bundle totals 337,745,414 bytes in the validation run.
[reference.json](reference.json) records the publisher identity, shapes,
versions, reference data hashes and native asset file hashes. This small manifest
is evidence of the Linux export, not a requirement that a Mac export have
byte-identical metadata/debug paths. Neither large checkpoint nor exported
model is committed. The bundle sits under an ignored app resource directory;
staging happens outside that resource tree.

`--verify-only` successfully re-read and hashed the prepared bundle without a
new download or export. Integrity tests cover checksum mismatch, missing hash
coverage, invalid reference size/path and preservation of existing outputs.

**Not verified:** compilation of the new Swift decoder probe with the Apple
SDK, model specialization/inference on iPhone, decoder parity across CoreAI and
PyTorch, physical memory peak, latency, accelerator placement or musical quality.
Seeded latent output is for numerical testing; it is not generated accompaniment.
The first iPhone gate uses CPU-only and a 0.2-second window. The affine device
success remains separate evidence.

User steps: [Mac preparation and iPhone probe](../../docs/coreai-pretrained-decoder.md).

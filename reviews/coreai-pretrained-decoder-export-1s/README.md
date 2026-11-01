# One-second pretrained decoder: Linux export

The same pinned VAE checkpoint was reused from the local cache and verified
against its full publisher SHA-256. Only pretrained decoder tensors were loaded
as FP32. A seed-0 standard-normal latent [1, 64, 25] produced finite PyTorch
output [1, 2, 48000] (1 s, stereo, 48 kHz). Folding weight normalization left
that reference exactly unchanged. The output peak is 1.0829030275344849 and
RMS 0.2406689003545956; the raw numerical reference is not clipped or normalized.
This latent is not an input recording or musically meaningful accompaniment.

Export with coreai-torch 0.4.3/coreai-core 1.0.0b3 produced a native
`decoder.aimodel`. A separate `--frames 25 --verify-only` pass checked all prepared
file hashes and binary sizes. [reference.json](reference.json) records the Linux
artifact identities; Mac exports need not have byte-identical metadata.

The ignored local folder is
`prototypes/ios27-coreai/ModelCandidates/decoder-pretrained-1s/`.
The existing 0.2 s bundle also passed integrity verification and is preserved.
Checkpoint and exported weights are not committed.

This is export evidence only. At export time, Apple compilation of the updated profile,
CoreAI device parity, timings and memory for this one-second window were pending.
The [0.2 s device result](../coreai-iphone17pro-pretrained-decoder-02s/README.md)
does not establish these. Follow the [Mac/device steps](../../docs/coreai-pretrained-decoder.md).

**Subsequent evidence:** the user has now reported successful one-second
device parity and timings in the
[one-second iPhone report](../coreai-iphone17pro-pretrained-decoder-1s/README.md).
That device evidence does not contain the Mac asset hashes; this Linux manifest
remains independent export evidence.

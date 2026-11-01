# Pretrained decoder: user-reported iPhone parity

The user supplied [this device report](report.txt) after running the native
pretrained ACE VAE decoder probe. The device was previously identified by the
user as iPhone 17 Pro; this report itself identifies iOS 27.0.1 (24A446), CoreAI
architecture h18p and an iOS device build. It was not reproduced in Linux.
[Structured measurements](report.json) preserve that provenance.

Native asset validation, model load and all three numerical runs passed for a
seeded Float32 latent [1, 64, 5] → stereo [1, 2, 9600] at 48 kHz (0.2 s).
Maximum absolute error was 4.902482032775879e-6; normalized RMSE was
4.677160889274361e-6, below the unchanged 1e-3 limit. The reference binaries
passed local SHA-256 checks; the Mac preparation had verified the full publisher
checkpoint. The report does not contain the actual local artifact hashes or
source commit, so Linux export manifests are not asserted to identify this
device's package byte-for-byte.

Load/specialization took 859.562375 ms, first inference plus read-back
51.658833 ms, and the two warm observations 33.858416 and 33.897625 ms
(arithmetic mean 33.8780205 ms). Memory baseline was 17.751754760742188 MiB
and sampled maximum was 792.3152618408203 MiB. This is whole-process footprint
sampled every 10 ms, not exact peak or isolated model allocation. Cache state,
thermal state and actual accelerator placement were not measured. CPU-only was
the requested policy.

This establishes one pretrained decoder's numerical parity for one repeated
input, not full VAE coverage, musical quality, accompaniment generation or an
end-to-end real-time claim. Encoder, diffusion generator and conditioning remain
separate work. At the time of this result, the next device gate was the independently prepared
[one-second profile](../../docs/coreai-pretrained-decoder.md); the 0.2 s package
is retained. The profile's Swift changes were not compiled with the Apple SDK
in this Linux workspace.

The user subsequently reported
[one-second decoder parity PASS](../coreai-iphone17pro-pretrained-decoder-1s/README.md).
The next component is the [encoder](../../docs/coreai-pretrained-encoder.md).

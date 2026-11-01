# Pretrained DiT block 0: user-reported iPhone result

[report.txt](report.txt) is the user's pasted device output;
[report.json](report.json) preserves measurements and provenance. The user
previously identified the device as iPhone 17 Pro. This report identifies iOS
27.0.1 (24A446), architecture h18p and an iOS device build. Source commit and
local Mac asset hashes are not emitted. This run was not reproduced in Linux.

Native asset validation, model load and all three numerical runs passed for
one pretrained Base DiT block, `decoder.layers.0.`. The seven Float32 input
descriptors match the prepared 16-hidden-token / 8-condition-token contract;
output is [1,16,2048]. Inputs are fixed synthetic activations, RoPE and masks,
not a piano recording or a full conditioning pipeline.

Maximum absolute error was 0.000179290771484375, RMSE
1.2832379143245418e-5 and normalized RMSE 1.7727500149824247e-6.
The unchanged elementwise rule is `abs(error) <= 1e-4 + 1e-3 * abs(reference)`,
plus normalized RMSE <= 1e-3. A maximum error above the standalone absolute
term can therefore still pass the combined rule; it is not an absolute-only
test. The device harness reported PASS for all three runs.

Load/specialization took 322.957625 ms. First run plus read-back took
18.869667 ms, followed by 4.377792 and 4.637042 ms (mean 4.507417 ms).
Process baseline was 20.36115264892578 MiB and sampled maximum
168.92408752441406 MiB at 10 ms intervals; total probe time was 406.333292 ms.
The footprint is not exact peak, isolated model storage or whole-device usage.
Cache, residency, sharing, thermal state and actual accelerator placement were
not measured. CPU-only was requested. Two warm observations do not establish
sustained performance or full-model scaling.

This gate does not cover a sequence longer than the local attention radius,
the next full-attention block, the full denoiser or diffusion sampling. The
next profile chains blocks 0 and 1 with 160 hidden / 64 conditioning tokens,
using the publisher's distinct local/full masks. It retains this short baseline.
See the [context-profile steps](../../docs/coreai-dit-context.md).

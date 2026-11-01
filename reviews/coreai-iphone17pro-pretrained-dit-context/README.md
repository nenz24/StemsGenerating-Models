# Two pretrained DiT blocks, longer context: iPhone result

The user supplied [report.txt](report.txt). [report.json](report.json) preserves
the values and provenance. The user previously identified iPhone 17 Pro; the
report itself identifies iOS 27.0.1 (24A446), h18p and an iOS device build.
Exact source commit and Mac artifact hashes were not emitted. This result was
not independently reproduced in Linux.

Asset validation, model load and three numerical runs passed for blocks
0 (local radius 128) → 1 (full), using 160 hidden / 64 condition tokens.
All eight Float32 descriptors match the contract, including the distinct local
mask; output is [1,160,2048]. Activations are fixed and synthetic, with no audio
duration claim, source-recording conditioning or diffusion sampling loop.

Maximum absolute error was 0.001678466796875, RMSE 2.895436184720318e-5 and
normalized RMSE 2.1317941668269713e-6. The original combined elementwise bound
`abs(error) <= 1e-4 + 1e-3 * abs(reference)` and normalized RMSE <= 1e-3 apply.
The absolute term alone is not the pass rule. The harness reports PASS for all
three repeated runs; the report does not contain output tensors for independent
elementwise recomputation.

Load/specialization took 571.069792 ms; first run plus read-back 60.230333 ms.
Warm observations were 31.217875 and 30.991417 ms (mean 31.104646 ms).
Total probe time was 960.758667 ms. Whole-process footprint baseline was
21.673629760742188 MiB and sampled maximum 174.11193084716797 MiB at 10 ms
intervals. This is not exact peak, isolated model allocation or whole-device
usage; the prepared native resource remains about 507 MB. Cache/residency,
thermal state and actual accelerator placement were not measured. CPU-only
was requested. These two warm observations do not predict full-stack latency.

The longer local/full attention gate passed. All 24 layers, complete velocity
prediction, conditioning, sampling and final music still need separate work.
The next storage experiment retains FP32 calculations while storing weights in
FP16; activation FP16 is audited separately rather than silently adopted.

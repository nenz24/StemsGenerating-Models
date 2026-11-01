# Pretrained encoder: one-second iPhone result

The user supplied [report.txt](report.txt); [report.json](report.json) preserves
the values and provenance. The device was previously identified by the user as
iPhone 17 Pro. This report identifies iOS 27.0.1 (24A446), architecture h18p and
an iOS device build. Exact source commit and Mac asset hashes were not emitted;
the run was not independently reproduced in Linux.

The native pretrained encoder validated, loaded and passed all three numerical
runs for Float32 [1, 2, 48000] → [1, 64, 25]. This is one fixed synthetic
stereo input and the deterministic posterior mean, not the full stochastic
encoder contract, user-recording quality or generated accompaniment. Local
reference binaries passed SHA-256 checks; checkpoint verification was performed
by Mac preparation. The separate Linux export manifest need not identify the
device's exported asset byte-for-byte.

Maximum absolute error was 6.597861647605896e-5, RMSE 1.197700916225508e-5 and
normalized RMSE 1.169841896856174e-5. The original tolerances were unchanged.
Load/specialization took 424.414375 ms, first inference plus read-back
149.549792 ms, and the two warm observations 129.006 and 133.645917 ms
(mean 131.3259585 ms). Total probe time was 875.614958 ms.

The process footprint baseline was 20.43927764892578 MiB and sampled maximum
97.0020751953125 MiB, using 10 ms sampling. This is not an exact peak or model
allocation and must not be described as a 97 MiB encoder: its separately
exported FP32 resource is about 338 MB. Cache state, memory residency, mapping
and sharing were not measured, so the much smaller footprint cannot establish
the full deployment memory budget or explain its difference from decoder runs.
CPU-only was requested; actual accelerator placement was not measured.

Together with [decoder parity](../coreai-iphone17pro-pretrained-decoder-1s/README.md),
this establishes separate codec-component gates. The two latency means cannot
be presented as measured end-to-end performance: their inputs, process runs and
cache states differ. The remaining generative path includes conditioning,
diffusion-transformer blocks and sampling; music quality is still a separate gate.

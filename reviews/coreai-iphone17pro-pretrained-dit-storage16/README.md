# FP16 stored weights / FP32 calculations: iPhone result

The user supplied [report.txt](report.txt); [report.json](report.json) preserves
the values and provenance. The user previously identified iPhone 17 Pro; the
report itself identifies iOS 27.0.1 (24A446), h18p and an iOS device build.
Source commit and exact Mac asset hashes were not emitted. This device run
was not independently reproduced in Linux.

Native validation, load and all three runs passed **both** unchanged numerical
gates: against the storage16 PyTorch reference and the original FP32 baseline.
The same eight Float32 inputs, local/full blocks 0→1 and 160/64-token context
were used. These are fixed synthetic activations, not instrument generation.

Against the storage16 reference, maximum absolute error was 0.0013427734375,
RMSE 2.8891748854000516e-5 and normalized RMSE 2.127184248442011e-6. Against
the original FP32 baseline, they were 0.001922607421875,
2.8958450833713256e-5 and 2.1320952225931604e-6. The combined elementwise
absolute/relative bound still applies; these maxima are not absolute-only tests.

| Two-block profile | Load ms | First run + read-back ms | Mean of two warm runs ms | Sampled process maximum MiB |
| --- | ---: | ---: | ---: | ---: |
| [FP32 context](../coreai-iphone17pro-pretrained-dit-context/README.md) | 571.07 | 60.23 | 31.10 | 174.11 |
| FP16 storage / FP32 math | 718.09 | 143.19 | 32.46 | 638.17 |

Native file storage was approximately halved in the export evidence. These
separate device reports do **not** demonstrate reduced RAM or faster inference:
the sampled process maximum was higher for this storage16 run. Cache/residency,
compiler allocation, thermal state and actual hardware placement were not
measured, so the reports do not isolate a cause or establish a controlled
performance regression. CPU-only was requested. FP32 upcasts may materialize
constants, but that mechanism has not been profiled on the device.

The storage16 process baseline was 16.361083984375 MiB and total probe time
1380.613167 ms. Footprint sampling is every 10 ms for the whole process; it is
not exact peak, isolated model allocation or whole-device memory. Neither
174 MiB nor 638 MiB predicts whether the full denoiser will fit.

The next gate is a [complete 24-layer velocity forward](../../docs/coreai-full-denoiser.md),
including input/time/condition/output projections, with a short synthetic latent
window. Smaller serialized weights are used for file size, without claiming
runtime RAM savings. Full conditioning and the diffusion sampling loop remain
separate work.

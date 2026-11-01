# Pretrained decoder: one-second iPhone result

The user supplied [report.txt](report.txt) from an iOS device build after the
one-second profile was pushed. [report.json](report.json) preserves the measured
values and provenance. The device was previously identified by the user as
iPhone 17 Pro; the emitted report identifies iOS 27.0.1 (24A446) and h18p.
The exact source commit and local asset hashes are not emitted, and this device
run was not independently reproduced in Linux.

Native asset validation, load and three numerical runs passed. Float32
[1, 64, 25] → [1, 2, 48000] corresponds to one second of stereo audio at 48 kHz.
Maximum absolute error was 7.502734661102295e-6 and normalized RMSE
3.1238491721560804e-6. The same unchanged tolerances apply as in the shorter test.
The report confirms local reference-binary SHA-256 checks and checkpoint
verification by Mac preparation; it does not identify the Mac asset as
byte-identical to the separately recorded Linux export.

| Decoder profile | Load/specialization ms | First run + read-back ms | Mean of two warm runs ms | Sampled process maximum MiB |
| --- | ---: | ---: | ---: | ---: |
| [0.2 s](../coreai-iphone17pro-pretrained-decoder-02s/README.md) | 859.56 | 51.66 | 33.88 | 792.32 |
| 1 s | 754.26 | 178.70 | 159.77 | 849.27 |

The one-second process baseline was 19.75177764892578 MiB; total probe time was
1329.297208 ms. These are separate reports with unmeasured cache/thermal states
and two warm observations each, not a controlled scaling benchmark. The 10 ms
sampled whole-process maximum is not exact peak or isolated model memory.
CPU-only was requested; actual accelerator placement was not measured.

This validates the pretrained decoder on one repeated seeded latent, not full
VAE coverage or accompaniment generation. The next component is the
[pretrained encoder](../../docs/coreai-pretrained-encoder.md), with its own input,
reference and device gate. Both decoder profiles remain available.

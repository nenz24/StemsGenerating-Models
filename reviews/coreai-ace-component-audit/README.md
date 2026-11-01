# ACE-Step audio component audit for CoreAI

This audit uses the pinned ACE-Step commit and publisher model manifests from
the accepted CUDA experiments. [Machine-readable results](report.json) were
produced by `scripts/audit_coreai_audio.py`. No full checkpoints were downloaded
or loaded. VAE configuration is verified against the publisher Git-blob digest;
bounded safetensors prefixes establish shapes/dtypes at pinned revisions, not
full-file checksum verification. Prefix reads may contain partial weight bytes.

## Architecture and precision

| Component | Element/parameter count | Checkpoint precision | Dense FP32 storage estimate |
| --- | ---: | --- | ---: |
| Base generator checkpoint tensors | 2,393,872,518 | BF16 | 9,575,490,072 bytes |
| VAE total | 168,695,426 | BF16 | 674,781,704 bytes |
| VAE encoder | 84,281,344 | Part of VAE | 337,125,376 bytes |
| VAE decoder | 84,414,082 | Part of VAE | 337,656,328 bytes |

Header inventories count serialized tensor elements, which may include buffers;
VAE counts independently match the instantiated meta-device architecture.
Dense storage is not device peak RAM. Conversion precision, runtime workspaces,
weights folded during conversion, text conditioning and live activations affect
the real budget. Full FP32 Base+VAE alone totals about 10.25 GB of dense state,
before those other allocations. The 13.86 GiB download manifest also contains
other components (including Turbo and a language model); it is not a RAM estimate
or a claim that every file is needed simultaneously for this generation path.

The selected VAE is Diffusers AutoencoderOobleck: stereo 48 kHz, 64 latent
channels, ratios [2, 4, 4, 6, 10], combined factor 1920. This means 25 VAE frames
per second; do not confuse it with the separate 5 Hz language/code sequence.
The full encoder posterior includes a mean and scale and samples noise. A future
conversion must preserve that contract or accept explicit noise for repeatable
comparison, not silently replace sampling with the mean.

## Memory implications for window size

Meta-device shape propagation gives decoder [1, 64, 25] -> [1, 2, 48000] for one
second, and a largest logical feature tensor of [1, 128, 48000]. Scaling that
unfused tensor's FP32 storage gives:

| Audio window | One logical feature tensor |
| --- | ---: |
| 0.2 s | 4.9152 MB |
| 1 s | 24.576 MB |
| 10 s | 245.76 MB |
| 90 s | 2,211.84 MB |

These are logical tensor sizes, not measured iPhone peaks or latency. Fusion may
avoid materializing some intermediates; multiple live tensors/workspaces may
increase memory. The 90-second musical generation baseline is not a requirement
to decode 90 seconds in one VAE call. Short codec windows need enough context
and separate seam/parity checks; changing codec windows does not solve generator
noise or musical continuity automatically.

## Actual native export probes

Four synthetic OobleckDecoderBlock probes successfully traversed
`torch.export` -> decomposition -> `TorchConverter` -> native `.aimodel` export:

- Channels 128, stride 2: actual final-stage width, 462,336 synthetic parameters.
- Channels 16, strides 4, 6 and 10: reduced-channel probes of the other stride patterns.

They include Snake exp/sin/power/reciprocal operations, weight-normalization
lowering, Conv1d, ConvTranspose1d and residual units with dilations 1/3/9.
Reported ATen names are after decomposition (transposed and regular convolution
share `aten.convolution.default` with different arguments).

This demonstrates export success for those synthetic shapes and patterns. It
does not demonstrate pretrained codec quality, full-width first-block support,
FP16/BF16 numerical parity, accelerator placement or device execution. Generated
synthetic assets are retained in the ignored audit output, not shipped as music
models or substituted for the approved sound baseline.

## Selected next gate

Start with the **pretrained VAE decoder**, using a fixed 0.2–1 second latent
window and an identical PyTorch/CoreAI input. Download/verify only the pinned
VAE checkpoint (337,431,388 bytes, publisher SHA-256 in the manifest), load the
real weights, and verify export/outputs before device profiling. No new training
is required for that gate. Measure load, first/warm execution, peak RAM and
numerical error; then test longer codec windows and boundaries.

Only after a representative pretrained audio component passes should we attempt
the generator's attention/conditioning/diffusion path and a short single-role
generation. Quantization or distillation may be needed, but neither its benefit
nor preservation of the accepted playing feel has been verified. The tiny affine
iPhone timings cannot predict any of these audio costs.

## Reproduce

Use a separate environment with `requirements-coreai-audit.txt`, then:

```bash
python scripts/audit_coreai_audio.py --out reports/coreai-audio-audit-new --fetch-headers
```

`--fetch-headers` performs bounded HTTPS prefix reads, not complete weight
downloads. Omit it for offline architecture/export probes. An existing nonempty
output directory is refused so previous evidence/assets are preserved. Versions
used here: Torch 2.8.0+cpu, Diffusers 0.37.0, coreai-torch 0.4.3, coreai-core
1.0.0b3. Diffusers emits a weight_norm deprecation warning; it is not an export
failure, and this audit does not rewrite the architecture to suppress it.
